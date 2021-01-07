from typing import List, Iterable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from exp2.classifier import Classifier
from exp2.controller import Controller, create_controller
from exp2.data import create_loader, PartialDataset
from exp2.feature_extractor import create_models
from exp2.lr_scheduler import get_classifier_lr_scheduler, get_controller_lr_scheduler
from exp2.memory import Memory
from exp2.model_state import ModelState, init_states
from exp2.predictor import Predictor, ByCtrl, FilteredController
from exp2.reporter import SourceReporter, ControllerReporter, create_test_reporter, ClassifierReporter
from exp2.reporter_strings import CTRL_EPOCH, CLF_EPOCH, CTRL_LR, CLF_LR
from exp2.utils import train_test_split
from logger import Logger
from utils import get_default_device, TrainingStopper

DEVICE = get_default_device()


def get_device():
    return DEVICE


def get_class_weights(config, newset, otherset=None):
    """Return class weights for training a new classifier.

    If otherset is None, it will return equal weight for all classes. Otherwise it will return weights
    according to the config, for all classes and "other"(at the last index).

    Args:
        newset (PartialDataset): the dataset with samples of new classes
        otherset (PartialDataset, optional): the dataset with samples of old classes, all of which will be considered
            as one class, namely "other".
    """

    n_new_classes = len(newset.classes)
    if otherset is None:
        # return equal weights
        weight = torch.ones(n_new_classes)
    elif len(otherset) == 0:
        weight = torch.tensor([1.] * n_new_classes + [0.])
    else:
        otherset_size = len(otherset)
        newset_size = len(newset)
        n_other_classes = len(otherset.classes)
        r = 1

        # account for otherset size
        if config.balance_other_samplesize:
            r *= otherset_size / newset_size

        # account for num of classes
        if config.balance_other_classsize:
            r *= n_new_classes / n_other_classes

        weight = [1.] * n_new_classes + [1 / r]
        weight = torch.tensor(weight)

    # normalize weights
    weight = weight / weight.sum()
    return weight


def get_classification_criterion(config, newset, otherset, device):
    """Create classification criterion given the new classes samples (newset) and old class samples (otherset)."""
    weight = get_class_weights(config, newset, otherset)
    weight = weight.to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    return criterion


class Model:
    predictors: List[Predictor] = [ByCtrl(), FilteredController()]
    controller = None

    def __init__(self, config, logger: Logger):
        self.logger = logger
        self.device = get_device()
        self.classifiers: List[Classifier] = []
        self.feature_extractor, self._classifier_constructor = create_models(config, device=self.device)
        self.logger.log({'feature_extractor': str(self.feature_extractor.net)}, commit=True)
        self.config = config
        self._classes = []

    def phase_start(self, phase):
        self.phase = phase

    def phase_end(self):
        pass

    @property
    def classes(self):
        return self._classes

    def _create_classifier_optimizer(self, classifier):
        config = self.config
        return optim.SGD(classifier.parameters(), lr=config.clf['lr'], momentum=0.9, weight_decay=config.weight_decay)

    def _create_classifier(self, classes) -> Classifier:
        """Create a new classifier and add it. Then return it."""
        idx = len(self.classifiers)
        classifier = self._classifier_constructor(classes, idx=idx)
        # add new class labels to classifiers mapping
        self._classes.extend(classes)
        self.classifiers.append(classifier)
        return classifier

    def _set_controller_train(self, train):
        if self.controller is not None:
            self.controller.train(train)

    def _set_classifiers_train(self, train):
        for clf in self.classifiers:
            clf.train(train)

    def set_train(self, train=False):
        """Set the model and its parts into training or evaluation mode."""
        self._set_controller_train(train)
        self._set_classifiers_train(train)

    def _train_new_classifier_start(self):
        pass

    def _train_new_classifier_end(self):
        pass

    def _get_ctrl_idx(self):
        """Return current controller's index. If it doesnt exists yet, return None.
        The controller's index signifies the phase it has been created and trained. So if the
        index is 2, this controller can predict between the first two classifiers.
        """
        if self.controller is None:
            return None
        return self.controller.idx

    def _create_new_controller(self) -> Controller:
        """Create and assign the new controller.
        The index assigned to the controller will be the current phase."""
        idx = self.phase
        controller = create_controller(self.config, idx, self.classifiers, device=self.device)
        self.controller = controller
        return controller

    def _train_controller_epoch(self, loader, optimizer, epoch, reporter: SourceReporter):
        """Train the controller. Only train the controller. Classifiers are kept frozen.
        """
        # set everything in eval mode except for the controller
        self.set_train(False)
        self.controller.train()

        for mstate in self.feature_extractor.feed(loader):
            ctrl_state = self.controller.feed(state=mstate)
            outputs, labels = ctrl_state.outputs, ctrl_state.parent.labels
            loss = self.controller.compute_loss(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ctrl_state.loss = loss
            ctrl_state.epoch = epoch
            reporter.update(mstate)

    @torch.no_grad()
    def _val_controller(self, loader, epoch, reporter: SourceReporter):
        self.set_train(False)
        for mstate in self.feature_extractor.feed(loader):
            ctrl_state = self.controller.feed(state=mstate)
            ctrl_state.epoch = epoch
            ctrl_outs = ctrl_state.outputs
            labels = mstate.labels
            ctrl_state.loss = self.controller.compute_loss(ctrl_outs, labels)
            reporter.update(mstate)

    def _train_a_new_controller_start(self):
        pass

    def _train_a_new_controller_epoch_start(self, epoch, lr_scheduler):
        self.logger.log({CTRL_EPOCH: epoch})
        lr = lr_scheduler.get_last_lr()[0]
        self.logger.log({CTRL_LR: lr})

    def _train_a_new_controller_epoch_end(self, epoch):
        self.logger.commit()

    def _train_a_new_controller_end(self):
        pass

    def train_a_new_controller(self, dataset):
        """Create a new controller and train it. It will replace the current controller with the new one."""
        config = self.config
        self._create_new_controller()
        optimizer = self.controller.get_optimizer()
        train_loader, val_loader = train_test_split(config, dataset, config.val_size)
        stopper = TrainingStopper(config.ctrl)
        lr_scheduler = get_controller_lr_scheduler(config, optimizer)

        self._train_a_new_controller_start()
        for epoch in range(1, config.ctrl['epochs'] + 1):
            if stopper.do_stop():
                break
            self._train_a_new_controller_epoch_start(epoch, lr_scheduler)
            source_reporter = SourceReporter()
            train_reporter = ControllerReporter(self.logger, source_reporter, self.controller, 'train')
            self._train_controller_epoch(train_loader, optimizer, epoch, source_reporter)
            source_reporter.end()
            if val_loader:
                source_reporter = SourceReporter()
                validation_reporter = ControllerReporter(self.logger, source_reporter, self.controller, 'validation')
                self._val_controller(val_loader, epoch, source_reporter)
                source_reporter.end()
                loss = validation_reporter.get_average_loss()
                lr_scheduler.step(loss)
                stopper.update(loss)
            self._train_a_new_controller_epoch_end(epoch)
        self._train_a_new_controller_end()

    def _test_start(self):
        pass

    def _test_end(self):
        self.logger.commit()

    @torch.no_grad()
    def test(self, dataset):
        """
        Test the model in whole and in parts.

        Args:
            dataset (PartialDataset): cumulative testset, containing all seen classes
        """
        self.set_train(False)
        source_reporter = SourceReporter()
        create_test_reporter(self.logger, source_reporter, self.classifiers, self.controller, self.classes)
        loader = create_loader(self.config, dataset)

        for mstate in self._feed_everything(loader):
            source_reporter.update(mstate)

        source_reporter.end()
        self._test_end()

    def _train_classifier_epoch(self, classifier: Classifier, loader, criterion, optimizer, epoch,
                                reporter: SourceReporter):
        """Train classifier for one epoch."""
        self.set_train(True)
        for mstate in self.feature_extractor.feed(loader):
            clf_state = classifier.feed(state=mstate)
            outputs, labels, labels_np = clf_state.outputs, clf_state.labels, clf_state.parent.labels_np
            loss = classifier.get_loss(outputs, labels, criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            clf_state.loss = loss
            clf_state.epoch = epoch
            reporter.update(clf_state)

    def _val_classifier(self, classifier: Classifier, loader, criterion, reporter: SourceReporter):
        for mstate in self.feature_extractor.feed(loader):
            clf_state = classifier.feed(state=mstate)
            outputs, labels, labels_np = clf_state.outputs, clf_state.labels, clf_state.labels_np
            loss = classifier.get_loss(outputs, labels, criterion)
            clf_state.loss = loss
            mstate = clf_state.parent
            reporter.update(mstate)

    def _train_classifier_epoch_start(self, epoch, lr_scheduler):
        self.logger.log({CLF_EPOCH: epoch})
        lr = lr_scheduler.get_last_lr()[0]
        self.logger.log({CLF_LR: lr})

    def _train_classifier_epoch_end(self, epoch):
        self.logger.commit()

    def _train_classifier(self, classifier, train_loader, val_loader, criterion, optimizer):
        """Train the classifier for a number of epochs."""
        config = self.config
        stopper = TrainingStopper(config.clf)
        lr_scheduler = get_classifier_lr_scheduler(config, optimizer)

        for epoch in range(1, config.clf['epochs'] + 1):
            if stopper.do_stop():
                break
            self._train_classifier_epoch_start(epoch, lr_scheduler)
            source_reporter = SourceReporter()
            ClassifierReporter(self.logger, source_reporter, classifier, 'train')
            self._train_classifier_epoch(classifier, train_loader, criterion, optimizer, epoch, source_reporter)
            source_reporter.end()
            if val_loader is not None:
                source_reporter = SourceReporter()
                val_reporter = ClassifierReporter(self.logger, source_reporter, classifier, 'validation')
                self._val_classifier(classifier, val_loader, criterion, source_reporter)
                source_reporter.end()
                loss = val_reporter.get_average_loss()
                stopper.update(loss)
                lr_scheduler.step(loss)
            self._train_classifier_epoch_end(epoch)

    def train_new_classifier(self, newset: PartialDataset, ctrl_memory: Memory, clf_memory: Memory):
        """Train a new classifier on the dataset. Optionally given otherset that contains
        examples from unknown classes.

        Algorithm:
            1. if training with other:
                newset_tr, newset_val = slit(newset)
                if shared memory (ctrl_memory is clf_memory):
                    otherset_tr, otherset_val = split(ctrl_memory)  # (or clf_memory, doesn't matter)
                else:
                    otherset_tr = clf_memory
                    otherset_val = ctrl_memory

            2. train with newset_tr + otherset_tr
                and validate with newset_val + otherset_val

        Args:
            ctrl_memory: memory storage of controller
            clf_memory: memory storage of classifiers
            newset (PartialDataset): dataset of new class samples
        """
        new_classes = newset.classes
        val_size = self.config.val_size
        # split new dataset into train and validation
        newset_tr, newset_val = newset.split(val_size)

        # make up otherset, dataset consisting of "other" categories
        otherset_tr, otherset_val = None, None
        if self.config.other:
            # if share same memory:
            if ctrl_memory is clf_memory:  # TODO: imho can still overfit if classifiers update
                otherset_tr, otherset_val = clf_memory.get_dataset().split(test_size=val_size)
            else:  # if separate memories, ctrl_memory becomes validation set
                otherset_tr = clf_memory.get_dataset()
                otherset_val = ctrl_memory.get_dataset()

        # create the new classifier, prepared for new classes
        classifier = self._create_classifier(new_classes)

        # define criterion; includes class-balancing logic
        criterion = get_classification_criterion(self.config, newset_tr, otherset_tr, self.device)

        # create optimizer
        optimizer = self._create_classifier_optimizer(classifier)

        # create final training set, combination of new and other categories
        trainset = newset_tr.mix(otherset_tr) if otherset_tr else newset_tr
        train_loader = create_loader(self.config, trainset)

        # create final validation set, combination of new and other categories
        valset = newset_tr.mix(otherset_val) if otherset_val else newset_val
        val_loader = create_loader(self.config, valset)

        # start training
        self._train_new_classifier_start()
        self._train_classifier(classifier, train_loader, val_loader, criterion, optimizer)
        self._train_new_classifier_end()

    @torch.no_grad()
    def _feed_everything(self, loader: DataLoader) -> Iterable[ModelState]:
        """Feed everything (controller, clasifiers) and return generator of ModelState"""
        states = init_states(loader, self.device)
        for state in states:
            state = self.feature_extractor.feed(state=state)
            self.controller.feed(state=state)
            for clf in self.classifiers:
                clf.feed(state=state)
            yield state
