import itertools
from typing import List

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn

from exp2.classifier import Classifier
from exp2.controller import GrowingController
from exp2.data import create_loader, PartialDataset
from exp2.lr_scheduler import _get_lr_scheduler
from exp2.memory import MemoryManagerBasic
from exp2.model_state import init_states
from exp2.models.splitting import create_models, log_architectures
from logger import Logger, get_accuracy
from utils import get_default_device, TrainingStopper, AverageMeter

DEVICE = get_default_device()


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
    # if batch_memory_samples are specified positive integer, then we don't apply class weights, we go with dual
    # dataloader, that is to include a certain number of memory samples in each batch
    if config.batch_memory_samples > 0:
        criterion = nn.CrossEntropyLoss()
    else:
        weight = get_class_weights(config, newset, otherset)
        weight = weight.to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)
    return criterion


class CIModelBase:
    phase: int

    def __init__(self, config, logger: Logger):
        self.logger = logger
        self.device = DEVICE
        self.feature_extractor, self._classifier_constructor = create_models(config, device=self.device)
        log_architectures(config, self.feature_extractor, self._classifier_constructor, self.device)
        self.config = config
        self.classifiers: List[Classifier] = []

    def phase_start(self, phase):
        self.phase = phase

    def phase_end(self):
        pass

    def _create_classifier(self, classes) -> Classifier:
        """Create a new classifier and add it. Then return it."""
        idx = self.phase - 1
        classifier = self._classifier_constructor(classes, idx=idx)
        # add new class labels to classifiers mapping
        self.classifiers.append(classifier)
        return classifier

    def set_classifiers_train(self, train):
        for clf in self.classifiers:
            clf.train(train)


class JointModel(CIModelBase):

    def __init__(self, config, train_source, train_transform, test_transform, logger: Logger):
        # establish memory
        super().__init__(config, logger)
        mem_tr_size = int(config.memory_size * (1 - config.val_size))
        mem_val_size = config.memory_size - mem_tr_size
        sizes = [mem_tr_size, mem_val_size]
        memory_manager = MemoryManagerBasic(sizes=sizes, source=train_source, train_transform=train_transform,
                                            test_transform=test_transform, names=('train', 'val'))
        self.train_memory, self.val_memory = memory_manager['train'], memory_manager['val']
        self.memory_manager = memory_manager
        self.controller = GrowingController(config)
        self.controller.to(self.device)

    def on_receive_phase_data(self, trainset):
        self._introduce_new_classes(trainset.classes)
        config = self.config

        # prepare train and validation sets
        if self.phase > 1:
            memory_valset = self.val_memory.get_dataset(train=False)
            memory_trainset = self.train_memory.get_dataset(train=True)
        else:
            memory_valset = None
            memory_trainset = None

        trainset_train, trainset_val = trainset.split(test_size=config.val_size)
        train_loader = create_loader(config, main_dataset=trainset_train, memoryset=memory_trainset, shuffle=True)
        val_loader = create_loader(config, trainset_val, memory_valset)

        # start training
        self.logger.console.info(f'Training the model')
        self._train(train_loader, val_loader)

        # Updates memory
        self.logger.console.info('Updating memory')
        # only trained samples can go into train memory to maintain unbiased validation in the future
        self.train_memory.update(trainset_train.ids, trainset.classes)
        self.val_memory.update(trainset_val.ids, trainset.classes)
        self.memory_manager.log_memory_sizes()

    def _introduce_new_classes(self, classes):
        classifier = self._classifier_constructor(classes, idx=self.phase - 1)
        self.classifiers.append(classifier)
        self.controller.extend(classes, classifier.output_size)

    def _train_epoch_start(self, epoch, lr_scheduler):
        self.logger.log({'epoch': epoch})
        lr = lr_scheduler.get_last_lr()[0]
        self.logger.log({'lr': lr})

    def _train_epoch_end(self, epoch):
        self.logger.commit()

    def _train(self, train_loader, val_loader):
        config = self.config
        stopper = TrainingStopper(config)
        optimizer = self.create_optimizer()
        lr_scheduler = _get_lr_scheduler(config, optimizer)

        for epoch in range(1, config.epochs + 1):
            if stopper.do_stop():
                break
            self._train_epoch_start(epoch, lr_scheduler)
            train_loss = self._train_epoch(epoch, train_loader, optimizer)

            # validate if validation dataset is not empty
            if len(val_loader.dataset) > 0:
                loss = self._validate(val_loader)
            else:
                loss = train_loss
            self._checkpoint(optimizer, loss, epoch)
            lr_scheduler.step(loss)
            stopper.update(loss)
            self._train_epoch_end(epoch)

        # load best checkpoint
        self._load_best()

    @property
    def last_classifier(self):
        return self.classifiers[-1]

    def _checkpoint(self, optimizer, loss, epoch):
        """Checkpoints the last classifier and the final layer."""
        # we don't save the optimizer
        self.controller.checkpoint(None, loss, epoch)
        self.last_classifier.checkpoint(None, loss, epoch)

    def _load_best(self):
        """Loads the best weights for the last classifier and the final layer."""
        self.controller.load_best()
        self.last_classifier.load_best()

    def create_optimizer(self):
        """Creates an optimizer for parameters of the last classifier and the final layer."""
        config = self.config
        last_classifier = self.classifiers[-1]
        optimizer = torch.optim.SGD(params=itertools.chain(last_classifier.parameters(), self.controller.parameters()),
                                    lr=config.lr, momentum=0.9)
        return optimizer

    def _train_epoch(self, epoch, loader, optimizer):
        assert isinstance(loader.dataset, PartialDataset) and loader.dataset.is_train()
        # set everything to eval mode except the last classifier and the final layer
        self.set_train(False)
        last_classifier = self.classifiers[-1]
        last_classifier.train()
        self.controller.train()
        loss_meter, ctrl_loss_meter = AverageMeter(), AverageMeter()

        for mstate in self._init_states(loader):
            total_loss = self._get_total_loss(mstate)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            loss_meter.update(total_loss, mstate.batch_size)
            ctrl_loss_meter.update(mstate.ctrl_state.loss, mstate.batch_size)

        self.logger.log({'train_loss': loss_meter.avg, 'ctrl_train_loss': ctrl_loss_meter.avg})
        return loss_meter.avg

    def _init_states(self, loader):
        return init_states(self.config, self, loader, self.device)

    @torch.no_grad()
    def _validate(self, loader):
        assert isinstance(loader.dataset, PartialDataset) and not loader.dataset.is_train()
        self.set_train(False)
        loss_meter, ctrl_loss_meter = AverageMeter(), AverageMeter()
        for mstate in self._init_states(loader):
            total_loss = self._get_total_loss(mstate)
            batch_size = mstate.batch_size
            loss_meter.update(total_loss, batch_size)
            ctrl_loss_meter.update(mstate.ctrl_state.loss, batch_size)
        self.logger.log({'val_loss': loss_meter.avg, 'ctrl_val_loss': ctrl_loss_meter.avg})
        return loss_meter.avg

    @torch.no_grad()
    def test(self, dataset):
        """Test on a balanced dataset."""
        self.set_train(False)
        loader = create_loader(self.config, dataset)
        loss_meter, ctrl_loss_meter = AverageMeter(), AverageMeter()

        # gather stuff
        all_predictions = []
        all_labels = []
        for mstate in self._init_states(loader):
            final_predictions = mstate.ctrl_state.predictions
            all_predictions.append(final_predictions)
            all_labels.append(mstate.labels_np)
            loss_meter.update(self._get_total_loss(mstate), mstate.batch_size)
            ctrl_loss_meter.update(mstate.ctrl_state.loss, mstate.batch_size)

        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        cm = confusion_matrix(all_labels, all_predictions, labels=self.controller.classes)
        self.logger.log_confusion_matrix(cm, labels=self.controller.classes)
        accuracy = get_accuracy(cm)
        self.logger.log({'test_loss': loss_meter.avg, 'ctrl_test_loss': ctrl_loss_meter.avg, 'test_acc': accuracy})
        self.logger.commit()

    def on_phase_end(self, phase):
        assert self.phase == phase
        # freeze the last learned classifier
        last_classifier = self.classifiers[-1]
        last_classifier.eval()
        for p in last_classifier.parameters():
            p.requires_grad = False

    def feed_final_layer(self, mstate):
        if mstate.final_outputs is not None:
            return mstate.final_outputs

        mstate.classes = self.controller.classes
        all_clf_outputs = [mstate.get_classifier_state(clf).outputs for clf in self.classifiers]
        final_outputs = self.controller(all_clf_outputs)
        mstate.final_outputs = final_outputs
        return final_outputs

    def set_train(self, train):
        self.set_classifiers_train(train)
        self.controller.train(train)

    def _get_total_loss(self, mstate):
        lam = self.config['lam']
        last_clf_loss = mstate.classifier_states[self.last_classifier].loss
        ctrl_loss = mstate.ctrl_state.loss
        return lam * last_clf_loss + ctrl_loss
