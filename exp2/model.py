import copy
import itertools
from collections import defaultdict, namedtuple
from functools import reduce, partial
from typing import List, Union

import sklearn
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

from exp2 import models
from exp2.data import create_loader, PartialDataset
from exp2.predictor import Predictor, ByCtrl, FilteredController
from logger import Logger
from utils import AverageMeter, np_a_in_b, get_default_device, TrainingStopper

DEVICE = get_default_device()


def get_device():
    return DEVICE


def load_model(model, path):
    model.load_state_dict(torch.load(path))


class FeatureExtractor(nn.Module):
    def __init__(self, config, net):
        super().__init__()
        self.net = net

    def forward(self, input):
        return self.net(input)


class Classifier(nn.Module):
    def __init__(self, config, net, classes, id):
        super().__init__()
        self.net = net
        self.id = id
        self._cls_idx = {cls: i for i, cls in enumerate(classes)}
        self.config = config
        self.classes = list(classes)

    def forward(self, input):
        return self.net(input)

    def localize_labels(self, labels: torch.Tensor):
        """Convert labels to local labels in range 0, ..., n, where n-1 is the output units.
        'Other' category is mapped to n, the last output unit."""
        loc = []
        device = labels.device
        n = len(self.classes)
        for lbl in labels.tolist():
            loc.append(self._cls_idx.get(lbl, n))
        return torch.tensor(loc, device=device)

    def get_predictions(self, outputs: torch.Tensor, open=True) -> List[int]:
        """Get classifier predictions. The predictions contain actual class labels, not the local ones.
        If open, the 'other' category will be considered and will have label -1.
        """
        if outputs.size(0) == 0:
            return []

        # get local predictions
        if open or not self.config.other:
            loc = torch.argmax(outputs, 1)
        else:
            loc = torch.argmax(outputs[:, :-1], 1)  # skip the last unit

        r = []
        n_classes = len(self.classes)
        for ll in loc.tolist():
            if ll == n_classes:  # other category, add it as -1
                r.append(-1)
            else:
                r.append(self.classes[ll])
        return r

    def map_other(self, labels: np.ndarray, excl_idx=None):
        """Map labels of 'other' category to -1.

        Args:
            labels: the labels
            excl_idx (optional, np.ndarray): the indices where known labels reside, if provided, will
                help avoid unnecessary computation
        """
        result = np.empty_like(labels)
        result.fill(-1)
        if excl_idx is None:
            excl_idx = np_a_in_b(result, self.classes)
        result[excl_idx] = labels[excl_idx]
        return result


class Controller(nn.Module):
    def __init__(self, config, n_classifiers, net=None):
        super().__init__()
        self.config = config
        self.net = net
        self.n_classifiers = n_classifiers

    def get_optimizer(self):
        return optim.SGD(params=self.parameters(), lr=self.config.ctrl_lr, momentum=0.9)

    def forward(self, input):
        assert self.net is not None
        return self.net(input)


class CNNController(Controller):
    pass


class LinearController(Controller):
    def __init__(self, config, n_classifiers):
        super().__init__(config, n_classifiers, None)
        in_features = n_classifiers * (config.n_classes_per_phase + config.other)
        self.net = nn.Linear(in_features=in_features, out_features=n_classifiers)

    def forward(self, clf_outs):
        """Assumes the classifier raw outputs in a list."""
        # run softmax
        clf_outs = [F.softmax(i, dim=1) for i in clf_outs]
        clf_outs = torch.cat(clf_outs, dim=1)
        return self.net(clf_outs)


@torch.no_grad()
def _reset_parameters(net: nn.Module):
    def fn(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    net.apply(fn)


def split_model(config, model):
    """Split the model into back and head.
    The back for being used as feature extractor, and frozen.
    The head is in a constructor form taking number of classes
    as parameter.
    It assumes the model is a sequential. It assumes last layer as the linear classification layer.
    Note: it does not clone the feature extractor.
    """
    assert isinstance(model, nn.Sequential)

    back: nn.Sequential = model[:config.split_pos]
    head: nn.Sequential = model[config.split_pos:]

    # freeze the feature extractor
    back.eval()
    for param in back.parameters():
        param.requires_grad = False

    # constructor for the head
    clf_layer: nn.Linear = head[-1]
    assert isinstance(clf_layer, nn.Linear)
    del head[-1]
    in_features = clf_layer.in_features

    def head_constructor(n_classes):
        # replace the classification layer from the head with the one matching number of classes
        layer = nn.Linear(in_features=in_features, out_features=n_classes)
        newhead = copy.deepcopy(head)
        newhead = nn.Sequential(*newhead, layer)
        if not config.clone_head:
            _reset_parameters(newhead)
        return newhead

    return back, head_constructor


PRETRAINED = None


def create_models(config, device) -> (FeatureExtractor, callable):
    global PRETRAINED
    if PRETRAINED is None:
        PRETRAINED = models.simple_net(n_classes=20)
    load_model(PRETRAINED, config.pretrained)
    fe, head_constructor = split_model(config, PRETRAINED)
    fe = FeatureExtractor(config, fe)
    fe.eval()
    fe = fe.to(device)

    def classifier_constructor(classes, id=None) -> Classifier:
        """id: the id assigned to the classifier"""
        n_classes = len(classes)
        net = head_constructor(n_classes=n_classes + config.other)
        return Classifier(config, net, classes, id).to(device)

    return fe, classifier_constructor


def create_controller(config, n_classifiers, device) -> Controller:
    if config.ctrl_pos == 'before':
        _, head_constructor = split_model(config, PRETRAINED)
        net = head_constructor(n_classes=n_classifiers)
        net = CNNController(config, n_classifiers, net)
    else:
        net = LinearController(config, n_classifiers)
    net = net.to(device)
    return net


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


def get_lr_scheduler(cfg, optimizer):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=cfg.lr_decay, patience=cfg.lr_patience,
                                                verbose=True, min_lr=0.00001)


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
        self.cls_to_clf_id = {}

    @property
    def classes(self):
        return list(self.cls_to_clf_id.keys())

    def _group_labels(self, labels: Union[torch.Tensor, List[int]]):
        """ Transform labels to their corresponding classifier ids."""
        if isinstance(labels, torch.Tensor):
            return torch.tensor([self.cls_to_clf_id[i] for i in labels.tolist()], device=labels.device)
        else:
            return [self.cls_to_clf_id[i] for i in labels]

    def _create_classifier_optimizer(self, classifier):
        config = self.config
        return optim.SGD(classifier.parameters(), lr=config.clf_lr, momentum=0.9, weight_decay=config.weight_decay)

    def _create_classifier(self, classes):
        """The created classifier is in train mode."""
        clf_id = len(self.classifiers)
        classifier = self._classifier_constructor(classes, id=clf_id)
        self.classifiers.append(classifier)
        return classifier

    def _train_classifiers(self, classifiers, loader, criterion, optimizers):
        self.set_train(True)
        alosses = None
        for i, loss, alosses in self._feed_classifiers(classifiers, loader, criterion):
            optimizers[i].zero_grad()
            loss.backward()
            optimizers[i].step()

        avg_losses = [alosses[i].avg for i in range(len(classifiers))]
        self.logger.log({'clf/' + str(clf.id) + '/tr_loss': avg_losses[i] for i, clf in enumerate(classifiers)})
        return avg_losses

    @torch.no_grad()
    def _val_classifiers(self, classifiers, loader, criterion):
        for clf in classifiers:
            clf.eval()

        alosses = None
        for i, loss, alosses in self._feed_classifiers(classifiers, loader, criterion):
            pass
        avg_losses = [alosses[i].avg for i in range(len(classifiers))]
        self.logger.log({'clf/' + str(clf.id) + '/val_loss': avg_losses[i] for i, clf in enumerate(classifiers)})
        return avg_losses

    def _feed_feature_extractor(self, loader):
        """Feed the samples to the feature extractor and yield namedtuple(['ids', 'inputs', 'features', 'labels', 'labels_np'])"""
        fields = ['ids', 'inputs', 'features', 'labels', 'labels_np']
        feat_out = namedtuple('FeatureExtractorOut', fields)
        for inputs, labels, ids in loader:
            labels_np = labels.numpy()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            features = self.feature_extractor(inputs)
            yield feat_out(ids, inputs, features, labels, labels_np)

    def _feed_classifiers(self, classifiers: List[Classifier], loader, criterion):
        alosses = [AverageMeter() for _ in classifiers]
        for nt in self._feed_feature_extractor(loader):
            features, labels, inputs = nt.features, nt.labels, nt.inputs
            for i, clf in enumerate(classifiers):
                output = clf(features)
                loc_labels = clf.localize_labels(labels)
                loss = criterion(output, loc_labels)
                alosses[i].update(loss, inputs.size(0))
                yield i, loss, alosses

    def train_new_classifier(self, newset: PartialDataset, otherset: PartialDataset = None):
        """Train a new classifier on the dataset. Optionally given otherset that contains
        examples from unknown classes.

        Args:
            newset (PartialDataset): dataset of new class samples
            otherset (optional, PartialDataset): dataset of old class samples( serving as a "other" category)
        """
        cfg = self.config
        epoch_tol = cfg.clf_new_epochs_tol
        n_epochs = cfg.clf_new_epochs
        new_classes = newset.classes
        classifier = self._create_classifier(new_classes)
        weight = get_class_weights(self.config, newset, otherset)
        weight = weight.to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = self._create_classifier_optimizer(classifier)
        dataset = newset.mix(otherset) if otherset else newset
        train_loader, val_loader = self._split(dataset)
        stopper = TrainingStopper(tol=epoch_tol)
        lr_scheduler = get_lr_scheduler(cfg, optimizer)
        for epoch in range(1, n_epochs + 1):
            if stopper.do_stop():
                break
            self.logger.log({'clf/epoch': epoch})
            self._train_classifiers([classifier], train_loader, criterion, [optimizer])
            if val_loader is not None:
                losses = self._val_classifiers([classifier], val_loader, criterion)
                stopper.update(losses[0])
                lr_scheduler.step(losses[0])
            self.logger.commit()
        # add new class labels to classifiers mapping
        self.cls_to_clf_id.update({cls: classifier.id for cls in new_classes})

    def update_prev_classifiers(self, dataset):
        train_loader, val_loader = self._split(dataset)
        n_epochs = self.config.clf_update_epochs
        criterion = nn.CrossEntropyLoss()
        classifiers = np.array(self.classifiers[:-1])
        optimizers = [self._create_classifier_optimizer(clf) for clf in classifiers]
        stoppers = np.array([TrainingStopper(tol=self.config.clf_update_epochs_tol) for _ in classifiers])
        for epoch in range(n_epochs):
            # filter out stopping classifiers and stoppers
            indices = [not stopper.do_stop() for stopper in stoppers]
            if sum(indices) == 0:
                break
            self.logger.log({'clf/epoch': epoch})
            self._train_classifiers(classifiers[indices], train_loader, criterion, optimizers)
            if val_loader:
                losses = self._val_classifiers(classifiers[indices], val_loader, criterion)
                for stopper, loss in zip(stoppers[indices], losses):
                    stopper.update(loss)
            self.logger.commit()

    def _forward_classifiers(self, features):
        out = []
        for clf in self.classifiers:
            out.append(clf(features))
        return out

    def _forward_controller(self, features, clf_no_grad=False):
        if self.config.ctrl_pos == 'before':
            return self.controller(features)
        with torch.set_grad_enabled(clf_no_grad):
            _clf_outs = self._forward_classifiers(features)
        return self.controller(_clf_outs)

    def _feed_controller(self, loader, criterion=None):
        """Feed controller with the given dataloader, and yield results as a generator. The classifiers
        are forwarded with no gradients.
        Yields:
            if criterion is not None, yields namedtuple(loss, average loss meter); else namedtuple(ctrl_outs, class labels, ids),
        """
        nt = namedtuple('CtrlOut', ['ids', 'labels', 'ctrl_outs'])
        for f in self._feed_feature_extractor(loader):
            labels, features, ids = f.labels, f.features, f.ids
            ctrl_outs = self._forward_controller(features, clf_no_grad=True)
            yield nt(ids, labels, ctrl_outs)

    def _get_controller_criterion(self):
        return self._controller_criterion

    def _create_new_controller(self) -> Controller:
        self._controller_criterion = nn.CrossEntropyLoss()
        controller = create_controller(self.config, n_classifiers=len(self.classifiers), device=self.device)
        return controller

    def _train_controller(self, loader, criterion, optimizer):
        """Train the controller. Only train the controller. Classifiers are kept frozen."""
        # set everything to eval except for the controller
        self.set_train(False)
        self.controller.train()

        for cout in self._feed_controller(loader, criterion):
            ctrl_outs, labels = cout.ctrl_outs, cout.labels
            loss = self._compute_controller_loss(ctrl_outs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def _val_controller(self, loader, criterion):
        self.set_train(False)
        aloss = AverageMeter()
        for cout in self._feed_controller(loader, criterion):
            ctrl_outs, labels = cout.ctrl_outs, cout.labels
            loss = self._compute_controller_loss(ctrl_outs, labels)
            aloss.update(loss, n=labels.size(0))
        avg_loss = aloss.avg
        self.logger.log({'ctrl/val_loss': avg_loss})
        return avg_loss

    def train_a_new_controller(self, dataset):
        """Create a new controller and train it. It will replace the current controller with the new one."""
        cfg = self.config
        n_epochs = cfg.ctrl_epochs
        epoch_tol = cfg.ctrl_epochs_tol
        self.controller = self._create_new_controller()
        criterion = self._get_controller_criterion()
        optimizer = self.controller.get_optimizer()
        train_loader, val_loader = self._split(dataset)
        stopper = TrainingStopper(tol=epoch_tol)
        lr_scheduler = get_lr_scheduler(cfg, optimizer)
        for epoch in range(1, n_epochs + 1):
            if stopper.do_stop():
                break
            self.logger.log({'ctrl/epoch': epoch})
            self._train_controller(train_loader, criterion, optimizer)
            if val_loader:
                loss = self._val_controller(val_loader, criterion)
                lr_scheduler.step(loss)
                stopper.update(loss)
            self.logger.commit()

    def _forward_controller_and_classifiers(self, features):
        clf_outs = self._forward_classifiers(features)
        if self.config.ctrl_pos == 'before':
            ctrl_outs = self.controller(features)
        else:
            ctrl_outs = self.controller(clf_outs)
        return clf_outs, ctrl_outs

    def _feed_everything(self, loader):
        """
        Feed the samples and yield namedtuple(['ids', 'inputs', 'clf_out', 'ctrl_outs', 'labels_np', 'labels', ])
        for every batch.
        """
        fields = ['ids', 'inputs', 'clf_outs', 'ctrl_outs', 'labels_np', 'labels', ]
        nt = namedtuple('EverythingOut', fields)
        for f in self._feed_feature_extractor(loader):
            ids, inputs, labels_np, labels, features = f.ids, f.inputs, f.labels_np, f.labels, f.features
            clf_outs, ctrl_outs = self._forward_controller_and_classifiers(features)
            yield nt(ids, inputs, clf_outs, ctrl_outs, labels_np, labels)

    @torch.no_grad()
    def test(self, dataset):
        """
        Test the model in whole and in parts.

        Reported metrics:
        - overall accuracy
        - controller
            - accuracy
            - confusion matrix
        - per classifier
            - open accuracy (with class weights)
            - closed accuracy
            - confusion matrices (open and closed)

        Args:
            dataset (PartialDataset): cumulative testset, containing all seen classes
        """
        self.set_train(False)

        loader = create_loader(self.config, dataset)
        all_clf_preds = defaultdict(lambda: [list() for _ in self.classifiers])
        all_labels_per_clf = defaultdict(lambda: [list() for _ in self.classifiers])
        # contains predictions of only known classes examples, per classifier
        all_ctrl_preds = []
        all_ctrl_outs = []
        all_labels = []

        for e in self._feed_everything(loader):
            labels_np, labels, clf_outs, ctrl_outs = e.labels_np, e.labels, e.clf_outs, e.ctrl_outs
            all_labels.extend(labels_np)
            all_ctrl_outs.extend(ctrl_outs.tolist())
            all_ctrl_preds.extend(torch.argmax(ctrl_outs, 1).tolist())

            for i, (clf_out, clf) in enumerate(zip(clf_outs, self.classifiers)):
                excl_idx = np_a_in_b(labels_np, clf.classes)  # indices that contain known labels to this classifier
                lbl = clf.map_other(labels_np, excl_idx)
                for open, excl in itertools.product([True, False], [True, False]):
                    pred = clf.get_predictions(clf_out[excl_idx] if excl else clf_out, open=open)
                    all_clf_preds[(open, excl)][i].extend(pred)
                    all_labels_per_clf[(open, excl)][i].extend(lbl[excl_idx] if excl else lbl)

        # report controller metrics
        self._report_controller_metrics(all_ctrl_preds, all_labels)

        # report classifier metrics
        self._report_classifiers_metrics(self.classifiers, all_clf_preds, all_labels_per_clf)

        # overall accuracy with different predictors
        for predictor in self.predictors:
            open = excl = True
            final_predictions = predictor(self.classifiers,
                                          all_ctrl_outs,
                                          all_ctrl_preds,
                                          all_clf_preds[(open, not excl)],
                                          all_clf_preds[(not open, not excl)])
            self._report_predictor_metrics(predictor, all_labels, final_predictions)
        self.logger.commit()

    def _split(self, dataset: PartialDataset):
        """Split into train and test sets, and return dataloaders"""
        val_size = self.config.val_size
        if val_size > 0:
            trainset, valset = dataset.split(test_size=val_size)
            val_loader = create_loader(self.config, valset)
        else:
            trainset, valset = dataset, None
            val_loader = None
        train_loader = create_loader(self.config, trainset)
        return train_loader, val_loader

    def _report_controller_metrics(self, preds, labels: List[int]):
        controller_labels = self._group_labels(labels)
        contr_confmatrix = sklearn.metrics.confusion_matrix(controller_labels, preds)
        with self.logger.prefix('ctrl'):
            classnames = list(range(len(self.classifiers)))
            self.logger.log_accuracies(contr_confmatrix, classnames, log_recalls=False)
            self.logger.log_confusion_matrix(contr_confmatrix, classnames, title='Confusion(controller)')

    def _report_classifiers_metrics(self, classifiers, preds, labels):
        """Compute and log classifier metrics.

        For each classifier the following metrics:
            open, incl: open predictions vs labels( on all testset)
            open, excl: open predictions vs labels( on exclusive testset, containing known classes)
                Here the samples don't contain "other", but the classifier is allowed to predict "other"
            closed, incl: closed predictions vs labels( on all testset)
                Here the classifier doesn't predict "other", but the samples can contain "other".
            closed, excl: closed predictions vs labels( on exclusive testset, containing known classes)
                Here the classifier doesn't predict "other", also the samples don't contain "other"

        Note that the labels should already be mapped suitable to the classifier ("other" category mapped to -1)

        Args:
            classifiers: a list of classifiers
            pred: a dictionary [(open, excl)] -> List[predictions per classifier]
            labels: a dictionary [(open, excl)] -> List[labels per classifier]
        """
        # why closed_excl and closed are observably same?

        assert preds.keys() == labels.keys()
        for (open, excl) in preds.keys():
            for pr, lbl, clf in zip(preds[(open, excl)], labels[(open, excl)], classifiers):
                cm_labels = clf.classes + [-1]
                cm = sklearn.metrics.confusion_matrix(lbl, pr, labels=cm_labels)
                pref = f'clf/{clf.id}/{"open" if open else "closed"}{"_excl" if excl else ""}'
                with self.logger.prefix(pref):
                    self.logger.log_accuracies(cm, cm_labels, log_recalls=False)
                    self.logger.log_confusion_matrix(cm, cm_labels, title=f'Confusion (classifier {clf.id})')

    def _report_predictor_metrics(self, predictor, all_labels, final_predictions):
        classes = self.classes
        cm = sklearn.metrics.confusion_matrix(all_labels, final_predictions, labels=classes)
        with self.logger.prefix("final/" + predictor.name):
            self.logger.log_accuracies(cm, classnames=classes, log_recalls=False)
            self.logger.log_confusion_matrix(cm, classes, title='Confusion (predictor)')

    @torch.no_grad()
    def _compute_importance_scores(self, dataset, score_function):
        """
        Compute importance scores based on the classifiers and controller's output.
        The score_function is a function receiving the arguments (clf_outputs, ctrl_outputs, labels)
        and returning score value for each batch sample.
        Args:
            dataset (Dataset):
            score_function (callable): function(clf_outputs: list[torch.tensor], ctrl_outputs: torch.tensor, labels: torch.tensor) -> scores (np.ndarray)

        Returns:
            (np.ndarray, np.ndarray): ids and scores
        """
        # set the model into evaluation mode
        self.set_train(False)

        loader = create_loader(self.config, dataset)
        all_scores = []
        all_ids = []

        for eout in self._feed_everything(loader):
            ids, clf_outs, ctrl_outs, labels = eout.ids, eout.clf_outs, eout.ctrl_outs, eout.labels
            scores = score_function(clf_outs, ctrl_outs, labels)
            all_scores.extend(scores.tolist())
            all_ids.extend(ids.tolist())

        all_ids = np.array(all_ids)
        all_scores = np.array(all_scores)
        return all_ids, all_scores

    def compute_fresh_importance_scores(self, dataset):
        """
        Computes importance scores for the new class samples using only the classifiers (not the controller).
        Here, the importance scores are computed by taking the maximum value of all previous classifiers outputs
        excluding their "other" output units (if present).

        Warning: this function must be called before the previous classifier are updated. Otherwise it would be meaningless.
        """

        def _score_function(clf_outs, ctrl_outs, labels, ignore_last_unit=False):
            # 1. for each sample loop through the all classifiers all their outputs, conditionally omitting last units
            # 2. get the maximum value
            # the following is a list, whose elements correspond to batch samples
            if ignore_last_unit:
                clf_outs = [outs[:, :-1] for outs in clf_outs]
            max_per_classifier_sample = [torch.max(outs, 1)[0] for outs in clf_outs]
            max_per_sample = reduce(torch.maximum, max_per_classifier_sample)
            max_per_sample = max_per_sample.cpu().numpy()
            return max_per_sample

        score_function = partial(_score_function, ignore_last_unit=self.config.other)
        return self._compute_importance_scores(dataset, score_function)

    def recompute_importance_scores(self, dataset):
        """This will recompute importance scores. It assumes the contoller is already familiar with classes inside.
        The importance scores here are computed as the loss of the controller.
        """
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        def score_function(clf_outs, ctrl_outs, labels):
            labels = self._group_labels(labels)
            loss = criterion(ctrl_outs, labels)
            return loss

        return self._compute_importance_scores(dataset, score_function)

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

    def _compute_controller_loss(self, ctrl_outs, labels):
        """Compute controller loss given its outputs and class labels.
        This will first map the labels into classifier ids and then apply Cross-entropy loss"""
        labels = self._group_labels(labels)
        criterion = self._controller_criterion
        return criterion(ctrl_outs, labels)
