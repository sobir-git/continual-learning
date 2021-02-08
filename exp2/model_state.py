from collections import OrderedDict
from typing import TYPE_CHECKING, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from exp2.classifier import Classifier
    from exp2.controller import GrowingController
    from exp2.model import JointModel


def get_class_weights(config, phase, balance_classes):
    """Class weights used for controller classification criterion."""
    cpp = config.n_classes_per_phase  # classes per phase
    nc = cpp * phase  # number of classes
    if balance_classes:
        bs = config.batch_size
        bms = config.batch_memory_samples
        noc = cpp * (phase - 1)  # number of old classes
        nnc = cpp  # number of new classes
        old_cls_weight = noc / bms
        new_cls_weight = nnc / (bs - bms)
        return torch.tensor([old_cls_weight] * noc + [new_cls_weight] * nnc, dtype=torch.float32)
    else:
        return torch.ones(nc, dtype=torch.float32)


def get_classification_criterion(config, phase):
    """Create classification criterion for classifier."""
    if config.clf_balance_classes and config.other:
        weight = get_class_weights(config, phase, config.clf_balance_classes)
        nnc = config.n_classes_per_phase  # classes per phase, number of new classes
        weight = torch.cat((weight[-nnc:], weight[:-nnc].sum().view(1, )))
    else:
        weight = None
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    return criterion


def lazy_property(fn):
    '''Decorator that makes a property lazy-evaluated.
    '''
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


class LazyClassifierState:
    def __init__(self, classifier: 'Classifier', parent: 'LazyModelState', criterion):
        self.parent = parent
        self.classifier = classifier
        self.criterion = criterion

    @lazy_property
    def outputs(self) -> torch.Tensor:
        features = self.parent.features
        return self.classifier(features)

    @lazy_property
    def loss(self) -> torch.Tensor:
        outputs = self.outputs
        labels = self.parent.labels
        return self.classifier.get_loss(outputs, labels, self.criterion)


class LazyControllerState:

    def __init__(self, controller: 'GrowingController', parent: 'LazyModelState'):
        self.controller = controller
        self.parent = parent

    @lazy_property
    def outputs(self) -> torch.Tensor:
        clf_states = self.parent.classifier_states
        clf_outputs = [state.outputs for state in clf_states.values()]
        return self.controller(clf_outputs)

    @lazy_property
    def loss(self) -> torch.Tensor:
        outputs = self.outputs
        labels = self.parent.labels
        return self.controller.get_loss(outputs, labels)

    @lazy_property
    def predictions(self) -> np.ndarray:
        outputs = self.outputs
        return self.controller.get_predictions(outputs)


class LazyModelState:
    ids: np.ndarray
    inputs: torch.Tensor
    classes: np.ndarray
    labels_np: np.ndarray
    labels: torch.Tensor
    phase: int

    def __init__(self, model: 'JointModel', inputs, labels_np=None, labels=None, classes=None,
                 phase=None, ids=None):
        self.feature_extractor = model.feature_extractor
        self.controller = model.controller
        self.classifiers = model.classifiers
        self.ids = ids
        self.phase = phase
        self.inputs = inputs
        self.labels_np = labels_np
        self.labels = labels
        self.classes = classes
        self.clf_criterion = get_classification_criterion(model.config, model.phase).to(model.device)
        self.classifier_states = OrderedDict(
            (clf, LazyClassifierState(clf, self, self.clf_criterion)) for clf in self.classifiers)
        self.ctrl_state = LazyControllerState(self.controller, self)

    @lazy_property
    def features(self) -> torch.Tensor:
        return self.feature_extractor(self.inputs)

    @property
    def final_outputs(self) -> torch.Tensor:
        return self.ctrl_state.outputs

    @property
    def batch_size(self):
        return len(self.inputs)

    @staticmethod
    def init_states(config, model, loader: DataLoader, device, **kwargs) -> Iterable['LazyModelState']:
        non_blocking = config.torch['non_blocking']
        for inputs, labels, ids in loader:
            labels_np = labels.numpy()
            inputs, labels = inputs.to(device, non_blocking=non_blocking), labels.to(device, non_blocking=non_blocking)
            yield LazyModelState(model, inputs, labels=labels, labels_np=labels_np, **kwargs)
