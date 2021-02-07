from collections import OrderedDict
from typing import TYPE_CHECKING, Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from exp2.classifier import Classifier
    from exp2.controller import GrowingController
    from exp2.model import JointModel


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
        self.clf_criterion = torch.nn.CrossEntropyLoss()
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
