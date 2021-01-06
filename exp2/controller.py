from typing import List, Iterable, Union

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from exp2.classifier import Classifier
from exp2.feature_extractor import PRETRAINED
from exp2.model_state import ControllerState, ModelState
from exp2.utils import split_model


class Controller(nn.Module):
    def __init__(self, config, idx, classifiers: List[Classifier], net: nn.Module = None):
        super().__init__()
        self.classifiers = classifiers
        self.idx = idx
        self.config = config
        self.net = net
        self.cls_to_clf_idx = self._create_class_to_classifier_mapping()
        self.criterion = nn.CrossEntropyLoss()

    def _create_class_to_classifier_mapping(self):
        """Create a mapping that maps class to range 0...n-1 where n is the number of classifiers"""
        cls_to_clf = dict()
        for i, clf in enumerate(self.classifiers):
            for cls in clf.classes:
                cls_to_clf[cls] = i

        return cls_to_clf

    @property
    def n_classifiers(self):
        return len(self.classifiers)

    def get_optimizer(self):
        return optim.SGD(params=self.parameters(), lr=self.config.ctrl_lr, momentum=0.9)

    def forward(self, input):
        return self.net(input)

    def get_predictions(self, outputs):
        if outputs.size(0) == 0:
            return []
        return torch.argmax(outputs, 1)

    def group_labels(self, labels: torch.Tensor):
        """Transform labels to their corresponding classifier ids in range 0...n-1 where n is the number of classifiers."""
        if isinstance(labels, torch.Tensor):
            return torch.tensor([self.cls_to_clf_idx[i] for i in labels.tolist()], device=labels.device)
        else:
            return np.array([self.cls_to_clf_idx[i] for i in labels])

    def compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor):
        """Compute controller loss given its outputs and class labels.
        This will first map the labels into classifier ids and then apply Cross-entropy loss

        Args:
            outputs: controller outputs, raw
            labels: class labels
        """

        labels = self.group_labels(labels)
        return self.criterion(outputs, labels)

    def _ensure_state(self, state: ModelState):
        if state.controller_state is None:
            ctrl_state = ControllerState(self)
            state.set_controller_state(ctrl_state)
            return ctrl_state

    def _feed_with_state(self, mstate: ModelState) -> ControllerState:
        """Feed controller with frozen classifiers(no grad)"""
        ctrl_state = self._ensure_state(mstate)
        if ctrl_state.outputs is not None:
            return ctrl_state

        if self.config.ctrl_pos == 'after':
            # gather classifier outputs
            clf_outputs = []
            with torch.no_grad():
                for clf in self.classifiers:
                    clf_state = clf.feed(state=mstate)
                    clf_outputs.append(clf_state.outputs)
            ctrl_inputs = clf_outputs
        else:
            ctrl_inputs = mstate.features

        ctrl_outputs = self(ctrl_inputs)
        ctrl_state.outputs = ctrl_outputs
        return ctrl_state

    def feed(self, state: ModelState = None,
             states: Iterable[ModelState] = None) -> Union[Iterable[ControllerState], ControllerState]:
        """Feed controller with the given dataloader, and yield results as a generator. The classifiers
        are forwarded with no gradients. If state is given, the controller state will be added to it as well.
        """
        if state:
            return self._feed_with_state(state)
        else:
            return map(self._feed_with_state, states)


class CNNController(Controller):
    pass


class MLPController(Controller):
    """A two layer MLP network."""

    def __init__(self, config, idx, classifiers):
        super().__init__(config, idx, classifiers, None)
        n_classifiers = self.n_classifiers
        in_features = n_classifiers * (config.n_classes_per_phase + config.other)
        self.net = self._create_net(config, n_classifiers, in_features)

    def _create_net(self, config, n_classifiers, in_features):
        hidden_layer_scale = config.ctrl_hidden_layer_scale
        activation = config.ctrl_hidden_activation
        hidden_layer_size = int(n_classifiers * hidden_layer_scale)
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_layer_size),
            getattr(nn, activation)(),
            nn.Linear(in_features=hidden_layer_size, out_features=n_classifiers),
        )

    def forward(self, clf_outs: List[torch.Tensor]):
        """Assumes the classifier raw outputs in a list."""
        # run softmax
        inputs = [F.softmax(i, dim=1) for i in clf_outs]
        inputs = torch.cat(inputs, dim=1)
        return self.net(inputs)


class LinearController(MLPController):
    def _create_net(self, config, n_classifiers, in_features):
        in_features = n_classifiers * (config.n_classes_per_phase + config.other)
        return nn.Linear(in_features=in_features, out_features=n_classifiers)


def create_controller(config, idx, classifiers, device) -> Controller:
    if config.ctrl_pos == 'before':
        _, head_constructor = split_model(config, PRETRAINED)
        net = head_constructor(n_classes=classifiers)
        net = CNNController(config, classifiers, net)
    else:
        if config.ctrl_hidden_layer_scale > 0:
            # that's MLP
            net = MLPController(config, idx, classifiers)
        else:
            net = LinearController(config, idx, classifiers)
    net = net.to(device)
    return net
