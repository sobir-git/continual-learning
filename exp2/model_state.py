from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, TYPE_CHECKING, Iterable

import numpy as np
import torch

if TYPE_CHECKING:
    from exp2.classifier import Classifier
    from exp2.controller import Controller


class ChildState:
    parent = None

    def __getattr__(self, item):
        return getattr(self.parent, item)


@dataclass
class ControllerState(ChildState):
    self: Controller
    outputs: torch.Tensor = None
    parent: 'ModelState' = None
    loss: torch.Tensor = None
    epoch: int = None


@dataclass
class ClassifierState(ChildState):
    self: Classifier
    outputs: torch.Tensor = None
    parent: 'ModelState' = None
    loss: torch.Tensor = None
    epoch: int = None


@dataclass
class ModelState:
    ids: np.ndarray
    inputs: torch.Tensor
    features: torch.Tensor = None
    controller_state: ControllerState = None
    classifier_states: Dict[Classifier, ClassifierState] = field(default_factory=dict)
    labels_np: np.ndarray = None
    labels: torch.Tensor = None
    phase: int = None
    mode: str = None  # "train", "val", or "test"

    def add_classifier(self, clf: ClassifierState):
        self.classifier_states[clf.self] = clf
        clf.parent = self

    def get_classifiers(self):
        return list(self.classifier_states.keys())

    def get_classifier_state(self, classifier: Classifier) -> ClassifierState:
        return self.classifier_states[classifier]

    def set_controller_state(self, controller_state: ControllerState):
        if self.controller_state is not None:
            raise ValueError("Controller is already present")
        self.controller_state = controller_state
        controller_state.parent = self

    def get_controller(self):
        return self.controller_state.self

    def get_controller_state(self) -> ControllerState:
        return self.controller_state

    @property
    def batchsize(self) -> int:
        return self.labels.size(0)


def init_states(loader, device) -> Iterable[ModelState]:
    for inputs, labels, ids in loader:
        labels_np = labels.numpy()
        inputs, labels = inputs.to(device), labels.to(device)
        yield ModelState(ids, inputs, labels=labels, labels_np=labels_np)
