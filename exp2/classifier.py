from typing import List, Iterable

import numpy as np
import torch
from torch import nn

from exp2.model_state import ClassifierState, ModelState
from utils import np_a_in_b


class Classifier(nn.Module):
    other_label = -1

    def __init__(self, config, net, classes, idx):
        super().__init__()
        self.net = net
        self.idx: int = idx
        self._cls_idx = {cls: i for i, cls in enumerate(classes)}
        self.config = config
        self.classes: List[int] = list(classes)

    def forward(self, inputs):
        return self.net(inputs)

    def localize_labels(self, labels: torch.Tensor):
        """Convert labels to local labels in range 0, ..., n, where n-1 is the output units.
        'Other' category is mapped to n, the last output unit."""
        loc = []
        device = labels.device
        n = len(self.classes)
        for lbl in labels.tolist():
            loc.append(self._cls_idx.get(lbl, n))
        return torch.tensor(loc, device=device)

    def get_predictions(self, outputs: torch.Tensor, is_open=True) -> np.ndarray:
        """Get classifier predictions. The predictions contain actual class labels, not the local ones.
        If `is_open`, the 'other' category will be considered and will have label self.other_label.
        Useful for computing accuracy.
        """
        if outputs.size(0) == 0:
            return np.array([])

        # get local predictions
        if is_open or not self.config.other:
            loc = torch.argmax(outputs, 1)
        else:
            loc = torch.argmax(outputs[:, :-1], 1)  # skip the last unit, i.e. corresponds to 'other'

        other_label = self.other_label
        n_classes = len(self.classes)
        predictions = np.fromiter((other_label if ll == n_classes else self.classes[ll] for ll in loc.tolist()),
                                  dtype=int)
        return np.array(predictions)

    def map_other(self, labels_np: np.ndarray) -> np.ndarray:
        """Map labels of 'other' category to self.other_label. Returns a new tensor.
        Useful for computing accuracies and confusion matrices.

        Args:
            labels_np: class labels
        """
        result = np.full_like(labels_np, self.other_label)
        excl_idx = np_a_in_b(labels_np, self.classes)
        result[excl_idx] = labels_np[excl_idx]
        return result

    def get_loss(self, outputs: torch.Tensor, labels: torch.Tensor, criterion) -> torch.Tensor:
        """Compute loss given the classifier outputs, class labels(unmapped), and criterion

        Args:
            outputs: classifier outputs
            labels: class labels (unmapped)
        """
        local_labels = self.localize_labels(labels)
        loss = criterion(outputs, local_labels)
        return loss

    def _ensure_state(self, state: ModelState):
        if self in state.get_classifiers():
            return state.get_classifier_state(self)
        else:
            clf_state = ClassifierState(self)
            state.add_classifier(clf_state)
            return clf_state

    def _feed_with_state(self, state: ModelState) -> ClassifierState:
        clf_state = self._ensure_state(state)
        if clf_state.outputs is None:
            clf_state.outputs = self(state.features)
        return clf_state

    def feed(self, state: ModelState = None, states: Iterable[ModelState] = None):
        """Feed classifier and return state object. If state is given, the classifier state will be added to it."""
        if state:
            return self._feed_with_state(state)
        else:
            return (self._feed_with_state(state) for state in states)