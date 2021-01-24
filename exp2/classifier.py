import os
import random
import string
from typing import List, Iterable

import numpy as np
import torch
import wandb
from torch import nn

from exp2.model_state import ClassifierState, ModelState
from utils import np_a_in_b, get_console_logger

console_logger = get_console_logger(__name__)


def upload_classifier(classifier):
    checkpoint_file = classifier.get_checkpoint_file()
    if os.path.exists(checkpoint_file):
        console_logger.info(f'uploading classifier {classifier.idx}')
        artifact = wandb.Artifact(f'classifier-{classifier.idx}-{wandb.run.id}', type='model')
        artifact.add_file(checkpoint_file)
        wandb.log_artifact(artifact)


class Checkpoint(nn.Module):
    _min_val_loss = -float('inf')

    def __init__(self, checkpoint_file=None):
        super().__init__()
        if checkpoint_file is None:
            letters = string.ascii_lowercase + string.digits
            self._checkpoint_file = ''.join(random.choice(letters) for _ in range(10)) + '.pt'
        else:
            self._checkpoint_file = checkpoint_file

    @staticmethod
    def wrap(model: nn.Module, checkpoint_file) -> "Checkpoint":
        class Patched(model.__class__, Checkpoint):
            pass

        model.__class__ = Patched
        Checkpoint.__init__(model, checkpoint_file)
        return model

    def checkpoint(self, optimizer, val_loss, epoch):
        """Checkpoint if val_loss is the minimum"""
        if self._min_val_loss > val_loss:
            d = {
                'val_loss': val_loss,
                'state_dict': self.state_dict(),
                'optimizer': optimizer,
                'epoch': epoch
            }
            torch.save(d, self._checkpoint_file)
            self._min_val_loss = val_loss

    def load_from_checkpoint(self, checkpoint_file):
        d = torch.load(checkpoint_file)
        self._min_val_loss = d['val_loss']
        self.load_state_dict(d['state_dict'])
        console_logger.debug('Loaded checkpoint %s, (from epoch %s)', checkpoint_file, d['epoch'])
        return d

    def load_best(self):
        """Load if checkpoint exists."""
        if not os.path.isfile(self._checkpoint_file):
            return
        d = self.load_from_checkpoint(self._checkpoint_file)
        wandb.log({f'clf/{self.idx}/best_epoch': d["epoch"]})
        wandb.log({f'clf/{self.idx}/min_val_loss': d["val_loss"]})
        return d

    def get_checkpoint_file(self):
        return self._checkpoint_file


class Classifier(Checkpoint, nn.Module):
    other_label = -1
    _min_val_loss = float('inf')

    def __init__(self, config, net, classes, idx):
        super().__init__(checkpoint_file=config.logdir + '/' + f'classifier_{idx}.pt')
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
