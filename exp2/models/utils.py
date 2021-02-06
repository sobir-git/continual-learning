import os
import random
import string
import tempfile
from functools import partial

import torch
from torch import nn

from utils import get_console_logger

console_logger = get_console_logger()


class MultiOutputNet(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs


@torch.no_grad()
def get_output_shape(model, input_shape=None, inputs=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if inputs is None:
        inputs = torch.randn(input_shape).unsqueeze(0)
    training = model.training
    model.eval()
    inputs = inputs.to(device)
    model = model.to(device)
    outputs = model(inputs)
    model.train(training)
    return outputs.shape[1:]


class ClassMapping:
    def __init__(self, *args, **kwargs):
        super(ClassMapping, self).__init__(*args, **kwargs)
        self._classes = []
        self._classes_inv = dict()

    def extend(self, classes):
        self._classes.extend(classes)
        self._classes_inv = {c: i for i, c in enumerate(self._classes)}

    @property
    def classes(self):
        return self._classes[:]

    def localize_labels(self, labels):
        """Convert labels to range 0 ... n"""
        classes_inv = self._classes_inv
        local_labels = torch.tensor([classes_inv[ll] for ll in labels.tolist()], dtype=labels.dtype,
                                    device=labels.device)
        return local_labels

    def globalize_labels(self, labels, device=None):
        """Convert local labels from range 0 ... n to actual labels"""
        if device is None:
            device = labels.device
        classes = self._classes
        global_labels = torch.tensor([classes[ll] for ll in labels.tolist()], dtype=labels.dtype, device=device)
        return global_labels


class Checkpoint(nn.Module):
    _min_val_loss = float('inf')

    def __init__(self, checkpoint_file=None):
        super().__init__()
        self._init(checkpoint_file)
        if checkpoint_file is None:
            self._istempfile = True
        self._istempfile = False

    def _init(self, checkpoint_file):
        tempdir = tempfile.gettempdir()
        if checkpoint_file is None:
            letters = string.ascii_lowercase + string.digits
            self._checkpoint_file = tempdir + '/' + ''.join(random.choice(letters) for _ in range(10)) + '.pt'
        else:
            self._checkpoint_file = checkpoint_file

    def __del__(self):
        if self._istempfile:
            os.remove(self._checkpoint_file)

    def remove_checkpoint(self):
        self._min_val_loss = float('inf')
        try:
            os.remove(self._checkpoint_file)
        except FileNotFoundError:
            pass
        except Exception as e:
            console_logger.error(f'Failed to remove checkpoint file: {e}')

    @staticmethod
    def wrap(model: nn.Module, checkpoint_file) -> 'Checkpoint':
        Checkpoint._init(model, checkpoint_file)
        model._min_val_loss = float('inf')
        model.checkpoint = partial(Checkpoint.checkpoint, model)
        model.load_from_checkpoint = partial(Checkpoint.load_from_checkpoint, model)
        model.get_checkpoint_file = partial(Checkpoint.get_checkpoint_file, model)
        model.load_best = partial(Checkpoint.load_best, model)
        model.remove_checkpoint = partial(Checkpoint.remove_checkpoint, model)
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
        console_logger.debug(f'Loaded checkpoint {self.__class__.__name__} (from epoch %s): %s', d['epoch'],
                             checkpoint_file)
        return d

    def load_best(self):
        """Load if checkpoint exists."""
        if not os.path.isfile(self._checkpoint_file):
            return
        d = self.load_from_checkpoint(self._checkpoint_file)
        return d

    def get_checkpoint_file(self):
        return self._checkpoint_file


class DeviceTracker(nn.Module):
    @property
    def device(self):
        try:
            return next(self.parameters()).data.device
        except StopIteration:
            return torch.device('cpu')
