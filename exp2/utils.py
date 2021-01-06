import copy
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from exp2.data import PartialDataset, create_loader


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


def load_model(model, path):
    model.load_state_dict(torch.load(path))


@contextmanager
def evaluating(net: nn.Module):
    '''Temporarily switch to evaluation mode.'''
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        net.train(istrain)


def train_test_split(config, dataset: PartialDataset, val_size):
    """Split into train and test sets, and return dataloaders. If config.val_size is 0, validation
    dataset will be None."""
    if val_size > 0:
        trainset, valset = dataset.split(test_size=val_size)
        val_loader = create_loader(config, valset)
    else:
        trainset, valset = dataset, None
        val_loader = None
    train_loader = create_loader(config, trainset)
    return train_loader, val_loader


def get_number(v):
    """Try to get the number from v, whether itself or something wrapped inside it.
    If not a number raises TypeError
    """
    if 'float' in str(type(v)):
        return float(v)
    if type(v) in (float, int, str, tuple, list, bool):
        return v

    if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
        return v.item()

    raise TypeError('Not a number')
