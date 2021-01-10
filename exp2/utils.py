import copy
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn
from torch.hub import load_state_dict_from_url


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

    def head_constructor(n_classes: int):
        # replace the classification layer from the head with the one matching number of classes
        layer = nn.Linear(in_features=in_features, out_features=n_classes)
        newhead = copy.deepcopy(head)
        newhead = nn.Sequential(*newhead, layer)
        if not config.clone_head:
            _reset_parameters(newhead)
        return newhead

    return back, head_constructor


def uri_validator(x):
    from urllib.parse import urlparse
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False


def load_state_dict_from_url_or_path(path):
    if uri_validator(path):
        return load_state_dict_from_url(path)
    return torch.load(path)


@contextmanager
def evaluating(net: nn.Module):
    '''Temporarily switch to evaluation mode.'''
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        net.train(istrain)


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
