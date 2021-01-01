import copy

import torch
from torch import nn


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