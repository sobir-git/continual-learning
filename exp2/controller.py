from typing import List

import torch
from torch import nn, optim
from torch.nn import functional as F

from exp2.feature_extractor import PRETRAINED
from exp2.utils import split_model


class Controller(nn.Module):
    def __init__(self, config, idx, n_classifiers, net=None):
        super().__init__()
        self.idx = idx
        self.config = config
        self.net = net
        self.n_classifiers = n_classifiers

    def get_optimizer(self):
        return optim.SGD(params=self.parameters(), lr=self.config.ctrl_lr, momentum=0.9)

    def forward(self, input):
        assert self.net is not None
        return self.net(input)


class CNNController(Controller):
    pass


class MLPController(Controller):
    """A two layer MLP network."""

    def __init__(self, config, idx, n_classifiers):
        super().__init__(config, idx, n_classifiers, None)
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


def create_controller(config, idx, n_classifiers, device) -> Controller:
    if config.ctrl_pos == 'before':
        _, head_constructor = split_model(config, PRETRAINED)
        net = head_constructor(n_classes=n_classifiers)
        net = CNNController(config, n_classifiers, net)
    else:
        if config.ctrl_hidden_layer_scale > 0:
            # that's MLP
            net = MLPController(config, idx, n_classifiers)
        else:
            net = LinearController(config, idx, n_classifiers)
    net = net.to(device)
    return net