from abc import abstractmethod
from typing import List

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F, Parameter

from exp2.classifier import Classifier
from exp2.models.utils import ClassMapping, Checkpoint, DeviceTracker
from exp2.models.splitting import PRETRAINED
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
        opt_name = self.config.ctrl['optimizer']
        common = dict(params=self.parameters(), lr=self.config.ctrl['lr'])
        if opt_name == 'SGD':
            return optim.SGD(**common, momentum=0.9)
        elif opt_name == 'Adam':
            return optim.Adam(**common)
        else:
            raise ValueError("Optimizer should be one of 'SGD' or 'Adam'")

    @abstractmethod
    def forward(self, input):
        ...

    def get_predictions(self, outputs) -> np.ndarray:
        if outputs.size(0) == 0:
            return []
        return torch.argmax(outputs, 1).cpu().numpy()

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
        hidden_layer_scale = config.ctrl['hidden_layer_scale']
        activation = config.ctrl['hidden_activation']
        hidden_layer_size = int(n_classifiers * hidden_layer_scale)
        return nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features=in_features, out_features=hidden_layer_size),
            nn.BatchNorm1d(hidden_layer_size),
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
        return nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features=in_features, out_features=n_classifiers),
        )


def create_controller(config, idx, classifiers, device) -> Controller:
    if config.ctrl['pos'] == 'before':
        _, head_constructor = split_model(config, PRETRAINED)
        net = head_constructor(n_classes=classifiers)
        net = CNNController(config, classifiers, net)
    else:
        if config.ctrl['hidden_layer_scale'] > 0:
            # that's MLP
            net = MLPController(config, idx, classifiers)
        else:
            net = LinearController(config, idx, classifiers)
    net = net.to(device)
    return net


class ZeroInitLinear(nn.Linear):
    def reset_parameters(self) -> None:
        torch.fill_(self.weight, 0)
        if self.bias is not None:
            torch.fill_(self.bias, 0)


class GrowingLinear(DeviceTracker, nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self._use_bias = True
        self._fin = [0, in_features]
        self._fout = [0, out_features]
        self.in_features = in_features
        self.out_features = out_features
        self._initlin(0, 0)
        self._initlin(0, 1)
        self._initlin(1, 0)
        self._initlin(1, 1)

    @torch.no_grad()
    def _initlin(self, i, j):
        """Initialize weights transforming input group i to output group j"""
        lin = ZeroInitLinear(self._fin[i], self._fout[j], bias=self._use_bias)
        self._setlin(i, j, lin)

    def _setlin(self, i, j, lin: ZeroInitLinear):
        lin = lin.to(self.device)
        self.add_module(f'l{i}{j}', lin)

    @torch.no_grad()
    def append(self, in_features, out_features):
        # combine existing weights and biases
        fin0 = self._fin[0]
        fout0 = self._fout[0]
        lin = ZeroInitLinear(self.in_features, self.out_features).to(self.device)
        w = lin.weight.data  # weights
        b = lin.bias.data  # biases
        w[:fout0, :fin0] = self.l00.weight.data
        w[:fout0, fin0:] = self.l10.weight.data
        w[fout0:, :fin0] = self.l01.weight.data
        w[fout0:, fin0:] = self.l11.weight.data

        torch.fill_(b, 0)
        b[:fout0] += self.l00.bias.data
        b[:fout0] += self.l10.bias.data
        b[fout0:] += self.l01.bias.data
        b[fout0:] += self.l11.bias.data
        self._setlin(0, 0, lin)

        # new weights
        self._fout = [sum(self._fout), out_features]
        self._fin = [sum(self._fin), in_features]
        self.in_features = sum(self._fin)
        self.out_features = sum(self._fout)
        self._initlin(0, 1)
        self._initlin(1, 0)
        self._initlin(1, 1)

    def forward(self, inputs):
        inputs0, inputs1 = inputs[:, :self._fin[0]], inputs[:, self._fin[0]:]
        o0 = self.l00(inputs0) + self.l10(inputs1)
        o1 = self.l01(inputs0) + self.l11(inputs1)
        o = torch.cat([o0, o1], dim=1)
        return o

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


class GrowingController(DeviceTracker, Checkpoint, ClassMapping, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = GrowingLinear(0, 0, bias=config.ctrl_bias)
        self.bn = nn.BatchNorm1d(0)
        self.criterion = nn.CrossEntropyLoss()

    def get_predictions(self, outputs) -> np.ndarray:
        """Get predictions given controller outputs."""
        local_predictions = torch.argmax(outputs, 1)
        return self.globalize_labels(local_predictions, 'cpu')

    def get_loss(self, outputs, labels):
        labels = self.localize_labels(labels)
        loss = self.criterion(outputs, labels)
        return loss

    def set_warmup(self, warmup=True):
        """Freeze linear.l00 if warmup, else unfreeze."""
        for p in self.linear.l00.parameters():
            p.requires_grad = not warmup

    def extend(self, new_classes, n_new_inputs):
        super(GrowingController, self).extend(new_classes)
        self.linear.append(n_new_inputs, len(new_classes))
        self.bn = nn.BatchNorm1d(self.bn.num_features + n_new_inputs).to(self.device)

        # checkpoint becames incompatible, so we remove it
        self.remove_checkpoint()

    def forward(self, clf_outs: List[torch.Tensor]):
        """Assumes the classifier raw outputs in a list."""
        # apply softmax
        inputs = [F.softmax(i, dim=1) for i in clf_outs]
        # apply batchnorm
        x = torch.cat(inputs, dim=1)
        x = self.bn(x)
        x = self.linear(x)
        return x
