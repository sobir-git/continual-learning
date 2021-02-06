from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F, Parameter

from exp2.models.utils import ClassMapping, Checkpoint, DeviceTracker


class ZeroInitLinear(nn.Linear):
    def reset_parameters(self) -> None:
        torch.fill_(self.weight, 0)
        if self.bias is not None:
            torch.fill_(self.bias, 0)


class GrowingLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device='cpu'):
        super().__init__()
        self.device = device
        self._use_bias = bias
        self._fin = [0, in_features]
        self._fout = [0, out_features]
        self.in_features = in_features
        self.out_features = out_features
        self._initw(0, 0)
        self._initw(0, 1)
        self._initw(1, 0)
        self._initw(1, 1)

    @torch.no_grad()
    def _initw(self, i, j):
        """Initialize weights transforming input group i to output group j"""
        w = torch.zeros(self._fout[j], self._fin[i], device=self.device)
        if self._use_bias:
            b = torch.zeros(self._fout[j], device=self.device)
        else:
            b = None
        self._setw(i, j, w, b)

    @torch.no_grad()
    def _setw(self, i, j, w, b=None):
        w = Parameter(w, requires_grad=True).to(self.device)
        self.register_parameter(f'w{i}{j}', w)
        if b is not None:
            b = Parameter(b, requires_grad=True).to(self.device)
        self.register_parameter(f'b{i}{j}', b)

    @torch.no_grad()
    def append(self, in_features, out_features):
        # combine existing weights
        fin0 = self._fin[0]
        fout0 = self._fout[0]
        w00 = torch.empty(self.out_features, self.in_features, device=self.device)
        w00[:fout0, :fin0] = self.w00
        w00[:fout0, fin0:] = self.w10
        w00[fout0:, :fin0] = self.w01
        w00[fout0:, fin0:] = self.w11

        if self._use_bias:
            b00 = torch.empty(self.out_features, device=self.device)
            b00[:fout0] = self.b00 + self.b10
            b00[fout0:] = self.b01 + self.b11
        else:
            b00 = None

        self._setw(0, 0, w00, b00)

        # new weights
        self._fout = [sum(self._fout), out_features]
        self._fin = [sum(self._fin), in_features]
        self.in_features = sum(self._fin)
        self.out_features = sum(self._fout)
        self._initw(0, 1)
        self._initw(1, 0)
        self._initw(1, 1)

    def forward(self, inputs):
        inputs0, inputs1 = inputs[:, :self._fin[0]], inputs[:, self._fin[0]:]
        o0 = F.linear(inputs0, self.w00, self.b00) + F.linear(inputs1, self.w10, self.b10)
        o1 = F.linear(inputs0, self.w01, self.b01) + F.linear(inputs1, self.w11, self.b11)
        o = torch.cat([o0, o1], dim=1)
        return o

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


class GrowingController(Checkpoint, ClassMapping, nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        self.device = device
        self.config = config
        self.linear = GrowingLinear(0, 0, bias=config.ctrl_bias, device=self.device)
        self.bn = nn.BatchNorm1d(0).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

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
        self.linear.w00.requires_grad = not warmup
        if self.linear.b00 is not None:
            self.linear.b00.requires_grad = not warmup

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
