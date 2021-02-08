from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F, Parameter

from exp2.model_state import get_class_weights
from exp2.models.utils import ClassMapping, Checkpoint, DeviceTracker
from utils import AverageMeter


class ZeroInitLinear(nn.Linear):
    def reset_parameters(self) -> None:
        torch.fill_(self.weight, 0)
        if self.bias is not None:
            torch.fill_(self.bias, 0)


class GrowingLinear(DeviceTracker, nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
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
            b = torch.zeros(self._fout[j])
        else:
            b = None
        self._setw(i, j, w, b)

    @torch.no_grad()
    def _setw(self, i, j, w, b=None):
        w = w.to(self.device)
        w = Parameter(w, requires_grad=True)
        self.register_parameter(f'w{i}{j}', w)
        if b is not None:
            b = b.to(self.device)
            b = Parameter(b, requires_grad=True)
        self.register_parameter(f'b{i}{j}', b)

    @torch.no_grad()
    def append(self, in_features, out_features):
        # combine existing weights
        fin0 = self._fin[0]
        fout0 = self._fout[0]
        w00 = torch.empty(self.out_features, self.in_features)
        w00[:fout0, :fin0] = self.w00
        w00[:fout0, fin0:] = self.w10
        w00[fout0:, :fin0] = self.w01
        w00[fout0:, fin0:] = self.w11

        if self._use_bias:
            b00 = torch.empty(self.out_features)
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
        if self._fin[0] > 0:
            o0 = F.linear(inputs0, self.w00, self.b00) + F.linear(inputs1, self.w10, self.b10)
            o1 = F.linear(inputs0, self.w01, self.b01) + F.linear(inputs1, self.w11, self.b11)
        else:
            o0 = F.linear(inputs1, self.w10, self.b10)
            o1 = F.linear(inputs1, self.w11, self.b11)
        o = torch.cat([o0, o1], dim=1)
        return o

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


class BiC(Checkpoint, ClassMapping, DeviceTracker, nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = Parameter(torch.tensor([1.], dtype=torch.float32, requires_grad=True))
        self.beta = Parameter(torch.tensor([0.], dtype=torch.float32, requires_grad=True))
        self.mask = None

    @torch.no_grad()
    def reset_params(self):
        torch.fill_(self.alpha, 1.)
        torch.fill_(self.beta, 0.)

    def set_biased_classes(self, classes):
        idx_on = [self._classes_inv[cls] for cls in classes]
        mask = torch.zeros(len(self.classes), dtype=torch.bool)
        mask[idx_on] = 1
        self.set_mask(mask)

    def set_mask(self, mask: torch.Tensor):
        self.mask = mask.to(self.device)

    def forward(self, inputs):
        outputs = inputs.clone()
        mapped_outputs = outputs[:, self.mask] * self.alpha + self.beta
        outputs[:, self.mask] = mapped_outputs
        return outputs

    def train_(self, config, loader, forward_fn, logger):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=self.parameters(), lr=config.bic_lr, momentum=0.9)
        self.remove_checkpoint()

        device = self.device
        for epoch in range(1, config.bic_epochs + 1):
            loss_meter = AverageMeter()
            for batch in loader:  # inputs, labels, ids
                inputs, labels, _ = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = forward_fn(batch)
                local_labels = self.localize_labels(labels)
                outputs = self(outputs)
                loss = criterion(outputs, local_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_size = len(labels)
                loss_meter.update(loss, batch_size)

            # end of epoch, checkpoint and log metrics
            self.checkpoint(None, loss_meter.avg, epoch)
            logger.log({'bic_train_loss': loss_meter.avg})
            logger.commit()

        # load best bic checkpoint
        self.load_best()
        logger.console.info(f"BiC parameters: {self.alpha.item(), self.beta.item()}")


class GrowingController(DeviceTracker, Checkpoint, ClassMapping, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = GrowingLinear(0, 0, bias=config.ctrl_bias)
        self.bn = nn.BatchNorm1d(0)
        self.bic = BiC()
        self._bic_active = False
        self.activation = self._get_activation(config.ctrl_activation)

    def _get_activation(self, activation):
        if activation == 'softmax':
            def softmax(inputs):
                return torch.softmax(inputs, dim=1)

            return softmax
        elif activation in [None, 'identity']:
            return nn.Identity()
        if hasattr(torch, activation):
            return getattr(torch, activation)
        return getattr(F, activation)

    def set_bic_state(self, activated):
        self._bic_active = activated

    def phase_start(self, phase):
        balance = self.config.ctrl_balance_classes
        self.criterion = nn.CrossEntropyLoss(weight=get_class_weights(self.config, phase, balance)).to(self.device)

    def get_predictions(self, outputs) -> np.ndarray:
        """Get predictions given controller outputs."""
        local_predictions = torch.argmax(outputs, 1)
        return self.globalize_labels(local_predictions, 'cpu').numpy()

    def get_loss(self, outputs, labels):
        labels = self.localize_labels(labels)
        loss = self.criterion(outputs, labels)
        return loss

    def set_warmup(self, warmup=True):
        """Freeze linear.l00 if warmup, else unfreeze."""
        level = self.config.warmup_level
        if level >= 1:
            self.linear.w00.requires_grad = not warmup
        if level >= 2:
            self.linear.w10.requires_grad = not warmup
        if level >= 3:
            self.linear.w01.requires_grad = not warmup

    def extend(self, new_classes, n_new_inputs):
        super(GrowingController, self).extend(new_classes)
        self.bic.extend(new_classes)
        self.bic.set_biased_classes(new_classes)
        self.linear.append(n_new_inputs, len(new_classes))
        self.bn = nn.BatchNorm1d(self.bn.num_features + n_new_inputs).to(self.device)

        # checkpoint becames incompatible, so we remove it
        self.remove_checkpoint()

    def forward(self, clf_outs: List[torch.Tensor]):
        """Assumes the classifier raw outputs in a list."""
        # apply softmax
        act = self.activation
        inputs = [act(i) for i in clf_outs]
        # apply batchnorm
        x = torch.cat(inputs, dim=1)
        x = self.bn(x)
        if self.config.ctrl_linear_layer:
            x = self.linear(x)
        if self._bic_active:
            x = self.bic(x)
        return x
