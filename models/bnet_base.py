import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from models.layers import FCBlock


class LossEstimator(nn.Module):
    def __init__(self, opt, in_shape, hidden_layers=None):
        super(LossEstimator, self).__init__()
        layers = [torch.nn.Flatten()]
        self.in_shape = in_shape
        in_size = np.prod(in_shape)
        if hidden_layers:
            layers.append(FCBlock(in_size, *hidden_layers, bn=opt.bn))
        layers.append(nn.Linear(hidden_layers[-1] if hidden_layers else in_size, 1))
        layers.append(nn.Softplus())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out


class Branch(nn.Module):
    def __init__(self, opt, stem: nn.Module, in_shape):
        super(Branch, self).__init__()
        self.stem = stem
        self.le = LossEstimator(opt, in_shape=in_shape)

    def estimate_loss(self, x) -> torch.Tensor:
        return self.le(x)

    def forward(self, x):
        return self.stem(x)


def loss_estimation_loss(est, actual, reduction='mean'):
    l = F.mse_loss(est, actual)
    if reduction == 'none':
        return l
    if reduction == 'mean':
        return l.mean()
    elif reduction == 'sum':
        return l.sum()
    raise ValueError(f'Invalid value for reduction: {reduction}')


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x


class BranchNet(nn.Module):
    pass
