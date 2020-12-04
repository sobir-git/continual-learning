import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class LossEstimator(nn.Module):
    def __init__(self, in_shape, hidden_layers=None):
        super(LossEstimator, self).__init__()
        layers = [Flatten()]
        self.in_shape = in_shape
        in_size = np.prod(in_shape)
        if hidden_layers:
            for size in hidden_layers:
                layers.append(nn.Linear(in_size, size))
                layers.append(nn.ReLU())
                in_size = size
        layers.append(nn.Linear(in_size, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, np.prod(self.in_shape))
        out = self.layers(x)
        return out


class Branch(nn.Module):
    def __init__(self, stem: nn.Module, in_shape):
        super(Branch, self).__init__()
        self.stem = stem
        self.le = LossEstimator(in_shape=in_shape)

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


class Flatten(nn.Module):
    def forward(self, input):
        """
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0) # of batches,
        will flatten to be 1 * nb_elements.
        """
        batch_size = input.size(0)
        out = input.view(batch_size, -1)
        return out  # (batch_size, *size)


class BranchNet(nn.Module):
    pass
