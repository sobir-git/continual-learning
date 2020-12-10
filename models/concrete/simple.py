import torch
from torch.nn import Sequential, Linear, Conv2d, ReLU, BatchNorm2d, Flatten, Tanh, ConstantPad2d, MaxPool2d

from models.bnet_base import BranchNet, Branch
from models.concrete import quick_test


def very_simple0(opt=None):
    return torch.nn.Sequential(
        Conv2d(3, 8, 5),  # 4x28x28
        MaxPool2d(2),  # 14x14
        ReLU(),
        BatchNorm2d(8),

        Conv2d(8, 8, 3),  # 12x12
        MaxPool2d(2),  # 6x6
        ReLU(),
        BatchNorm2d(8),

        Flatten(),
        Linear(8 * 6 * 6, 10),

    )  # 83  40s  58.6% 95s


def very_simple1(opt=None):
    return torch.nn.Sequential(
        ConstantPad2d((2, 1, 2, 1), 0),
        Conv2d(3, 8, 5, stride=2),  # 16 x 16
        ReLU(),
        BatchNorm2d(8),

        Conv2d(8, 16, 4, stride=2),  # 7x7
        ReLU(),
        BatchNorm2d(16),

        Conv2d(16, 32, 3),  # 5x5
        ReLU(),
        BatchNorm2d(32),

        Flatten(),
        Linear(32 * 5 * 5, 16),
        Tanh(),
        Linear(16, 10),
    )


def _simple_base():
    return Sequential(
        ConstantPad2d((2, 1, 2, 1), 0),
        Conv2d(3, 8, 5, stride=2),  # 16 x 16
        ReLU(),
        BatchNorm2d(8),

        Conv2d(8, 16, 4, stride=2),  # 7x7
        ReLU(),
        BatchNorm2d(16)
    )


def _simple_branch():
    return Sequential(
        Conv2d(16, 32, 3),  # 5x5
        ReLU(),
        BatchNorm2d(32),

        Flatten(),
        Linear(32 * 5 * 5, 16),
        Tanh(),
        Linear(16, 10),
    )


def simple_double_branch(opt):
    base = _simple_base()
    branches = [
        Branch(opt, _simple_branch(), in_shape=(16, 7, 7)),
        Branch(opt, _simple_branch(), in_shape=(16, 7, 7)),
    ]
    return BranchNet(base, branches=branches)


if __name__ == '__main__':
    quick_test.test(very_simple1())
