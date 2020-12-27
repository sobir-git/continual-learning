import torch
from torch.nn import ConstantPad2d, Conv2d, ReLU, BatchNorm2d, Flatten, Linear, Tanh


def simple_net(n_classes=10):
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

        Conv2d(32, 64, 3),  # 3x3
        ReLU(),
        BatchNorm2d(64),

        Flatten(),
        Linear(64 * 3 * 3, n_classes*2),
        Tanh(),
        Linear(n_classes*2, n_classes),
    )