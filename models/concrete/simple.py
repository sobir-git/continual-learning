import torch

from models.concrete import quick_test


def very_simple0(opt=None):
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 5),  # 4x28x28
        torch.nn.MaxPool2d(2),  # 14x14
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(8),

        torch.nn.Conv2d(8, 8, 3),  # 12x12
        torch.nn.MaxPool2d(2),  # 6x6
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(8),

        torch.nn.Flatten(),
        torch.nn.Linear(8 * 6 * 6, 10),

    )  # 83  40s  58.6% 95s


def very_simple1(opt=None):
    return torch.nn.Sequential(
        torch.nn.ConstantPad2d((2, 1, 2, 1), 0),
        torch.nn.Conv2d(3, 8, 5, stride=2),  # 16 x 16
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(8),

        torch.nn.Conv2d(8, 16, 4, stride=2),  # 7x7
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(16),

        torch.nn.Conv2d(16, 32, 3),  # 5x5
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(32),

        torch.nn.Flatten(),
        torch.nn.Linear(32 * 5 * 5, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 10),
    )


if __name__ == '__main__':
    quick_test.test(very_simple1())
