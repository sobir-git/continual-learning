import torch

from models.concrete import quick_test


def very_simple(opt=None):
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


if __name__ == '__main__':
    quick_test.test(very_simple())
