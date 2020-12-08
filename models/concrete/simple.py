import torch

from models.concrete import quick_test


def very_simple(opt=None):
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),  # 8 x 32 x 32
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),  # 8 x 16 x 16
        torch.nn.Conv2d(8, 3, 1),  # 3 x 16 x 16
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),  # 3 x 8 x 8
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 8 * 8, 10),
        torch.nn.ReLU()
    )


if __name__ == '__main__':
    quick_test.test(very_simple())
