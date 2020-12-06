from unittest.mock import Mock

import torch

from training import Trainer
from .common import *


@pytest.fixture
def simple_classifier():
    n_classes = 10

    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.MaxPool2d((2, 2), stride=2),  # 8 x 16 x 16
        torch.nn.ReLU(),

        torch.nn.Conv2d(8, 16, 3, padding=1),
        torch.nn.MaxPool2d((2, 2), stride=2),  # 16 x 8 x 8
        torch.nn.ReLU(),

        torch.nn.Flatten(),
        torch.nn.Linear(16 * 8 * 8, n_classes),
        torch.nn.ReLU()
    )


@pytest.fixture
def trainer(opt):
    logger, device = Mock(), torch.device('cpu')
    return Trainer(opt, logger, device, type='pre')


def test_train(opt, trainer, simple_classifier, monkeypatch):
    import wandb
    n_samples = 280
    opt.batch_size = 4
    monkeypatch.setattr(wandb, 'log', print)
    loader = Mock()
    universal_inp = torch.randn(opt.batch_size, 3, 32, 32)
    labels = torch.randint(0, 10, (opt.batch_size,))
    loader.__len__ = Mock(return_value=n_samples)
    loader.__iter__ = Mock(return_value=((universal_inp, labels) for _ in range(len(loader))))

    trainer.train(loader=loader, optimizer=Mock(), model=simple_classifier, step=1)


def test_test(trainer):
    assert False
