import pytest
import torch

from models.cifar10cnn import Cifar10Cnn
from .common import opt


@pytest.fixture
def cifar10cnn(opt, request):
    n_initial_filters, n_blocks, bn = request.param
    opt.bn = bn
    model = Cifar10Cnn(opt, n_initial_filters=n_initial_filters, n_blocks=n_blocks)
    return model


@pytest.fixture
def cifar10batch():
    batch_size = 16
    return torch.rand(batch_size, 3, 32, 32)


@pytest.mark.parametrize('cifar10cnn', [(4, 1, True), (4, 1, False), (16, 3, True), (16, 3, False), (16, 4, True)], indirect=True)
def test_forward(cifar10batch, cifar10cnn):
    batch_size = cifar10batch.size(0)
    assert batch_size == 16
    output = cifar10cnn(cifar10batch)
    assert output.shape == (batch_size, 10)
