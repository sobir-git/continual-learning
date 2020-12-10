from models.concrete.single import double_branched4
from .common import *
from .test_cifar10cnn import cifar10batch


def test_double_branched4(opt, cifar10batch):
    model = double_branched4(opt)
    outputs, estimated_losses = model(cifar10batch)
    assert outputs.shape == (16, 2, 10)
    assert estimated_losses.shape == (16, 2)
