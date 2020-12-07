from models.concrete.single import double_branched4
from .common import *
from .test_cifar10cnn import cifar10batch


def test_double_branched4(opt, cifar10batch):
    model = double_branched4(opt)
    output = model(cifar10batch)
    assert output.shape == (16, 10)
