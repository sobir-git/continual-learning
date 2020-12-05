import pytest
import torch

from models.bnet_base import LossEstimator
from .common import opt

in_shape = (32, 4, 4)



@pytest.fixture(params=[None, (128, 64)])
def le(opt, request):
    hidden_layers = request.param
    return LossEstimator(opt, in_shape=in_shape, hidden_layers=hidden_layers)


class TestLossEstimator:

    def test_forward(self, le):
        batch_size = 16
        fake_input = torch.rand(batch_size, *in_shape)
        out = le(fake_input)
        assert out.shape == (batch_size, 1)
        assert torch.Tensor.all(torch.gt(out, 0))
