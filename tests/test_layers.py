import itertools

import pytest
import torch

from models.layers import FCBlock, ConvBlock


def test_fcblock():
    out_features = 10
    fc = FCBlock(*[64, 32, out_features], bn=True)
    fake_input = torch.randn(16, 64)
    out = fc(fake_input)
    assert out.shape == (fake_input.size(0), out_features)


@pytest.mark.parametrize("bn,pool", itertools.product([True, False], [None, 'max']))
def test_conv_block(bn, pool):
    in_channels = 64
    out_channels = 128
    size = 8
    batch_size = 10
    conv = ConvBlock(in_channels, out_channels, bn=bn, pool=pool)
    fake_input = torch.randn(batch_size, in_channels, size, size)
    out = conv(fake_input)
    out_size = size if pool is None else size//2
    assert out.shape == (batch_size, out_channels, out_size, out_size)
