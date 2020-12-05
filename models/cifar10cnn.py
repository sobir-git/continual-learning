import torch
from torch import nn

from models.layers import ConvBlock, FCBlock


def block(opt, in_channels, out_channels, pool):
    return nn.Sequential(
        ConvBlock(in_channels, out_channels, bn=opt.bn),
        ConvBlock(out_channels, out_channels, bn=opt.bn, pool=pool)
    )


def final_block(opt, n_channels, spatial_size):
    """
    Final block of Cifar10Cnn model, consisting of three dense layers.
    Args:
        opt:
        n_channels:
        spatial_size:
    """
    fc_in1 = n_channels * spatial_size * spatial_size
    fc_out1 = max(fc_in1 // 4, 256)
    fc_out2 = fc_out1 // 2
    return nn.Sequential(
        FCBlock(fc_in1, fc_out1, bn=opt.bn, flatten=True),
        FCBlock(fc_out1, fc_out2, bn=opt.bn),
        nn.Linear(fc_out2, 10)  # output layer
    )



class Cifar10Cnn(nn.Module):
    def __init__(self, opt, n_initial_filters=16, n_blocks=2):
        """
        Cnn model for Cifar10
        @param opt:
        @param n_initial_filters: number of initial filters, this gets doubled at each block.
        @param n_blocks: number of blocks, this doesn't include the last classification block
        """
        super().__init__()
        assert n_blocks > 0
        self.initial_block = block(opt, 3, n_initial_filters, pool='max')
        self.blocks = nn.ModuleList()

        spatial_size = 16
        f = n_initial_filters
        for i in range(2, n_blocks + 1):
            self.blocks.append(block(opt, f, f * 2, pool='max'))  # output: f*2 x 32/2**(i) x 32/2**(i)
            spatial_size //= 2
            f *= 2

        # create final block
        self.final_block = final_block(opt, n_channels=f, spatial_size=spatial_size)

    def forward(self, xb):
        out = self.initial_block(xb)
        for block in self.blocks:
            out = block(out)
        out = self.final_block(out)
        return out


if __name__ == '__main__':
    from tmp import opt
    model = Cifar10Cnn(opt, 16, 2)
    print(model)
    fake_batch = torch.rand(16, 3, 32, 32)
    model(fake_batch)
