from torch import nn

from models.bnet1 import BranchNet1
from models.bnet_base import Branch
from models.cifar10cnn import block, final_block, Cifar10Cnn

__all__ = ['single_branched3', 'double_branched4', 'cnn_model3', 'cnn_model4']
__n_initial_channels = 16


def single_branched3(opt):
    f = __n_initial_channels
    base = nn.Sequential(
        block(opt, 3, f, pool='max'),  # f x 32 x 32
        block(opt, f, f * 2, pool='max')  # f*2 x 8 x 8
    )
    spatial_size = 32 // 2 ** len(base)  # 8
    branch_stem = nn.Sequential(
        block(opt, f * 2, f * 4, pool='max'),  # f*4, 4, 4
        final_block(opt, n_channels=f * 4, spatial_size=spatial_size // 2)  # divide by two because of previous block
    )
    branch = Branch(opt, stem=branch_stem, in_shape=(f * 2, 8, 8))
    net = BranchNet1(base, branches=[branch])
    return net


def double_branched4(opt):
    f = __n_initial_channels
    base = nn.Sequential(
        block(opt, 3, f, pool='max'),  # f x 32 x 32
        block(opt, f, f * 2, pool='max'),  # f*2 x 8 x 8
        block(opt, f * 2, f * 4, pool='max')  # f*4 x 4 x 4
    )
    spatial_size = 32 // 2 ** len(base)  # 4

    # create branches
    n_branches = 2
    branches = []
    for i in range(n_branches):
        branch_stem = nn.Sequential(
            nn.Conv2d(f * 4, f, kernel_size=1),  # n_parameters = f*4*f
            block(opt, f, f * 2, pool='max'),  # f*2 x 2 x 2
            # in the conventional one the number of parameters of a block is fin * (fin + fout) * k^2
            final_block(opt, n_channels=f * 2, spatial_size=spatial_size // 2)
            # divide by two because of previous block
        )
        branch = Branch(opt, stem=branch_stem, in_shape=(f * 2 ** (len(base) - 1), spatial_size, spatial_size))
        branches.append(branch)

    net = BranchNet1(base, branches=branches)
    return net


def cnn_model3(opt):
    return Cifar10Cnn(opt, n_initial_filters=__n_initial_channels, n_blocks=3)


def cnn_model4(opt):
    return Cifar10Cnn(opt, n_initial_filters=__n_initial_channels, n_blocks=4)
