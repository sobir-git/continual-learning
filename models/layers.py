from torch import nn


def ConvBlock(in_channels, out_channels, bn=False, pool=None):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
    ]
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if pool == 'max':
        layers.append(nn.MaxPool2d(2, 2))
    else:
        assert pool is None
    return nn.Sequential(*layers)


def FCBlock(*layers: int, flatten=False, bn=False):
    assert len(layers) > 1
    seq = []
    if flatten: seq.append(nn.Flatten())
    in_size = layers[0]
    for size in layers[1:]:
        seq.append(nn.Linear(in_size, size))
        seq.append(nn.ReLU())
        if bn:
            seq.append(nn.BatchNorm1d(size))
        in_size = size
    return nn.Sequential(*seq)