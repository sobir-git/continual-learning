import copy

import torch
from torch import nn
from torch.nn import ConstantPad2d, Conv2d, ReLU, BatchNorm2d, Flatten, Linear, Tanh

from exp2.models.efficientnet import EfficientNetHead
from exp2.models.utils import MultiOutputNet, get_output_shape


def simple_net(n_classes=10):
    return torch.nn.Sequential(
        ConstantPad2d((2, 1, 2, 1), 0),
        Conv2d(3, 8, 5, stride=2),  # 16 x 16
        ReLU(),
        BatchNorm2d(8),

        Conv2d(8, 16, 4, stride=2),  # 7x7
        ReLU(),
        BatchNorm2d(16),

        Conv2d(16, 32, 3),  # 5x5
        ReLU(),
        BatchNorm2d(32),

        Conv2d(32, 64, 3),  # 3x3
        ReLU(),
        BatchNorm2d(64),

        Flatten(),
        Linear(64 * 3 * 3, n_classes * 2),
        Tanh(),
        Linear(n_classes * 2, n_classes),
    )


def simple_net_20_classes(num_classes=20):
    return simple_net(num_classes)


class SimpleNetHead(EfficientNetHead):
    def __init__(self, split_pos, split_pos_lower, n_classes, model):
        nn.Module.__init__(self)
        self.neck = copy.deepcopy(model[split_pos:])
        if split_pos_lower is not None:
            self.lower_neck = copy.deepcopy(model[split_pos_lower:])
        else:
            self.lower_neck = None

        neck_input_shape = get_output_shape(model[:split_pos], input_shape=(3, 32, 32))
        neck_output_shape = get_output_shape(self.neck, neck_input_shape)
        in_features = neck_output_shape[0]
        if split_pos_lower is not None:
            lower_neck_input_shape = get_output_shape(model[:split_pos_lower], input_shape=(3, 32, 32))
            lower_neck_output_shape = get_output_shape(self.lower_neck, lower_neck_input_shape)
            in_features += lower_neck_output_shape[0]
        self.final = nn.Linear(in_features=in_features, out_features=n_classes)


def split_simple_net_20_classes(config, model):
    """
    Split Efficientnet model into feature extractor and head constructor.
    """

    if config.split_pos_lower is not None:
        fe0 = model[:config.split_pos_lower]
        fe1 = model[config.split_pos_lower:config.split_pos]
        fe = MultiOutputNet(fe0, fe1)
    else:
        fe = model[:config.split_pos]

    # freeze the feature extractor
    fe.eval()
    for param in fe.parameters():
        param.requires_grad = False

    @torch.no_grad()
    def head_constructor(n_classes: int):
        head = SimpleNetHead(config.split_pos, config.split_pos_lower, n_classes, model)
        return head

    return fe, head_constructor


if __name__ == '__main__':
    from types import SimpleNamespace
    from torchinfo import summary

    model = simple_net_20_classes(20)
    config = SimpleNamespace(split_pos=-7, split_pos_lower=4)
    fe, head_constructor = split_simple_net_20_classes(config, model)

    print('Feature extractor')
    print(fe)
    summary(fe, input_size=(1, 3, 32, 32))

    fe_outputs = fe(torch.randn(1, 3, 32, 32))
    head = head_constructor(5)
    print('Head')
    print(head)
    summary(head, input_data=fe_outputs)
