"""
Sourced with modifications from: https://colab.research.google.com/github/pytorch/ignite/blob/master/examples/notebooks/EfficientNet_Cifar100_finetuning.ipynb
"""

import math
from collections import OrderedDict

import torch
from torch import nn

from exp2.models.utils import MultiOutputNet, get_output_shape
from exp2.utils import load_state_dict_from_url_or_path


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out')
    elif isinstance(module, nn.Linear):
        init_range = 1.0 / math.sqrt(module.weight.shape[1])
        nn.init.uniform_(module.weight, a=-init_range, b=init_range)


# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class SqueezeExcitation(nn.Module):

    def __init__(self, inplanes, se_planes):
        super(SqueezeExcitation, self).__init__()
        self.reduce_expand = nn.Sequential(
            nn.Conv2d(inplanes, se_planes,
                      kernel_size=1, stride=1, padding=0, bias=True),
            MemoryEfficientSwish(),
            nn.Conv2d(se_planes, inplanes,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_se = torch.mean(x, dim=(-2, -1), keepdim=True)
        x_se = self.reduce_expand(x_se)
        return x_se * x


class MBConv(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, stride,
                 expand_rate=1.0, se_rate=0.25,
                 drop_connect_rate=0.2):
        super(MBConv, self).__init__()

        expand_planes = int(inplanes * expand_rate)
        se_planes = max(1, int(inplanes * se_rate))

        self.expansion_conv = None
        if expand_rate > 1.0:
            self.expansion_conv = nn.Sequential(
                nn.Conv2d(inplanes, expand_planes,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(expand_planes, momentum=0.01, eps=1e-3),
                MemoryEfficientSwish()
            )
            inplanes = expand_planes

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(inplanes, expand_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=expand_planes,
                      bias=False),
            nn.BatchNorm2d(expand_planes, momentum=0.01, eps=1e-3),
            MemoryEfficientSwish()
        )

        self.squeeze_excitation = SqueezeExcitation(expand_planes, se_planes)

        self.project_conv = nn.Sequential(
            nn.Conv2d(expand_planes, planes,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes, momentum=0.01, eps=1e-3),
        )

        self.with_skip = stride == 1
        self.drop_connect_rate = torch.tensor(drop_connect_rate, requires_grad=False)

    def _drop_connect(self, x):
        keep_prob = 1.0 - self.drop_connect_rate
        drop_mask = torch.rand(x.shape[0], 1, 1, 1) + keep_prob
        drop_mask = drop_mask.type_as(x)
        drop_mask.floor_()
        return drop_mask * x / keep_prob

    def forward(self, x):
        z = x
        if self.expansion_conv is not None:
            x = self.expansion_conv(x)

        x = self.depthwise_conv(x)
        x = self.squeeze_excitation(x)
        x = self.project_conv(x)

        # Add identity skip
        if x.shape == z.shape and self.with_skip:
            if self.training and self.drop_connect_rate is not None:
                self._drop_connect(x)
            x += z
        return x


class EfficientNet(nn.Module):
    name = 'efficientnet-b0'

    @classmethod
    def from_pretrained(cls, url):
        # works with state dicts like 'http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth'
        # A basic remapping is required
        model = cls()
        state_dict = load_state_dict_from_url_or_path(url)

        mapping = {
            k: v for k, v in zip(state_dict.keys(), model.state_dict().keys())
        }
        mapped_state_dict = OrderedDict([
            (mapping[k], v) for k, v in state_dict.items()
        ])
        model.load_state_dict(mapped_state_dict)
        return model

    def _setup_repeats(self, num_repeats):
        return int(math.ceil(self.depth_coefficient * num_repeats))

    def _setup_channels(self, num_channels):
        num_channels *= self.width_coefficient
        new_num_channels = math.floor(num_channels / self.divisor + 0.5) * self.divisor
        new_num_channels = max(self.divisor, new_num_channels)
        if new_num_channels < 0.9 * num_channels:
            new_num_channels += self.divisor
        return new_num_channels

    def __init__(self, num_classes=1000,
                 width_coefficient=1.0,
                 depth_coefficient=1.0,
                 se_rate=0.25,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2):
        super(EfficientNet, self).__init__()

        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.divisor = 8

        list_channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        list_channels = [self._setup_channels(c) for c in list_channels]

        list_num_repeats = [1, 2, 2, 3, 3, 4, 1]
        list_num_repeats = [self._setup_repeats(r) for r in list_num_repeats]

        expand_rates = [1, 6, 6, 6, 6, 6, 6]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]

        # Define stem:
        self.stem = nn.Sequential(
            nn.Conv2d(3, list_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(list_channels[0], momentum=0.01, eps=1e-3),
            MemoryEfficientSwish()
        )

        # Define MBConv blocks
        blocks = []
        counter = 0
        num_blocks = sum(list_num_repeats)
        for idx in range(7):

            num_channels = list_channels[idx]
            next_num_channels = list_channels[idx + 1]
            num_repeats = list_num_repeats[idx]
            expand_rate = expand_rates[idx]
            kernel_size = kernel_sizes[idx]
            stride = strides[idx]
            drop_rate = drop_connect_rate * counter / num_blocks

            name = "MBConv{}_{}".format(expand_rate, counter)
            blocks.append((
                name,
                MBConv(num_channels, next_num_channels,
                       kernel_size=kernel_size, stride=stride, expand_rate=expand_rate,
                       se_rate=se_rate, drop_connect_rate=drop_rate)
            ))
            counter += 1
            for i in range(1, num_repeats):
                name = "MBConv{}_{}".format(expand_rate, counter)
                drop_rate = drop_connect_rate * counter / num_blocks
                blocks.append((
                    name,
                    MBConv(next_num_channels, next_num_channels,
                           kernel_size=kernel_size, stride=1, expand_rate=expand_rate,
                           se_rate=se_rate, drop_connect_rate=drop_rate)
                ))
                counter += 1

        self.blocks = nn.Sequential(OrderedDict(blocks))

        # Define head
        self.head = nn.Sequential(
            nn.Conv2d(list_channels[-2], list_channels[-1],
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(list_channels[-1], momentum=0.01, eps=1e-3),
            MemoryEfficientSwish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(list_channels[-1], num_classes)
        )

        self.apply(init_weights)

    def forward(self, x):
        f = self.stem(x)
        f = self.blocks(f)
        y = self.head(f)
        return y

    def _get_sequence(self):
        return [
            self.stem,
            *self.blocks,
            self.head[:-3],  # ends with AdaptiveAvgPool2D
            self.head[-3:]  # flatten, dropout, linear
        ]

    def __getitem__(self, item):
        """Convenient indexing for splitting the model into feature extractor and head."""
        sequence = self._get_sequence()

        if isinstance(item, int):
            return sequence[item]

        items = sequence[item]
        ll = []
        for i in items:
            if isinstance(i, nn.Sequential):
                ll.extend(i)
            else:
                ll.append(i)
        return nn.Sequential(*ll)

    def __len__(self):
        return len(self._get_sequence())


def flatten_cnn(input_shape, n_classes, expand_rate=1.5, h_threshold=7):
    """Aguments CNN until it is not larger that 7x7 spatially. The number of parameters
    will be O(n_classes)."""

    def dw_block(ch_in, ch_out):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out,
                      kernel_size=3, stride=2,
                      padding=1, groups=ch_in,
                      bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch_out, out_channels=ch_in, kernel_size=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

    def conv_block(ch_in, ch_out):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

    def init_block(ch_in, ch_out):
        return nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
        )

    h, w = input_shape[1:]  # C, H, W
    if h <= h_threshold:
        neck = [init_block(input_shape[0], n_classes * 4)]
    else:
        c = int(math.sqrt(n_classes))
        c_out = (c + 2) * 2
        neck = [init_block(input_shape[0], c_out)]
        c = c_out

        while h > h_threshold:
            c_out = int(c * expand_rate)
            neck.extend([
                conv_block(c, c_out),
                dw_block(c_out, c_out),
            ])
            h = (h + 1) // 2
            c = c_out

    neck.extend([
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    ])

    return nn.Sequential(*neck)


class EfficientNetHead(nn.Module):
    def __init__(self, input_shape, n_classes, lower_input_shape=None):
        super().__init__()
        self.neck = flatten_cnn(input_shape, n_classes)
        self.lower_neck = None

        neck_output_shape = get_output_shape(self.neck, input_shape)
        in_features = neck_output_shape[0]
        if lower_input_shape is not None:
            self.lower_neck = flatten_cnn(lower_input_shape, n_classes)
            in_features += get_output_shape(self.lower_neck, lower_input_shape)[0]

        self.final = nn.Linear(in_features=in_features, out_features=n_classes)

    def forward(self, lower_inputs, inputs=None):
        if type(lower_inputs) in [list, tuple]:
            lower_inputs, inputs = lower_inputs
        if inputs is None:
            outputs = self.neck(lower_inputs)
        else:
            outputs = self.neck(inputs)
            lower_outputs = self.lower_neck(lower_inputs)
            outputs = torch.cat([outputs, lower_outputs], 1)
        return self.final(outputs)


def split_efficientnet(config, model: EfficientNet):
    """
    Split Efficientnet model into feature extractor and head constructor.
    The returned feature extractor is frozen and in "eval" mode.
    """

    head_input_shape = None
    head_lower_input_shape = None

    if config.split_pos_lower is not None:
        fe0 = model[:config.split_pos_lower]
        fe1 = model[config.split_pos_lower:config.split_pos]

        fe = MultiOutputNet(fe0, fe1)
        batch = torch.randn(1, 3, 224, 224)
        outputs = fe(batch)
        head_input_shape = outputs[1].shape[1:]
        head_lower_input_shape = outputs[0].shape[1:]
    else:
        fe = model[:config.split_pos]
        head_input_shape = get_output_shape(fe, input_shape=(3, 224, 224))

    # freeze the feature extractor
    fe.eval()
    for param in fe.parameters():
        param.requires_grad = False

    @torch.no_grad()
    def head_constructor(n_classes: int):
        head = EfficientNetHead(input_shape=head_input_shape, n_classes=n_classes,
                                lower_input_shape=head_lower_input_shape)
        return head

    return fe, head_constructor


if __name__ == '__main__':
    from torchinfo import summary

    url = 'http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth'
    model = EfficientNet.from_pretrained(url)
    batch = torch.rand(5, 3, 224, 224)
    # outputs = model.stem(batch)
    # print('input shape:', batch.shape[1:])
    # print('model.stem output shape:', outputs.shape[1:])

    # n = len(model.blocks)
    # for i in range(n):
    #     outputs = model.blocks[i](outputs)
    #     print(f'model.block[{i}] output shape:', outputs.shape[1:])

    split_pos = -2
    split_pos_lower = 0
    input_shape = model[:split_pos](batch).shape[1:]
    lower_input_shape = model[:split_pos_lower](batch).shape[1:]

    head = EfficientNetHead(input_shape, 10, lower_input_shape)

    feat0 = model[:split_pos_lower]
    feat1 = model[split_pos_lower:split_pos]
    out0 = feat0(batch)
    out1 = feat1(out0)
    print(head)
    print(out0.shape)
    print(out1.shape)
    summary(head, input_data=(out0, out1))

    #
    # print(summary(model, input_data=batch, depth=5))
