from torch import nn


class FeatureExtractor(nn.Module):
    def __init__(self, config, net, device):
        super().__init__()
        self.device = device
        self.net = net
        self.config = config

    def forward(self, inputs):
        return self.net(inputs)
