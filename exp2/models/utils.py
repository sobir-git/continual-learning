import torch
from torch import nn


class MultiOutputNet(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs


@torch.no_grad()
def get_output_shape(model, input_shape=None, inputs=None):
    model_device = next(model.parameters()).device
    training = model.training
    if inputs is None:
        inputs = torch.randn(input_shape).unsqueeze(0).to(model_device)
    model.eval()
    outputs = model(inputs)
    model.train(training)
    return outputs.shape[1:]
