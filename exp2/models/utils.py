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
def get_output_shape(model, input_shape=None, inputs=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if inputs is None:
        inputs = torch.randn(input_shape).unsqueeze(0)
    training = model.training
    model.eval()
    inputs = inputs.to(device)
    model = model.to(device)
    outputs = model(inputs)
    model.train(training)
    return outputs.shape[1:]
