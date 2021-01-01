from torch import nn

from exp2 import models
from exp2.classifier import Classifier
from exp2.utils import load_model, split_model

PRETRAINED = None


class FeatureExtractor(nn.Module):
    def __init__(self, config, net):
        super().__init__()
        self.net = net

    def forward(self, input):
        return self.net(input)


def create_models(config, device) -> (FeatureExtractor, callable):
    global PRETRAINED
    if PRETRAINED is None:
        PRETRAINED = models.simple_net(n_classes=20)
    load_model(PRETRAINED, config.pretrained)
    fe, head_constructor = split_model(config, PRETRAINED)
    fe = FeatureExtractor(config, fe)
    fe.eval()
    fe = fe.to(device)

    def classifier_constructor(classes, id=None) -> Classifier:
        """id: the id assigned to the classifier"""
        n_classes = len(classes)
        net = head_constructor(n_classes=n_classes + config.other)
        return Classifier(config, net, classes, id).to(device)

    return fe, classifier_constructor