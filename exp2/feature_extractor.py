from typing import Iterable

from torch import nn
from torch.utils.data import DataLoader

from exp2 import models
from exp2.classifier import Classifier
from exp2.model_state import ModelState, init_states
from exp2.utils import load_model, split_model

PRETRAINED = None


class FeatureExtractor(nn.Module):
    def __init__(self, config, net, device):
        super().__init__()
        self.device = device
        self.net = net

    def forward(self, inputs):
        return self.net(inputs)

    def _feed_with_state(self, state: ModelState):
        if state.features is None:
            state.features = self(state.inputs)
        return state

    def _feed_with_states(self, states: Iterable[ModelState]):
        yield from map(self._feed_with_state, states)

    def feed(self, loader: DataLoader = None, state: ModelState = None, states: Iterable[ModelState] = None):
        """Feed the samples to the feature extractor."""
        if loader:
            states = init_states(loader, self.device)
            return self.feed(states=states)
        elif state:
            return self._feed_with_state(state)
        else:
            return self._feed_with_states(states)


def create_models(config, device) -> (FeatureExtractor, callable):
    global PRETRAINED
    if PRETRAINED is None:
        PRETRAINED = models.simple_net(n_classes=20)
    load_model(PRETRAINED, config.pretrained)
    fe, head_constructor = split_model(config, PRETRAINED)
    fe = FeatureExtractor(config, fe, device)
    fe.eval()
    fe = fe.to(device)

    def classifier_constructor(classes, idx=None) -> Classifier:
        """idx: the index assigned to the classifier"""
        n_classes = len(classes)
        net = head_constructor(n_classes=n_classes + config.other)
        return Classifier(config, net, classes, idx).to(device)

    return fe, classifier_constructor
