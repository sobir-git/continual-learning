from typing import Iterable

from torch import nn
from torch.utils.data import DataLoader

from exp2.classifier import Classifier
from exp2.model_state import ModelState, init_states
from exp2.models import model_mapping, url_mapping
from exp2.utils import split_model, load_state_dict_from_url_or_path

PRETRAINED = None


class FeatureExtractor(nn.Module):
    def __init__(self, config, net, device):
        super().__init__()
        self.device = device
        self.net = net
        self.config = config

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
            states = init_states(self.config, loader, self.device)
            return self.feed(states=states)
        elif state:
            return self._feed_with_state(state)
        else:
            return self._feed_with_states(states)


def load_pretrained(model_name):
    # construct model
    model_cls = model_mapping[model_name]
    url = url_mapping[model_name]
    if hasattr(model_cls, 'from_pretrained'):
        return model_cls.from_pretrained(url)
    state_dict = load_state_dict_from_url_or_path(url)
    model = model_cls()
    model.load_state_dict(state_dict)
    return model


def create_models(config, device) -> (FeatureExtractor, callable):
    global PRETRAINED
    if PRETRAINED is None:
        PRETRAINED = load_pretrained(config.pretrained)
    fe, head_constructor = split_model(config, PRETRAINED)
    fe = FeatureExtractor(config, fe, device)
    fe.eval()
    if config.torch['half']:
        fe = fe.half()
    fe = fe.to(device)

    def classifier_constructor(classes, idx=None) -> Classifier:
        """idx: the index assigned to the classifier"""
        n_classes = len(classes)
        net = head_constructor(n_classes=n_classes + config.other)
        if config.torch['half']:
            net = net.half()
        return Classifier(config, net, classes, idx).to(device)

    return fe, classifier_constructor
