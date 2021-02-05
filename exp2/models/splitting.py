import torch
import wandb

from exp2.classifier import Classifier
from exp2.feature_extractor import FeatureExtractor
from exp2.models import model_mapping, url_mapping, EfficientNet
from exp2.models.efficientnet import split_efficientnet
from exp2.models.simple_net import split_simple_net_20_classes
from exp2.utils import load_state_dict_from_url_or_path

PRETRAINED = None


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
    if isinstance(PRETRAINED, EfficientNet):
        fe, head_constructor = split_efficientnet(config, PRETRAINED)
    else:  # TODO: check if it is really simplenet
        fe, head_constructor = split_simple_net_20_classes(config, PRETRAINED)

    fe = FeatureExtractor(config, fe, device)
    fe.eval()
    fe = fe.to(device)

    def classifier_constructor(classes, idx=None) -> Classifier:
        """idx: the index assigned to the classifier"""
        n_classes = len(classes)
        net = head_constructor(n_classes=n_classes + config.other)
        return Classifier(config, net, classes, idx).to(device)

    return fe, classifier_constructor


def _log_architecture(name, model, input_data=None):
    from torchinfo import summary
    with open(wandb.run.dir + '/' + f'model_{name}.txt', 'w', encoding="utf-8") as f:
        f.write(str(model) + '\n')
        if input_data is not None:
            model_stats_str = str(summary(model, input_data=input_data, verbose=0, depth=5))
            f.write(model_stats_str)


def log_architectures(config, fe, classifier_constructor, device):
    x = torch.randn(1, *config.input_size, device=device)
    _log_architecture('feature_extractor', fe, x)
    x = fe(x)
    clf = classifier_constructor(range(config.n_classes_per_phase))
    _log_architecture('classifier', clf, x)
