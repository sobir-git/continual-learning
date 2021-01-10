from types import SimpleNamespace

import torch

from exp2.feature_extractor import create_models


def test_create_models():
    config = SimpleNamespace(split_pos=-1, other=True, pretrained='efficientnet-b0', clone_head=False)
    fe, clf_constructor = create_models(config, torch.device('cpu'))
    # print(fe)
    clf = clf_constructor([1, 2, 3])
    print(clf)
    assert len(clf.net) == 3
