import torch
import numpy as np
from sklearn.metrics import confusion_matrix

from utils import wandb_confusion_matrix


def test_wandb_confusion_matrix1():
    """Test by sending computed confusion matrix"""
    y_true = torch.tensor([2, 2, 2, 3, 3, 3, 4, 4])
    y_pred = torch.tensor([2, 2, 3, 3, 3, 3, 2, 4])
    classnames = ['plane', 'truck', 'cat', 'dog', 'bird', 'unicorn']
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classnames)))

    assert np.allclose(cm, [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 2, 1, 0, 0],
        [0, 0, 0, 3, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ])

    cm_copy = cm.copy()
    wcm = wandb_confusion_matrix(confmatrix=cm, classnames=classnames)
    assert wcm is not None

    # in conf matrix you should see smth like:
    # plane 0    0    0    0    0    0
    # truck 0    0    0    0    0    0
    # cat   0    0    2    1    0    0
    # dog   0    0    0    3    0    0
    # bird  0    0    1    0    1    0
    # unic  0    0    0    0    0    0
    #      plan truc cat  dog  bird unic
    # e.g cat was misclassified as dog once

    # check that it has not been modified
    assert np.allclose(cm, cm_copy)
