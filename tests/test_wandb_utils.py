import torch
import numpy as np
from sklearn.metrics import confusion_matrix

from utils import wandb_confusion_matrix


def test_wandb_confusion_matrix1():
    """Test by sending computed confusion matrix"""
    y_true = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2,])
    y_pred = torch.tensor([0, 0, 1, 1, 1, 1, 0, 2])
    cm = confusion_matrix(y_true, y_pred)

    assert np.allclose(cm, [
        [2, 1, 0],
        [0, 3, 0],
        [1, 0, 1]
    ])

    cm_copy = cm.copy()
    wcm = wandb_confusion_matrix(confmatrix=cm, labels=('cat', 'dog', 'bird'))
    assert wcm is not None

    # in conf matrix you should see smth like:
    #
    # cat   2    1    0
    # dog   0    2    0
    # bird  1    0    1
    #      cat  dog  bird
    # e.g cat was misclassified as dog once

    # check that it has not been modified
    assert np.allclose(cm, cm_copy)


def test_wandb_confusion_matrix2():
    """Test by sending predictions and true labels"""
    y_true = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2,])
    y_pred = torch.tensor([0, 0, 1, 1, 1, 1, 0, 2])
    wcm = wandb_confusion_matrix(y_true, y_pred, labels=('cat', 'dog', 'bird'))
    assert wcm is not None

    # in conf matrix you should see smth like:
    #
    # cat   2    1    0
    # dog   0    2    0
    # bird  1    0    1
    #      cat  dog  bird
    # e.g cat was misclassified as dog once

