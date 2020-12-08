import torch

from utils import wandb_confusion_matrix


def test_wandb_confusion_matrix():
    y_true = torch.tensor([1, 1, 1, 0, 0])
    y_pred = torch.tensor([1, 1, 0, 0, 0])
    cm = wandb_confusion_matrix(y_true=y_true, y_pred=y_pred, labels=('cat', 'dog'))
    assert cm['confusion_matrix']
