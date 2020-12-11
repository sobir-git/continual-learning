import time
from unittest.mock import Mock

import numpy as np
import torch
import wandb
from sklearn.metrics import confusion_matrix
import webbrowser

from logger import Logger, traverse_dict

from .common import *


@pytest.fixture
def logger(opt):
    console_logger = Mock()
    return Logger(opt, console_logger, pref='test-logger')


def open_url(url):
    webbrowser.open(url, new=0)


class TestLogger:
    def test_log_heatmap(self, logger):
        wandb.init(project='test')
        for i in range(3):
            time.sleep(0.3)
            data = np.random.rand(3, 2)
            logger.log_heatmap('test-heatmap', data, rows=['cat', 'dog', 'bird'], columns=['1', 'main'],
                               vmin=0., vmax=1.,
                               title='Lovely test heatmap')
            logger.commit()

        open_url(wandb.run.url)
        wandb.finish()
        print('Go and check W&B dashboard if these heatmaps are correct')

    def test_log_confusion_matrix(self, logger):
        wandb.init(project='test')
        y_true = torch.tensor([2, 2, 2, 3, 3, 3, 4, 4])
        y_pred = torch.tensor([2, 2, 3, 3, 3, 3, 2, 4])
        classnames = ['plane', 'truck', 'cat', 'dog', 'bird', 'unicorn']
        cm = confusion_matrix(y_true, y_pred, labels=range(len(classnames)))
        logger.log_confusion_matrix(cm, classnames, title='Lovely confusion matrix', commit=True)

        open_url(wandb.run.url)
        wandb.finish()

    def test_log_accuracies(self, logger):
        wandb.init(project='test')

        # log single network accuracies
        y_true = torch.tensor([2, 2, 2, 3, 3, 3, 4, 4])
        y_pred = torch.tensor([2, 2, 3, 3, 3, 3, 2, 4])  # this guys is perfect to recall dogs(3)
        classnames = ['plane', 'truck', 'cat', 'dog', 'bird', 'unicorn']
        cm = confusion_matrix(y_true, y_pred, labels=range(len(classnames)))
        logger.log_accuracies(cm, classnames, commit=True)

        # log multiple branch accuracies
        confmatrices = {'main': cm}
        y_true = torch.tensor([2, 2, 2, 3, 3, 3, 4, 4])
        y_pred = torch.tensor([2, 2, 3, 3, 2, 4, 4, 4])  # this guys is perfect to recall birds(4)
        confmatrices['branch0'] = confusion_matrix(y_true, y_pred, labels=range(len(classnames)))
        logger.log_accuracies(confmatrices, classnames, commit=True)

        open_url(wandb.run.url)
        wandb.finish()

    def test__console_commit(self, logger):
        logger.log({'a': {'0': 212.2, '1': 23.0}, 'b': True, 'mock': Mock()})
        msg = logger._console_commit()
        assert 'a.0' in msg and 'a.1' in msg
        assert 'b' in msg and 'True' in msg
        assert 'mock' not in msg
        print(msg)


def test_traverse_dict():
    d = {'a': {'0': 0, '1': 1, 'two': {'2': 2}}, 'b': -1}
    x = list(traverse_dict(d))
    assert x == [
        (['a', '0'], 0),
        (['a', '1'], 1),
        (['a', 'two', '2'], 2),
        (['b'], -1)
    ]

    # test with skip_fn
    skip_fn = lambda d: '2' in d
    x1 = list(traverse_dict(d, skip_fn))
    assert x1 == [
        (['a', '0'], 0),
        (['a', '1'], 1),
        (['b'], -1)
    ]
