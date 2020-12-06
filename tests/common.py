import pytest

from opts import parse_args


@pytest.fixture
def opt():
    args = ['--model', 'SimpleNet',
            '--dataset', 'CIFAR10',
            '--data_dir', '../data',
            '--num_classes_per_task', '1',
            '--num_tasks', '3',
            '--num_pretrain_classes', '3',
            '--num_pretrain_passes', '1'
            ]
    args = parse_args(args)
    return args