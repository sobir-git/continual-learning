from unittest.mock import Mock

import numpy as np
import torch
from torch.utils.data import Dataset

from dataloader import VisionDataset, ReorderTargets
from utils import Timer
from .common import *


def partial_dataloader(loader, n_batches, fixed=False):
    '''
    Create a loader that only has n_batches.
    Fixed is whether to make the loader produce same set of batches everytime iterated.
    '''
    m_loader = Mock(loader)
    i = iter(loader)

    if not fixed:
        batches = [next(i) for _ in range(n_batches)]

        def it(self):
            return iter(batches)
    else:
        def it(self):
            return iter(next(i) for _ in range(n_batches))

    m_loader.__len__ = lambda self: n_batches
    m_loader.__iter__ = it
    return m_loader


def test_partial_dataloader(vision_dataset):
    loader = partial_dataloader(vision_dataset.pretest_loader, 3)
    datatime = Timer()
    for x, y in datatime.get_timed_generator(loader):
        pass
    assert datatime.total > 0


@pytest.fixture
def vision_dataset(opt) -> VisionDataset:
    class_order = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    opt.num_pretrain_classes = 3
    opt.num_tasks = 3
    opt.num_classes_per_task = 1

    v = VisionDataset(opt, class_order=class_order)
    assert v.class_order == class_order
    return v


def test_init(vision_dataset):
    assert torch.allclose(vision_dataset.pretrain_mask, torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float))
    assert vision_dataset.n_classes_in_whole_dataset == 10


def test__get_dataset(vision_dataset):
    # create one without target transform
    dataset = vision_dataset._get_dataset(train=True, target_transform=None)
    assert dataset.target_transform is None
    targets = dataset.labels if hasattr(dataset, 'labels') else dataset.targets
    assert len(targets) == 50000  # CIFAR10

    # create the other with target transform
    target_transform = Mock()
    dataset = vision_dataset._get_dataset(train=True, target_transform=target_transform)
    assert dataset.target_transform is target_transform
    assert isinstance(dataset[1][1], Mock)


def test__get_loader(vision_dataset: VisionDataset):
    target_transform = Mock()
    dataloader = vision_dataset._get_loader(train=True, class_list=[1, 2, 9], target_transform=target_transform)
    assert isinstance(dataloader.dataset, Dataset)
    assert dataloader.dataset.target_transform is target_transform


def test__gen_cl_mapping(vision_dataset):
    assert vision_dataset.pretrain_class_list == vision_dataset.class_order[:3]
    assert len(vision_dataset.pretrain_mask) == 10
    assert vision_dataset.pretrain_mask.sum() == 3
    assert list(vision_dataset.pretest_loader.dataset.target_transform.class_order) == list(vision_dataset.class_order)


def check_mask(class_list, mask, target_transform):
    assert len(set(class_list)) == len(class_list), "class list contains duplicates"
    assert mask.sum() == len(class_list)
    for cls in range(max(class_list) + 1):
        if cls in class_list:
            assert mask[target_transform(cls)] == 1.


def test_get_ci_dataloaders(vision_dataset):
    loaders = vision_dataset.get_ci_dataloaders()
    target_transform = loaders[0][0].dataset.target_transform
    isinstance(target_transform, ReorderTargets)
    assert list(target_transform.class_order) == vision_dataset.class_order

    cumul_class_list = vision_dataset.pretrain_class_list.copy()
    for trainloader, testloader, class_list, mask in loaders:
        target_transform = trainloader.dataset.target_transform
        assert len(mask) == vision_dataset.n_classes_in_whole_dataset
        check_mask(class_list, mask, target_transform)

        # make sure testloader contains classes from all previous tasks
        # sample some indices
        cumul_class_list += class_list
        indices = list(testloader.sampler)
        assert set(np.array(testloader.dataset.targets)[indices]) == set(cumul_class_list)


def test_class_names(vision_dataset):
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    assert vision_dataset.opt.dataset == 'CIFAR10', "Sorry, this test applies to cifar-10 for now"

    for c in classes:
        i = classes.index(c)
        assert vision_dataset.class_names[vision_dataset.target_transform(i)] == c
