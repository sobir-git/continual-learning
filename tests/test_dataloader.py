import pytest
from unittest.mock import Mock

import torch
from torch.utils.data import Dataset

from dataloader import VisionDataset, ReorderTargets
from opts import parse_args


@pytest.fixture
def opt():
    args = ['--model', 'SimpleNet',
            '--dataset', 'CIFAR10',
            '--data_dir', '../data',
            '--num_classes_per_task', '1',
            '--num_tasks', '3',
            '--num_pretrain_classes', '3',
            '--num_pretrain_passes', '1',
            '--num_loops', '1']
    args = parse_args(args)
    return args


@pytest.fixture
def vd(opt) -> VisionDataset:
    class_order = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    v = VisionDataset(opt, class_order=class_order)
    assert v.class_order == class_order
    return v


def test_init(vd):
    assert torch.allclose(vd.pretrain_mask, torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float))
    assert vd.n_classes_in_whole_dataset == 10


def test__get_dataset(vd):
    # create one without target transform
    dataset = vd._get_dataset(train=True)
    assert dataset.target_transform is None
    targets = dataset.labels if hasattr(dataset, 'labels') else dataset.targets
    assert len(targets) == 50000  # CIFAR10

    # create the other with target transform
    target_transform = Mock()
    dataset = vd._get_dataset(train=True, target_transforms=target_transform)
    assert dataset.target_transform is target_transform
    assert isinstance(dataset[1][1], Mock)


def test__get_loader(vd: VisionDataset):
    target_transform = Mock()
    dataloader = vd._get_loader(train=True, class_list=[1, 2, 9], target_transform=target_transform)
    assert dataloader.dataset.target_transform is target_transform

    dataloader = vd._get_loader(train=True, class_list=[9, 1, 2], reorder_targets=True, target_start=2)
    assert isinstance(dataloader.dataset, Dataset)
    target_transform = dataloader.dataset.target_transform
    assert isinstance(target_transform, ReorderTargets)
    target_start = target_transform.start
    assert target_start == 2
    assert target_transform(9) == target_start
    assert target_transform(1) == target_start + 1
    assert target_transform(2) == target_start + 2


def test__gen_cl_mapping(vd):
    assert vd.pretrain_class_list == vd.class_order[:3]
    assert len(vd.pretrain_mask) == 10
    assert vd.pretrain_mask.sum() == 3
    assert list(vd.pretest_loader.dataset.target_transform.class_order) == list(vd.pretrain_class_list)


def check_mask(class_list, mask, target_transform):
    assert len(set(class_list)) == len(class_list), "class list contains duplicates"
    assert mask.sum() == len(class_list)
    for cls in range(max(class_list) + 1):
        if cls in class_list:
            assert mask[target_transform(cls)] == 1.


def test_get_ci_dataloaders(vd):
    loaders = vd.get_ci_dataloaders()
    target_transform = loaders[0][0].dataset.target_transform
    isinstance(target_transform, ReorderTargets)
    assert list(target_transform.class_order) == vd.cl_class_list

    for trainloader, testloader, class_list, mask in loaders:
        target_transform = trainloader.dataset.target_transform
        assert len(mask) == vd.n_classes_in_whole_dataset
        check_mask(class_list, mask, target_transform)

