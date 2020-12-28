import copy
from typing import Tuple

import numpy as np
import sklearn
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from dataloader import get_statistics, get_augment_transforms
from exp2.config import Config
from utils import np_a_in_b
import exp2.tiny_data


class PartialDataset(Dataset):

    def __init__(self, source, ids, transform, classes=None, test_transform=None):
        assert transform is not None
        self.transform = transform
        self.test_transform = test_transform
        self.source = source
        self.ids = ids  # only self ids, no otherset when involved
        self.labels = self.get_labels(self.source)[self.ids]
        self._classes = classes if classes is not None \
            else list(set(self.labels))
        self.otherset = None

    @property
    def classes(self):
        """Number of classes only in self (not including otherset)"""
        return self._classes

    def __getitem__(self, item):
        """From self and otherset(if exists)."""
        if item < len(self.ids):
            id = self.ids[item]
            input, label = self.source[id]
            input = self.transform(input)
            return input, label, id
        else:
            return self.otherset[item - len(self.ids)]  # input, label id

    def __len__(self):
        """Length of self + otherset"""
        n = len(self.ids)
        if self.otherset:
            n += len(self.otherset)
        return n

    def _remove_otheret(self):
        self.otherset = None

    def set_otherset(self, dataset):
        assert isinstance(dataset, PartialDataset)
        assert dataset.otherset is None, "Otherset shouldn't have otherset"
        assert dataset.source is self.source
        assert set(dataset.classes).isdisjoint(self._classes)
        self.otherset = dataset

    @classmethod
    def from_classes(cls, source, transform, classes, **kwargs):
        labels = cls.get_labels(source)
        ids = np_a_in_b(labels, classes)
        return cls(source, ids, transform, classes=classes, **kwargs)

    @staticmethod
    def get_labels(dataset) -> np.ndarray:
        """Only self labels"""
        if hasattr(dataset, 'labels'):
            labels = dataset.labels
        else:
            labels = dataset.targets

        # make sure labels are numpy arrays
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        return labels

    def split(self, train_size=None, test_size=None):
        """Split into train and test set. If it has otherset, it will also be split in the same ratio.
        Also set val augmentation set to test ones.
        """
        source = self.source

        train_ids, test_ids = \
            train_test_split(self.ids, train_size=train_size, test_size=test_size, stratify=self.labels)
        self_train = PartialDataset(source, train_ids, self.transform)
        self_test = PartialDataset(source, test_ids, self.test_transform)

        if self.otherset:
            o_train_ids, o_test_ids = \
                train_test_split(self.otherset.ids, train_size=train_size, test_size=test_size,
                                 stratify=self.otherset.labels)
            other_train = PartialDataset(source, o_train_ids, self.transform, test_transform=self.test_transform)
            other_test = PartialDataset(source, o_test_ids, self.test_transform)
            self_train.set_otherset(other_train)
            self_test.set_otherset(other_test)

        return self_train, self_test

    def without_otherset(self):
        """Return a new copy of the self without otherset."""
        new_self = copy.deepcopy(self)
        new_self._remove_otheret()
        return new_self


class CIData:
    def __init__(self, train_source, test_source, class_order, n_classes_per_phase, n_phases, train_transform,
                 test_transform):
        """Note: Assumes the train source and test source without tranforms."""
        self.class_order = class_order
        self.n_classes_per_phase = n_classes_per_phase
        self.n_phases = n_phases
        self.train_source = train_source
        self.test_source = test_source
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.data = []
        cumul_classes = []
        for phase in range(n_phases):
            classes = self.class_order[phase * n_classes_per_phase:(phase + 1) * n_classes_per_phase]
            train = PartialDataset.from_classes(train_source, train_transform, classes,
                                                test_transform=test_transform)
            test = PartialDataset.from_classes(test_source, test_transform, classes)
            cumul_classes.extend(classes)
            cumul_test = PartialDataset.from_classes(test_source, test_transform, cumul_classes)
            self.data.append((train, test, cumul_test))

    def get_phase_data(self, phase) -> Tuple[PartialDataset, PartialDataset, PartialDataset]:
        """Phases start from 1."""
        assert phase < self.n_phases + 1
        return self.data[phase - 1]


def create_loader(config, dataset):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True)


def _get_dataset(config: Config, train: bool, transforms):
    """Create the train or test with the optionally given target_transform."""
    # Support for *some* pytorch default loaders is provided. Code is made such that adding new datasets is super easy, given they are in ImageFolder format.
    if config.dataset in ['CIFAR10', 'CIFAR100', 'MNIST', 'KMNIST', 'FashionMNIST', 'TinyCIFAR10']:
        dataset = getattr(torchvision.datasets, config.dataset)(root=config.datadir, train=train,
                                                                download=config.download,
                                                                transform=transforms)
    elif config.dataset == 'SVHN':
        split = 'train' if train else 'test'
        dataset = getattr(torchvision.datasets, config.dataset)(root=config.datadir, split=split,
                                                                download=config.download,
                                                                transform=transforms)
    else:
        subfolder = 'train' if train else 'test'  # ImageNet 'val' is labled as 'test' here.
        dataset = torchvision.datasets.ImageFolder(config.datadir + '/' + config.dataset + '/' + subfolder,
                                                   transform=transforms)
    return dataset


def prepare_data(config) -> CIData:
    n_classes_per_phase = config.n_classes_per_phase
    n_phases = config.n_phases
    mean, std, n_classes, inp_size, in_channels = get_statistics(config.dataset)
    train_augment, test_augment = get_augment_transforms(dataset=config.dataset, inp_sz=inp_size)
    train_transforms = torchvision.transforms.Compose(
        train_augment + [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
    test_transforms = torchvision.transforms.Compose(
        test_augment + [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
    train_source = _get_dataset(config, train=True, transforms=None)
    test_source = _get_dataset(config, train=False, transforms=None)
    class_order = np.array(list(range(n_classes)))
    if config.class_order_seed != -1:
        _rs = np.random.RandomState(config.class_order_seed)
        _rs.shuffle(class_order)

    return CIData(train_source, test_source, class_order, n_classes_per_phase, n_phases,
                  train_transform=train_transforms, test_transform=test_transforms)
