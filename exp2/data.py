from typing import Tuple

import numpy as np
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from utils import np_a_in_b
from dataloader import get_statistics, get_augment_transforms
from exp2.config import Config
import exp2.tiny_data


class PartialDataset(Dataset):

    def __init__(self, source, ids, transform, classes=None, test_transform=None):
        assert transform is not None
        self.transform = transform
        self.test_transform = test_transform
        self.source = source
        self.ids: np.ndarray = ids  # only self ids, no otherset when involved
        self.labels = self.get_labels(self.source)[self.ids]
        self._classes = classes if classes is not None \
            else list(set(self.labels))

    @property
    def classes(self):
        """The list of classes only in self (not including otherset)"""
        return self._classes

    def __getitem__(self, item):
        """From self and otherset(if exists)."""
        id = self.ids[item]
        input, label = self.source[id]
        input = self.transform(input)
        return input, label, id

    def __len__(self):
        """Length of self + otherset"""
        return len(self.ids)

    @classmethod
    def from_classes(cls, source, transform, classes, **kwargs):
        labels = cls.get_labels(source)
        ids = np_a_in_b(labels, classes)
        return cls(source, ids, transform, classes=classes, **kwargs)

    @staticmethod
    def get_labels(dataset) -> np.ndarray:
        """Only self labels"""
        if hasattr(dataset, 'labels'):
            labels = dataset._labels
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
        train_ids, test_ids = \
            train_test_split(self.ids, train_size=train_size, test_size=test_size, stratify=self.labels)
        self_train = PartialDataset(self.source, train_ids, self.transform)
        self_test = PartialDataset(self.source, test_ids, self.test_transform)
        return self_train, self_test

    def mix(self, otherset):
        """Create a dataset of mixture of samples. Other properties are inherited from self.
        Warning: it assumes that both datasets have different sets of samples.
        """
        # create list of classes from both dataset, the list starts with classes of self in the same order
        classes = list(self.classes)
        for cls in otherset.classes:
            if cls not in classes:
                classes.append(cls)
        source = self.source
        ids = np.concatenate([self.ids, otherset.ids])
        transform = self.transform
        test_transform = self.test_transform
        return self.__class__(source, ids, transform, classes=classes, test_transform=test_transform)


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


def create_loader(config, dataset) -> DataLoader:
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
