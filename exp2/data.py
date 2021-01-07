from typing import Tuple, List, Callable

import numpy as np
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

from utils import np_a_in_b, get_console_logger
from dataloader import get_statistics, get_augment_transforms
import exp2.tiny_data

console_logger = get_console_logger(__name__)


class PartialDataset(Dataset):
    _train: bool
    _transform: Callable

    def __init__(self, source: Dataset, ids: np.ndarray, train: bool, train_transform: Compose,
                 test_transform: Compose, classes: List[int]):
        assert len(train_transform.transforms) >= len(test_transform.transforms)
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.source = source
        self._classes = classes
        self.ids = ids if isinstance(ids, np.ndarray) else np.array(ids, dtype=int)
        self._set_train(train)

        # get corresponding labels
        self.labels = self.get_labels(self.source)[ids] if len(ids) > 0 \
            else np.array([])

    def _set_train(self, train):
        """Set the train mode."""
        assert type(train) is bool
        self._train = train
        if train:
            self._transform = self.train_transform
        else:
            self._transform = self.test_transform

    @property
    def classes(self):
        """The list of classes"""
        return self._classes

    def __getitem__(self, item):
        """From self and otherset(if exists)."""
        id = self.ids[item]
        input, label = self.source[id]
        input = self._transform(input)
        return input, label, id

    def __len__(self):
        """Length of self + otherset"""
        return len(self.ids)

    @classmethod
    def from_classes(cls, source, train: bool, train_transform, test_transform, classes):
        """Create dataset from given classes. The samples are taken from source dataset.
        """
        labels = cls.get_labels(source)
        ids = np_a_in_b(labels, classes)
        return cls(source, ids, train=train, train_transform=train_transform, test_transform=test_transform,
                   classes=classes)

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

    def split_by_sizes(self, *sizes):
        sizes = list(sizes)
        if -1 not in sizes:
            assert sum(sizes) <= len(self.ids)
        assert sizes.count(-1) in [0, 1]

        if -1 in sizes:
            sizes[sizes.index(-1)] = len(self.ids) - sum(sizes)

        # shuffle, then split ids
        ids = self.ids.copy()
        np.random.shuffle(ids)
        i = 0
        for size in sizes:
            part = ids[i:i + size]
            PartialDataset(self.source, ids=part, train=self._train, )

            i += size

    def split(self, test_size=None, train_size=None):
        """Split into train and test set. Also set test transforms for the testset.
        if test_size == 0, then testset will still be a dataset but empty.
        """
        if test_size is not None: assert test_size <= 0.5
        if train_size is not None: assert train_size >= 0.5

        # handle two special cases, finally split
        if len(self.ids) == 0:
            console_logger.warning('Trying to split empty dataset. Will split into two empties anyways.')
            train_ids, test_ids = [], []
        elif test_size == 0:
            train_ids, test_ids = self.ids, []
        else:
            train_ids, test_ids = \
                train_test_split(self.ids, train_size=train_size, test_size=test_size, stratify=self.labels)

        self_train = PartialDataset(self.source, train_ids, train=True, train_transform=self.train_transform,
                                    test_transform=self.test_transform, classes=self._classes)
        self_test = PartialDataset(self.source, test_ids, train=False, train_transform=self.train_transform,
                                   test_transform=self.test_transform, classes=self._classes)
        return self_train, self_test

    def mix(self, otherset):
        """Create a dataset of mixture of samples. Other properties are inherited from self.
        Warning: it assumes that both datasets have different sets of samples.
        Train and test transforms of otherset will be ignored.
        The resulting dataset will have train/test transforms and train mode as self.
        """
        # create list of classes from both dataset, the list starts with classes of self in the same order
        classes = list(self.classes)
        for cls in otherset.classes:
            if cls not in classes:
                classes.append(cls)
        source = self.source
        ids = np.concatenate([self.ids, otherset.ids])
        test_transform = self.test_transform
        train = self._train
        return self.__class__(source, ids, train, train_transform=self.train_transform,
                              test_transform=test_transform, classes=classes)

    def subset(self, ids):
        """Return a new dataset with samples of those ids"""
        return PartialDataset(source=self.source, ids=ids, train=self._train, train_transform=self.train_transform,
                              test_transform=self.test_transform, classes=self._classes)

    def is_train(self):
        return self._train


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
            train = PartialDataset.from_classes(train_source, train=True, train_transform=train_transform,
                                                test_transform=test_transform, classes=classes)
            test = PartialDataset.from_classes(test_source, train=False, train_transform=train_transform,
                                               test_transform=test_transform, classes=classes)
            cumul_classes.extend(classes)
            cumul_test = PartialDataset.from_classes(test_source, train=False, train_transform=train_transform,
                                                     test_transform=test_transform, classes=cumul_classes)
            self.data.append((train, test, cumul_test))

    def get_phase_data(self, phase) -> Tuple[PartialDataset, PartialDataset, PartialDataset]:
        """Phases start from 1."""
        assert phase < self.n_phases + 1
        return self.data[phase - 1]


def create_loader(config, dataset: PartialDataset) -> DataLoader:
    shuffle = True
    if len(dataset) == 0:  # in this case we don't shuffle because it causes an error in Dataloader code
        shuffle = False
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle)


def _get_dataset(config, train: bool, transforms):
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
