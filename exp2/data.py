import itertools
from typing import Tuple, List, Callable, Sequence

import numpy as np
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, RandomSampler, Sampler, SequentialSampler, BatchSampler
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
        if isinstance(train_transform, Compose) and isinstance(test_transform, Compose):
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

    def concat(self, otherset: 'PartialDataset'):
        """Concatenate with other dataset and produce a new one.
        The ids are just concatenated. Other properties are inherited from self.
        Warning: it assumes that both datasets have different sets of samples (ids).
        Train and test transforms of otherset will be ignored.
        The resulting dataset will have train/test transforms and train mode as self.
        """
        assert self.source == otherset.source, "Both datasets should have same sources."

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

    # alias for concat
    mix = concat

    def subset(self, ids):
        """Return a new dataset with samples of those ids"""
        return PartialDataset(source=self.source, ids=ids, train=self._train, train_transform=self.train_transform,
                              test_transform=self.test_transform, classes=self._classes)

    def is_train(self):
        return self._train


class CIData:
    def __init__(self, train_source: Dataset, test_source: Dataset, class_order: List[int], n_classes_per_phase: int,
                 n_phases: int, train_transform,
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


class ContinuousRandomSampler(Sampler[int]):
    indices: Sequence[int]

    def __init__(self, indices, generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def get_perm(self):
        perm = torch.randperm(len(self.indices), generator=self.generator)
        return perm.tolist()

    def __iter__(self):
        i = 0
        perm = self.get_perm()
        while True:
            if i == len(perm):
                perm = self.get_perm()
                i = 0
            yield self.indices[perm[i]]
            i += 1

    def __len__(self):
        return float('inf')


class ContinuousSequentialSampler(Sampler[int]):
    indices = Sequence[int]

    def __init__(self, indices) -> None:
        self.indices = indices

    def __iter__(self):
        return itertools.cycle(self.indices)

    def __len__(self):
        return float('inf')


class DualBatchSampler(BatchSampler):
    def __init__(self, main_sampler, continuous_random_sampler: ContinuousRandomSampler, sizes: Tuple[int, int],
                 drop_last=False):
        self.sizes = sizes
        self.continuous_random_sampler = continuous_random_sampler
        self.main_sampler = main_sampler
        self.drop_last = drop_last

    def __len__(self):
        if self.drop_last:
            return len(self.main_sampler) // self.sizes[0]  # type: ignore
        else:
            return (len(self.main_sampler) + self.sizes[0] - 1) // self.sizes[0]  # type: ignore

    def __iter__(self):
        batch = []
        it_continuous_random_sampler = iter(self.continuous_random_sampler)
        for idx in self.main_sampler:
            batch.append(idx)
            if len(batch) == self.sizes[0]:
                # add from the other sampler
                batch.extend(next(it_continuous_random_sampler) for _ in range(self.sizes[1]))
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


def create_loader(config, main_dataset: PartialDataset, memoryset: PartialDataset = None, shuffle=True):
    if len(main_dataset) == 0:  # in this case we don't shuffle because it causes an error in Dataloader code
        assert len(memoryset) == 0, "Got main dataset empty but memory non-empty"
        return DataLoader(main_dataset, batch_size=config.batch_size, shuffle=False)

    common_kwargs = {'num_workers': config.num_workers}
    if memoryset is not None:
        concatenated_dataset = main_dataset.concat(memoryset)
        if config.batch_memory_samples > 0:
            if len(memoryset) == 0:
                raise ValueError('Memory dataset is empty while config.batch_memory_samples > 0')
            sampler_cls = [SequentialSampler, RandomSampler][shuffle]
            main_sampler = sampler_cls(main_dataset)
            sampler_cls = [ContinuousSequentialSampler, ContinuousRandomSampler][shuffle]
            offset = len(main_dataset)
            memory_sampler = sampler_cls(np.fromiter(range(len(memoryset)), dtype=int) + offset)
            batch_sizes = (config.batch_size - config.batch_memory_samples, config.batch_memory_samples)
            batch_sampler = DualBatchSampler(main_sampler=main_sampler, continuous_random_sampler=memory_sampler,
                                             sizes=batch_sizes)

            return DataLoader(concatenated_dataset, batch_sampler=batch_sampler, **common_kwargs)
        else:  # we just merge two datasets
            return DataLoader(main_dataset.concat(memoryset), batch_size=config.batch_size, shuffle=shuffle,
                              **common_kwargs)
    else:  # no memory
        return DataLoader(main_dataset, batch_size=config.batch_size, shuffle=shuffle, **common_kwargs)


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
    if config.resize_input:
        inp_size = config.resize_input
    train_augment, test_augment = get_augment_transforms(dataset=config.dataset, inp_sz=inp_size)
    train_transforms = torchvision.transforms.Compose(
        train_augment + [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
    test_transforms = torchvision.transforms.Compose(
        test_augment + [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
    train_source = _get_dataset(config, train=True, transforms=None)
    test_source = _get_dataset(config, train=False, transforms=None)
    class_order = list(range(n_classes))
    if config.class_order_seed != -1:
        _rs = np.random.RandomState(config.class_order_seed)
        _rs.shuffle(class_order)

    return CIData(train_source, test_source, class_order, n_classes_per_phase, n_phases,
                  train_transform=train_transforms, test_transform=test_transforms)
