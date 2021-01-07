from typing import List

import numpy as np
from torch.utils.data import Dataset

from exp2.data import PartialDataset
from utils import get_console_logger

console_logger = get_console_logger(__name__)


class Memory:
    # TODO: write tests
    def __init__(self, max_size, source: Dataset, train_transform, test_transform):
        self.train_transform = train_transform
        self.test_transform = test_transform
        self._scores = dict()
        self._ids = dict()
        assert not isinstance(source, PartialDataset)
        assert isinstance(source, Dataset)
        self.source = source
        self.max_size = max_size

    def get_classes(self):
        return list(self._ids.keys())

    def get_all_ids(self):
        ids = []
        classes = self.get_classes()
        for cls in classes:
            ids.extend(self._ids[cls])
        return np.array(ids)

    def get_dataset(self, train: bool):
        """Create dataset of existing samples in memory.

        Args:
            train: whether the resulting dataset should have train transforms
        """
        ids = self.get_all_ids()
        return PartialDataset(self.source, ids, train=train,
                              train_transform=self.train_transform,
                              test_transform=self.test_transform,
                              classes=self.get_classes())

    def update(self, ids: np.ndarray, new_classes: List[int], scores: np.ndarray = None):
        """ Update memory with new classes. If 'scores' is None, assign equal scores.
        """
        labels = PartialDataset.get_labels(self.source)[ids]
        assert set(labels) == set(new_classes)

        existing_classes = self.get_classes()
        per_class = int(self.max_size / (len(new_classes) + len(existing_classes)))
        residuals = self.max_size - per_class * (len(existing_classes) + len(new_classes))

        # reduce existing classes
        for cls in existing_classes:
            size = per_class + (residuals > 0)
            self._reduce_cls(cls, size)
            residuals -= 1

        # if scores is None, assign equal scores
        if scores is None:
            scores = np.ones_like(ids)

        # add new classes
        for cls in new_classes:
            size = per_class + (residuals > 0)
            self._add_new_class(cls, ids[labels == cls], scores[labels == cls], size)
            residuals -= 1

        # make sure we don't exceed the limit
        self._assert_maxsize()

    def update_scores(self, ids, scores):
        scores = dict(zip(ids, scores))
        for cls in self.get_classes():
            ids = self._ids[cls]
            cls_scores = self._scores[cls]
            for i, id in enumerate(ids):
                cls_scores[i] = scores.get(id, cls_scores[i])

        self._sort()

    def _sort(self):
        """Sort ids by scores. Should be called everytime after updating scores."""
        for cls in self.get_classes():
            ids = self._ids[cls]
            scores = self._scores[cls]
            sorted_idx = np.argsort(scores)[::-1]
            self._scores[cls] = scores[sorted_idx]
            self._ids[cls] = ids[sorted_idx]

    def _reduce_cls(self, cls, newsize):
        """Reduce the number of class samples down to new size."""
        current_size = len(self._ids[cls])
        if newsize > current_size:
            console_logger.warning(
                f'Wanted to reduce the memory samples of class {cls} but they already fewer ({current_size} <= {newsize})')
            return
        self._ids[cls] = self._ids[cls][:newsize]
        self._scores[cls] = self._scores[cls][:newsize]

    def _add_new_class(self, cls, ids: np.ndarray, scores: np.ndarray, per_class):
        n = len(ids)
        sorted_idx = np.argsort(scores)[n:n - per_class - 1:-1]  # sort descending
        self._ids[cls] = ids[sorted_idx]
        self._scores[cls] = scores[sorted_idx]

    def _assert_maxsize(self):
        size = self.get_n_samples()
        assert size <= self.max_size, f'memory size > maxsize ({size} > {self.max_size})'

    def get_n_samples(self):
        return sum(len(ids) for ids in self._ids.values())


def create_memory_storages(config, data):
    """Return memory storages for controller and for classifiers."""
    memory_size = config.memory_size
    ctrl_memory_size = config.ctrl['memory_size']
    clf_memory_size = memory_size - ctrl_memory_size
    if ctrl_memory_size == 0:
        # both use the same memory
        clf_memory = ctrl_memory = Memory(memory_size, data.train_source, data.train_transform, data.test_transform)
    else:
        clf_memory = Memory(clf_memory_size, data.train_source, data.train_transform,
                            data.test_transform)
        ctrl_memory = Memory(ctrl_memory_size, data.train_source, data.train_transform, data.test_transform)

    return clf_memory, ctrl_memory


def update_memories(ctrl_memory: Memory, clf_memory: Memory, trainset: PartialDataset):
    if clf_memory is not ctrl_memory:
        middle = len(trainset.ids) // 2
        clf_memory.update(ids=trainset.ids[:middle], new_classes=trainset.classes)
        ctrl_memory.update(ids=trainset.ids[middle:], new_classes=trainset.classes)
    else:
        clf_memory.update(ids=trainset.ids, new_classes=trainset.classes)


def log_total_memory_sizes(ctrl_memory: Memory, clf_memory: Memory):
    size = ctrl_memory.get_n_samples()
    if ctrl_memory is not clf_memory:
        size += clf_memory.get_n_samples()
    console_logger.info('Memory size (number of samples): %s', size)
    return size
