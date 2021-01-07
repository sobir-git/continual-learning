from typing import List, Dict

import numpy as np
from torch.utils.data import Dataset

from exp2.data import PartialDataset


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
        return ids

    def get_dataset(self):
        ids = self.get_all_ids()
        return PartialDataset(self.source, ids, self.train_transform, test_transform=self.test_transform,
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
            self._reduce_cls(cls, per_class + (residuals > 0))
            residuals -= 1

        # if scores is None, assign equal scores
        if scores is None:
            scores = np.ones_like(ids)

        # add new classes
        for cls in new_classes:
            self._add_new_class(cls, ids[labels == cls], scores[labels == cls], per_class + (residuals > 0))
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

    def _reduce_cls(self, cls, per_class):
        assert per_class <= len(self._ids[cls])  # TODO: is here a bug?
        self._ids[cls] = self._ids[cls][:per_class]
        self._scores[cls] = self._scores[cls][:per_class]

    def _add_new_class(self, cls, ids: np.ndarray, scores: np.ndarray, per_class):
        n = len(ids)
        sorted_idx = np.argsort(scores)[n:n - per_class - 1:-1]  # sort descending
        self._ids[cls] = ids[sorted_idx]
        self._scores[cls] = scores[sorted_idx]

    def _assert_maxsize(self):
        assert sum(len(ids) for ids in self._ids.values()) <= self.max_size


def create_memory_storages(config, data):
    """Return memory storages for controller and for classifiers."""
    memory_size = config.memory_size
    ctrl_memory_size = config.ctrl.memory_size
    clf_memory_size = memory_size - ctrl_memory_size
    if ctrl_memory_size == 0:
        # both use the same memory
        clf_memory = ctrl_memory = Memory(memory_size, data.train_source, data.train_transform, data.test_transform)
    else:
        clf_memory = Memory(clf_memory_size, data.train_source, data.train_transform,
                            data.test_transform)
        ctrl_memory = Memory(ctrl_memory_size, data.train_source, data.train_transform, data.test_transform)

    return clf_memory, ctrl_memory


def update_memories(ctrl_memory, clf_memory, trainset: PartialDataset):
    middle = len(trainset.ids) // 2
    clf_memory.update(ids=trainset.ids[:middle], new_classes=trainset.classes)
    ctrl_memory.update(ids=trainset.ids[middle:], new_classes=trainset.classes)
