from typing import List, Dict

import numpy as np
from torch.utils.data import Dataset

from exp2.data import PartialDataset


class Memory:
    # TODO: write tests
    def __init__(self, config, source: Dataset):
        self._scores = dict()
        self._ids = dict()
        assert not isinstance(source, PartialDataset)
        assert isinstance(source, Dataset)
        self.source = source
        self.max_size = config.memory_size

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
        return PartialDataset(self.source, ids, classes=self.get_classes())

    def update(self, ids: np.ndarray, scores, new_classes: List[int]):
        """ Update memory with new classes """
        labels = PartialDataset.get_labels(self.source)[ids]
        assert set(labels) == set(new_classes)

        existing_classes = self.get_classes()
        per_class = int(self.max_size / (len(new_classes) + len(existing_classes)))
        residuals = self.max_size - per_class * (len(existing_classes) + len(new_classes))

        # reduce existing classes
        for cls in existing_classes:
            self._reduce_cls(cls, per_class + (residuals > 0))
            residuals -= 1

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
        assert per_class <= len(self._ids[cls])
        self._ids[cls] = self._ids[cls][:per_class]
        self._scores[cls] = self._scores[cls][:per_class]

    def _add_new_class(self, cls, ids: np.ndarray, scores: np.ndarray, per_class):
        n = len(ids)
        sorted_idx = np.argsort(scores)[n:n - per_class - 1:-1]  # sort descending
        self._ids[cls] = ids[sorted_idx]
        self._scores[cls] = scores[sorted_idx]

    def _assert_maxsize(self):
        assert sum(len(ids) for ids in self._ids.values()) <= self.max_size