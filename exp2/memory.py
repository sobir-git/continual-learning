import pickle
from pathlib import Path
from typing import List

import numpy as np
import wandb
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
        return np.array(ids, dtype=int)

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
        assert set(labels).issubset(new_classes)

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
        added_ids = []
        for cls in new_classes:
            size = per_class + (residuals > 0)
            added_ids.append(self._add_new_class(cls, ids[labels == cls], scores[labels == cls], size))
            residuals -= 1

        # make sure we don't exceed the limit
        self._assert_maxsize()

        # handle degenerate case
        if len(added_ids) == 0:
            return np.array([])
        return np.concatenate(added_ids)

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
        self._ids[cls] = added_ids = ids[sorted_idx]
        self._scores[cls] = scores[sorted_idx]
        return added_ids

    def _assert_maxsize(self):
        size = self.get_n_samples()
        assert size <= self.max_size, f'memory size > maxsize ({size} > {self.max_size})'

    def get_n_samples(self):
        return sum(len(ids) for ids in self._ids.values())

    def get_state(self):
        return {'ids': self._ids, 'max_size': self.max_size, 'scores': self._scores}

    def load_state(self, state):
        self._ids = state['ids']
        self._scores = state['scores']
        assert state['max_size'] == self.max_size


class MemoryManager:
    def __init__(self, config, data):
        """Return memory storages for controller and for classifiers."""
        clf_memory_size = config.clf['memory_size']
        ctrl_memory_size = config.ctrl['memory_size']
        shared_memory_size = config.shared_memory_size
        self.clf_memory = Memory(clf_memory_size, data.train_source, data.train_transform, data.test_transform)
        self.ctrl_memory = Memory(ctrl_memory_size, data.train_source, data.train_transform, data.test_transform)
        self.shared_memory = Memory(shared_memory_size, data.train_source, data.train_transform, data.test_transform)
        self.artifact = wandb.Artifact(f'memory-ids-{wandb.run.id}', 'ids')

    def get_memories(self):
        return self.clf_memory, self.ctrl_memory, self.shared_memory

    def update_memories(self, dataset: PartialDataset, phase: int):
        # to prevent overlap we choose a splitting index
        split_idx = int(
            len(dataset) * self.clf_memory.max_size / (self.clf_memory.max_size + self.shared_memory.max_size))
        self.clf_memory.update(ids=dataset.ids[:split_idx], new_classes=dataset.classes)
        self.shared_memory.update(ids=dataset.ids[split_idx:], new_classes=dataset.classes)
        with self.artifact.new_file(f'ids-{phase}.pkl', 'wb') as f:
            pickle.dump({
                'clf': self.clf_memory.get_state(),
                'shared': self.shared_memory.get_state(),
                'ctrl': self.ctrl_memory.get_state()
            }, f)
        self.log_total_memory_sizes()

    def load_from_artifact(self, artifact: wandb.Artifact, phase: int):
        folder = Path(artifact.download())
        with open(folder / f'ids-{phase}.pkl', 'rb') as f:
            d = pickle.load(f)

        for pref in ('clf', 'ctrl', 'shared'):
            state = d[pref]
            mem: Memory = getattr(self, pref + '_memory')
            mem.load_state(state)

    def log_total_memory_sizes(self):
        ct = self.ctrl_memory.get_n_samples()
        cf = self.clf_memory.get_n_samples()
        sh = self.shared_memory.get_n_samples()
        tot = ct + cf + sh
        console_logger.info('Memory sizes (classifier, controller, shared): %s + %s + %s = %s', cf, ct, sh, tot)
        return tot

    def on_training_end(self):
        console_logger.info('Uploading memory indices')
        wandb.run.log_artifact(self.artifact)
