import pickle
from pathlib import Path
from typing import List

import numpy as np
import wandb
from torch.utils.data import Dataset

from exp2.data import PartialDataset, CIData
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

    def update(self, ids: np.ndarray, new_classes: List[int], scores: np.ndarray = None, max_size=None):
        """ Update memory with new classes. If 'scores' is None, assign equal scores.
        If max_size is specified, the memory will be treated to have only that much size, so reductions and
        additions of samples will be done according to that.

        Args:
            max_size (int): the effective maxsize, should be no more than self.max_size
            ids: indices belonging to the `source` dataset.
            scores: array of the same size as ids, representing scores for each element
            max_size: additional size limit on the memory. If specified, after this update
                the memory size will is not greater than it.
        """
        # handle degenerate case
        if len(ids) == 0:
            return np.array([], dtype=int)  # added ids
        if max_size is None:
            max_size = self.max_size
        assert max_size <= self.max_size

        labels = PartialDataset.get_labels(self.source)[ids]
        assert set(labels).issubset(new_classes)

        existing_classes = self.get_classes()
        per_class = int(max_size / (len(new_classes) + len(existing_classes)))
        residuals = max_size - per_class * (len(existing_classes) + len(new_classes))

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
        self._assert_maxsize(max_size)

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
        if per_class >= n:
            sorted_idx = np.argsort(scores)[::-1]  # sort descending
        else:
            sorted_idx = np.argsort(scores)[n:n - per_class - 1:-1]  # sort descending
        self._ids[cls] = added_ids = ids[sorted_idx]
        self._scores[cls] = scores[sorted_idx]
        return added_ids

    def _assert_maxsize(self, max_size=None):
        if max_size is None:
            max_size = self.max_size
        max_size = min(max_size, self.max_size)
        size = self.get_n_samples()
        assert size <= max_size, f'memory size > maxsize ({size} > {max_size})'

    def get_n_samples(self):
        return sum(len(ids) for ids in self._ids.values())

    def get_state(self):
        return {'ids': self._ids, 'max_size': self.max_size, 'scores': self._scores}

    def load_state(self, state):
        self._ids = state['ids']
        self._scores = state['scores']
        assert state['max_size'] == self.max_size


class MemoryManagerBasic:
    def __init__(self, sizes: List[int], source, train_transform, test_transform, names=None):
        self.memories = []
        for sz in sizes:
            self.memories.append(
                Memory(source=source, max_size=sz, train_transform=train_transform, test_transform=test_transform))
        self.names = names if names is not None else ['mem' + str(i) for i in range(1, len(sizes) + 1)]

    def get_memories(self):
        return self.memories

    def __getitem__(self, item):
        if isinstance(item, str):
            i = self.names.index(item)
            return self.memories[i]
        return self.memories[item]

    def items(self):
        return zip(self.names, self.memories)

    def update_memories(self, dataset: PartialDataset, memories: List[Memory] = None):
        if memories is None:
            memories = self.memories
        total_sz = sum(m.max_size for m in memories)
        if total_sz > 0:
            shuffled_ids = np.random.permutation(dataset.ids)
            split_sizes = [int(len(dataset) * m.max_size / total_sz) for m in memories]
            cumul_sz = 0
            for memory, sz in zip(memories, split_sizes):
                memory.update(ids=shuffled_ids[cumul_sz:cumul_sz + sz], new_classes=dataset.classes)
                cumul_sz += sz

    def log_memory_sizes(self):
        sizes = [m.get_n_samples() for m in self.memories]
        console_logger.info(f'Memory sizes {tuple(self.names)}: {" + ".join(map(str, sizes))} = {sum(sizes)}')
        return sum(sizes)


class MemoryManager(MemoryManagerBasic):
    def __init__(self, config, data: CIData):
        """Return memory storages for controller and for classifiers."""
        sizes = [config.clf['memory_size'], config.ctrl['memory_size'], config.shared_memory_size]
        super().__init__(sizes, data.train_source, train_transform=data.train_transform,
                         test_transform=data.test_transform, names=['clf', 'ctrl', 'shared'])
        self.artifact = wandb.Artifact(f'memory-ids-{wandb.run.id}', 'ids')

    def get_memories(self):
        return self['clf'], self['ctrl'], self['shared']

    def update_memories(self, dataset: PartialDataset, phase: int):
        """Update classifier and shared memory.
        Then upload artifacts, and log total memory sizes"""
        super(MemoryManager, self).update_memories(dataset, memories=[self['clf'], self['shared']])
        self.upload_artifacts(phase)
        self.log_memory_sizes()

    def upload_artifacts(self, phase: int):
        with self.artifact.new_file(f'ids-{phase}.pkl', 'wb') as f:
            pickle.dump({
                name: memory.get_state() for name, memory in self.items()
            }, f)

    def load_from_artifact(self, artifact: wandb.Artifact, phase: int):
        folder = Path(artifact.download())
        with open(folder / f'ids-{phase}.pkl', 'rb') as f:
            d = pickle.load(f)

        for name in self.names:
            state = d[name]
            mem: Memory = getattr(self, name)
            mem.load_state(state)

    def on_training_end(self):
        console_logger.info('Uploading memory indices')
        wandb.run.log_artifact(self.artifact)
