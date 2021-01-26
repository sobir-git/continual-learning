from functools import reduce

import numpy as np

from exp2.data import PartialDataset
from exp2.memory import Memory, MemoryManagerBasic
from tests.test_data import cifar100_10_test_data


def test_memory():
    cidata = cifar100_10_test_data()
    source = cidata.train_source
    memory = Memory(max_size=500, source=source, train_transform=cidata.train_transform,
                    test_transform=cidata.test_transform)

    cumul_train_ids = []

    for phase in range(1, 11):
        trainset, _, _ = cidata.get_phase_data(phase)
        cumul_train_ids.extend(trainset.ids)
        added_ids = memory.update(trainset.ids, new_classes=trainset.classes)
        in1d = lambda a, b: np.alltrue(np.in1d(a, b))
        assert in1d(added_ids, trainset.ids)
        for train in [True, False]:
            ds = memory.get_dataset(train=train)
            assert ds.is_train() == train
            assert ds.train_transform is cidata.train_transform
            assert ds.test_transform is cidata.test_transform
            assert in1d(memory.get_all_ids(), cumul_train_ids)
            assert np.alltrue(ds.ids == memory.get_all_ids())

        # returns ids correctly
        assert [ds[i][2] for i in range(len(ds))] == ds.ids.tolist()

        # labels match with the source dataset
        source_labels = PartialDataset.get_labels(source)
        assert np.alltrue(ds.labels == source_labels[ds.ids])

        # labels still match when accessed via range 0..n
        assert [ds[i][1] for i in range(len(ds))] == source_labels[ds.ids].tolist()


def test_multiple_memories():
    cidata = cifar100_10_test_data()
    sizes = [400, 100]
    memman = MemoryManagerBasic(sizes, source=cidata.train_source, train_transform=cidata.train_transform,
                                test_transform=cidata.test_transform)

    cumul_train_ids = []
    for phase in range(1, 11):
        trainset, _, _ = cidata.get_phase_data(phase)
        cumul_train_ids.extend(trainset.ids)

        memman.update_memories(trainset)
        intersection = reduce(np.intersect1d, (mem.get_all_ids() for mem in memman.memories))
        assert len(intersection) == 0

        for train in [True, False]:
            for i, mem in enumerate(memman.memories):
                assert mem.get_n_samples() == sizes[i]
                ds = mem.get_dataset(train=train)
                assert ds.is_train() == train
                assert ds.train_transform is cidata.train_transform
                assert ds.test_transform is cidata.test_transform

                in1d = lambda a, b: np.alltrue(np.in1d(a, b))
                assert in1d(mem.get_all_ids(), cumul_train_ids)
                assert in1d(ds.ids, mem.get_all_ids())
