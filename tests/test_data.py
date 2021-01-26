from collections import Counter
from functools import reduce
from random import shuffle
from types import SimpleNamespace

import numpy as np
import torchvision
from torch.utils.data import SequentialSampler, RandomSampler, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, ToTensor

from exp2.data import ContinuousRandomSampler, DualBatchSampler, create_loader, PartialDataset, CIData
from exp2.memory import Memory


def test_continous_random_sampler():
    indices = [1, 2, 3]
    sampler = ContinuousRandomSampler(indices)
    it = iter(sampler)
    sampled_indices = [next(it) for _ in range(3 * 10)]
    assert len(sampled_indices) == 30
    assert sampled_indices != indices * 10  # should be permuted
    assert not all(
        sampled_indices[i:i + 3] == sampled_indices[i - 3:i] for i in range(3, 10)), "All cycles shouldn't be same"
    for i in range(0, 30, 3):
        assert set(sampled_indices[i:i + 3]) == set(indices)  # contain all samples at each cycle


def _test_batches(main_ids, memory_ids, sizes, batch_ids):
    assert len(batch_ids) == len(main_ids) / sizes[0]
    assert all(len(b) == sum(sizes) for b in batch_ids)
    assert reduce(set.union, batch_ids, set()).issubset(set(main_ids).union(memory_ids))


def test_dual_batch_sampler():
    main_ids = datasource0 = list(range(4))
    main_sampler = SequentialSampler(datasource0)
    memory_ids = [5, 6, 7, 8]
    sizes = (2, 3)  # batch contains two samples from main and 3 from the other
    continuous_random_sampler = ContinuousRandomSampler(memory_ids)
    batch_sampler = DualBatchSampler(main_sampler, continuous_random_sampler, sizes=sizes)
    batches = list(batch_sampler)
    _test_batches(main_ids, memory_ids, sizes, batches)
    assert set(datasource0).intersection(batches[0]) == {0, 1}
    assert set(datasource0).intersection(batches[1]) == {2, 3}


def test_create_loader():
    config = SimpleNamespace(batch_memory_samples=2, batch_size=4,
                             torch=dict(non_bloking=True, num_workers=0, pin_memory=False))
    source = CIFAR10('../data')
    transform = ToTensor()
    main_ids = np.fromiter(range(10), dtype=int) * 10  # 0, 10, ..., 90
    memory_ids = np.array([1, 2, 3], dtype=int)
    dataset = PartialDataset(source, ids=main_ids, train=True, train_transform=transform, test_transform=transform,
                             classes=[])
    memoryset = PartialDataset(source, ids=memory_ids, train=True, train_transform=transform, test_transform=transform,
                               classes=[])
    loader = create_loader(config, main_dataset=dataset, memoryset=memoryset)

    batch_main_samples = config.batch_size - config.batch_memory_samples
    batches = list(loader)
    for b in batches:
        b[2] = b[2].tolist()
    sizes = (batch_main_samples, config.batch_memory_samples)
    _test_batches(main_ids, memory_ids, sizes, [b[2] for b in batches])

    all_ids = []
    for batch in batches:
        ids = batch[2]
        assert len(ids) == 4  # batch size correct
        assert set(ids[:batch_main_samples]).issubset(
            main_ids), "First n samples of batch should belong to the main dataset"
        assert set(ids[batch_main_samples:]).issubset(
            memory_ids), "Second part of the batch should contain only memory samples"
        all_ids.extend(ids)

    assert set(main_ids).union(memory_ids).issuperset(all_ids), "The batches should contain all samples at least ones"
    for id_ in main_ids:
        assert all_ids.count(id_) == 1, "All samples in the main dataset should appear exactly once"


def cifar100_10_test_data():
    train_source = CIFAR100(root='../data', train=True)
    test_source = CIFAR100(root='../data', train=False)
    class_order = list(range(100))
    shuffle(class_order)
    cidata = CIData(train_source=train_source, test_source=test_source, class_order=class_order,
                    n_classes_per_phase=10, n_phases=10, train_transform=torchvision.transforms.ToTensor(),
                    test_transform=torchvision.transforms.ToTensor())
    return cidata


def test_get_phase_data():
    cidata = cifar100_10_test_data()
    class_order = cidata.class_order
    cumul_classes = []
    for phase in range(1, 11):
        trainset, testset, cumul_testset = cidata.get_phase_data(phase)
        assert len(trainset.labels) == 5000
        assert len(testset.labels) == 1000
        assert set(testset.labels) == set(trainset.labels) == set(class_order[(phase - 1) * 10:phase * 10])
        assert set(cumul_testset.labels) == set(class_order[:phase * 10])
        assert trainset.classes == testset.classes == class_order[(phase - 1) * 10: phase * 10]
        cumul_classes.extend(trainset.classes)
        assert cumul_testset.classes == cumul_classes
        assert cumul_testset.source == testset.source == cidata.test_source
        assert trainset.source == cidata.train_source


def test_partial_dataset():
    source = CIFAR10(root='../data', train=True)
    classes = [2, 3, 4]
    transform = ToTensor()
    dataset = PartialDataset.from_classes(source, train=True, train_transform=transform, test_transform=transform,
                                          classes=classes)
    source_labels = np.asarray(source.targets)

    # dataset ids only belong to selected classes
    assert all(source_labels[i] in classes for i in dataset.ids)

    # all ids belonging to selected classes are present
    cc = Counter(source_labels)
    assert len(dataset) == sum(cc[cls] for cls in classes)

    # ====== dataloaders ======
    for shuffle in [True, False]:
        loader = DataLoader(dataset, batch_size=32, shuffle=shuffle)
        all_labels = []
        all_ids = []
        all_inputs = []
        for inputs, labels, ids in loader:
            all_labels.extend(labels)
            all_ids.extend(ids)
            all_inputs.append(inputs)
        all_inputs = np.concatenate(all_inputs, axis=0)

        sorting = np.argsort(all_ids)
        all_ids = np.array(all_ids)[sorting]
        all_inputs = all_inputs[sorting]
        all_labels = np.array(all_labels)[sorting]
        original_inputs = np.stack([transform(dataset.source[i][0]) for i in sorted(dataset.ids)])

        assert np.allclose(all_ids, sorted(dataset.ids))
        assert np.allclose(all_labels, source_labels[all_ids])
        assert np.allclose(all_inputs, original_inputs)


def _gather_batches(loader):
    all_inputs = []
    all_labels = []
    all_ids = []
    for inputs, labels, ids in loader:
        all_inputs.append(inputs)
        all_labels.append(labels)
        all_ids.append(ids)

    return np.concatenate(all_inputs), np.concatenate(all_labels), np.concatenate(all_ids)


def are_loaders_same(loader0, loader1):
    data0 = _gather_batches(loader0)
    data1 = _gather_batches(loader1)

    def _sort_data(data):
        data = list(data)
        sorted_ids = np.argsort(data[2])
        data[0] = data[0][sorted_ids]
        data[1] = data[1][sorted_ids]
        data[2] = data[2][sorted_ids]
        return data

    data0 = _sort_data(data0)
    data1 = _sort_data(data1)

    for i, j in zip(data0, data1):
        assert np.allclose(i, j)

    return True


def test_dataloader_with_partial_dataset():
    source = CIFAR10(root='../data', train=True)
    classes = [0, 1]
    transform = ToTensor()
    dataset = PartialDataset.from_classes(source, train=True, train_transform=transform, test_transform=transform,
                                          classes=classes)
    mem = Memory(len(dataset), source=source, train_transform=transform, test_transform=transform)
    mem.update(dataset.ids, new_classes=dataset.classes)
    mem_dataset = mem.get_dataset(train=True)

    assert sorted(mem_dataset.ids) == sorted(dataset.ids)

    loader0 = DataLoader(dataset, shuffle=False)
    loader1 = DataLoader(mem_dataset, shuffle=False)
    dataset.ids = dataset.ids.copy()

    assert are_loaders_same(loader0, loader1)
