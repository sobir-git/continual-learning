from functools import reduce
from types import SimpleNamespace

import numpy as np
from torch.utils.data import SequentialSampler, RandomSampler
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor

from exp2.data import ContinuousRandomSampler, DualBatchSampler, create_loader, PartialDataset


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
    config = SimpleNamespace(batch_memory_samples=2, batch_size=4, num_workers=0)
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
