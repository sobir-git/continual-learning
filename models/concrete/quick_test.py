import torch

from utils import Timer


def test(model):
    timer = Timer()
    batch = torch.rand(16, 3, 32, 32)

    n_runs = 10
    for i in range(n_runs):
        with timer:
            output = model(batch)
    assert output.shape == (16, 10)
    print('Running time: ', timer.total / n_runs)