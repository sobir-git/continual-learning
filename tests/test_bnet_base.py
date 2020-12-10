from unittest.mock import Mock

import torch
import numpy as np

from models.bnet_base import LossEstimator, BranchNet, Branch, BnetTrainer
from .common import *
from .test_dataloader import vision_dataset, partial_dataloader

INPUT_SHAPE = (3, 32, 32)
BASE_OUT_SHAPE = (8, 6, 6)
BATCH_SIZE = 16
NUM_CLASSES = 10


@pytest.fixture(params=[None, (128, 64)])
def le(opt, request):
    hidden_layers = request.param
    return LossEstimator(opt, in_shape=BASE_OUT_SHAPE, hidden_layers=hidden_layers)


@pytest.fixture
def branch(opt):
    return new_branch(opt)


@pytest.fixture
def fake_base_out():
    return torch.rand(BATCH_SIZE, *BASE_OUT_SHAPE)


@pytest.fixture
def fake_input():
    return torch.rand(BATCH_SIZE, *INPUT_SHAPE)


def new_branch(opt):
    insize = int(np.prod(BASE_OUT_SHAPE))
    stem = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(insize, 20),
        torch.nn.Linear(20, NUM_CLASSES),
    )
    return Branch(opt, stem, BASE_OUT_SHAPE)


@pytest.fixture
def branchnet(opt):
    base = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=2, stride=3),  # 8 x 12 x 12
        torch.nn.MaxPool2d(2, 2)  # 8 x 6 x 6
    )
    branches = [new_branch(opt), new_branch(opt)]
    return BranchNet(base=base, branches=branches)


class TestLossEstimator:

    def test_forward(self, le):
        batch_size = 16
        fake_base_out = torch.rand(batch_size, *BASE_OUT_SHAPE)
        out = le(fake_base_out)
        assert out.shape == (batch_size, 1)
        assert torch.Tensor.all(torch.gt(out, 0))


class TestBranch:
    def test_out_and_est_loss(self, opt, fake_base_out):
        branch = new_branch(opt)
        output, est_loss = branch.out_and_est_loss(fake_base_out)
        assert output.shape == (BATCH_SIZE, NUM_CLASSES)
        assert est_loss.shape == (BATCH_SIZE, 1)


class TestBranchNet:
    def test_forward(self, branchnet: BranchNet, fake_input):
        outputs, estimation_losses = branchnet.forward(fake_input)
        B = len(branchnet.branches)
        assert outputs.shape == (BATCH_SIZE, B, NUM_CLASSES)
        assert estimation_losses.shape == (BATCH_SIZE, B)


@pytest.fixture
def trainer(opt, branchnet):
    logger = Mock()
    optimizer = torch.optim.SGD(branchnet.parameters(), lr=0.01)
    return BnetTrainer(opt, model=branchnet, logger=logger, device=torch.device('cpu'), optimizer=optimizer)


class TestBnetTrainer:
    def test_get_branch_probs(self, trainer):
        model = trainer.model
        B = len(model.branches)
        cross_entropy_loss = torch.rand(BATCH_SIZE, B)
        # set it for branch 0 large
        cross_entropy_loss[:, 0] = 100.
        est_loss = cross_entropy_loss.clone()
        probs = trainer.get_branch_probs(cross_entropy_loss, est_loss)
        assert probs.shape == est_loss.shape
        assert np.all(probs[:, 0] < probs[:, 1])

    def test__train(self, trainer, vision_dataset):
        x, y = next(iter(vision_dataset.pretest_loader))
        trainer.optimizer = Mock()
        cross_entropy_loss, branch_mask, le_loss = trainer._train(x, y)
        assert trainer.optimizer.step.called, "Optimizer step has not been called"
        N, B = BATCH_SIZE, len(trainer.model.branches)
        assert cross_entropy_loss.shape == (N, B)
        assert branch_mask.shape == (N, B)
        assert le_loss.shape == (N, B)
        assert torch.allclose(branch_mask, branch_mask ** 2), "Branch mask should be binary mask"

    def test_train(self, trainer, vision_dataset):
        loader = partial_dataloader(vision_dataset.pretrain_loader, 10)  # a small loader with three batches
        data_time = trainer.train(loader, num_loops=2)
        assert isinstance(data_time, float) and data_time > 0

    def test_test(self, trainer, vision_dataset):
        datatime = trainer.train(vision_dataset.pretest_loader)
        assert isinstance(datatime, float) and datatime > 0

    def test_backprop(self, trainer: BnetTrainer, fake_input):
        # it's just a weak test
        outputs, estimated_losses = trainer.model.forward(fake_input)
        cross_entropy_loss = outputs.mean(2).clone()  # just because
        assert cross_entropy_loss.shape == (BATCH_SIZE, len(trainer.model.branches)) == estimated_losses.shape
        lel = trainer.lel_function(cross_entropy_loss, estimated_losses, reduction='none')  # ok this is fine
        branch_mask = torch.randint_like(cross_entropy_loss, low=0, high=1)
        trainer.backprop(cross_entropy_loss, branch_mask, lel)

        # check if model parameters have gradients
        for param in trainer.model.parameters(recurse=True):
            if param.requires_grad:
                assert param.grad is not None
