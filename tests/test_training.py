from training import StandardTrainer
from .common import *
from .test_dataloader import *


@pytest.fixture
def simple_classifier():
    n_classes = 10

    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.MaxPool2d((2, 2), stride=2),  # 8 x 16 x 16
        torch.nn.ReLU(),

        torch.nn.Conv2d(8, 16, 3, padding=1),
        torch.nn.MaxPool2d((2, 2), stride=2),  # 16 x 8 x 8
        torch.nn.ReLU(),

        torch.nn.Flatten(),
        torch.nn.Linear(16 * 8 * 8, n_classes),
        torch.nn.ReLU()
    )


def test_train(opt, simple_classifier, vision_dataset):
    loader = partial_dataloader(vision_dataset.pretrain_loader, 10)  # a small loader with three batches
    optimizer = torch.optim.SGD(simple_classifier.parameters(), lr=0.001)
    trainer = StandardTrainer(opt, model=simple_classifier, optimizer=optimizer, logger=Mock(),
                              device=torch.device('cpu'))
    data_time = trainer.train(loader, num_loops=2)
    assert isinstance(data_time, float) and data_time > 0


def test_test(opt, simple_classifier, vision_dataset):
    loader = partial_dataloader(vision_dataset.pretest_loader, 3)  # a small loader with three batches
    optimizer = torch.optim.SGD(simple_classifier.parameters(), lr=0.001)
    trainer = StandardTrainer(opt, model=simple_classifier, optimizer=optimizer, logger=Mock(),
                              device=torch.device('cpu'))
    loss, accuracy = trainer.test(loader, Mock(), torch.ones(10))
    assert loss and accuracy
