import time

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

from dataloader import get_augment_transforms, get_statistics
from exp2.data import PartialDataset
from exp2.models import simple_net
from utils import utils, AverageMeter

##########################
### SETTINGS
##########################

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 258
DATASET = 'CIFAR100'
MODELPATH = 'simple_net.pty'
DOWNLOAD = True

# Architecture
NUM_CLASSES = 20
DEVICE = utils.get_default_device()

mean, std, n_classes_in_whole_dataset, inp_size, in_channels = get_statistics(DATASET)
train_augment, test_augment = get_augment_transforms(dataset=DATASET, inp_sz=32)
train_transforms = torchvision.transforms.Compose(
    train_augment + [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
test_transforms = torchvision.transforms.Compose(
    test_augment + [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)])
train_source = CIFAR100('./data', train=True, transform=train_transforms, download=DOWNLOAD)
test_source = CIFAR100('./data', train=False, transform=test_transforms, download=DOWNLOAD)

classes = list(range(20))
trainset = PartialDataset.from_classes(train_source, classes)
testset = PartialDataset.from_classes(test_source, classes)
train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
test_loader = DataLoader(testset, batch_size=32, shuffle=True)

net = simple_net(n_classes=len(classes))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)


def compute_accuracy(net, data_loader):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        logits = net(features)
        preds = torch.argmax(logits, 1)
        num_examples += targets.size(0)
        correct_pred += (preds == targets).sum()
    return correct_pred / num_examples * 100


start_time = time.time()
all_test_acc = []
acc_tol = 10


def save_model(model, path=MODELPATH):
    print("Saving model in", path)
    torch.save(model.state_dict(), path)


def load_model(model, path=MODELPATH):
    model.get_state_dict(torch.load(path))


for epoch in range(NUM_EPOCHS):
    atest_acc = AverageMeter()
    for batch_idx, (features, targets) in enumerate(train_loader):

        ### PREPARE MINIBATCH
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        ### FORWARD AND BACK PROP
        logits = net(features)
        loss = criterion(logits, targets)
        optimizer.zero_grad()

        loss.backward()

        ### UPDATE MODEL PARAMETERS
        optimizer.step()

        ### LOGGING
        if not batch_idx % 120:
            print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} | '
                  f'Batch {batch_idx:03d}/{len(train_loader):03d} |'
                  f' Cost: {loss:.4f}')

    with torch.no_grad():
        train_acc = compute_accuracy(net, train_loader)
        test_acc = compute_accuracy(net, test_loader)
        all_test_acc.append(test_acc)
        print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} Train Acc.: {train_acc:.2f}%'
              f' | Test Acc.: {test_acc:.2f}%')

    elapsed = (time.time() - start_time) / 60
    print(f'Time elapsed: {elapsed:.2f} min')

    # early stopping
    if len(all_test_acc) > acc_tol and max(all_test_acc[-acc_tol:]) < max(all_test_acc):
        save_model(net)
        print('Exiting by early stopping.')
        break

elapsed = (time.time() - start_time) / 60
print(f'Total Training Time: {elapsed:.2f} min')
