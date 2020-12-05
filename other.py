import torch
from torch import nn
from torch.nn import functional as F

from models.bnet1 import BranchNet1
from models.bnet_base import Branch, Base
from models.cifar10cnn import Cifar10Cnn


def new_branch0():
    in_shape = (16, 5, 5)
    return Branch(nn.Sequential(
        torch.nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10),
    ), in_shape=in_shape)


def new_branch1():
    in_shape = (16, 5, 5)
    return Branch(nn.Sequential(
        torch.nn.Flatten(),
        nn.Linear(16 * 5 * 5, 10)
    ), in_shape=in_shape)


def branchnet1():
    base = Base()
    branch0 = new_branch0()
    branch1 = new_branch1()
    return BranchNet1(base, [branch0, branch1])


def branchnet110():
    base = Base()
    branch0 = new_branch0()
    branch10 = new_branch1()
    branch11 = new_branch1()
    return BranchNet1(base, [branch0, branch10, branch11])


def simple_model():
    return nn.Sequential(Base(), new_branch0())


def simple_train(net, loader, n_epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    datasize = len(loader)
    log_every = datasize // 20

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % log_every == log_every - 1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / log_every))
                running_loss = 0.0

        print('Finished Training')


def simple_test(net, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.eq(predicted, labels).sum().item()

    print('Accuracy  %.2f %%' % (100 * correct / total))


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    pass