import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

import utils


class LossEstimator(nn.Module):
    def __init__(self, in_shape, hidden_layers=None):
        super(LossEstimator, self).__init__()
        layers = [Flatten()]
        self.in_shape = in_shape
        in_size = np.prod(in_shape)
        if hidden_layers:
            for size in hidden_layers:
                layers.append(nn.Linear(in_size, size))
                layers.append(nn.ReLU())
                in_size = size
        layers.append(nn.Linear(in_size, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, np.prod(self.in_shape))
        out = self.layers(x)
        return out


class Branch(nn.Module):
    def __init__(self, stem: nn.Module, in_shape):
        super(Branch, self).__init__()
        self.stem = stem
        self.le = LossEstimator(in_shape=in_shape)

    def estimate_loss(self, x) -> torch.Tensor:
        return self.le(x)

    def forward(self, x):
        return self.stem(x)


def loss_estimation_loss(est, actual, reduction='mean'):
    l = F.mse_loss(est, actual)
    if reduction == 'none':
        return l
    if reduction == 'mean':
        return l.mean()
    elif reduction == 'sum':
        return l.sum()
    raise ValueError(f'Invalid value for reduction: {reduction}')


class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x


class Flatten(nn.Module):
    def forward(self, input):
        """
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0) # of batches,
        will flatten to be 1 * nb_elements.
        """
        batch_size = input.size(0)
        out = input.view(batch_size, -1)
        return out  # (batch_size, *size)


def train_bnet(net, loader, n_epochs=1, branch_idx=None):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train branchnet
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        datasize = len(loader)
        log_every = datasize // 20
        loss_meter = utils.AverageMeter()
        lel_meters = [utils.AverageMeter() for _ in net.branches]
        clf_loss_meter = utils.AverageMeter()

        for i, data in enumerate(loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss, lels, clf_loss = net.loss(inputs, labels, branch_idx=branch_idx)
            loss.backward()
            optimizer.step()

            # update average-meters
            loss_meter.update(loss.detach())
            clf_loss_meter.update(clf_loss.detach())
            for j in range(len(lels)):
                lel_meters[j].update(lels[j])

            # print statistics
            if i % log_every == log_every - 1:
                _formatted_lels = ''.join(['%.3f ' % l.mean() for l in lel_meters])
                print('[%d, %.1f] loss: %.3f, lels: %s, clf_loss: %.3f' %
                      (epoch + 1, (i + 1) / datasize * 100, loss_meter.mean(), _formatted_lels, clf_loss_meter.mean())
                      )
                # reset average-meters
                loss_meter.reset()
                clf_loss_meter.reset()
                for _l in lel_meters: _l.reset()

    print('Finished Training')


def test_bnet(net, loader, branch_idx=None):
    correct = 0
    total = 0
    branch_ids = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            images, labels = data
            with torch.no_grad():
                out, br_ids = net.forward(images, branch_idx=branch_idx, return_id=True)
            branch_ids.extend(br_ids)
            all_labels.extend(labels)
            _, predicted = torch.max(out, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy: %0.3f %%' % (100 * correct / total))
    return correct / total, np.array(branch_ids), np.array(all_labels)


