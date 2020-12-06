import time

import numpy as np
import torch
import wandb

import models.bnet_base
from utils import AverageMeter, get_accuracy, to_device


class Trainer:
    def __init__(self, opt, logger, device, tag=''):
        self.opt = opt
        self.logger = logger
        assert not opt.regularization == 'cutmix', "we cannot apply cutmix"
        self.default_criterion = torch.nn.CrossEntropyLoss().to(device)
        self.device = device
        if tag:
            self.tag = tag + '/'
        else:
            self.tag = ''

    def train(self, loader, optimizer, model, epoch=None, phase=None, branch_idx=None):
        assert epoch is None or epoch > 0
        assert phase is None or phase > 0
        assert epoch or phase

        """
        This trains either:
            - one epoch for pretraining
            - one phase for continual learning
                - in this case there is option to loop over the data multiple times (opt.num_loops)

            In both cases either epoch number of phase number is given, just for the purpose of logging.
        """
        w_tag = self.tag + 'train/'

        is_bnet = isinstance(model, models.bnet_base.BranchNet)

        datasize = len(loader)
        log_every = min(np.ceil(datasize / 5), max(np.ceil(datasize / 20), 1000))  # log every n batches

        model.train()
        loss_meter, data_time, batch_time = [AverageMeter() for _ in range(3)]

        if is_bnet:
            lel_meters = [AverageMeter() for _ in model.branches]
            clf_loss_meter = AverageMeter()

        start = time.time()

        for loop in range(self.opt.num_loops if phase else 1):  # loop only applies for CL training
            for batch_idx, (inputs, labels) in enumerate(loader, start=1):

                # Tweak inputs
                inputs, labels = to_device((inputs, labels), self.device)
                data_time.update(time.time() - start)

                # forward + backward + optimize
                if is_bnet:
                    loss, lels, clf_loss = model.loss(inputs, labels, branch_idx=branch_idx)
                else:
                    outputs = model(inputs)
                    loss = self.default_criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.opt.clip)  # Always be safe than sorry
                optimizer.step()

                # Log losses
                batch_time.update(time.time() - start)
                loss_meter.update(loss.item())
                if is_bnet:
                    clf_loss_meter.update(clf_loss.item())
                    for j in range(len(lels)):
                        lel_meters[j].update(lels[j].item())

                # print statistics
                if batch_idx % log_every == 0:
                    wandb.log({w_tag + 'loss': loss_meter.avg}, step=batch_idx)
                    msg = f'[{epoch or phase}, {batch_idx / datasize * 100:.0f}%]\t loss: {loss_meter.avg:.3f}'
                    if is_bnet:
                        wandb.log(
                            dict(**{w_tag + 'lel' + str(j): lm.avg for j, lm in enumerate(lel_meters)},
                                 **{w_tag + 'clf_loss': clf_loss_meter.avg},
                                 **{'epoch': epoch or phase}),
                            step=batch_idx
                        )
                        _formatted_lels = ''.join(['%.3f ' % l.avg for l in lel_meters])
                        msg += f'\t clf_loss: {clf_loss_meter.avg:.3f}\t lels: {_formatted_lels}'
                    self.logger.info(msg)

                start = time.time()

        self.logger.info(
            f'==> Train[{epoch or phase}]:\tTime:{batch_time.sum:.4f}\tData:{data_time.sum:.4f}\tLoss:{loss_meter.avg:.4f}\t')

    def test(self, loader, model, mask, phase):
        """Tests the model and return the accuracy"""
        w_tag = self.tag + 'test/'

        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        model.eval()
        losses, batch_time, accuracy = AverageMeter(), AverageMeter(), AverageMeter()
        mask = to_device(mask, self.device)

        with torch.no_grad():
            start = time.time()
            for inputs, labels in loader:
                # Get outputs
                inputs, labels = to_device((inputs, labels), self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                losses.update(loss.data, inputs.size(0))

                # Measure accuracy
                prob = torch.softmax(outputs, dim=1)
                acc = get_accuracy(prob, labels, mask)
                accuracy.update(acc, labels.size(0))
                batch_time.update(time.time() - start)
                start = time.time()

        wandb.log({w_tag + 'loss': losses.avg, w_tag + 'acc': accuracy.avg, 'epoch': phase})
        self.logger.info(
            f'==> Test [{phase}]:\tTime:{batch_time.sum:.4f}\tLoss:{losses.avg:.4f}\tAcc:{accuracy.avg:.4f}')
        return accuracy.avg
