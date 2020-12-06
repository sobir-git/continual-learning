import time
from typing import Iterator, Dict

import numpy as np
import torch
import wandb

import models.bnet_base
from utils import AverageMeter, get_accuracy, to_device, Timer


class Trainer:
    _default_criterion = torch.nn.CrossEntropyLoss()

    def __init__(self, opt, logger, device, tag='', criterion=None):
        self.opt = opt
        self.logger = logger
        assert not opt.regularization == 'cutmix', "we cannot apply cutmix"
        self.device = device
        self.criterion = (criterion if criterion else self._default_criterion).to(device)
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
        # create metrics
        _mn = ['loss']
        if is_bnet:
            _mn.append('clf_loss')
            _mn.extend('lel' + str(i) for i in range(len(model.branches)))
        metrics: Dict[str, AverageMeter] = {name: AverageMeter() for name in _mn}

        # create timers
        epoch_time = Timer()  # the time it takes for one epoch
        data_time = Timer()  # the time it takes for forward+backward+step
        epoch_loss = AverageMeter()
        loader_iter: Iterator = iter(loader)
        data_time.attach(loader_iter.__iter__)

        epoch_time.start()
        for batch_idx, (inputs, labels) in enumerate(loader_iter, start=1):

            # Tweak inputs
            with data_time:
                inputs, labels = to_device((inputs, labels), self.device)

            # forward + backward + optimize
            if is_bnet:
                loss, lels, clf_loss = model.loss(inputs, labels, branch_idx=branch_idx)
            else:
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.opt.clip)  # Always be safe than sorry
            optimizer.step()

            # Log losses
            metrics['loss'].update(loss.item())
            epoch_loss.update(loss.item())
            if is_bnet:
                metrics['clf_loss'].update(clf_loss.item())
                for j in range(len(lels)):
                    metrics['lel' + str(j)].update(lels[j].item())


            # print statistics
            if batch_idx % log_every == 0:

                wandb.log({
                    **{w_tag + k: v.avg for k, v in metrics.items()},
                    **{'epoch': epoch or phase}})
                msg = f'[{epoch or phase}, {batch_idx / datasize * 100:.0f}%]' + \
                      '\t '.join(f'{k}: {metrics[k].avg:.3f}' for k in metrics.keys())
                self.logger.info(msg)

                # reset metrics
                for v in metrics.values():
                    v.reset()

        epoch_time.finish()
        self.logger.info(
            f'==> Train[{epoch or phase}]:\tTime:{epoch_time.total:.4f}\tData:{data_time.total:.4f}\tLoss:{epoch_loss.avg:.4f}\t')

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
