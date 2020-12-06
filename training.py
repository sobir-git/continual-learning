from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

import models.bnet_base
from utils import AverageMeter, get_accuracy, to_device, Timer


class Trainer:
    _default_criterion = torch.nn.CrossEntropyLoss()

    def __init__(self, opt, logger, device, type, criterion=None):
        self.opt = opt
        self.logger = logger
        self.type = type
        assert not opt.regularization == 'cutmix', "we cannot apply cutmix"
        self.device = device
        self.criterion = (criterion if criterion else self._default_criterion).to(device)
        if type in ['cl', 'pre']:
            self._tag = type + '/'
            self._step_name = 'epoch' if type == 'pre' else 'phase'
        else:
            raise ValueError('Type should be on of cl or pre')

    def train(self, loader: DataLoader, optimizer, model, step, branch_idx=None):
        """
        This trains either:
            - one epoch for pretraining
            - one phase for continual learning
                - in this case there is option to loop over the data multiple times (opt.num_loops)

            In both cases either epoch number of phase number is given, just for the purpose of logging.
        """
        w_tag = self._tag + 'train/'  # tag to preprend wandb metric name
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
        timed_loader = data_time.get_timed_generator(loader)

        epoch_time.start()
        for batch_idx, (inputs, labels) in enumerate(timed_loader, start=1):

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
                    **{self._step_name: step}})
                msg = f'[{step}, {batch_idx / datasize * 100:.0f}%]\t' + \
                      '\t '.join(f'{k}: {metrics[k].avg:.3f}' for k in metrics.keys())
                self.logger.info(msg)

                # reset metrics
                for v in metrics.values():
                    v.reset()

        epoch_time.finish()
        self.logger.info(
            f'==> Train[{step}]:\tTime:{epoch_time.total:.4f}\tData:{data_time.total:.4f}\tLoss:{epoch_loss.avg:.4f}\t')

    def test(self, loader, model, mask, step):
        """Tests the model and return the accuracy"""
        w_tag = self._tag + 'test/'

        model.eval()
        losses, accuracy = AverageMeter(), AverageMeter()
        mask = to_device(mask, self.device)

        epoch_time = Timer().start()
        with torch.no_grad():
            for inputs, labels in loader:
                # Get outputs
                inputs, labels = to_device((inputs, labels), self.device)
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                losses.update(loss.data, inputs.size(0))

                # Measure accuracy
                prob = torch.softmax(outputs, dim=1)
                acc = get_accuracy(prob, labels, mask)
                accuracy.update(acc, labels.size(0))

        epoch_time.finish()
        wandb.log({w_tag + 'loss': losses.avg, w_tag + 'acc': accuracy.avg, self._step_name: step})
        self.logger.info(
            f'==> Test [{step}]:\tTime:{epoch_time.total:.4f}\tLoss:{losses.avg:.4f}\tAcc:{accuracy.avg:.4f}')
        return accuracy.avg
