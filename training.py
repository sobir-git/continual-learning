from typing import Dict
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import wandb

import models.bnet_base
from utils import AverageMeter, get_accuracy, to_device, Timer, get_prediction, wandb_confusion_matrix


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

        epoch_time.start()
        num_loops = self.opt.num_loops if self.type == 'cl' else 1  # for CL we can loop many times over the batches
        for i in range(num_loops):
            for batch_idx, (inputs, labels) in enumerate(data_time.get_timed_generator(loader), start=1):

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

    @torch.no_grad()
    def test(self, loader, model, mask, classnames, step):
        """Tests the model and return the accuracy"""
        w_tag = self._tag + 'test/'

        model.eval()
        losses, accuracy = AverageMeter(), AverageMeter()
        mask = to_device(mask, self.device)
        all_preds = torch.tensor([])
        all_trues = torch.tensor([])

        epoch_time = Timer().start()
        for inputs, labels in loader:
            # Get outputs
            inputs, labels = to_device((inputs, labels), self.device)
            outputs = model(inputs)
            loss = self.criterion(outputs, labels)
            losses.update(loss.data, inputs.size(0))

            # Measure accuracy
            probs = torch.softmax(outputs, dim=1)
            acc = get_accuracy(probs, labels, mask)
            accuracy.update(acc, labels.size(0))

            all_preds = torch.cat((all_preds, get_prediction(probs, mask=mask).cpu()), dim=0)
            all_trues = torch.cat((all_trues, labels.cpu()), dim=0)
        epoch_time.finish()

        cm = confusion_matrix(all_trues, all_preds, labels=range(len(classnames)))
        wandb.log(
            {
                w_tag + 'loss': losses.avg, w_tag + 'acc': accuracy.avg,
                w_tag + 'conf_mtx': wandb_confusion_matrix(confmatrix=cm, classnames=classnames)
            },
            commit=False
        )
        if isinstance(model, models.bnet_base.BranchNet):
            branch_preds = defaultdict(lambda: torch.tensor([]))
            all_trues = torch.tensor([])
            # go through all batches and gather the branch outputs
            for inputs, labels in loader:
                inputs, labels = to_device((inputs, labels), self.device)
                all_trues = torch.cat((all_trues, labels.cpu()), dim=0)
                for br_idx, probs in model.full_forward(inputs):
                    branch_preds[br_idx] = torch.cat(
                        (branch_preds[br_idx], get_prediction(probs, mask=mask).cpu()), dim=0)

            br_conf_mtxs = dict()
            # for each branch, compute the confusion matrix
            for br_idx, branch in model.branch_dict.items():
                cm = confusion_matrix(all_trues, branch_preds[br_idx], labels=range(len(classnames)))
                br_conf_mtxs[str(br_idx) + '/' + 'conf_mtx'] = \
                    wandb_confusion_matrix(confmatrix=cm, classnames=classnames)

            wandb.log({w_tag + '/' + 'br_conf_mtxs': br_conf_mtxs}, commit=False)

        wandb.log({self._step_name: step}, commit=True)
        self.logger.info(
            f'==> Test [{step}]:\tTime:{epoch_time.total:.4f}\tLoss:{losses.avg:.4f}\tAcc:{accuracy.avg:.4f}')
        return accuracy.avg
