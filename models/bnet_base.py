import itertools
from typing import List, Union

import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch import nn
import numpy as np
from torch.utils.data import DataLoader

from logger import Logger
from models.layers import FCBlock
from training import TrainerBase
from utils import AverageMeter, Timer, get_prediction


class LeLFunction:
    def __init__(self, opt):
        self.loss_fn = getattr(self, opt.lel_fn)
        self.detach = opt.lel_detach

    def __call__(self, actual_loss, estimated_loss):
        if self.detach:
            actual_loss = actual_loss.detach()
        r = self.loss_fn(actual_loss, estimated_loss)
        return r

    def mse(self, acl, esl):  # (N, B)
        return F.mse_loss(acl, esl, reduction='none')


class LossEstimator(nn.Module):
    def __init__(self, opt, in_shape, hidden_layers=None):
        super(LossEstimator, self).__init__()
        layers = [torch.nn.Flatten()]
        self.in_shape = in_shape
        in_size = np.prod(in_shape)
        if hidden_layers:
            layers.append(FCBlock(in_size, *hidden_layers, bn=opt.bn))
        layers.append(nn.Linear(hidden_layers[-1] if hidden_layers else in_size, 1))
        layers.append(nn.Softplus())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out


class Branch(nn.Module):
    def __init__(self, opt, stem: nn.Module, in_shape):
        super(Branch, self).__init__()
        self.stem = stem
        self.le = LossEstimator(opt, in_shape=in_shape)

    def out_and_est_loss(self, x):
        return self.stem(x), self.le(x)

    def estimate_loss(self, x) -> torch.Tensor:
        return self.le(x)

    def forward(self, x):
        return self.stem(x)


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


class BranchNet(nn.Module):
    _branches: Union[List[Branch], nn.ModuleList]
    _base: nn.Module
    beta = 0.

    def __init__(self, base, branches):
        super().__init__()
        self._base = base
        self._branches = nn.ModuleList(branches)

    @property
    def branch_dict(self):
        return {'br' + str(i): br for i, br in enumerate(self._branches)}

    @property
    def branches(self):
        return list(self._branches)

    @property
    def base(self):
        return self._base

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def B(self):
        """Number of branches"""
        return len(self._branches)

    def forward(self, x, br_idx=None):
        """
        Return outputs(shape N, B, C), est_losses(shape N, B).
        If br_idx is given the shapes would be (N, C) and (N, 1)
        """
        N = x.size(0)
        B = self.B

        base_out = self._base(x)
        if br_idx is not None:
            br = self._branches[br_idx]
            out, est_loss = br.out_and_est_loss(base_out)
            return out, est_loss

        est_losses = []
        outputs = []
        for br_idx, br in enumerate(self._branches):
            out, est_loss = br.out_and_est_loss(base_out)  # (N, C)
            assert est_loss.shape == (N, 1)
            est_losses.append(est_loss)
            outputs.append(out)

        est_losses = torch.cat(est_losses, dim=1)  # (N, B)
        assert est_losses.shape == (N, B)
        outputs = torch.stack(outputs, dim=1)  # (N, B, C)
        assert outputs.dim() == 3, outputs.shape[:2] == (N, B)

        return outputs, est_losses


def gen_branch_mask(br_probs):
    # this is a mask that will capture only the selected branches
    N, B = br_probs.shape
    branch_mask = torch.zeros(br_probs.shape, dtype=torch.bool)
    for i in range(N):  # loop over all batch items
        # randomly choose a branch
        _br_idx = np.random.choice(B, p=br_probs[i])
        branch_mask[i, _br_idx] = 1
    return branch_mask


def get_cross_entropy_loss(criterion, outputs, y):  # (N, B, C), (N,)
    cross_entropy_loss = []
    for br_idx in range(outputs.size(1)):
        loss_ = criterion(outputs[:, br_idx, :], y)  # (N,)
        cross_entropy_loss.append(loss_)
    cross_entropy_loss = torch.stack(cross_entropy_loss, 1)  # (N,B)
    return cross_entropy_loss


class BnetTrainer(TrainerBase):
    _default_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    beta = 0.

    def __init__(self, opt, model: BranchNet, logger: Logger, device, optimizer, backprop=None, lel_function=None):
        super().__init__(opt, model, logger, device, optimizer)
        self.lel_function = lel_function or LeLFunction(opt)
        self.criterion = self._default_criterion.to(device)
        self.backprop = backprop or Backprop(opt)

    def get_branch_probs(self, cross_entropy_loss, est_loss):
        """Return probabilities of selecting branches given their classification losses."""
        assert cross_entropy_loss.shape == est_loss.shape
        losses = (est_loss + self.beta * cross_entropy_loss).detach().cpu().numpy()

        # always be safe from zero division
        losses += 1e-16
        p: np.ndarray = 1 / losses
        p = p / p.sum(axis=1).reshape(-1, 1)
        assert p.shape == losses.shape
        return p

    def _train(self, x, y):
        """
        Trains the model with given the given batch (x, y). Assumes x, y are in the same device as the model
        Args:
            x: input
            y: labels

        Returns:
            A tuple containing (cross_entropy_loss, branch_mask, le_loss).
             Where cross_entropy_loss (N, B) --
             branch_mask (N, B) -- branch mask of selected branches per sample
             le_loss (N, B) -- loss estimation loss
        """
        N = x.size(0)
        B = len(self.model.branches)
        # C = number of classes

        outputs, estimated_loss = self.model(x)
        assert outputs.dim() == 3 and outputs.shape[:2] == (N, B)
        assert estimated_loss.shape == (N, B)

        # construct cross-entropy losses for all sample, batch pairs
        cross_entropy_loss = get_cross_entropy_loss(self.criterion, outputs, y)

        # construct a probability matrix for choosing branches
        br_probs = self.get_branch_probs(cross_entropy_loss, estimated_loss)  # (N, B)
        branch_mask = gen_branch_mask(br_probs)

        # construct loss estimation loss; this is a loss for each sample, branch pairs
        le_loss = self.lel_function(cross_entropy_loss, estimated_loss)

        # backprop classification loss
        self.optimizer.zero_grad()
        self.backprop(self.model, cross_entropy_loss, branch_mask, le_loss)

        # backprop loss estimation loss
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.clip)  # Always be safe than sorry
        self.optimizer.step()

        return cross_entropy_loss, branch_mask, le_loss  # shapes all (N, B)

    def train(self, dataloader: DataLoader, num_loops=1):
        """
        Trains the model on the given dataloader for num_loops passes.
        Args:
            dataloader:
            num_loops: number of passes over the dataloader

        Returns:
            The total time it took for data-loading/processing stuff.
        """
        self._before_train()

        def get_averages(d):
            return {k: v.avg for k, v in d.items()}

        B = len(self.model.branches)
        device = self.device
        data_time = Timer()  # the time it takes for forward+backward+step
        clf_losses = {'main': AverageMeter(), **{str(br_idx): AverageMeter() for br_idx in range(B)}}
        le_losses = {str(br_idx): AverageMeter() for br_idx in range(B)}
        log_every = self._get_log_every(len(dataloader))

        for i in range(num_loops):
            for batch_idx, (inputs, labels) in enumerate(data_time.get_timed_generator(dataloader)):
                inputs, label = inputs.to(device), labels.to(device)
                cross_entropy_loss, branch_mask, lel = self._train(inputs, labels)
                clf_losses['main'].update((cross_entropy_loss * branch_mask).sum() / branch_mask.sum(), inputs.size(0))
                for br_idx in range(B):
                    clf_losses[str(br_idx)].update(cross_entropy_loss[:, br_idx].mean(), inputs.size(0))
                    le_losses[str(br_idx)].update(lel[:, br_idx].mean(), inputs.size(0))

                if batch_idx % log_every == 0:
                    self.logger.log(
                        {'clf_losses': get_averages(clf_losses), 'le_losses': get_averages(le_losses),
                         'percent': (batch_idx + 1) / len(dataloader)}, commit=True)
                    # reset meters
                    for meter in itertools.chain(clf_losses.values(), le_losses.values()):
                        meter.reset()

        self._after_train()
        return data_time.total

    @torch.no_grad()
    def test(self, loader: DataLoader, classnames, mask):
        """
        Test the model.
        Returns classification loss, accuracy.
        """
        self._before_test()

        B = len(self.model.branches)
        device = self.device
        loss_est_heatmap = torch.zeros(len(classnames), B, device=device)

        # holds predictions of main(combined) and indiviual branches
        predictions = [list() for _ in range(len(self.model.branches) + 1)]
        trues = []
        main_loss_meter = AverageMeter()  # this is a loss meter for the main output (the 'best' branches combined)
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, estimated_losses = self.model(inputs)  # (N, B, C), (N, B)

            # ==== branch predictions
            for br_idx in range(B):
                br_output = outputs[:, br_idx, :]
                preds = get_prediction(y_prob=torch.softmax(br_output, 1), mask=mask)
                predictions[br_idx].extend(preds)

            # ==== main prediction
            # for each sample select the branch with minimum estimated loss
            branch_ids = torch.argmin(estimated_losses, 1)  # (N, 1)
            assert branch_ids.shape == (inputs.size(0),)
            main_output = []
            for i in range(inputs.size(0)):  # N
                main_output.append(outputs[i, branch_ids[i], :])  # (C,)
            main_output = torch.stack(main_output, 0)  # (N, C)
            main_loss_meter.update(self.criterion(main_output, labels).mean(), inputs.size(0))
            predictions[-1].extend(get_prediction(y_prob=torch.softmax(main_output, 1), mask=mask))

            # ==== update loss-estimation heatmap
            for i in range(inputs.size(0)):
                cls = labels[i]
                loss_est_heatmap[cls, :] += estimated_losses[i, :]
            # normalize
            trues.append(labels)

        trues = torch.cat(trues)

        # normalize
        cls_counter = torch.bincount(trues)
        cls_counter = torch.cat((cls_counter, torch.zeros(len(classnames) - len(cls_counter))))
        cls_counter[cls_counter == 0] = 1  # replace zero to prevent zero division error
        loss_est_heatmap = loss_est_heatmap / cls_counter.view(-1, 1)

        # ==== for each main + branches predictions
        # report confusion matrix and accuracies/recalls
        confusion_matrices = {}
        for i, preds in enumerate(predictions):
            cm = confusion_matrix(trues, preds)
            confusion_matrices[str(i) if i < B else 'main'] = cm
            self.logger.log_confusion_matrix(cm, classnames)

        accuracies = self.logger.log_accuracies(confusion_matrices, classnames)
        # report estimated_losses for each branch-class pair
        self.logger.log_heatmap('estimated_loss', data=loss_est_heatmap, rows=classnames,
                                columns=[str(i) for i in range(B)],
                                title='Average estimated loss per class', vmin=0)
        # report classification loss
        self.logger.log({'clf_loss': main_loss_meter.avg})

        self._after_test()
        return main_loss_meter.avg, accuracies['main']


class Backprop:
    def __init__(self, opt):
        self.method = getattr(self, opt.backprop)
        self._frozen_base_params = []

    def __call__(self, model, cross_entropy_loss, branch_mask, lel):
        self.model = model
        self.cross_entropy_loss = cross_entropy_loss
        self.branch_mask = branch_mask
        self.lel = lel
        return self.method()

    def only_clf(self):
        clf_loss = self.cross_entropy_loss * self.branch_mask
        clf_loss.mean().backward()

    def freeze_model_base(self):
        if len(self._frozen_base_params) > 0:
            raise Exception("Already some base params are frozen")

        for p in self.model.base.parameters(recurse=True):
            if p.requires_grad:
                p.requires_grad = False
                self._frozen_base_params.append(p)

    def undo_model_base_freeze(self):
        for p in self._frozen_base_params:
            assert not p.requires_grad, "Did someone else unfroze some base params after me?"
            p.requires_grad = True
        self._frozen_base_params.clear()

    def clf_and_le(self):
        # shapes: (N, B), (N, B), (N, B)

        clf_loss = (self.cross_entropy_loss * self.branch_mask).sum() / self.branch_mask.sum()

        # backprop clf_loss
        clf_loss.mean().backward(retain_graph=True)

        # freeze the model base and backprop lel
        self.freeze_model_base()
        self.lel.mean().backward()

        # unfreeze the model base back
        self.undo_model_base_freeze()
