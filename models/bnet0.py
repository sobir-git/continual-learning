from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.bnet_base import Branch, loss_estimation_loss


class BranchNet0(nn.Module):
    def __init__(self, base: nn.Module, branches: List[Branch]):
        super(BranchNet0, self).__init__()
        self.base = base
        self.branches: nn.ModuleList[branches] = nn.ModuleList(branches)

    def forward(self, x, branch_idx=None, return_id=False):
        N = x.size(0)
        B = len(self.branches)

        base_out = self.base(x)
        if branch_idx is not None:
            out = self.branches[branch_idx](base_out)
            if return_id:
                return out, [branch_idx] * N
            else:
                return out

        est_losses = [self.calc_estimation_loss(br_idx, base_out=base_out) for br_idx in range(B)]
        # choose a branch with minimum estimated loss for each batch item
        po = np.hstack([e.detach().numpy().reshape(N, 1) for e in est_losses])  # shape (N, B)
        assert po.shape == (N, B), po.shape

        # from each row get argmin -> branch_idx
        branch_ids = np.argmin(po, axis=1)
        assert branch_ids.shape == (N,), branch_ids.shape

        # get item indices that will be fed to each branch
        outputs = []
        ids = []
        for i, br in enumerate(self.branches):
            # select items that go to branch i
            item_ids = branch_ids == i

            if item_ids.sum() > 0:
                items = base_out[item_ids]
                outputs.append(br(items))
                ids.append(item_ids)

        # organize outputs of each branch in a single output with correct input order
        output = torch.empty(N, *outputs[0].shape[1:])
        for item_ids, out in zip(ids, outputs):
            output[item_ids] = out
        if return_id:
            return output, branch_ids
        return output

    def calc_estimation_loss(self, branch_idx, x=None, base_out=None):

        if (x is None) and (base_out is None):
            raise ValueError('x and base_out both absent')
        if base_out is None:
            base_out = self.base(x)

        N = base_out.size(0)
        est_loss = self.branches[branch_idx].estimate_loss(base_out)  # shape = (N,1)
        assert est_loss.shape == (N, 1), est_loss.shape
        est_loss = est_loss.view(N)
        assert est_loss.shape == (N,), est_loss.shape
        return est_loss

    def loss(self, x, y, branch_idx=None):
        clf_losses = []
        lels = []
        N = x.size(0)
        B = len(self.branches)

        base_out = self.base(x)
        if branch_idx is not None:
            br_out = self.branches[branch_idx](base_out)
            clf_loss = F.nll_loss(F.log_softmax(br_out, dim=1), y)
            loss = clf_loss
            return loss, [0, 0], clf_loss.detach()

        for i, br in enumerate(self.branches):
            # forward through the branch
            br_out = br(base_out)

            # actual branch loss for each batch item
            clf_loss = F.nll_loss(F.log_softmax(br_out, dim=1), y, reduction='none')
            assert clf_loss.shape == (N,), clf_loss.shape
            clf_losses.append(clf_loss)

            # loss_estimation_loss averaged for each sample
            est_loss = self.calc_estimation_loss(i, base_out=base_out)
            lels.append(
                loss_estimation_loss(est_loss, clf_loss, reduction='mean')
            )

        # choose a branch randomly with probabilities of actual_loss
        a = np.hstack([i.detach().numpy().reshape(N, 1) for i in clf_losses])  # shape = (N, B)
        assert a.shape == (N, B), a.shape

        chosen_branches = []
        for i in range(N):  # loop over all batch items
            # normalize actual losses into probabilities
            # probabilities are proportional to 1/estimated_loss
            p = 1/a[i,:]
            # normalize
            p = p / p.sum()
            _br_idx = np.random.choice(B, p=p)
            chosen_branches.append(_br_idx)
        chosen_branches = np.array(chosen_branches)

        # compute the loss
        # the loss consists of sum{loss estimation losses} + mean{classification losses for corresponding branches}
        # , the first term consists of B elements, one for each branch
        # , the second term consists of N elements, one for each item; since they are
        # calculated from different branch outputs, multiple branches will be trained from one batch
        clf_loss = 0
        for i in range(B):
            item_ids = chosen_branches == i
            # get the classification loss for item ids at the chosen branch
            clf_loss += clf_losses[i][item_ids].sum()
        loss = sum(lels)/B + clf_loss / N
        # loss = clf_loss / N
        return loss, [l.detach() for l in lels], (clf_loss / N).detach()