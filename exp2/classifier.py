from typing import List

import numpy as np
import torch
from torch import nn

from utils import np_a_in_b


class Classifier(nn.Module):
    def __init__(self, config, net, classes, id):
        super().__init__()
        self.net = net
        self.id = id
        self._cls_idx = {cls: i for i, cls in enumerate(classes)}
        self.config = config
        self.classes = list(classes)

    def forward(self, input):
        return self.net(input)

    def localize_labels(self, labels: torch.Tensor):
        """Convert labels to local labels in range 0, ..., n, where n-1 is the output units.
        'Other' category is mapped to n, the last output unit."""
        loc = []
        device = labels.device
        n = len(self.classes)
        for lbl in labels.tolist():
            loc.append(self._cls_idx.get(lbl, n))
        return torch.tensor(loc, device=device)

    def get_predictions(self, outputs: torch.Tensor, open=True) -> List[int]:
        """Get classifier predictions. The predictions contain actual class labels, not the local ones.
        If open, the 'other' category will be considered and will have label -1.
        """
        if outputs.size(0) == 0:
            return []

        # get local predictions
        if open or not self.config.other:
            loc = torch.argmax(outputs, 1)
        else:
            loc = torch.argmax(outputs[:, :-1], 1)  # skip the last unit

        r = []
        n_classes = len(self.classes)
        for ll in loc.tolist():
            if ll == n_classes:  # other category, add it as -1
                r.append(-1)
            else:
                r.append(self.classes[ll])
        return r

    def map_other(self, labels: np.ndarray, excl_idx=None):
        """Map labels of 'other' category to -1.

        Args:
            labels: the labels
            excl_idx (optional, np.ndarray): the indices where known labels reside, if provided, will
                help avoid unnecessary computation
        """
        result = np.empty_like(labels)
        result.fill(-1)
        if excl_idx is None:
            excl_idx = np_a_in_b(result, self.classes)
        result[excl_idx] = labels[excl_idx]
        return result