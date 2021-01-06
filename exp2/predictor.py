from abc import abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from exp2.classifier import Classifier


@dataclass
class Givens:
    """Givens for a predictor to use and do predictions"""
    classifiers: List[Classifier]
    ctrl_outputs: torch.Tensor
    ctrl_predictions: np.ndarray
    clf_preds_open: List[np.ndarray]
    clf_preds_closed: List[np.ndarray]


class Predictor:
    name: str

    @abstractmethod
    def predict(self, givens: Givens) -> np.ndarray:
        pass

    def __call__(self, givens):
        return self.predict(givens)


class ByCtrl(Predictor):
    """
    First chooses a classifier using the controller. Then does a closed prediction with that classifier.
    """
    name = 'pred_0'

    def predict(self, givens: Givens):
        ctrl_preds = givens.ctrl_predictions
        clf_preds_closed = givens.clf_preds_closed
        preds = []
        for i in range(len(ctrl_preds)):
            clf_idx = ctrl_preds[i]
            preds.append(clf_preds_closed[clf_idx][i])
        return np.array(preds)


class FilteredController(Predictor):
    """
    Choose one classifier among all those that have non-other output, and do prediction with it.
    If multiple choices(or none) for classifier, then choose the one with highest controller score.
    """
    name = 'pred_1'

    def predict(self, givens: Givens):
        """First checks if only a single classifier is predicting non-other"""
        ctrl_preds = givens.ctrl_predictions
        clf_preds_closed = givens.clf_preds_closed
        clf_preds_open = givens.clf_preds_open
        classifiers = givens.classifiers
        ctrl_outs = givens.ctrl_outputs

        other_label = classifiers[0].other_label
        predictions = []
        # for each sample
        for i in range(len(ctrl_preds)):
            # filter all classifiers that predicted non-other
            filtered = [j for j in range(len(classifiers)) if clf_preds_open[j][i] != other_label]

            # if none is left, include all back
            if len(filtered) == 0:
                filtered = list(range(len(classifiers)))

            # choose one with the highest controller score
            chosen = torch.argmax(ctrl_outs[i, filtered]).item()
            prediction = clf_preds_closed[chosen][i]
            predictions.append(prediction)
        return np.array(predictions)
