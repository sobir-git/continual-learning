import numpy as np


class Predictor:
    name = None

    def method(self, classifiers, ctrl_outs, ctlr_preds, clf_preds_open, clf_preds_closed):
        pass

    def __call__(self, classifiers, ctrl_outs, ctlr_preds, clf_open_preds, clf_closed_preds):
        return self.method(classifiers, ctrl_outs, ctlr_preds, clf_open_preds, clf_closed_preds)


class ByCtrl(Predictor):
    """
    First chooses a classifier using the controller. Then does a closed prediction with that classifier.
    """
    name = 'pred_0'

    def method(self, classifiers, ctrl_outs, ctlr_preds, clf_preds_open, clf_preds_closed):
        preds = []
        for i in range(len(ctlr_preds)):
            clf_id = ctlr_preds[i]
            preds.append(clf_preds_closed[clf_id][i])
        return np.array(preds)


class FilteredController(Predictor):
    """
    Choose one classifier among all those that have non-other output, and do prediction with it.
    If multiple choices(or none) for classifier, then choose the one with highest controller score.
    """
    name = 'pred_1'

    def method(self, classifiers, ctrl_outs, ctlr_preds, clf_preds_open, clf_preds_closed):
        """First checks if only a single classifier is predicting non-other"""
        preds = []
        # loop through the samples
        for i in range(len(ctlr_preds)):
            # choose a
            best_clf_id = None
            max_score = -float('inf')
            for clf_idx in range(len(classifiers)):
                # if classifier have predicted non-other, compare its controller score with the best one
                if clf_preds_open[clf_idx][i] != -1:
                    if ctrl_outs[i][clf_idx] > max_score:
                        best_clf_id = clf_idx
            if best_clf_id is None:
                # no classifier found with non-other prediction, choose one what controller have predicted
                best_clf_id = ctlr_preds[i]
            preds.append(clf_preds_closed[best_clf_id][i])
        return np.array(preds)
