import numpy as np


class Predictor:
    name = None

    def method(self, classifiers, ctrl_outs, ctlr_preds, clf_preds_open, clf_preds_closed):
        pass

    def __call__(self, classifiers, ctrl_outs, ctlr_preds, clf_open_preds, clf_closed_preds):
        return self.method(classifiers, ctrl_outs, ctlr_preds, clf_open_preds, clf_closed_preds)


class ByCtrl(Predictor):
    name = 'pred_0'

    def method(self, classifiers, ctrl_outs, ctlr_preds, clf_preds_open, clf_preds_closed):
        preds = []
        for i in range(len(ctlr_preds)):
            clf_id = ctlr_preds[i]
            preds.append(clf_preds_closed[clf_id][i])
        return np.array(preds)


class FilteredController(Predictor):
    name = 'pred_1'

    def method(self, classifiers, ctrl_outs, ctlr_preds, clf_preds_open, clf_preds_closed):
        """First checks if only a single classifier is predicting non-other"""
        preds = []
        for i in range(len(ctlr_preds)):
            # choose one classifier among all those that have non-other output, the ones with higher controller score
            best_clf_id = None
            max_score = -float('inf')
            for clf_idx in range(len(classifiers)):
                if clf_preds_open[clf_idx][i] != -1:
                    if ctrl_outs[i][clf_idx] > max_score:
                        best_clf_id = clf_idx
            if best_clf_id is None:
                best_clf_id = ctlr_preds[i]
            preds.append(clf_preds_closed[best_clf_id][i])
        return np.array(preds)
