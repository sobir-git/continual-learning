import numpy as np
from scipy.sparse import coo_matrix
from sklearn.utils import check_consistent_length
from sklearn.utils.multiclass import unique_labels


def rectangular_confusion_matrix(y_true, y_pred, true_labels=None, pred_labels=None, sample_weight=None,
                                 normalize=None):
    labels = {'pred': pred_labels, 'true': true_labels}
    y = {'true': y_true, 'pred': y_pred}

    for g in ['pred', 'true']:
        if labels[g] is None:
            labels[g] = unique_labels(y[g])
        else:
            labels[g] = np.asarray(labels[g])
            if labels[g].size == 0:
                raise ValueError(f"'{g}_labels' should contains at least one label.")
            if y[g].size > 0 and np.all([l not in y[g] for l in labels[g]]):
                raise ValueError(f"At least one label specified must be in y_{g}")

    if y['true'].size == 0:
        return np.zeros((labels['true'].size, labels['pred'].size), dtype=int)

    if sample_weight is None:
        sample_weight = np.ones(y['true'].shape[0], dtype=np.int64)
    else:
        sample_weight = np.asarray(sample_weight)

    check_consistent_length(y['true'], y['pred'], sample_weight)

    if normalize not in ['true', 'pred', 'all', None]:
        raise ValueError("normalize must be one of {'true', 'pred', "
                         "'all', None}")

    labels_to_ind = {}
    for g in ('pred', 'true'):
        labels_to_ind[g] = {y: x for x, y in enumerate(labels[g])}

    # convert yt, yp into index
    for g in ('pred', 'true'):
        y[g] = np.array([labels_to_ind[g].get(x, labels[g].size + 1) for x in y[g]])

    # intersect y_pred, y_true with labels, eliminate items not in labels
    ind = np.logical_and(y['pred'] < labels['pred'].size, y['true'] < labels['true'].size)
    for g in ('pred', 'true'):
        y[g] = y[g][ind]
    # also eliminate weights of eliminated items
    sample_weight = sample_weight[ind]

    # Choose the accumulator dtype to always have high precision
    if sample_weight.dtype.kind in {'i', 'u', 'b'}:
        dtype = np.int64
    else:
        dtype = np.float64

    cm = coo_matrix((sample_weight, (y['true'], y['pred'])),
                    shape=(labels['true'].size, labels['pred'].size), dtype=dtype,
                    ).toarray()

    with np.errstate(all='ignore'):
        if normalize == 'true':
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == 'pred':
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif normalize == 'all':
            cm = cm / cm.sum()
        cm = np.nan_to_num(cm)

    return cm
