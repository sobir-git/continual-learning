import collections

import torch
import wandb
import numpy as np

from utils import wandb_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


def plotly_heatmap(data, rows=None, columns=None):
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=columns,
        y=rows,
        hoverongaps=False,
        colorscale='greens'))
    fig.update_xaxes(side='top')
    fig.update_yaxes(autorange='reversed')
    return fig


def traverse_dict(d, skip_fn=lambda d: False, pref=None):
    """
    Traverses nested dictionary returning nested keys and value pairs;
    """
    if skip_fn(d):
        return
    if pref is None:
        pref = []
    for key, val in d.items():
        key = pref + [key]
        if isinstance(val, dict):
            yield from traverse_dict(val, skip_fn=skip_fn, pref=key)
        else:
            yield key, val


def dict_deep_update(d, u):
    """
    Deep version of d1.update(d2).
    Args:
        d: Dictionary that needs updating
        u: Dictionary used to update

    Returns:
        First dictionary, same instance.

    Examples:
        >>> d1 = {'net1': {'pretrain': {'loss': 0.25}}}
        >>> d2 = {'net1': {'pretrain': {'accuracy': 0.87}}}
        >>> dict_deep_update(d, u)
        >>> assert d == {'net1': {'pretrain': {'loss': 0.25, 'accuracy': 0.87}}}

    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d



class Logger:
    _image_heatmaps = True
    epoch = None

    def __init__(self, opt, console_logger, pref=None):
        self.opt = opt
        self.prefs = [pref] if pref else []
        self.console = console_logger
        self._data = dict()

    def set_epoch(self, epoch):
        """This affects everything afterwards logged with epoch."""
        self.epoch = epoch

    def remove_epoch(self):
        self.epoch = None

    @property
    def pref_str(self):
        return '.'.join(self.prefs)

    def _make_consolable(self, val):
        if isinstance(val, torch.Tensor):
            try:
                val = val.item()
            except Exception:
                return None
        if type(val) not in (int, float, str, tuple, list):
            if 'float' not in str(type(val)):
                return None
        if type(val) in (float,):
            str_val = f'{val:.3f}'
        else:
            str_val = str(val)
        if str_val[0] + str_val[-1] == '<>':
            return None
        if len(str_val) > 16:
            return None
        return str_val

    def _console_commit(self):
        # traverse d and log everything that is text
        msg = f'[{self.pref_str}'
        if self.epoch is not None:
            msg += ' ' + str(self.epoch)
        msg += ']\t'

        skip_function = lambda d: len(set(d.keys()).intersection(('_type', 'path'))) > 1

        for keys, val in traverse_dict(self._data, skip_fn=skip_function):
            # remove prefs from keys
            keys = keys[len(self.prefs):]
            key = '.'.join(keys)
            str_val = self._make_consolable(val)
            if str_val is not None:
                msg += f'{key}: {str_val},\t'
        self.console.info(msg)
        return msg

    def push_pref(self, pref):
        self.prefs.append(pref)

    def pop_pref(self):
        self.prefs.pop()

    def commit(self):
        if not self._data:
            return # nothing to commit
        w_data = self._data
        if self.epoch is not None:
            w_data = {'epoch': self.epoch, **w_data}
        wandb.log(w_data, commit=True)
        self._console_commit()
        self._data.clear()

    def _commit_if(self, commit):
        if commit:
            self.commit()

    def log_heatmap(self, name, data, rows=None, columns=None, title=None, vmax=None, vmin=None):
        if self._image_heatmaps:
            fig, ax = plt.subplots()
            ax.set_title(title)
            sns.heatmap(data, xticklabels=columns or 'auto', yticklabels=rows or 'auto', annot=True, vmax=vmax,
                        linewidths=2,
                        cmap=sns.color_palette("light:g", as_cmap=True), vmin=vmin)
            self.log({name: wandb.Image(fig)})
            plt.close(fig)
        else:
            # Warning: this option is currently buggy because of wandb or plotly
            if vmax is not None or vmin is not None:
                raise NotImplementedError("vmin and vmax are currently not implemented when using with plotly")
            fig = plotly_heatmap(data, rows, columns)
            self.log({name: fig})

    def log_confusion_matrix(self, confmatrix, classnames, title='Confusion Matrix', name='conf_mtx', commit=False):
        # ==== log confusion matrix
        self.log({name: wandb_confusion_matrix(confmatrix, classnames, title=title)}, commit=commit)

    def _log_accuracy_one(self, confmatrix, classnames):
        diag = np.diag(confmatrix)
        accuracy = diag.sum() / confmatrix.sum()
        recalls = diag / confmatrix.sum(1)
        self.log({'accuracy': accuracy})
        self.log_heatmap('recalls', data=recalls.reshape(-1, 1), rows=classnames, title='Recalls', vmin=0, vmax=1)
        return accuracy

    def _log_accuracy_many(self, confmatrix, classnames):
        columns = sorted(list(confmatrix.keys()))
        recall_data = np.zeros((len(classnames), len(columns)))
        accuracies = {}
        for i, c in enumerate(columns):
            cm = confmatrix[c]
            diag = np.diag(cm)
            recall = diag / cm.sum(1)  # recall means e.g. how many of the dogs you recognized
            recall_data[:, i] = recall
            accuracies[c] = diag.sum() / cm.sum()
        # report all accuracies under the namespace 'accuracies'
        self.log({'accuracies': accuracies})
        self.log_heatmap('recalls', recall_data, rows=classnames, columns=columns, title='Recalls', vmin=0, vmax=1)
        return accuracies

    def log_accuracies(self, confmatrix, classnames, commit=False):
        """
        Reports accuracy(s) and recall(s).
        Args:
            confmatrix: confusion matrix, can be a dict of confusion matrices for each branch
            classnames:
            commit:

        Returns:
            accuracy(s)
        """
        if isinstance(confmatrix, dict):
            acc = self._log_accuracy_many(confmatrix=confmatrix, classnames=classnames)
        else:
            acc = self._log_accuracy_one(confmatrix=confmatrix, classnames=classnames)
        self._commit_if(commit)
        return acc

    def log(self, d, commit=False):
        if self.prefs:
            for pref in self.prefs[::-1]:
                d = {pref: d}
        dict_deep_update(self._data, d)
        self._commit_if(commit)
