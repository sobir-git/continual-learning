import numpy as np
import wandb
from sklearn.metrics import confusion_matrix
import plotly.graph_objs as go


def wandb_confusion_matrix(y_true, y_pred, labels=None):
    """
    Note: labels should be a list of strings for the corresponding class numbers
    """
    confmatrix = confusion_matrix(y_pred, y_true)  # i,j refers to number of items in class i predicted to be class j
    confdiag = np.eye(len(confmatrix)) * confmatrix
    np.fill_diagonal(confmatrix, 0)

    confmatrix = confmatrix.astype('float')
    n_confused = np.sum(confmatrix)
    confmatrix[confmatrix == 0] = np.nan
    confmatrix = go.Heatmap({'coloraxis': 'coloraxis1', 'x': labels, 'y': labels, 'z': confmatrix,
                             'hoverongaps': False,
                             'hovertemplate': 'Predicted %{y}<br>Instead of %{x}<br>On %{z} examples<extra></extra>'})

    confdiag = confdiag.astype('float')
    n_right = np.sum(confdiag)
    confdiag[confdiag == 0] = np.nan
    confdiag = go.Heatmap({'coloraxis': 'coloraxis2', 'x': labels, 'y': labels, 'z': confdiag,
                           'hoverongaps': False,
                           'hovertemplate': 'Predicted %{y} just right<br>On %{z} examples<extra></extra>'})

    fig = go.Figure((confdiag, confmatrix))
    transparent = 'rgba(0, 0, 0, 0)'
    n_total = n_right + n_confused
    fig.update_layout({'coloraxis1': {
        'colorscale': [[0, transparent], [0, 'rgba(180, 0, 0, 0.05)'],
                       [1, f'rgba(180, 0, 0, {max(0.2, (n_confused / n_total) ** 0.5)})']],
        'showscale': False}}
    )
    fig.update_layout({'coloraxis2': {
        'colorscale': [[0, transparent], [0, f'rgba(0, 180, 0, {min(0.8, (n_right / n_total) ** 2)})'],
                       [1, 'rgba(0, 180, 0, 1)']], 'showscale': False}})

    xaxis = {'title': {'text': 'y_true'}, 'showticklabels': True}
    yaxis = {'title': {'text': 'y_pred'}, 'showticklabels': True}

    fig.update_layout(title={'text': 'Confusion matrix', 'x': 0.5}, paper_bgcolor=transparent,
                      plot_bgcolor=transparent, xaxis=xaxis, yaxis=yaxis)

    return {'confusion_matrix': wandb.data_types.Plotly(fig)}
