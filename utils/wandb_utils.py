import numpy as np
import wandb
import plotly.graph_objs as go


def wandb_confusion_matrix(confmatrix=None, classnames=None, title=None):
    """
    Note: labels should be a list of strings for the corresponding class numbers
    Arguments:
        confmatrix: confusion matrix whose i,j entry denotes class i was predicted as j (like sklearn.confusion_matrix)
    """
    confmatrix = confmatrix.copy()  # copy to prevent modifications

    if classnames is not None:
        assert len(confmatrix) == len(classnames)

    # separate the diagonal from the matrix
    confmatrix = confmatrix.astype('float')
    confdiag = np.eye(len(confmatrix)) * confmatrix
    np.fill_diagonal(confmatrix, 0)

    n_confused = np.sum(confmatrix)
    n_right = np.sum(confdiag)
    confmatrix[confmatrix == 0] = np.nan
    confdiag[confdiag == 0] = np.nan
    confmatrix = go.Heatmap(
        {'coloraxis': 'coloraxis1', 'x': classnames, 'y': classnames[::-1], 'z': np.flipud(confmatrix),
         'hoverongaps': False,
         'hovertemplate': 'Predicted %{x}<br>Instead of %{y}<br>On %{z} examples<extra></extra>'})
    confdiag = go.Heatmap({'coloraxis': 'coloraxis2', 'x': classnames, 'y': classnames[::-1], 'z': np.flipud(confdiag),
                           'hoverongaps': False,
                           'hovertemplate': 'Predicted %{x} just right<br>On %{z} examples<extra></extra>'})

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

    xaxis = {'title': {'text': 'y_pred'}, 'showticklabels': True, 'side': 'top'}
    yaxis = {'title': {'text': 'y_true'}, 'showticklabels': True}

    fig.update_layout(title={'text': title, 'x': 0.5}, paper_bgcolor=transparent,
                      plot_bgcolor=transparent, xaxis=xaxis, yaxis=yaxis)

    return wandb.data_types.Plotly(fig)
