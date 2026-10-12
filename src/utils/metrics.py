"""Metrics in original sales units; percentages are explicitly labeled."""
import numpy as np
import torch


def _arrays(y_true, y_pred):
    values = []
    for value in [y_true, y_pred]:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        values.append(np.asarray(value, dtype=np.float64))
    true, pred = values
    if true.shape != pred.shape or true.size == 0:
        raise ValueError('Targets and predictions must have the same nonempty shape')
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError('Targets/predictions contain NaN or Inf')
    if (true < 0).any():
        raise ValueError('Retail sales targets must be nonnegative')
    return true, pred


def calculate_mae(y_true, y_pred):
    true, pred = _arrays(y_true, y_pred)
    return float(np.abs(true-pred).mean())


def calculate_rmse(y_true, y_pred):
    true, pred = _arrays(y_true, y_pred)
    return float(np.sqrt(np.square(true-pred).mean()))


def calculate_wmape(y_true, y_pred):
    """Return a ratio; undefined if total actual sales are zero."""
    true, pred = _arrays(y_true, y_pred)
    return float(np.abs(true-pred).sum()/true.sum()) if true.sum() > 0 else float('nan')


def calculate_smape(y_true, y_pred):
    true, pred = _arrays(y_true, y_pred)
    denom = np.abs(true) + np.abs(pred)
    terms = np.divide(2*np.abs(true-pred), denom, out=np.zeros_like(denom), where=denom > 0)
    return float(100*terms.mean())


def evaluate_all_metrics(y_true, y_pred):
    return {'MAE': calculate_mae(y_true, y_pred), 'RMSE': calculate_rmse(y_true, y_pred),
            'wMAPE (%)': 100*calculate_wmape(y_true, y_pred), 'sMAPE (%)': calculate_smape(y_true, y_pred)}


def metric_table(predictions, group_columns):
    import pandas as pd
    rows = []
    groups = predictions.groupby(group_columns, dropna=False) if group_columns else [((), predictions)]
    for key, frame in groups:
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, key))
        row.update(evaluate_all_metrics(frame.actual.to_numpy(), frame.prediction.to_numpy()))
        row['n_predictions'] = len(frame)
        rows.append(row)
    return pd.DataFrame(rows)
