"""Exact challenge metric: macro F_0.5 over all S1 entities (singletons included)."""
import numpy as np
import pandas as pd

BETA2 = 0.25


def fbeta_macro(pred_q, pred_s, true_s1, rec, by_country=False, s1_mask=None):
    """pred_q/pred_s: predicted (S2/S3 record, S1 record) links as global indices.
    true_s1: array from data.load_truth. Scores every S1 record in rec (or those with s1_mask True)."""
    src = rec.src.to_numpy()
    s1 = np.flatnonzero((src == 1) if s1_mask is None else ((src == 1) & s1_mask))
    n = len(rec)
    q_true = np.flatnonzero((true_s1 >= 0) & (src != 1))
    n_true = np.bincount(true_s1[q_true], minlength=n)[s1]
    n_pred = np.bincount(pred_s, minlength=n)[s1]
    tp = np.bincount(pred_s[true_s1[pred_q] == pred_s], minlength=n)[s1]
    fn, fp = n_true - tp, n_pred - tp
    denom = (1 + BETA2) * tp + BETA2 * fn + fp
    f = np.where(denom > 0, (1 + BETA2) * tp / np.maximum(denom, 1e-9), 0.0)
    f = np.where((n_true == 0) & (n_pred == 0), 1.0, f)
    if not by_country:
        return f.mean()
    s = pd.Series(f).groupby(rec.country.to_numpy()[s1])
    return f.mean(), s.mean().to_dict()
