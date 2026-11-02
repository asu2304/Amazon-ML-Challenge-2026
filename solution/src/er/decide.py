"""Turn pair probabilities into final links, optimizing F_0.5.

1. Assignment: every S2/S3 record keeps only its most probable S1 candidate (a record matches at most
   one S1), and only if p >= tau.
2. Per S1: sort its assigned records by p and keep the prefix with the highest *expected* F_0.5
   (k = 0, i.e. the empty list, is allowed -> protects singletons).
"""
import numpy as np
import pandas as pd

BETA2 = 0.25


def assign(q, s, p, tau):
    order = np.lexsort((-p, q))
    first = np.r_[True, q[order][1:] != q[order][:-1]]
    top = order[first]
    top = top[p[top] >= tau]
    return q[top], s[top], p[top]


def expected_f_prefix(q, s, p, n_extra=None):
    """For each S1 keep the prefix of its (desc-sorted) assigned records maximizing expected F_0.5.
    Expected counts: TP_k = sum_{i<=k} p_i, N_true = sum of all p_i of that S1 (+ optional n_extra)."""
    df = pd.DataFrame({'q': q, 's': s, 'p': p}).sort_values(['s', 'p'], ascending=[True, False])
    df['logq'] = np.log1p(-np.minimum(df.p.to_numpy(), 0.999999))
    g = df.groupby('s', sort=False)
    k = g.cumcount().to_numpy() + 1
    tp = g.p.cumsum().to_numpy()
    n_true = g.p.transform('sum').to_numpy()
    if n_extra is not None:
        n_true = n_true + n_extra
    p_empty = np.exp(g.logq.transform('sum').to_numpy())
    f_k = (1 + BETA2) * tp / ((1 + BETA2) * tp + BETA2 * (n_true - tp) + (k - tp))
    df['score'] = (1 - p_empty) * f_k
    best = df.groupby('s', sort=False).score.transform('max').to_numpy()
    df['kb'] = np.where(df.score.to_numpy() >= best, k, 1 << 30)
    best_k = df.groupby('s', sort=False).kb.transform('min').to_numpy()
    keep = (k <= best_k) & (best > p_empty)
    out = df[keep]
    return out.q.to_numpy(), out.s.to_numpy()


def decide(q, s, p, tau, mode='expf'):
    aq, as_, ap = assign(q, s, p, tau)
    if mode == 'assign':
        return aq, as_
    return expected_f_prefix(aq, as_, ap)
