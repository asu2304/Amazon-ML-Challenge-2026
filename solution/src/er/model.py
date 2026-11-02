"""LightGBM pair classifier with out-of-fold predictions grouped by true cluster."""
import lightgbm as lgb
import numpy as np

PARAMS = dict(
    objective='binary', learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=255, num_threads=40, force_col_wise=True, verbose=-1, seed=0,
)


def query_folds(q, true_s1, n_folds=2, seed=0):
    """Fold per candidate row, grouping every record with the other records of its true cluster
    (unmatched records are their own group) so no cluster is split across folds."""
    grp = np.where(true_s1[q] >= 0, true_s1[q], q + len(true_s1))
    return ((grp * 2654435761 + seed) % 2**32 % n_folds).astype(np.int8)


def fit(X, y, Xv, yv, rounds=800, log=print):
    dtr = lgb.Dataset(X, y, free_raw_data=True)
    dva = lgb.Dataset(Xv, yv, reference=dtr)
    m = lgb.train(PARAMS, dtr, num_boost_round=rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
    log(f'    best_iter={m.best_iteration} valid_logloss={m.best_score["valid_0"]["binary_logloss"]:.5f}')
    return m


def predict_chunked(m, X, cols, chunk=4_000_000):
    """Predict in row chunks as float32 -> memory stays ~chunk x cols x 4 bytes instead of a float64 copy of X."""
    out = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), chunk):
        out[i:i + chunk] = m.predict(X.iloc[i:i + chunk][cols].to_numpy(np.float32), num_threads=PARAMS['num_threads'])
    return out


def oof_multi(F, y, folds, fit_mask, extra_sets, cols, valid_rows=2_000_000, seed=0, rounds=1500, log=print):
    """OOF predictions for rows of F (fold k predicted by the model fit on the other folds, using only
    rows with fit_mask), plus fold-averaged predictions for each frame in extra_sets. Returns (oof, [preds], models)."""
    rng = np.random.default_rng(seed)
    oof = np.zeros(len(F), dtype=np.float32)
    extra = [np.zeros(len(E), dtype=np.float32) for E in extra_sets]
    n_folds = int(folds.max()) + 1
    models = []
    for k in range(n_folds):
        tr = np.flatnonzero((folds != k) & fit_mask)
        va = np.flatnonzero(folds == k)
        va_es = rng.choice(va, size=min(valid_rows, len(va)), replace=False)
        log(f'  fold {k}: fit rows={len(tr):,} pos={y[tr].mean():.4f}')
        m = fit(F.iloc[tr][cols], y[tr], F.iloc[va_es][cols], y[va_es], rounds=rounds, log=log)
        oof[va] = predict_chunked(m, F.iloc[va], cols)
        for e, E in zip(extra, extra_sets):
            e += predict_chunked(m, E, cols) / n_folds
        imp = sorted(zip(m.feature_importance('gain'), cols), reverse=True)[:20]
        log('    top features: ' + ', '.join(c for _, c in imp))
        models.append(m)
    return oof, extra, models


def oof_and_test(F_tr, y, folds, F_te, q_tr, fit_frac=0.5, valid_rows=2_000_000, seed=0, log=print):
    """Out-of-fold probabilities for train rows and fold-averaged probabilities for test rows."""
    rng = np.random.default_rng(seed)
    cols = list(F_tr.columns)
    oof = np.zeros(len(F_tr), dtype=np.float32)
    pte = np.zeros(len(F_te), dtype=np.float32)
    uq = np.unique(q_tr)
    fit_q = np.zeros(q_tr.max() + 1, dtype=bool)
    fit_q[rng.choice(uq, size=int(len(uq) * fit_frac), replace=False)] = True
    n_folds = int(folds.max()) + 1
    for k in range(n_folds):
        tr = np.flatnonzero((folds != k) & fit_q[q_tr])
        va = np.flatnonzero(folds == k)
        va_es = rng.choice(va, size=min(valid_rows, len(va)), replace=False)
        log(f'  fold {k}: fit rows={len(tr):,} pos={y[tr].mean():.3f}')
        m = fit(F_tr.iloc[tr][cols], y[tr], F_tr.iloc[va_es][cols], y[va_es], log=log)
        oof[va] = m.predict(F_tr.iloc[va][cols], num_threads=PARAMS['num_threads'])
        pte += m.predict(F_te[cols], num_threads=PARAMS['num_threads']) / n_folds
        if k == 0:
            imp = sorted(zip(m.feature_importance('gain'), cols), reverse=True)[:15]
            log('    top features: ' + ', '.join(f'{c}' for _, c in imp))
    return oof, pte
