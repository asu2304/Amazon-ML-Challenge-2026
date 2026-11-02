"""Fast re-prediction of the test set with models saved by a full run_v2.py run (no training).

Use it when only test-side inputs change (normalization patches, candidate additions for test, decision
settings). Train-side OOF predictions, the models, the decision mode and tau all come from the full run's work dir.

    python -u src/run_test_only.py --model_dir work_v4 --out output_v4b --extra work_v3 --ce_from work_v2 \
        --ce2 work_v31/ce2.pt --france_patch work_v4 --gpus 0,3

Steps: test features (v1 cache + patches + bi-encoder pairs + context + extra) -> stage-1 fold models ->
cross-encoders on the top-5 (keyed caches; only unseen pairs are scored) -> stage-2 features -> stage-2 fold
models -> decision -> outputs.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v2 as R  # noqa: E402
from er import data, decide, features, model  # noqa: E402

T0 = time.time()


def log(m):
    print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)


R.log = log


def predict_folds(prefix, F):
    ps = []
    k = 0
    while (Path(f'{prefix}{k}.txt')).exists():
        m = lgb.Booster(model_file=f'{prefix}{k}.txt')
        ps.append(model.predict_chunked(m, F, m.feature_name()))
        k += 1
    assert ps, f'no models {prefix}*'
    log(f'  {prefix}: {k} fold models averaged')
    return np.mean(ps, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../dataset')
    ap.add_argument('--work', default='work')
    ap.add_argument('--model_dir', required=True, help='work dir of the full run (models, summary.json, caches)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--extra', default=None)
    ap.add_argument('--ce_from', default=None)
    ap.add_argument('--ce2', default=None)
    ap.add_argument('--france_patch', default=None)
    ap.add_argument('--gpus', default='0,3')
    ap.add_argument('--ce_topk', type=int, default=5)
    ap.add_argument('--jobs', type=int, default=40)
    args = ap.parse_args()
    W, M = Path(args.work), Path(args.model_dir)
    model.PARAMS.update(num_threads=args.jobs)
    summ = json.loads((M / 'summary.json').read_text())
    mode, tau = summ['mode'], summ['tau']
    log(f'decision from {M}/summary.json: mode={mode} tau={tau}')

    rec_te = data.load_split(args.data, 'test')
    norm_te = pd.read_parquet(W / 'test_norm.parquet')
    P_te = pd.read_parquet(W / 'test_pairs.parquet')
    F_te = pd.read_parquet(W / 'test_feats.parquet')
    if args.france_patch:
        R.apply_france_patch(Path(args.france_patch), norm_te, P_te, F_te)
    if args.extra:
        P_te, F_te = R.add_bi_pairs('test', Path(args.extra), W, M, rec_te, norm_te, P_te, F_te,
                                    Path(args.france_patch) / 'test_bi_feats_fr.parquet' if args.france_patch else None)
        features.add_context(F_te, P_te.q.to_numpy(), P_te.s.to_numpy(), R.CTX_COLS)
    q, s = P_te.q.to_numpy(), P_te.s.to_numpy()
    if 'cos_bi' in F_te:
        features.add_context(F_te, q, s, ['cos_bi'])
    R.extra_features(F_te, q, s, rec_te, norm_te)
    log(f'test features ready: {len(F_te):,} pairs x {F_te.shape[1]} columns')

    p1 = predict_folds(str(M / 'stage1_fold'), F_te)
    r, _, _, _ = features._group_rank_gap(q, p1)
    rows = np.flatnonzero(r < args.ce_topk)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus)
    gpus = args.gpus.split(',')
    src_dir = Path(args.ce_from) if args.ce_from else M
    n = np.int64(len(rec_te))
    caches = [(src_dir / 'ce_pairs_test.npy', src_dir / 'ce_test.npy')]
    if args.extra:
        caches.append((Path(args.extra) / 'ce_missing_pairs_test.npy', Path(args.extra) / 'ce_missing_test.npy'))
    caches.append((M / 'ce_missing_pairs_test.npy', M / 'ce_missing_test.npy'))
    ce = np.full(len(q), np.nan, np.float32)
    ce[rows] = R.keyed_scores(q[rows], s[rows], n, caches, M, 'ce_test',
                              R.gpu_score_fn(f'{src_dir}/tok_test', src_dir / 'ce.pt', gpus, env))
    if args.ce2:
        c2 = [(M / 'ce2_pairs_test.npy', M / 'ce2_test.npy')]
        if args.extra:
            c2.append((Path(args.extra) / 'ce2_pairs_test.npy', Path(args.extra) / 'ce2_test.npy'))
        ce2 = np.full(len(q), np.nan, np.float32)
        ce2[rows] = R.keyed_scores(q[rows], s[rows], n, c2, M, 'ce2_test',
                                   R.gpu_score_fn(f'{src_dir}/tok_test', args.ce2, gpus, env))
        R.add_ce_features(F_te, q, ce2, 'ce2')
    features.add_stage2(F_te, q, s, p1)
    R.add_ce_features(F_te, q, ce)
    sib = R.sibling_features(rows, q, s, q, p1, norm_te)
    for c in sib.columns:
        col = np.full(len(F_te), np.nan, np.float32)
        col[rows] = sib[c].to_numpy()
        F_te[c] = col

    p2 = predict_folds(str(M / 'stage2_fold'), F_te)
    np.save(M / f'p2_te_{Path(args.out).name}.npy', p2)
    pred_q, pred_s = decide.decide(q, s, p2, tau, mode)
    R.write_outputs(Path(args.out), rec_te, P_te, pred_q, pred_s)


if __name__ == '__main__':
    main()
