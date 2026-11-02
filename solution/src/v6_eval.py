"""v6 step 2: score a synthetic labelled dataset with the full v4 pipeline (saved models, no retraining).

normalize -> TF-IDF blocking -> bi-encoder retrieval -> features -> stage-1 models -> cross-encoders 1+2 on top-5
-> stage-2 models -> v4 decision -> macro F0.5 against the synthetic truth.

    python -u src/v6_eval.py --synth work_v6/synth_France --gpus 0,3
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v2 as R  # noqa: E402
from er import biencoder, blocking, crossenc, data, decide, features, metric, model, normalize  # noqa: E402
from run_test_only import predict_folds  # noqa: E402

T0 = time.time()


def log(m):
    print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)


R.log = log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synth', required=True)
    ap.add_argument('--gpus', default='0,3')
    ap.add_argument('--model_dir', default='work_v4')
    args = ap.parse_args()
    D, M = Path(args.synth), Path(args.model_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    gpus = args.gpus.split(',')
    env = dict(os.environ)
    model.PARAMS.update(num_threads=40)

    rec = data.load_split(str(D), 'test')
    gt = data.read_tsv(D / 'test_truth.tsv')
    pos = pd.Series(np.arange(len(rec)), index=rec.id.to_numpy())
    t = np.full(len(rec), -1, np.int64)
    ex = gt.assign(m=gt.matched_entity_ids.str.split(',')).explode('m')
    ex = ex[ex.m.notna() & (ex.m != '')]
    t[pos[ex.m].to_numpy()] = pos[ex.source1_entity_id].to_numpy()
    s1 = np.flatnonzero(rec.src.to_numpy() == 1)
    t[s1] = s1
    log(f'synthetic records {len(rec):,} (S1 {len(s1):,}); true links {len(ex):,}')

    maps = json.loads(Path('work/maps.json').read_text())
    part_map = {tuple(k.split('\t')): v for k, v in maps['part'].items()}
    fr_maps = json.loads((Path('work_v4') / 'france_maps.json').read_text())
    norm, nums = normalize.normalize_records(rec, maps['tok'], part_map, fr_maps=fr_maps)

    P = blocking.block(rec, norm, k=10, max_df=0.05, max_cand=15, n_threads=40, log=log)
    r_tf, _ = blocking.blocking_recall(P, rec, t)
    F = features.pair_features(rec, norm, nums, P, log=log)

    # bi-encoder retrieval
    tok = D / 'tok'
    if not (tok / 'off.npy').exists():
        tok.mkdir(exist_ok=True)
        flat, off = crossenc.tokenize_records(crossenc.record_texts(rec))
        np.save(tok / 'flat.npy', flat)
        np.save(tok / 'off.npy', off)
    E = biencoder.encode(str(tok), np.arange(len(rec)), 'work_v3/bi.pt', gpus)
    src, ctry = rec.src.to_numpy(), rec.country.to_numpy()
    Bs = []
    for c in sorted(set(ctry)):
        si, qi = np.flatnonzero((ctry == c) & (src == 1)), np.flatnonzero((ctry == c) & (src != 1))
        I, V = biencoder.knn(E[qi], E[si], k=10, gpu=0)
        Bs.append(pd.DataFrame({'q': np.repeat(qi, I.shape[1]), 's': si[I.ravel()], 'cos_bi': V.ravel()}))
    B = pd.concat(Bs, ignore_index=True)
    kP = P.q.to_numpy() * len(rec) + P.s.to_numpy()
    new = B[~np.isin(B.q.to_numpy() * len(rec) + B.s.to_numpy(), kP)].reset_index(drop=True)
    cos_bi_old = (E[P.q.to_numpy()].astype(np.float32) * E[P.s.to_numpy()].astype(np.float32)).sum(1)
    Pn = pd.concat([new[['q', 's']], blocking.pair_cosines(rec, norm, new.q.to_numpy(), new.s.to_numpy(), log=log)], axis=1)
    Fn = features.pair_features(rec, norm, nums, Pn, log=log)
    F['cos_bi'], F['bi_only'] = cos_bi_old, np.int8(0)
    Fn['cos_bi'], Fn['bi_only'] = new.cos_bi.to_numpy(), np.int8(1)
    P = pd.concat([P[['q', 's']], new[['q', 's']]], ignore_index=True)
    F = pd.concat([F, Fn[F.columns]], ignore_index=True)
    q, s = P.q.to_numpy(), P.s.to_numpy()
    r_all, _ = blocking.blocking_recall(P, rec, t)
    log(f'blocking recall: TF-IDF {r_tf:.4f} -> +bi-encoder {r_all:.4f}; pairs {len(P):,}')

    features.add_context(F, q, s, R.CTX_COLS + ['cos_bi'])
    R.extra_features(F, q, s, rec, norm)

    p1 = predict_folds(str(M / 'stage1_fold'), F)
    r, _, _, _ = features._group_rank_gap(q, p1)
    rows = np.flatnonzero(r < 5)
    n = np.int64(len(rec))
    ce = np.full(len(q), np.nan, np.float32)
    ce[rows] = R.keyed_scores(q[rows], s[rows], n, [], D, 'ce', R.gpu_score_fn(str(tok), 'work_v2/ce.pt', gpus, env))
    ce2 = np.full(len(q), np.nan, np.float32)
    ce2[rows] = R.keyed_scores(q[rows], s[rows], n, [], D, 'ce2', R.gpu_score_fn(str(tok), 'work_v31/ce2.pt', gpus, env))
    R.add_ce_features(F, q, ce2, 'ce2')
    features.add_stage2(F, q, s, p1)
    R.add_ce_features(F, q, ce)
    sib = R.sibling_features(rows, q, s, q, p1, norm)
    for c in sib.columns:
        col = np.full(len(F), np.nan, np.float32)
        col[rows] = sib[c].to_numpy()
        F[c] = col
    p2 = predict_folds(str(M / 'stage2_fold'), F)

    summ = json.loads((M / 'summary.json').read_text())
    for name, p in [('stage 1', p1), ('cross-encoder alone', np.nan_to_num(ce)), ('stage 2 (v4)', p2)]:
        a, b = decide.decide(q, s, p, summ['tau'], summ['mode'])
        f = metric.fbeta_macro(a, b, t, rec)
        log(f'{name:22s} F0.5 = {f:.4f}  (v4 decision mode={summ["mode"]} tau={summ["tau"]})')
    best = max((metric.fbeta_macro(*decide.decide(q, s, p2, tau, mode), t, rec), mode, tau)
               for mode in ['assign', 'expf'] for tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    log(f'stage 2 with the best threshold for THIS data: F0.5 = {best[0]:.4f} (mode={best[1]} tau={best[2]})')
    np.savez(D / 'eval_preds.npz', q=q, s=s, p1=p1, p2=p2, ce=ce, t=t)


if __name__ == '__main__':
    main()
