"""v2 pipeline on top of the cached v1 stages (normalization, candidate pairs, base features).

What v2 adds
  * Test-like validation: test has ~2x more (one-off) decoys per S1 than train with the same cluster
    sizes. Train clusters are split into two universes, each with half the clusters plus ALL decoys
    (~41% decoys, like test). Training, OOF and threshold tuning all happen there, with context
    features recomputed per universe.
  * Cluster split of train: part A (25%) trains the cross-encoder, part B (75%) is the GBDT's OOF set,
    so the cross-encoder score used as a feature is never computed on its own training pairs.
  * Extra stage-1 features (street-only similarity, first-token agreement, IDF-style name rarity).
  * Cross-encoder (multilingual-e5-base, MIT) on the top-5 stage-1 candidates of every record.
  * Stage 2: stage-1 listwise features + cross-encoder score/rank/gap + sibling agreement
    (how similar this record is to other records confidently assigned to the same S1).
  * Optional France pseudo-labels for a final stage-2 refit.

    python -u src/run_v2.py --data ../dataset --work work --work2 work_v2 --out output_v2
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).resolve().parent))
from er import blocking, crossenc, data, decide, features, metric, model  # noqa: E402

T0 = time.time()


def log(msg):
    print(f'[{time.time() - T0:7.0f}s] {msg}', flush=True)


def uhash(x, salt):
    """splitmix64 hash of integer ids -> uniform [0, 1); different salts give independent splits."""
    with np.errstate(over='ignore'):
        z = np.asarray(x, dtype=np.uint64) + np.uint64(salt) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / 2.0**53


def cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.uint8)


# ------------------------------------------------------------------ features

CTX_COLS = ['cos_name', 'cos_addr', 'cos_combo', 'nm_tset', 'ad_tset']
DROP_COLS = {'n_cand_q', 'n_cand_s'}


def extra_features(F, q, s, rec, norm, chunk=10_000_000):
    addr = norm.addr_norm.str.replace(r'\b\d+\b', ' ', regex=True).str.split().str.join(' ').to_numpy(object)
    first = norm.name_core.str.split(n=1).str[0].fillna('')
    fid = pd.factorize(first)[0]
    core = norm.name_core.to_numpy(object)
    street, prat = np.empty(len(q), np.uint8), np.empty(len(q), np.uint8)
    for i in range(0, len(q), chunk):
        a, b = q[i:i + chunk], s[i:i + chunk]
        street[i:i + chunk] = cp(addr[a].tolist(), addr[b].tolist(), fuzz.token_set_ratio)
        prat[i:i + chunk] = cp(core[a].tolist(), core[b].tolist(), fuzz.partial_token_sort_ratio)
    F['street_tset'] = street
    F['nm_ptsort'] = prat
    F['first_tok_eq'] = (fid[q] == fid[s]).astype(np.int8)
    # rarity of the record's first name token among S1 records of the country (IDF-like)
    key = rec.country.to_numpy() + '|' + first.to_numpy()
    s1 = rec.src.to_numpy() == 1
    cnt = pd.Series(key[s1]).value_counts()
    F['first_tok_s1_cnt'] = cnt.reindex(key[s]).fillna(0).to_numpy().astype(np.int32)


def sibling_features(rows, gq, gs, q, p, norm, chunk=20_000_000):
    """For candidate rows `rows`: agreement of record q with the other records r that confidently picked
    the same S1 (their best candidate, p > 0.5). gq/gs: group ids (universe-aware); q: original record ids."""
    rank, _, _, _ = features._group_rank_gap(gq, p)
    conf = np.flatnonzero((rank == 0) & (p > 0.5))
    sib = pd.DataFrame({'s': gs[conf], 'r': q[conf]})
    T = pd.DataFrame({'row': rows, 'q': q[rows], 's': gs[rows]})
    M = T.merge(sib, on='s')
    M = M[M.r.to_numpy() != M.q.to_numpy()]
    core, addr = norm.name_core.to_numpy(object), norm.addr_norm.to_numpy(object)
    nm, ad = np.empty(len(M), np.uint8), np.empty(len(M), np.uint8)
    a, b = M.q.to_numpy(), M.r.to_numpy()
    for i in range(0, len(M), chunk):
        nm[i:i + chunk] = cp(core[a[i:i + chunk]].tolist(), core[b[i:i + chunk]].tolist(), fuzz.token_set_ratio)
        ad[i:i + chunk] = cp(addr[a[i:i + chunk]].tolist(), addr[b[i:i + chunk]].tolist(), fuzz.token_set_ratio)
    M = M.assign(nm=nm, ad=ad, both=np.minimum(nm, ad))
    g = M.groupby('row')
    out = pd.DataFrame(index=rows)
    out['sib_cnt'] = g.size().reindex(rows).fillna(0).to_numpy()
    for c in ['nm', 'ad', 'both']:
        out[f'sib_{c}_max'] = g[c].max().reindex(rows).to_numpy()
        out[f'sib_{c}_mean'] = g[c].mean().reindex(rows).to_numpy()
    return out.astype(np.float32)


def patch_rows(F, patch, rows):
    """Overwrite rows of F with the same columns of patch (dtype preserved)."""
    for c in F.columns:
        if c in patch:
            a = F[c].to_numpy(copy=True)
            a[rows] = patch[c].to_numpy().astype(a.dtype)
            F[c] = a


def apply_france_patch(Wp, norm_te, P_te, F_te):
    """v4: France test records re-normalized with French rules; their TF-IDF-pair features recomputed."""
    nf = pd.read_parquet(Wp / 'test_norm_fr.parquet')
    idx = nf.pop('idx').to_numpy()
    patch_rows(norm_te, nf, idx)
    Ff = pd.read_parquet(Wp / 'test_feats_fr.parquet')
    rows = Ff.pop('row').to_numpy()
    assert (Ff.columns.isin(F_te.columns)).all() and len(Ff) == len(rows)
    patch_rows(F_te, Ff, rows)
    patch_rows(P_te, Ff[['cos_name', 'cos_addr', 'cos_combo']], rows)
    log(f'France patch: {len(idx):,} records re-normalized, {len(rows):,} TF-IDF pair rows recomputed')


def add_bi_pairs(split, W3, W, W2, rec, norm, P, F, patch=None):
    """v3: append the bi-encoder candidates TF-IDF missed (with all base features) and add cos_bi to every pair."""
    path = W2 / f'{split}_bi_pairs_feats.parquet'
    if not path.exists() and (W3 / f'{split}_bi_pairs_feats.parquet').exists():
        path = W3 / f'{split}_bi_pairs_feats.parquet'           # reuse v3's cached features
    new = pd.read_parquet(W3 / f'{split}_bi_new_pairs.parquet')
    if path.exists():
        Fn = pd.read_parquet(path)
    else:
        log(f'{split}: features for {len(new):,} bi-encoder pairs')
        nums = np.load(W / f'{split}_nums.npy')
        Pn = pd.concat([new[['q', 's']], blocking.pair_cosines(rec, norm, new.q.to_numpy(), new.s.to_numpy(), log=log)], axis=1)
        Fn = features.pair_features(rec, norm, nums, Pn, log=log)
        Fn.to_parquet(path, index=False)
    if patch is not None:
        Fp = pd.read_parquet(patch)
        rows = Fp.pop('row').to_numpy()
        patch_rows(Fn, Fp, rows)
        log(f'France patch: {len(rows):,} bi-encoder pair rows recomputed')
    Pn = pd.concat([new[['q', 's']], Fn[['cos_name', 'cos_addr', 'cos_combo']]], axis=1)
    F['cos_bi'] = pd.read_parquet(W3 / f'{split}_tfidf_pairs_cos_bi.parquet').cos_bi.to_numpy()
    F['bi_only'] = np.int8(0)
    Fn['cos_bi'] = new.cos_bi.to_numpy()
    Fn['bi_only'] = np.int8(1)
    return (pd.concat([P[['q', 's', 'cos_name', 'cos_addr', 'cos_combo']], Pn], ignore_index=True),
            pd.concat([F, Fn[F.columns]], ignore_index=True))


def add_ce_features(F, q, ce, name='ce'):
    """ce: probability for scored rows, NaN elsewhere."""
    F[name] = ce
    v = np.where(np.isnan(ce), -1.0, ce).astype(np.float32)
    r, g, b, _ = features._group_rank_gap(q, v)
    F[f'{name}_rank_q'] = np.where(np.isnan(ce), np.nan, r).astype(np.float32)
    F[f'{name}_gap_q'] = np.where(np.isnan(ce), np.nan, g).astype(np.float32)
    F[f'{name}_max_q'] = b.astype(np.float32)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../dataset')
    ap.add_argument('--work', default='work')
    ap.add_argument('--work2', default='work_v2')
    ap.add_argument('--out', default='output_v2')
    ap.add_argument('--a_frac', type=float, default=0.25)
    ap.add_argument('--fit_frac', type=float, default=0.4)
    ap.add_argument('--ce_topk', type=int, default=5)
    ap.add_argument('--ce_train_pairs', type=int, default=6_000_000)
    ap.add_argument('--gpus', default='0,1,2,3')
    ap.add_argument('--jobs', type=int, default=40)
    ap.add_argument('--pseudo_france', action='store_true')
    ap.add_argument('--ce_test', default=None, help='override test cross-encoder scores (e.g. France-adapted)')
    ap.add_argument('--extra', default=None, help='v3: dir with bi-encoder candidate pairs to add (work_v3)')
    ap.add_argument('--ce_from', default=None, help='reuse cross-encoder checkpoint, tokens and cached scores from this dir')
    ap.add_argument('--ce2', default=None, help='v3.1: second cross-encoder checkpoint; its scores become extra features')
    ap.add_argument('--per_country_tau', action='store_true', help='tune decision threshold/mode per country')
    ap.add_argument('--france_patch', default=None, help='v4: dir with re-normalized France test rows/features')
    args = ap.parse_args()
    W, W2 = Path(args.work), Path(args.work2)
    W2.mkdir(parents=True, exist_ok=True)
    model.PARAMS.update(num_threads=args.jobs, learning_rate=0.05, num_leaves=255, min_data_in_leaf=100)

    # ---------------- load v1 caches
    rec_tr, rec_te = data.load_split(args.data, 'train'), data.load_split(args.data, 'test')
    t = data.load_truth(args.data, rec_tr)
    norm_tr, norm_te = pd.read_parquet(W / 'train_norm.parquet'), pd.read_parquet(W / 'test_norm.parquet')
    P_tr, P_te = pd.read_parquet(W / 'train_pairs.parquet'), pd.read_parquet(W / 'test_pairs.parquet')
    F_tr, F_te = pd.read_parquet(W / 'train_feats.parquet'), pd.read_parquet(W / 'test_feats.parquet')
    log(f'loaded: train pairs {len(P_tr):,}, test pairs {len(P_te):,}')
    if args.france_patch:
        apply_france_patch(Path(args.france_patch), norm_te, P_te, F_te)
    if args.extra:
        P_tr, F_tr = add_bi_pairs('train', Path(args.extra), W, W2, rec_tr, norm_tr, P_tr, F_tr)
        P_te, F_te = add_bi_pairs('test', Path(args.extra), W, W2, rec_te, norm_te, P_te, F_te,
                                  Path(args.france_patch) / 'test_bi_feats_fr.parquet' if args.france_patch else None)
        features.add_context(F_te, P_te.q.to_numpy(), P_te.s.to_numpy(), CTX_COLS)
        log(f'with bi-encoder candidates: train pairs {len(P_tr):,}, test pairs {len(P_te):,}')

    # ---------------- test-like train universes + cluster split
    # Train clusters (S1 + its true matches) are split into two halves. Universe u holds the clusters of
    # half u plus ALL unmatched records -> ~2x decoys per S1, same cluster sizes, like test.
    # Record i of universe u gets group index i + u*N, so every per-record / per-S1 operation stays inside
    # its universe; strings are still looked up by the original index.
    N = len(rec_tr)
    src = rec_tr.src.to_numpy()
    cl = np.where(t >= 0, t, np.arange(N) + N)                      # original cluster id per record
    half = (uhash(cl, 23) < 0.5).astype(np.int64)                   # universe of each cluster
    q0, s0 = P_tr.q.to_numpy(), P_tr.s.to_numpy()
    keep = (t[q0] < 0) | (half[q0] == half[s0])
    P_tr, F_tr = P_tr[keep].reset_index(drop=True), F_tr[keep].reset_index(drop=True)
    q_tr, s_tr = P_tr.q.to_numpy(), P_tr.s.to_numpy()
    u = half[s_tr]
    gq, gs = q_tr + u * N, s_tr + u * N
    t2 = np.full(2 * N, -1, np.int64)                              # truth in the doubled index space
    for k in (0, 1):
        m = (t >= 0) & (half == k)
        t2[np.flatnonzero(m) + k * N] = t[m] + k * N
    rec2 = pd.DataFrame({'src': np.concatenate([src, src]),
                         'country': np.concatenate([rec_tr.country.to_numpy()] * 2)})
    part_a = uhash(cl, 11) < args.a_frac                            # per original record
    y = (t[q_tr] == s_tr).astype(np.int8)
    rowA, rowB = part_a[q_tr], ~part_a[q_tr]
    folds = (uhash(cl[q_tr], 5) < 0.5).astype(np.int8)
    s1_eval = np.zeros(2 * N, bool)                                 # S1 of universe u, part B only
    for k in (0, 1):
        s1_eval[np.flatnonzero((src == 1) & (half == k) & ~part_a) + k * N] = True
    in_u = np.zeros(2 * N, bool); in_u[gq] = True
    log(f'universes: pairs {len(P_tr):,}; decoy share in universe records '
        f'{(t2[in_u] < 0).mean():.3f}; S1 per universe {((src == 1) & (half == 0)).sum():,} / {((src == 1) & (half == 1)).sum():,}; '
        f'rows A={rowA.sum():,} B={rowB.sum():,}')
    q_te, s_te = P_te.q.to_numpy(), P_te.s.to_numpy()

    # ---------------- stage-1 features
    log('recomputing context features inside each universe')
    features.add_context(F_tr, gq, gs, CTX_COLS + (['cos_bi'] if 'cos_bi' in F_tr else []))
    if 'cos_bi' in F_te:
        features.add_context(F_te, q_te, s_te, ['cos_bi'])
    key = rec_tr.country.to_numpy() + '|' + norm_tr.name_core.to_numpy()
    for k in (0, 1):
        cnt = pd.Series(key[(src == 1) & (half == k)]).value_counts()
        m = u == k
        F_tr.loc[m, 's_name_cnt'] = cnt.reindex(key[s_tr[m]]).fillna(0).to_numpy().astype(np.int32)
        F_tr.loc[m, 'q_name_cnt'] = cnt.reindex(key[q_tr[m]]).fillna(0).to_numpy().astype(np.int32)
    log('extra features')
    extra_features(F_tr, q_tr, s_tr, rec_tr, norm_tr)
    extra_features(F_te, q_te, s_te, rec_te, norm_te)
    # candidate-count features are distorted in the half-S1 universes (each record keeps ~half its
    # candidates) -> excluded; ranks / gaps / best-other values are exact there
    cols1 = [c for c in F_tr.columns if c not in DROP_COLS]

    fit_q = uhash(q_tr, 7) < args.fit_frac
    p1_path = W2 / 'p1.npz'
    if p1_path.exists():
        z = np.load(p1_path)
        p1_tr, p1_te = z['tr'], z['te']
    else:
        log('stage 1 GBDT (OOF on part B, predictions for part A and test)')
        FB = F_tr[rowB].reset_index(drop=True)
        oofB, (pA, pT), m1 = model.oof_multi(FB, y[rowB], folds[rowB], fit_q[rowB],
                                             [F_tr[rowA], F_te], cols1, log=log)
        for k, m in enumerate(m1):
            m.save_model(str(W2 / f'stage1_fold{k}.txt'))
        del FB
        p1_tr = np.empty(len(F_tr), np.float32)
        p1_tr[rowB], p1_tr[rowA] = oofB, pA
        p1_te = pT
        np.savez(p1_path, tr=p1_tr, te=p1_te)
    report('stage 1', gq, gs, p1_tr, rowB, t2, rec2, s1_eval)

    # ---------------- cross-encoder
    ce_tr, ce_te = run_cross_encoder(args, W2, rec_tr, rec_te, P_tr, P_te, F_tr, y, rowA, p1_tr, p1_te, gq)
    ce2 = score_ce2(args, W2, P_tr, P_te, p1_tr, p1_te, gq, ce_tr, ce_te) if args.ce2 else None

    # ---------------- stage 2
    log('stage 2 features')
    for i, (F, a, b, oq, p, ce, norm) in enumerate([(F_tr, gq, gs, q_tr, p1_tr, ce_tr, norm_tr),
                                                    (F_te, q_te, s_te, q_te, p1_te, ce_te, norm_te)]):
        if ce2 is not None:
            add_ce_features(F, a, ce2[i], 'ce2')
        features.add_stage2(F, a, b, p)
        add_ce_features(F, a, ce)
        rows = np.flatnonzero(~np.isnan(ce))
        sib = sibling_features(rows, a, b, oq, p, norm)
        for c in sib.columns:
            col = np.full(len(F), np.nan, np.float32)
            col[rows] = sib[c].to_numpy()
            F[c] = col
    cols2 = [c for c in F_tr.columns if c not in DROP_COLS]
    FB = F_tr[rowB].reset_index(drop=True)
    oofB, (p2_te,), models2 = model.oof_multi(FB, y[rowB], folds[rowB], fit_q[rowB], [F_te], cols2, seed=1, log=log)
    for k, m in enumerate(models2):
        m.save_model(str(W2 / f'stage2_fold{k}.txt'))
    p2_tr = np.full(len(F_tr), np.nan, np.float32)
    p2_tr[rowB] = oofB
    np.savez(W2 / 'p2.npz', trB=oofB, te=p2_te)
    f, mode, tau = report('stage 2', gq, gs, p2_tr, rowB, t2, rec2, s1_eval)
    per_c = tune_per_country(gq, gs, p2_tr, rowB, t2, rec2, s1_eval, (mode, tau)) if args.per_country_tau else None

    pred_q, pred_s = decide_by_country(q_te, s_te, p2_te, rec_te.country.to_numpy(), per_c, (mode, tau))
    write_outputs(Path(args.out), rec_te, P_te, pred_q, pred_s)
    if per_c:  # also the single pooled threshold, for comparison
        pred_q, pred_s = decide.decide(q_te, s_te, p2_te, tau, mode)
        write_outputs(Path(args.out + '_pooled'), rec_te, P_te, pred_q, pred_s)
    (W2 / 'summary.json').write_text(json.dumps({'oofB_f05': f, 'mode': mode, 'tau': float(tau), 'args': vars(args)}, indent=2))

    if args.pseudo_france:
        fm = fit_q[rowB]
        p2_ps = pseudo_france_refit(FB[fm], y[rowB][fm], F_te, rec_te, q_te, s_te, p2_te, cols2, log)
        np.save(W2 / 'p2_te_pseudo.npy', p2_ps)
        pred_q, pred_s = decide.decide(q_te, s_te, p2_ps, tau, mode)
        write_outputs(Path(args.out + '_pseudo'), rec_te, P_te, pred_q, pred_s)


def tune_per_country(q, s, p, rows, t_sim, rec, s1_eval, default):
    """Tune (mode, tau) separately for every country present in validation; returns {country: (mode, tau)}."""
    country = rec.country.to_numpy()
    out = {}
    for c in sorted(set(country[q[rows]])):
        rc = rows & (country[q] == c)
        mc = s1_eval & (country == c)
        f, mode, tau = report(f'  tau for {c}', q, s, p, rc, t_sim, rec, mc)
        out[c] = (mode, tau)
    rows_all = []
    for c, (mode, tau) in out.items():
        rc = rows & (country[q] == c) & (np.nan_to_num(p) >= 0.15)
        rows_all.append(decide.decide(q[rc], s[rc], p[rc], tau, mode))
    a = np.concatenate([x[0] for x in rows_all]); b = np.concatenate([x[1] for x in rows_all])
    f, by = metric.fbeta_macro(a, b, t_sim, rec, by_country=True, s1_mask=s1_eval)
    log(f'per-country tau: OOF F0.5={f:.4f} {by} {out} (pooled {default})')
    return out


def decide_by_country(q, s, p, country_of_record, per_c, default):
    """Decision per country with its own (mode, tau); countries not tuned (e.g. France) use the pooled one."""
    if not per_c:
        return decide.decide(q, s, p, default[1], default[0])
    cq = country_of_record[q]
    outs = []
    for c in np.unique(cq):
        m = cq == c
        mode, tau = per_c.get(c, default)
        outs.append(decide.decide(q[m], s[m], p[m], tau, mode))
    return np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs])


def score_ce2(args, W2, P_tr, P_te, p1_tr, p1_te, gq, ce_tr, ce_te):
    """Score the same rows the first cross-encoder scored with the second checkpoint (cached per split)."""
    src_dir = Path(args.ce_from) if args.ce_from else W2
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus)
    gpus = args.gpus.split(',')
    out = []
    for split, P, ce in [('train', P_tr, ce_tr), ('test', P_te, ce_te)]:
        rows = np.flatnonzero(~np.isnan(ce))
        n = np.int64(P.q.max() + 1 if len(P) else 1)
        n = np.int64(max(n, P.s.max() + 1))
        caches = [(W2 / f'ce2_pairs_{split}.npy', W2 / f'ce2_{split}.npy')]
        if args.extra:
            caches.append((Path(args.extra) / f'ce2_pairs_{split}.npy', Path(args.extra) / f'ce2_{split}.npy'))
        v = np.full(len(P), np.nan, np.float32)
        v[rows] = keyed_scores(P.q.to_numpy()[rows], P.s.to_numpy()[rows], n, caches, W2, f'ce2_{split}',
                               gpu_score_fn(f'{src_dir}/tok_{split}', args.ce2, gpus, env))
        out.append(v)
    return out


def report(name, q, s, p, rows, t_sim, rec, s1_eval):
    """Tune (mode, tau) on OOF rows and log macro F0.5 on the evaluation S1 records."""
    rows = rows & (np.nan_to_num(p) >= 0.15)   # rows below every tau can never be assigned
    qq, ss, pp = q[rows], s[rows], p[rows]
    best = (-1, None, None)
    for mode in ['assign', 'expf']:
        for tau in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            a, b = decide.decide(qq, ss, pp, tau, mode)
            f = metric.fbeta_macro(a, b, t_sim, rec, s1_mask=s1_eval)
            best = max(best, (f, mode, tau))
    f, mode, tau = best
    for d in (-0.05, 0.05):
        if 0.05 <= tau + d <= 0.95:
            a, b = decide.decide(qq, ss, pp, tau + d, mode)
            best = max(best, (metric.fbeta_macro(a, b, t_sim, rec, s1_mask=s1_eval), mode, round(tau + d, 3)))
    f, mode, tau = best
    a, b = decide.decide(qq, ss, pp, tau, mode)
    f, by = metric.fbeta_macro(a, b, t_sim, rec, by_country=True, s1_mask=s1_eval)
    log(f'{name}: OOF(B, test-like) F0.5={f:.4f} {by} mode={mode} tau={tau}')
    return f, mode, tau


def keyed_scores(q, s, n, caches, W2, tag, score_fn):
    """Scores for pairs (q, s), looked up by (record, S1) key in every existing cache (pairs.npy, scores.npy).
    Pairs found in no cache are scored once with score_fn and saved as a new cache file, so re-runs and changed
    candidate sets never misalign scores."""
    key = q.astype(np.int64) * n + s
    out = np.full(len(q), np.nan, np.float32)
    found = np.zeros(len(q), bool)
    k = 0
    while (W2 / f'{tag}_new{k}_pairs.npy').exists():
        caches = caches + [(W2 / f'{tag}_new{k}_pairs.npy', W2 / f'{tag}_new{k}.npy')]
        k += 1
    for pp, sp in caches:
        if not (Path(pp).exists() and Path(sp).exists()):
            continue
        cp, cs = np.load(pp), np.load(sp)
        assert len(cp) == len(cs), (pp, len(cp), len(cs))
        ck = cp[:, 0] * n + cp[:, 1]
        o = np.argsort(ck)
        ck, cs = ck[o], cs[o]
        m = np.flatnonzero(~found)
        pos = np.minimum(np.searchsorted(ck, key[m]), len(ck) - 1)
        hit = ck[pos] == key[m]
        out[m[hit]] = cs[pos[hit]]
        found[m[hit]] = True
    miss = np.flatnonzero(~found)
    log(f'{tag}: {found.sum():,} scores from cache, {len(miss):,} pairs to score')
    if len(miss):
        uk, first = np.unique(key[miss], return_index=True)
        pp, sp = W2 / f'{tag}_new{k}_pairs.npy', W2 / f'{tag}_new{k}.npy'
        np.save(pp, np.stack([q[miss][first], s[miss][first]], 1).astype(np.int64))
        score_fn(pp, sp)
        out[miss] = np.load(sp)[np.searchsorted(uk, key[miss])]
    assert not np.isnan(out).any()
    return out


def gpu_score_fn(tok_dir, ckpt, gpus, env):
    def run(pairs_path, out_path):
        subprocess.run([sys.executable, '-c',
                        'import sys; sys.path.insert(0, "src"); from er import crossenc; '
                        f'crossenc.predict("{tok_dir}", "{pairs_path}", "{ckpt}", "{out_path}", {gpus!r})'],
                       env=env, check=True)
    return run


def run_cross_encoder(args, W2, rec_tr, rec_te, P_tr, P_te, F_tr, y, rowA, p1_tr, p1_te, gq):
    gpus = args.gpus.split(',')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus)
    src_dir = Path(args.ce_from) if args.ce_from else W2          # checkpoint + tokens (+ cached scores)
    for split, rec in [('train', rec_tr), ('test', rec_te)]:
        d = src_dir / f'tok_{split}'
        if not (d / 'off.npy').exists():
            log(f'tokenizing {split} records')
            d.mkdir(exist_ok=True)
            flat, off = crossenc.tokenize_records(crossenc.record_texts(rec))
            np.save(d / 'flat.npy', flat)
            np.save(d / 'off.npy', off)
    q_tr, s_tr = P_tr.q.to_numpy(), P_tr.s.to_numpy()
    q_te, s_te = P_te.q.to_numpy(), P_te.s.to_numpy()

    ckpt = src_dir / 'ce.pt'
    if not ckpt.exists():
        # training pairs: part A records, their top candidates by blocking score (+ the true S1 if blocked)
        score = F_tr[['cos_name', 'cos_addr', 'cos_combo']].max(axis=1).to_numpy() + F_tr.cos_combo.to_numpy()
        r, _, _, _ = features._group_rank_gap(gq, score.astype(np.float32))
        sel = np.flatnonzero(rowA & ((r < args.ce_topk) | (y == 1)))
        _, first = np.unique(np.stack([q_tr[sel], s_tr[sel]], 1), axis=0, return_index=True)
        sel = sel[first]                                     # decoys appear in both universes
        rng = np.random.default_rng(0)
        if len(sel) > args.ce_train_pairs:
            sel = rng.choice(sel, args.ce_train_pairs, replace=False)
        tp = np.stack([q_tr[sel], s_tr[sel], y[sel]], 1).astype(np.int64)
        np.save(W2 / 'ce_train_pairs.npy', tp)
        log(f'cross-encoder training on {len(tp):,} pairs (pos={tp[:, 2].mean():.3f}) with GPUs {args.gpus}')
        cmd = ['torchrun', f'--nproc_per_node={len(gpus)}', 'src/train_ce.py', str(W2 / 'tok_train'),
               str(W2 / 'ce_train_pairs.npy'), str(ckpt)]
        subprocess.run([sys.executable, '-m', 'torch.distributed.run'] + cmd[1:], env=env, check=True)

    out = []
    for split, q, s, p, g in [('train', q_tr, s_tr, p1_tr, gq), ('test', q_te, s_te, p1_te, q_te)]:
        r, _, _, _ = features._group_rank_gap(g, p)
        rows = np.flatnonzero(r < args.ce_topk)
        ce = np.full(len(q), np.nan, np.float32)
        if args.ce_from:
            # reuse cached scores by (record, S1) key; score only pairs never scored before
            n = np.int64(len(rec_tr) if split == 'train' else len(rec_te))
            caches = [(src_dir / f'ce_pairs_{split}.npy',
                       Path(args.ce_test) if (split == 'test' and args.ce_test) else src_dir / f'ce_{split}.npy')]
            for d in [Path(args.extra)] if args.extra else []:
                caches.append((d / f'ce_missing_pairs_{split}.npy', d / f'ce_missing_{split}.npy'))
            caches.append((W2 / f'ce_missing_pairs_{split}.npy', W2 / f'ce_missing_{split}.npy'))
            ce[rows] = keyed_scores(q[rows], s[rows], n, caches, W2, f'ce_{split}',
                                    gpu_score_fn(f'{src_dir}/tok_{split}', ckpt, gpus, env))
        else:
            path = W2 / f'ce_{split}.npy'
            if not path.exists():
                np.save(W2 / f'ce_pairs_{split}.npy', np.stack([q[rows], s[rows]], 1).astype(np.int64))
                log(f'cross-encoder inference on {len(rows):,} {split} pairs')
                subprocess.run([sys.executable, '-c',
                                'import sys; sys.path.insert(0, "src"); from er import crossenc; '
                                f'crossenc.predict("{W2}/tok_{split}", "{W2}/ce_pairs_{split}.npy", "{ckpt}", '
                                f'"{path}", {gpus!r})'], env=env, check=True)
            ce[rows] = np.load(args.ce_test if (split == 'test' and args.ce_test) else path)
        out.append(ce)
    return out


def pseudo_france_refit(FB, yB, F_te, rec_te, q_te, s_te, p2_te, cols, log):
    """Add confident test-France decisions as labels and refit stage 2 on B + pseudo rows."""
    fr = rec_te.country.to_numpy()[q_te] == 'France'
    r, g, _, _ = features._group_rank_gap(q_te, p2_te)
    conf_q = np.unique(q_te[fr & (r == 0) & (p2_te > 0.97) & (g > 0.7)])
    neg_q = np.unique(q_te[fr & (r == 0) & (p2_te < 0.02)])
    isc = np.isin(q_te, conf_q)
    isn = np.isin(q_te, neg_q)
    rows = np.flatnonzero(isc | isn)
    yp = (isc[rows] & (r[rows] == 0)).astype(np.int8)
    log(f'pseudo-France: {len(conf_q):,} confident-match records, {len(neg_q):,} confident-none records, {len(rows):,} rows')
    X = pd.concat([FB[cols], F_te.iloc[rows][cols]], ignore_index=True)
    yy = np.concatenate([yB, yp])
    rng = np.random.default_rng(3)
    va = rng.choice(len(X), 2_000_000, replace=False)
    m = model.fit(X, yy, X.iloc[va], yy[va], rounds=1500, log=log)
    return model.predict_chunked(m, F_te, cols)


def write_outputs(out_dir, rec, pairs, pred_q, pred_s):
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = rec.id.to_numpy()
    s1 = np.flatnonzero(rec.src.to_numpy() == 1)

    def lists(q, s):
        df = pd.DataFrame({'s': s, 'id': ids[q]}).sort_values(['s', 'id'])
        return df.groupby('s').id.agg(','.join).reindex(s1).fillna('').to_numpy()

    pd.DataFrame({'source1_entity_id': ids[s1], 'matched_entity_ids': lists(pred_q, pred_s)}) \
        .to_csv(out_dir / 'matching_results.tsv', sep='\t', index=False)
    pd.DataFrame({'source1_entity_id': ids[s1],
                  'candidate_entity_ids': lists(pairs.q.to_numpy(), pairs.s.to_numpy())}) \
        .to_csv(out_dir / 'candidate_pairs.tsv', sep='\t', index=False)
    log(f'wrote {out_dir}/matching_results.tsv and candidate_pairs.tsv')


if __name__ == '__main__':
    main()
