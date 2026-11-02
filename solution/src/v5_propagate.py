"""v5: collective resolution by sibling propagation (a third stage on top of v4's predictions).

Analysis of v4's validation errors: 87% of the remaining F0.5 loss is missed matches, and 97.7% of missed
records have a sibling (another record of the same business) that v4 linked confidently and correctly; 64% share
its exact core name (vs 18.5% for decoys). So a record that is ambiguous on its own can be resolved through its
siblings.

  anchors   records v4 linked with best p2 >= ANCHOR_P
  targets   records v4 left unassigned
  candidates for a target Y: the S1s of anchors sharing Y's core name, no-space name or full address
             (within the same country; within the same universe on train)
  features  anchor support (count, confidence, similarity of Y to the anchors), Y's own stage-1/2 scores for that
            S1 (if it was a candidate), Y's best score elsewhere, competition (other S1s vouched for), cluster size,
            name ambiguity, direct Y-vs-S1 string similarity
  model     LightGBM, out-of-fold on part B (train), refit on all part-B rows for test
  decision  add Y -> argmax S1 if p3 >= tau3, tau3 tuned on the exact macro F0.5 (v4 links are kept unchanged)

    python -u src/v5_propagate.py            -> output_v5/
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).resolve().parent))
from er import data, decide, features, metric, model  # noqa: E402
from run_v2 import uhash, write_outputs  # noqa: E402

T0 = time.time()
ANCHOR_P = 0.9
MAX_ANCHORS_PER_KEY = 30
MAX_CAND_PER_TARGET = 10


def log(m):
    print(f'[{time.time() - T0:7.0f}s] {m}', flush=True)


def cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.uint8)


def build(qq, ss, p2, p1, grp_of, country_of, n_orig, norm, rec_src, tau, mode, truth=None):
    """Stage-3 candidate table. qq/ss: group-indexed record/S1 per candidate row (train: universe ids; test: plain);
    grp_of(x): universe of a group index; n_orig: records per universe (index % n_orig -> original record)."""
    aq, as_ = decide.decide(qq, ss, p2, tau, mode)
    n = int(max(qq.max(), ss.max())) + 1
    pred = np.full(n, -1, np.int64)
    pred[aq] = as_
    pbest = np.zeros(n, np.float32)
    np.maximum.at(pbest, qq, p2)
    recs = np.unique(qq)
    anchors = recs[(pred[recs] >= 0) & (pbest[recs] >= ANCHOR_P)]
    targets = recs[pred[recs] < 0]
    log(f'  anchors {len(anchors):,}, unassigned targets {len(targets):,}, base links {len(aq):,}')

    o = lambda x: x % n_orig
    name = norm.name_core.to_numpy(object)
    nosp = norm.name_nospace.to_numpy(object)
    addr = norm.addr_norm.to_numpy(object)
    cands = []
    for kind, arr, min_len in [('name', name, 3), ('nosp', nosp, 5), ('addr', addr, 12)]:
        def key(x):
            return pd.Series(grp_of(x).astype(str)) + '|' + country_of(o(x)) + '|' + arr[o(x)]
        ka = pd.DataFrame({'k': key(anchors).to_numpy(), 'a': anchors})
        ka = ka[pd.Series(arr[o(anchors)]).str.len().to_numpy() >= min_len]
        ka = ka.groupby('k', sort=False).head(MAX_ANCHORS_PER_KEY)
        kt = pd.DataFrame({'k': key(targets).to_numpy(), 'y': targets})
        kt = kt[pd.Series(arr[o(targets)]).str.len().to_numpy() >= min_len]
        j = kt.merge(ka, on='k')[['y', 'a']]
        j['via'] = kind
        cands.append(j)
        log(f'  via {kind}: {len(j):,} (target, anchor) links')
    J = pd.concat(cands, ignore_index=True)
    J = J[J.y.to_numpy() != J.a.to_numpy()]
    J['s'] = pred[J.a.to_numpy()]
    J['aconf'] = pbest[J.a.to_numpy()]
    yo, ao = o(J.y.to_numpy()), o(J.a.to_numpy())
    J['nm_ya'] = cp(name[yo].tolist(), name[ao].tolist(), fuzz.token_set_ratio)
    J['ad_ya'] = cp(addr[yo].tolist(), addr[ao].tolist(), fuzz.token_set_ratio)
    for v in ['name', 'nosp', 'addr']:
        J[f'via_{v}'] = (J.via == v).astype(np.int8)
    G = J.groupby(['y', 's'], sort=False).agg(
        n_links=('a', 'size'), n_anchors=('a', 'nunique'), aconf_max=('aconf', 'max'), aconf_mean=('aconf', 'mean'),
        nm_ya_max=('nm_ya', 'max'), ad_ya_max=('ad_ya', 'max'), ad_ya_mean=('ad_ya', 'mean'),
        via_name=('via_name', 'max'), via_nosp=('via_nosp', 'max'), via_addr=('via_addr', 'max')).reset_index()
    G = G.sort_values(['y', 'n_anchors', 'aconf_max'], ascending=[True, False, False])
    G = G.groupby('y', sort=False).head(MAX_CAND_PER_TARGET).reset_index(drop=True)
    y, s = G.y.to_numpy(), G.s.to_numpy()

    # competition and cluster context
    G['n_s1_vouched'] = G.groupby('y').s.transform('size').to_numpy()
    r, gap, _, _ = features._group_rank_gap(y, G.n_anchors.to_numpy().astype(np.float32))
    G['anchor_rank'], G['anchor_gap'] = r, gap
    G['s1_cluster_size'] = np.bincount(as_, minlength=n)[s]
    # the target's own stage-1/2 scores for this S1, and its best elsewhere
    key_rows = qq.astype(np.int64) * n + ss
    order = np.argsort(key_rows)
    kr = key_rows[order]
    kq = y.astype(np.int64) * n + s
    pos = np.minimum(np.searchsorted(kr, kq), len(kr) - 1)
    hit = kr[pos] == kq
    G['own_p2'] = np.where(hit, p2[order][pos], np.nan).astype(np.float32)
    G['own_p1'] = np.where(hit, p1[order][pos], np.nan).astype(np.float32)
    G['was_candidate'] = hit.astype(np.int8)
    G['y_best_p2'] = pbest[y]
    # direct string similarity target vs S1 record
    so = o(s)
    G['nm_ys'] = cp(name[o(y)].tolist(), name[so].tolist(), fuzz.token_set_ratio)
    G['ad_ys'] = cp(addr[o(y)].tolist(), addr[so].tolist(), fuzz.token_set_ratio)
    G['y_addr_empty'] = (pd.Series(addr[o(y)]).str.len().to_numpy() == 0).astype(np.int8)
    G['y_src'] = rec_src[o(y)]
    ncount = pd.Series(name[o(anchors)]).value_counts()
    G['name_anchor_cnt'] = ncount.reindex(name[o(y)]).fillna(0).to_numpy()
    if truth is not None:
        G['label'] = (truth[y] == s).astype(np.int8)
    log(f'  stage-3 candidates {len(G):,} for {G.y.nunique():,} targets'
        + (f', positives {G.label.mean():.3f}' if truth is not None else ''))
    return G, aq, as_


FEATS = ['n_links', 'n_anchors', 'aconf_max', 'aconf_mean', 'nm_ya_max', 'ad_ya_max', 'ad_ya_mean', 'via_name',
         'via_nosp', 'via_addr', 'n_s1_vouched', 'anchor_rank', 'anchor_gap', 's1_cluster_size', 'own_p2', 'own_p1',
         'was_candidate', 'y_best_p2', 'nm_ys', 'ad_ys', 'y_addr_empty', 'y_src', 'name_anchor_cnt']
PARAMS = dict(objective='binary', learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, num_threads=40, verbose=-1, seed=0)


def add_links(G, p3, tau3):
    """Each target gets at most one new S1: its best stage-3 candidate if p3 >= tau3."""
    r, _, _, _ = features._group_rank_gap(G.y.to_numpy(), p3)
    m = (r == 0) & (p3 >= tau3)
    return G.y.to_numpy()[m], G.s.to_numpy()[m]


def main():
    W, W3, W4 = Path('work'), Path('work_v3'), Path('work_v4')
    summ = json.loads((W4 / 'summary.json').read_text())
    tau, mode = summ['tau'], summ['mode']
    log(f'v4 decision: mode={mode} tau={tau}')

    # ---------------- train (part B, test-like universes; same construction as run_v2)
    rec = data.load_split('../dataset', 'train')
    t = data.load_truth('../dataset', rec)
    norm = pd.read_parquet(W / 'train_norm.parquet', columns=['name_core', 'name_nospace', 'addr_norm'])
    P = pd.concat([pd.read_parquet(W / 'train_pairs.parquet', columns=['q', 's']),
                   pd.read_parquet(W3 / 'train_bi_new_pairs.parquet', columns=['q', 's'])], ignore_index=True)
    N = len(rec)
    src = rec.src.to_numpy()
    cl = np.where(t >= 0, t, np.arange(N) + N)
    half = (uhash(cl, 23) < 0.5).astype(np.int64)
    q0, s0 = P.q.to_numpy(), P.s.to_numpy()
    keep = (t[q0] < 0) | (half[q0] == half[s0])
    q, s = q0[keep], s0[keep]
    gq, gs = q + half[s] * N, s + half[s] * N
    part_a = uhash(cl, 11) < 0.25
    rowB = ~part_a[q]
    t2 = np.full(2 * N, -1, np.int64)
    for k in (0, 1):
        m = (t >= 0) & (half == k)
        t2[np.flatnonzero(m) + k * N] = t[m] + k * N
    rec2 = pd.DataFrame({'src': np.r_[src, src], 'country': np.r_[rec.country.to_numpy(), rec.country.to_numpy()]})
    s1_eval = np.zeros(2 * N, bool)
    for k in (0, 1):
        s1_eval[np.flatnonzero((src == 1) & (half == k) & ~part_a) + k * N] = True
    p2 = np.load(W4 / 'p2.npz')['trB']
    p1 = np.load(W4 / 'p1.npz')['tr'][rowB]
    qq, ss = gq[rowB], gs[rowB]
    country = rec.country.to_numpy()
    log('building stage-3 candidates on train part B')
    G, aq, as_ = build(qq, ss, p2, p1, lambda x: x // N, lambda x: country[x], N, norm, src, tau, mode, truth=t2)
    base = metric.fbeta_macro(aq, as_, t2, rec2, by_country=True, s1_mask=s1_eval)
    log(f'v4 base (validation): F0.5={base[0]:.4f} {base[1]}')

    # ---------------- out-of-fold stage-3 model
    fold = (uhash(cl[G.y.to_numpy() % N], 5) < 0.5).astype(np.int8)
    p3 = np.zeros(len(G), np.float32)
    X = G[FEATS].astype(np.float32)
    yv = G.label.to_numpy()
    for k in (0, 1):
        tr, va = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        m = lgb.train(PARAMS, lgb.Dataset(X.iloc[tr], yv[tr]), 600, valid_sets=[lgb.Dataset(X.iloc[va], yv[va])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        p3[va] = m.predict(X.iloc[va], num_threads=40)
        log(f'  fold {k}: best_iter={m.best_iteration}')
    imp = sorted(zip(m.feature_importance('gain'), FEATS), reverse=True)[:10]
    log('  top stage-3 features: ' + ', '.join(c for _, c in imp))

    best = (base[0], None)
    for tau3 in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        ny, ns = add_links(G, p3, tau3)
        f = metric.fbeta_macro(np.r_[aq, ny], np.r_[as_, ns], t2, rec2, s1_mask=s1_eval)
        log(f'  tau3={tau3}: +{len(ny):,} links -> F0.5={f:.4f}')
        if f > best[0]:
            best = (f, tau3)
    f_best, tau3 = best
    if tau3 is None:
        log('stage 3 does not improve validation F0.5; not applying it')
        return
    ny, ns = add_links(G, p3, tau3)
    f, by = metric.fbeta_macro(np.r_[aq, ny], np.r_[as_, ns], t2, rec2, by_country=True, s1_mask=s1_eval)
    log(f'v5 (validation): F0.5={f:.4f} {by}  (v4 {base[0]:.4f}), tau3={tau3}')

    # ---------------- test
    mfull = lgb.train(PARAMS, lgb.Dataset(X, yv), int(m.best_iteration * 1.1) + 1)
    del G, X
    rec_te = data.load_split('../dataset', 'test')
    norm_te = pd.read_parquet(W / 'test_norm.parquet', columns=['name_core', 'name_nospace', 'addr_norm'])
    nf = pd.read_parquet(W4 / 'test_norm_fr.parquet', columns=['name_core', 'name_nospace', 'addr_norm', 'idx'])
    for c in ['name_core', 'name_nospace', 'addr_norm']:
        a = norm_te[c].to_numpy(copy=True)
        a[nf.idx.to_numpy()] = nf[c].to_numpy()
        norm_te[c] = a
    P_te = pd.concat([pd.read_parquet(W / 'test_pairs.parquet', columns=['q', 's']),
                      pd.read_parquet(W3 / 'test_bi_new_pairs.parquet', columns=['q', 's'])], ignore_index=True)
    zq, zs = P_te.q.to_numpy(), P_te.s.to_numpy()
    p2t = np.load(W4 / 'p2.npz')['te']
    p1t = np.load(W4 / 'p1.npz')['te']
    ctry_te = rec_te.country.to_numpy()
    log('building stage-3 candidates on test')
    Gt, aqt, ast = build(zq, zs, p2t, p1t, lambda x: np.zeros(len(x), np.int64), lambda x: ctry_te[x],
                         len(rec_te), norm_te, rec_te.src.to_numpy(), tau, mode)
    p3t = mfull.predict(Gt[FEATS].astype(np.float32), num_threads=40)
    ny, ns = add_links(Gt, p3t, tau3)
    log(f'test: v4 links {len(aqt):,} + stage-3 links {len(ny):,} '
        f'(by country {pd.Series(ctry_te[ny]).value_counts().to_dict()})')
    # candidate set = v4 candidates + stage-3 candidates (every output link is a candidate)
    cand = pd.concat([P_te, Gt[['y', 's']].rename(columns={'y': 'q'})], ignore_index=True).drop_duplicates()
    write_outputs(Path('output_v5'), rec_te, cand, np.r_[aqt, ny], np.r_[ast, ns])
    (W4 / 'v5_summary.json').write_text(json.dumps({'v4_val': base[0], 'v5_val': f, 'tau3': tau3,
                                                     'by_country': by, 'test_new_links': int(len(ny))}, indent=2))


if __name__ == '__main__':
    main()
