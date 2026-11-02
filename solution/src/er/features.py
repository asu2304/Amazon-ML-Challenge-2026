"""Pair features: string similarities, address-number agreement, legal form, ambiguity and context."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler


def _cp(a, b, scorer, dtype=np.uint8):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=dtype)


def _num_feats(nums, qi, si):
    Q, S = nums[qi], nums[si]
    nq, ns = (Q >= 0).sum(1), (S >= 0).sum(1)
    eq = (Q[:, :, None] == S[:, None, :]) & (Q[:, :, None] >= 0)
    ov = eq.any(2).sum(1)
    union = nq + ns - ov
    return {
        'n_num_q': nq.astype(np.int8), 'n_num_s': ns.astype(np.int8), 'num_overlap': ov.astype(np.int8),
        'num_jacc': np.where(union > 0, ov / np.maximum(union, 1), -1).astype(np.float32),
        'num_first_eq': np.where((nq > 0) & (ns > 0), (Q[:, 0] == S[:, 0]), -1).astype(np.int8),
        'num_first_in_s': np.where(nq > 0, eq[:, 0, :].any(1), -1).astype(np.int8),
    }


def pair_features(rec, norm, nums, pairs, chunk=4_000_000, log=print):
    core = norm.name_core.to_numpy(object)
    nosp = norm.name_nospace.to_numpy(object)
    addr = norm.addr_norm.to_numpy(object)
    legal = norm.legal.to_numpy(object)
    feats = []
    for start in range(0, len(pairs), chunk):
        qi = pairs.q.to_numpy()[start:start + chunk]
        si = pairs.s.to_numpy()[start:start + chunk]
        qn, sn = core[qi].tolist(), core[si].tolist()
        qa, sa = addr[qi].tolist(), addr[si].tolist()
        qs, ss = nosp[qi].tolist(), nosp[si].tolist()
        f = {
            'nm_ratio': _cp(qn, sn, fuzz.ratio),
            'nm_tset': _cp(qn, sn, fuzz.token_set_ratio),
            'nm_tsort': _cp(qn, sn, fuzz.token_sort_ratio),
            'nm_partial': _cp(qn, sn, fuzz.partial_ratio),
            'nm_jw': _cp(qn, sn, JaroWinkler.normalized_similarity, np.float32),
            'ns_ratio': _cp(qs, ss, fuzz.ratio),
            'ns_partial': _cp(qs, ss, fuzz.partial_ratio),
            'ad_ratio': _cp(qa, sa, fuzz.ratio),
            'ad_tset': _cp(qa, sa, fuzz.token_set_ratio),
            'ad_partial': _cp(qa, sa, fuzz.partial_ratio),
        }
        f.update(_num_feats(nums, qi, si))
        lq, ls = legal[qi], legal[si]
        f['legal_q_empty'] = (lq == '').astype(np.int8)
        f['legal_eq'] = ((lq == ls) & (lq != '')).astype(np.int8)
        f['legal_conflict'] = ((lq != ls) & (lq != '') & (ls != '')).astype(np.int8)
        feats.append(pd.DataFrame(f))
        log(f'  features {min(start + chunk, len(pairs)):,}/{len(pairs):,}')
    F = pd.concat(feats, ignore_index=True)

    q, s = pairs.q.to_numpy(), pairs.s.to_numpy()
    F['len_q'] = norm.name_core.str.len().to_numpy()[q].astype(np.int16)
    F['len_s'] = norm.name_core.str.len().to_numpy()[s].astype(np.int16)
    alen = norm.addr_norm.str.len().to_numpy()
    F['alen_q'], F['alen_s'] = alen[q].astype(np.int16), alen[s].astype(np.int16)
    F['q_src'] = rec.src.to_numpy()[q]
    F['q_nonlatin'] = norm.nonlatin.to_numpy()[q]
    F['q_domain'] = norm.is_domain.to_numpy()[q]

    # ambiguity: how many S1 records of the country share this core name
    s1 = rec.src.to_numpy() == 1
    key = rec.country.to_numpy() + '|' + norm.name_core.to_numpy()
    cnt = pd.Series(key[s1]).value_counts()
    F['s_name_cnt'] = cnt.reindex(key[s]).fillna(0).to_numpy().astype(np.int32)
    F['q_name_cnt'] = cnt.reindex(key[q]).fillna(0).to_numpy().astype(np.int32)

    for c in ['cos_name', 'cos_addr', 'cos_combo']:
        F[c] = pairs[c].to_numpy()
    add_context(F, q, s, ['cos_name', 'cos_addr', 'cos_combo', 'nm_tset', 'ad_tset'])
    return F


def _group_rank_gap(key, val):
    """Rank of val within its key group (0 = best), gap to the best *other* value, group max."""
    order = np.lexsort((-val, key))
    k, v = key[order], val[order]
    start = np.r_[True, k[1:] != k[:-1]]
    gid = np.cumsum(start) - 1
    first_pos = np.flatnonzero(start)
    rank = np.arange(len(k)) - first_pos[gid]
    best = v[first_pos[gid]]
    has2 = (first_pos[gid] + 1 < len(k))
    nxt_idx = np.minimum(first_pos[gid] + 1, len(k) - 1)
    second = np.where(has2 & (k[nxt_idx] == k), v[nxt_idx], 0.0)
    gap = np.where(rank == 0, v - second, v - best)
    size = np.bincount(gid)[gid]
    r, g, b, n = (np.empty_like(x) for x in (rank, gap, best, size))
    r[order], g[order], b[order], n[order] = rank, gap, best, size
    return r, g, b, n


def add_context(F, q, s, cols):
    for c in cols:
        v = F[c].to_numpy().astype(np.float32)
        r, g, b, n = _group_rank_gap(q, v)
        F[f'{c}_rank_q'], F[f'{c}_gap_q'] = r.astype(np.int16), g.astype(np.float32)
        r, g, b, n2 = _group_rank_gap(s, v)
        F[f'{c}_rank_s'], F[f'{c}_gap_s'] = r.astype(np.int16), g.astype(np.float32)
    F['n_cand_q'] = n.astype(np.int16)
    F['n_cand_s'] = n2.astype(np.int32)


def add_stage2(F, q, s, p):
    """Listwise features from stage-1 probabilities (who else competes for this record / this S1).
    Columns are added to F in place."""
    F['p1'] = p
    r, g, b, _ = _group_rank_gap(q, p)
    F['p1_rank_q'], F['p1_gap_q'], F['p1_max_q'] = r.astype(np.int16), g, b
    r, g, b, _ = _group_rank_gap(s, p)
    F['p1_rank_s'], F['p1_gap_s'], F['p1_max_s'] = r.astype(np.int16), g, b
    F['p1_sum_q'] = pd.Series(p).groupby(q).transform('sum').to_numpy(np.float32)
    top = (F.p1_rank_q.to_numpy() == 0) & (p > 0.5)
    F['p1_sum_s_top'] = pd.Series(np.where(top, p, 0)).groupby(s).transform('sum').to_numpy(np.float32)
    F['p1_cnt_s_top'] = pd.Series(top.astype(np.int16)).groupby(s).transform('sum').to_numpy(np.int16)
    return F
