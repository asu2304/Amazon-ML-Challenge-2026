"""Candidate generation: for every S2/S3 record, top-K S1 records of the same country from three TF-IDF channels.

  name   char 3-grams of the core name (+ the no-space form, so domains match spaced names)
  addr   char 3-grams of the normalized address
  combo  word tokens of name + address (rare street names / numbers / name words dominate;
         separates the many S1 records that share a generic name)

The union of the three top-K lists is the candidate set fed to the model. Exact cosines for all
three channels are computed for every candidate pair and used as features.
"""
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn


def _vec(kind, max_df):
    if kind == 'word':
        return TfidfVectorizer(analyzer='word', token_pattern=r'\S+', min_df=2, max_df=max_df,
                               sublinear_tf=True, dtype=np.float32)
    return TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=2, max_df=max_df,
                           sublinear_tf=True, dtype=np.float32)


def _rowwise_cos(A, B, qi, si, chunk=2_000_000):
    out = np.empty(len(qi), dtype=np.float32)
    for i in range(0, len(qi), chunk):
        out[i:i + chunk] = np.asarray(A[qi[i:i + chunk]].multiply(B[si[i:i + chunk]]).sum(1)).ravel()
    return out


def channel_texts(norm):
    name = (norm.name_core + ' ' + norm.name_nospace).to_numpy()
    addr = norm.addr_norm.to_numpy()
    combo = (norm.name_core + ' ' + norm.addr_norm).to_numpy()
    return {'name': (name, 'char'), 'addr': (addr, 'char'), 'combo': (combo, 'word')}


def block(rec, norm, k=10, max_df=0.05, threshold=0.05, max_cand=15, n_threads=60, log=print):
    """Returns DataFrame [q, s, cos_name, cos_addr, cos_combo] with global record indices."""
    src, country = rec.src.to_numpy(), rec.country.to_numpy()
    texts = channel_texts(norm)
    out = []
    for c in sorted(pd.unique(country)):
        s_idx = np.flatnonzero((country == c) & (src == 1))
        q_idx = np.flatnonzero((country == c) & (src != 1))
        mats, cand = {}, []
        for ch, (txt, kind) in texts.items():
            t0 = time.time()
            vec = _vec(kind, max_df)
            vec.fit(np.concatenate([txt[s_idx], txt[q_idx]]))
            S = vec.transform(txt[s_idx]).astype(np.float32).tocsr()
            Q = vec.transform(txt[q_idx]).astype(np.float32).tocsr()
            mats[ch] = (Q, S)
            C = sp_matmul_topn(Q, S.T.tocsr(), top_n=k, threshold=threshold, sort=True, n_threads=n_threads).tocoo()
            cand.append(np.stack([C.row, C.col]).T)
            log(f'  [{c}] {ch:5s}: |Q|={len(q_idx):,} |S1|={len(s_idx):,} vocab={len(vec.vocabulary_):,} '
                f'pairs={C.nnz:,} ({time.time() - t0:.0f}s)')
        pr = np.unique(np.concatenate(cand).astype(np.int64), axis=0)
        qi, si = pr[:, 0], pr[:, 1]
        df = pd.DataFrame({'q': q_idx[qi], 's': s_idx[si]})
        for ch, (Q, S) in mats.items():
            df[f'cos_{ch}'] = _rowwise_cos(Q, S, qi, si)
        # keep the best `max_cand` candidates per record (ranked by best channel + combo cosine)
        score = df[['cos_name', 'cos_addr', 'cos_combo']].max(axis=1) + df.cos_combo
        rank = score.groupby(df.q).rank(ascending=False, method='first')
        df = df[rank <= max_cand].reset_index(drop=True)
        log(f'  [{c}] union pairs={len(df):,}  per query={len(df) / max(1, len(q_idx)):.1f}')
        out.append(df)
    return pd.concat(out, ignore_index=True)


def pair_cosines(rec, norm, q, s, max_df=0.05, log=print):
    """TF-IDF cosines of the three channels for arbitrary (S2/S3, S1) pairs. Vectorizers are fit exactly as
    in block(), so values match the cosines of the blocked pairs."""
    src, country = rec.src.to_numpy(), rec.country.to_numpy()
    texts = channel_texts(norm)
    out = {f'cos_{ch}': np.zeros(len(q), np.float32) for ch in texts}
    for c in sorted(pd.unique(country)):
        s_idx = np.flatnonzero((country == c) & (src == 1))
        q_idx = np.flatnonzero((country == c) & (src != 1))
        rows = np.flatnonzero(country[q] == c)
        if len(rows) == 0:
            continue
        pos_q = np.full(len(rec), -1, np.int64)
        pos_s = np.full(len(rec), -1, np.int64)
        pos_q[q_idx], pos_s[s_idx] = np.arange(len(q_idx)), np.arange(len(s_idx))
        for ch, (txt, kind) in texts.items():
            vec = _vec(kind, max_df)
            vec.fit(np.concatenate([txt[s_idx], txt[q_idx]]))
            S = vec.transform(txt[s_idx]).astype(np.float32).tocsr()
            Q = vec.transform(txt[q_idx]).astype(np.float32).tocsr()
            out[f'cos_{ch}'][rows] = _rowwise_cos(Q, S, pos_q[q[rows]], pos_s[s[rows]])
        log(f'  [{c}] cosines for {len(rows):,} pairs')
    return pd.DataFrame(out)


def blocking_recall(pairs, rec, true_s1):
    """Fraction of true (S2/S3 -> S1) links present in the candidate set, overall and per country."""
    q_true = np.flatnonzero((true_s1 >= 0) & (rec.src.to_numpy() != 1))
    hit = pd.Series(False, index=q_true)
    ok = pairs[true_s1[pairs.q.to_numpy()] == pairs.s.to_numpy()].q.to_numpy()
    hit[ok] = True
    by = pd.Series(hit.to_numpy()).groupby(rec.country.to_numpy()[q_true]).mean()
    return hit.mean(), by.to_dict()
