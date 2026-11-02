"""v4 step 2: re-normalize France test records with the learned French rules and recompute every feature of
France test pairs (TF-IDF candidates and bi-encoder candidates). Only France rows are written, as patches that
run_v2.py --france_patch applies on top of the v1/v3 caches (train has no France, so nothing else changes).

    python -u src/v4_france_features.py
Writes work_v4/test_norm_fr.parquet, test_nums_fr.npy, test_feats_fr.parquet, test_bi_feats_fr.parquet.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from er import blocking, data, features, normalize  # noqa: E402

T0 = time.time()


def log(m):
    print(f'[{time.time() - T0:6.0f}s] {m}', flush=True)


def france_features(rec, norm, nums, P, fr_rows):
    """All v1 pair features (TF-IDF cosines + similarities + context) for the France rows of P."""
    q, s = P.q.to_numpy()[fr_rows], P.s.to_numpy()[fr_rows]
    Pf = pd.concat([pd.DataFrame({'q': q, 's': s}), blocking.pair_cosines(rec, norm, q, s, log=log)], axis=1)
    return features.pair_features(rec, norm, nums, Pf, log=log)


def main():
    W, W3, W4 = Path('work'), Path('work_v3'), Path('work_v4')
    rec = data.load_split('../dataset', 'test')
    maps = json.loads((W / 'maps.json').read_text())
    tok_map = maps['tok']
    part_map = {tuple(k.split('\t')): v for k, v in maps['part'].items()}
    fr_maps = json.loads((W4 / 'france_maps.json').read_text())
    fr = np.flatnonzero(rec.country.to_numpy() == 'France')

    log(f'normalizing {len(fr):,} France test records with French rules')
    nf, numf = normalize.normalize_records(rec.iloc[fr].reset_index(drop=True), tok_map, part_map, fr_maps=fr_maps)
    norm = pd.read_parquet(W / 'test_norm.parquet')
    nums = np.load(W / 'test_nums.npy')
    changed = (norm.addr_norm.to_numpy()[fr] != nf.addr_norm.to_numpy()).mean()
    for c in norm.columns:
        norm.loc[fr, c] = nf[c].to_numpy()
    nums[fr] = numf
    log(f'France addresses changed by the new rules: {changed:.1%}')
    nf.assign(idx=fr).to_parquet(W4 / 'test_norm_fr.parquet', index=False)
    np.save(W4 / 'test_nums_fr.npy', numf)

    ctry = rec.country.to_numpy()
    for name, path, cols in [('TF-IDF pairs', W / 'test_pairs.parquet', ['q', 's']),
                             ('bi-encoder pairs', W3 / 'test_bi_new_pairs.parquet', ['q', 's'])]:
        P = pd.read_parquet(path, columns=cols)
        rows = np.flatnonzero(ctry[P.q.to_numpy()] == 'France')
        log(f'{name}: recomputing features for {len(rows):,} France rows')
        F = france_features(rec, norm, nums, P, rows)
        F['row'] = rows
        out = W4 / ('test_feats_fr.parquet' if name == 'TF-IDF pairs' else 'test_bi_feats_fr.parquet')
        F.to_parquet(out, index=False)
        log(f'wrote {out}')


if __name__ == '__main__':
    main()
