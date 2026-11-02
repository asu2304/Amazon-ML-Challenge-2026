"""Loading the challenge TSVs into one record table per split."""
from pathlib import Path

import numpy as np
import pandas as pd


def read_tsv(path):
    return pd.read_csv(path, sep='\t', keep_default_na=False, dtype=str, quoting=3)


def load_split(data_dir, split):
    """All records of a split (S1, S2, S3) in one frame; row position is the global record index."""
    parts = []
    for i in (1, 2, 3):
        df = read_tsv(Path(data_dir) / split / f'{split}_source{i}.tsv')
        df['src'] = np.int8(i)
        parts.append(df)
    rec = pd.concat(parts, ignore_index=True)
    return rec.rename(columns={'entity_id': 'id', 'business_name': 'name', 'business_address': 'addr'})


def load_truth(data_dir, rec):
    """Array `true_s1` aligned with rec: index of the matching S1 record for S2/S3 rows, -1 if none.
    S1 rows point to themselves."""
    gt = read_tsv(Path(data_dir) / 'train' / 'train_ground_truth.tsv')
    pos = pd.Series(np.arange(len(rec)), index=rec['id'].to_numpy())
    ex = gt.assign(m=gt.matched_entity_ids.str.split(',')).explode('m')
    ex = ex[ex.m.notna() & (ex.m != '')]
    true_s1 = np.full(len(rec), -1, dtype=np.int64)
    true_s1[pos[ex.m].to_numpy()] = pos[ex.source1_entity_id].to_numpy()
    s1 = np.flatnonzero(rec.src.to_numpy() == 1)
    true_s1[s1] = s1
    return true_s1


def subsample_universe(rec, true_s1, frac, seed=0):
    """Keep `frac` of S1 entities with all their matches, plus `frac` of the unmatched S2/S3 records.
    Used only for quick development runs."""
    rng = np.random.default_rng(seed)
    is_s1 = rec.src.to_numpy() == 1
    keep_s1 = is_s1 & (rng.random(len(rec)) < frac)
    keep = keep_s1.copy()
    matched = true_s1 >= 0
    keep |= ~is_s1 & matched & keep_s1[np.where(matched, true_s1, 0)]
    keep |= ~is_s1 & ~matched & (rng.random(len(rec)) < frac)
    idx = np.flatnonzero(keep)
    remap = np.full(len(rec), -1, dtype=np.int64)
    remap[idx] = np.arange(len(idx))
    new_true = np.where(true_s1[idx] >= 0, remap[np.maximum(true_s1[idx], 0)], -1)
    return rec.iloc[idx].reset_index(drop=True), new_true
