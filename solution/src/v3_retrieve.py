"""v3 step 1: bi-encoder retrieval channel.

  1. Fine-tune the bi-encoder on part-A positive pairs (same cluster split as v2, so part B stays unseen).
  2. Embed every train and test record (GPU, shared memory).
  3. Per country, each S2/S3 record retrieves its top-K S1 records by cosine.
  4. Report blocking recall on part B: TF-IDF alone vs bi-encoder alone vs union.
  5. Save the candidate pairs the TF-IDF channels did not produce (with their cosine) for train and test,
     plus the bi-encoder cosine of the existing TF-IDF pairs (a new feature).

    python -u src/v3_retrieve.py --work work --work2 work_v2 --work3 work_v3 --gpus 2,3
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from er import biencoder, blocking, data  # noqa: E402
from run_v2 import uhash  # noqa: E402

T0 = time.time()


def log(msg):
    print(f'[{time.time() - T0:7.0f}s] {msg}', flush=True)


def pair_cos(E, q, s, chunk=2_000_000):
    out = np.empty(len(q), np.float32)
    for i in range(0, len(q), chunk):
        a = torch.from_numpy(E[q[i:i + chunk]]).float()
        b = torch.from_numpy(E[s[i:i + chunk]]).float()
        out[i:i + chunk] = (a * b).sum(1).numpy()
    return out


def retrieve(rec, E, k, gpu):
    src, country = rec.src.to_numpy(), rec.country.to_numpy()
    out = []
    for c in sorted(pd.unique(country)):
        si = np.flatnonzero((country == c) & (src == 1))
        qi = np.flatnonzero((country == c) & (src != 1))
        I, V = biencoder.knn(E[qi], E[si], k=k, gpu=gpu)
        out.append(pd.DataFrame({'q': np.repeat(qi, I.shape[1]), 's': si[I.ravel()], 'cos_bi': V.ravel()}))
        log(f'  [{c}] knn: {len(qi):,} queries x {len(si):,} S1')
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../dataset')
    ap.add_argument('--work', default='work')
    ap.add_argument('--work2', default='work_v2')
    ap.add_argument('--work3', default='work_v3')
    ap.add_argument('--gpus', default='2,3')
    ap.add_argument('--k', type=int, default=10)
    args = ap.parse_args()
    W, W2, W3 = Path(args.work), Path(args.work2), Path(args.work3)
    W3.mkdir(exist_ok=True)
    gpus = args.gpus.split(',')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus)

    rec_tr = data.load_split(args.data, 'train')
    t = data.load_truth(args.data, rec_tr)
    N = len(rec_tr)
    src = rec_tr.src.to_numpy()
    cl = np.where(t >= 0, t, np.arange(N) + N)
    part_a = uhash(cl, 11) < 0.25

    ckpt = W3 / 'bi.pt'
    if not ckpt.exists():
        qa = np.flatnonzero(part_a & (t >= 0) & (src != 1))
        np.save(W3 / 'bi_train_pairs.npy', np.stack([qa, t[qa]], 1).astype(np.int64))
        log(f'bi-encoder training on {len(qa):,} part-A positive pairs')
        subprocess.run([sys.executable, '-m', 'torch.distributed.run', f'--nproc_per_node={len(gpus)}', '--master_port=29533',
                        'src/train_bi.py', str(W2 / 'tok_train'), str(W3 / 'bi_train_pairs.npy'), str(ckpt)],
                       env=env, check=True)

    for split in ['train', 'test']:
        out_new = W3 / f'{split}_bi_new_pairs.parquet'
        if out_new.exists():
            continue
        rec = rec_tr if split == 'train' else data.load_split(args.data, 'test')
        log(f'encoding {len(rec):,} {split} records')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        E = biencoder.encode(str(W2 / f'tok_{split}'), np.arange(len(rec)), str(ckpt), gpus)
        log(f'knn {split}')
        B = retrieve(rec, E, args.k, gpu=0)
        P = pd.read_parquet(W / f'{split}_pairs.parquet', columns=['q', 's'])
        P['cos_bi'] = pair_cos(E, P.q.to_numpy(), P.s.to_numpy())
        P[['cos_bi']].to_parquet(W3 / f'{split}_tfidf_pairs_cos_bi.parquet', index=False)
        key_tf = P.q.to_numpy() * len(rec) + P.s.to_numpy()
        key_bi = B.q.to_numpy() * len(rec) + B.s.to_numpy()
        new = B[~np.isin(key_bi, key_tf)].reset_index(drop=True)
        new.to_parquet(out_new, index=False)
        log(f'{split}: bi-encoder pairs {len(B):,}, new (not from TF-IDF) {len(new):,} '
            f'({len(new) / max(1, (rec.src != 1).sum()):.2f} per record)')
        if split == 'train':
            qB = ~part_a
            tf = P[qB[P.q.to_numpy()]]
            bi = B[qB[B.q.to_numpy()]]
            r_tf, by_tf = blocking.blocking_recall(tf, rec_tr, np.where(qB, t, -1))
            r_bi, by_bi = blocking.blocking_recall(bi, rec_tr, np.where(qB, t, -1))
            r_un, by_un = blocking.blocking_recall(pd.concat([tf[['q', 's']], bi[['q', 's']]]), rec_tr, np.where(qB, t, -1))
            log(f'part-B blocking recall: TF-IDF {r_tf:.4f} {by_tf}')
            log(f'                        bi-encoder {r_bi:.4f} {by_bi}')
            log(f'                        union {r_un:.4f} {by_un}')
        del E, B, P


if __name__ == '__main__':
    main()
