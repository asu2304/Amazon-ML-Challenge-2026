"""v3.1 step 1: retrain the cross-encoder on the candidates it will actually judge in v3.

v2's cross-encoder only ever saw TF-IDF candidates. In v3, bi-encoder candidates enter the top-5 of many
records. Here the training pairs are part-A records' top-5 candidates ranked by v3's stage-1 model (plus the
true S1 when it was blocked), i.e. the hardest decisions, and the model continues from v2's checkpoint.

    python -u src/v31_ce.py --gpus 0
Writes work_v31/ce2.pt.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from er import data, features  # noqa: E402
from run_v2 import uhash  # noqa: E402


def train_rows(data_dir, W, W3):
    """Rebuild v3's train candidate rows exactly as run_v2 does (TF-IDF pairs + bi-encoder pairs, universes)."""
    rec = data.load_split(data_dir, 'train')
    t = data.load_truth(data_dir, rec)
    P = pd.concat([pd.read_parquet(W / 'train_pairs.parquet', columns=['q', 's']),
                   pd.read_parquet(W3 / 'train_bi_new_pairs.parquet', columns=['q', 's'])], ignore_index=True)
    N = len(rec)
    cl = np.where(t >= 0, t, np.arange(N) + N)
    half = (uhash(cl, 23) < 0.5).astype(np.int64)
    q0, s0 = P.q.to_numpy(), P.s.to_numpy()
    keep = (t[q0] < 0) | (half[q0] == half[s0])
    q, s = q0[keep], s0[keep]
    gq = q + half[s] * N
    part_a = uhash(cl, 11) < 0.25
    return q, s, gq, (t[q] == s).astype(np.int64), part_a[q]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../dataset')
    ap.add_argument('--gpus', default='0')
    ap.add_argument('--pairs', type=int, default=4_000_000)
    ap.add_argument('--lr', type=float, default=2e-5)
    args = ap.parse_args()
    W, W2, W3, W31 = Path('work'), Path('work_v2'), Path('work_v3'), Path('work_v31')
    W31.mkdir(exist_ok=True)

    q, s, gq, y, rowA = train_rows(args.data, W, W3)
    p1 = np.load(W3 / 'p1.npz')['tr']
    assert len(p1) == len(q), (len(p1), len(q))
    r, _, _, _ = features._group_rank_gap(gq, p1)
    sel = np.flatnonzero(rowA & ((r < 5) | (y == 1)))
    _, first = np.unique(np.stack([q[sel], s[sel]], 1), axis=0, return_index=True)
    sel = sel[first]
    rng = np.random.default_rng(1)
    sel = rng.choice(sel, min(args.pairs, len(sel)), replace=False)
    tp = np.stack([q[sel], s[sel], y[sel]], 1).astype(np.int64)
    np.save(W31 / 'ce2_train_pairs.npy', tp)
    print(f'ce2 training pairs: {len(tp):,} (pos={tp[:, 2].mean():.3f}) from part-A top-5 by v3 stage 1', flush=True)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus)
    subprocess.run([sys.executable, '-m', 'torch.distributed.run', f'--nproc_per_node={len(args.gpus.split(","))}',
                    '--master_port=29541', 'src/train_ce.py', str(W2 / 'tok_train'), str(W31 / 'ce2_train_pairs.npy'),
                    str(W31 / 'ce2.pt'), str(args.lr), str(W2 / 'ce.pt')], env=env, check=True)
    print('wrote', W31 / 'ce2.pt', flush=True)


if __name__ == '__main__':
    main()
