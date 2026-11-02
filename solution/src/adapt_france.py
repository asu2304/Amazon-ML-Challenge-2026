"""France adaptation of the cross-encoder by self-training (v2.1).

France has no training labels. From the v2 stage-2 test probabilities we take France records the model is
very sure about (best candidate p > 0.97 with a large gap -> positive; that record's other top candidates
-> negatives; records whose best candidate has p < 0.02 -> negatives), mix them 50/50 with the original
part-A training pairs (so US/India are not forgotten), continue fine-tuning the v2 cross-encoder at a lower
learning rate, and re-score the France test pairs. No external data is used.

    python -u src/adapt_france.py --work work --work2 work_v2 --gpus 2,3
Writes work_v2/ce_test_fr.npy (v2 cross-encoder test scores with France rows replaced).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from er import crossenc, data, features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../dataset')
    ap.add_argument('--work', default='work')
    ap.add_argument('--work2', default='work_v2')
    ap.add_argument('--gpus', default='2,3')
    ap.add_argument('--max_pos', type=int, default=600_000)
    ap.add_argument('--neg_per_pos', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-5)
    args = ap.parse_args()
    W, W2 = Path(args.work), Path(args.work2)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus)
    rng = np.random.default_rng(0)

    rec_te = data.load_split(args.data, 'test')
    P = pd.read_parquet(W / 'test_pairs.parquet', columns=['q', 's'])
    q, s = P.q.to_numpy(), P.s.to_numpy()
    p2 = np.load(W2 / 'p2.npz')['te']
    fr = rec_te.country.to_numpy()[q] == 'France'
    rank, gap, best, _ = features._group_rank_gap(q, p2)

    pos = np.flatnonzero(fr & (rank == 0) & (p2 > 0.97) & (gap > 0.7))
    pos = rng.choice(pos, min(args.max_pos, len(pos)), replace=False)
    pos_q = np.zeros(len(rec_te), bool)
    pos_q[q[pos]] = True
    neg_hard = np.flatnonzero(pos_q[q] & (rank >= 1) & (rank <= 4))           # other candidates of sure records
    neg_none = np.flatnonzero(fr & (best < 0.02) & (rank <= 2))                # records sure to match nothing
    n_neg = args.neg_per_pos * len(pos)
    neg = np.concatenate([neg_hard, neg_none])
    neg = rng.choice(neg, min(n_neg, len(neg)), replace=False)
    print(f'France pseudo pairs: {len(pos):,} positive, {len(neg):,} negative '
          f'(hard {len(neg_hard):,}, none {len(neg_none):,} available)', flush=True)

    # combined token store: train records [0, N_tr), test records [N_tr, N_tr + N_te)
    mix = W2 / 'tok_mix'
    tr_off = np.load(W2 / 'tok_train' / 'off.npy')
    n_tr = len(tr_off) - 1
    if not (mix / 'off.npy').exists():
        mix.mkdir(exist_ok=True)
        te_off = np.load(W2 / 'tok_test' / 'off.npy')
        flat = np.concatenate([np.load(W2 / 'tok_train' / 'flat.npy'), np.load(W2 / 'tok_test' / 'flat.npy')])
        np.save(mix / 'flat.npy', flat)
        np.save(mix / 'off.npy', np.concatenate([tr_off[:-1], te_off + tr_off[-1]]))
        del flat

    fr_pairs = np.concatenate([
        np.stack([q[pos] + n_tr, s[pos] + n_tr, np.ones(len(pos), np.int64)], 1),
        np.stack([q[neg] + n_tr, s[neg] + n_tr, np.zeros(len(neg), np.int64)], 1)])
    orig = np.load(W2 / 'ce_train_pairs.npy')
    orig = orig[rng.choice(len(orig), min(len(orig), len(fr_pairs)), replace=False)]
    mix_pairs = np.concatenate([fr_pairs, orig]).astype(np.int64)
    np.save(W2 / 'ce_fr_train_pairs.npy', mix_pairs)
    print(f'adaptation training pairs: {len(mix_pairs):,} (France {len(fr_pairs):,} + original {len(orig):,})', flush=True)

    ckpt = W2 / 'ce_fr.pt'
    if not ckpt.exists():
        subprocess.run([sys.executable, '-m', 'torch.distributed.run', f'--nproc_per_node={len(args.gpus.split(","))}',
                        'src/train_ce.py', str(mix), str(W2 / 'ce_fr_train_pairs.npy'), str(ckpt),
                        str(args.lr), str(W2 / 'ce.pt')], env=env, check=True)

    # re-score France rows of the v2 test cross-encoder pairs
    cp = np.load(W2 / 'ce_pairs_test.npy')
    ce = np.load(W2 / 'ce_test.npy')
    frr = np.flatnonzero(rec_te.country.to_numpy()[cp[:, 0]] == 'France')
    sub = W2 / 'fr_tmp'
    sub.mkdir(exist_ok=True)
    np.save(sub / 'pairs.npy', cp[frr])
    print(f're-scoring {len(frr):,} France test pairs', flush=True)
    subprocess.run([sys.executable, '-c',
                    'import sys; sys.path.insert(0, "src"); from er import crossenc; '
                    f'crossenc.predict("{W2}/tok_test", "{sub}/pairs.npy", "{ckpt}", "{sub}/ce.npy", '
                    f'{args.gpus.split(",")!r})'], env=env, check=True)
    new = np.load(sub / 'ce.npy')
    print(f'France ce: mean before {ce[frr].mean():.4f} after {new.mean():.4f}; '
          f'share >0.5 before {(ce[frr] > 0.5).mean():.4f} after {(new > 0.5).mean():.4f}', flush=True)
    ce[frr] = new
    np.save(W2 / 'ce_test_fr.npy', ce)
    print('wrote', W2 / 'ce_test_fr.npy', flush=True)


if __name__ == '__main__':
    main()
