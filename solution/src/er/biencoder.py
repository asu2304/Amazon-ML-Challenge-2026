"""Bi-encoder retrieval channel (v3): multilingual-e5-base fine-tuned as a Siamese encoder.

Each record is encoded on its own (mean-pooled, L2-normalized); training pulls an S2/S3 record towards its
true S1 and away from the other S1s in the batch (in-batch negatives, symmetric InfoNCE). At inference every
S2/S3 record retrieves its top-K S1 records of the same country by cosine (exact search on GPU), giving
candidates that character TF-IDF misses (other scripts, truncated addresses, trade names).
Reuses the per-record token ids from crossenc.tokenize_records.
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .crossenc import CLS, MODEL, PAD, SEP


def batch_ids(flat, off, idx):
    seqs = [np.concatenate(([CLS], flat[off[i]:off[i + 1]], [SEP])) for i in idx]
    L = max(len(s) for s in seqs)
    ids = np.full((len(seqs), L), PAD, dtype=np.int64)
    for j, s in enumerate(seqs):
        ids[j, :len(s)] = s
    ids = torch.from_numpy(ids)
    return ids, (ids != PAD).long()


class BiEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(MODEL, add_pooling_layer=False)

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return Fn.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1)


def train_ddp(tok_dir, pairs_path, out_path, bs=512, lr=2e-5, temp=0.05, epochs=1):
    """pairs: (n, 2) int64 [s2s3_record, s1_record] positives."""
    import torch.distributed as dist
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(dev)
    flat, off = np.load(f'{tok_dir}/flat.npy', mmap_mode='r'), np.load(f'{tok_dir}/off.npy')
    P = np.load(pairs_path)
    model = nn.parallel.DistributedDataParallel(BiEncoder().to(dev), device_ids=[dev.index])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * (len(P) // (bs * world))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, 0.05 * steps)) * max(0.0, 1 - s / max(1, steps)))
    rng = np.random.default_rng(0)
    step, t0 = 0, time.time()
    for _ in range(epochs):
        perm = rng.permutation(len(P))
        per = len(perm) // world
        mine = perm[rank * per:(rank + 1) * per]
        for i in range(0, len(mine) - bs + 1, bs):
            b = P[mine[i:i + bs]]
            ids, mask = batch_ids(flat, off, np.concatenate([b[:, 0], b[:, 1]]))  # one DDP forward for both sides
            with torch.autocast('cuda', dtype=torch.bfloat16):
                e = model(ids.to(dev), mask.to(dev)).float()
            eq, es = e[:len(b)], e[len(b):]
            logits = eq @ es.T / temp
            tgt = torch.arange(len(b), device=dev)
            # duplicate S1s in a batch would be false negatives -> mask them out
            same = torch.from_numpy(b[:, 1][:, None] == b[:, 1][None, :]).to(dev) & ~torch.eye(len(b), dtype=torch.bool, device=dev)
            logits = logits.masked_fill(same, -1e4)
            loss = (Fn.cross_entropy(logits, tgt) + Fn.cross_entropy(logits.T, tgt)) / 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if rank == 0 and step % 100 == 0:
                print(f'step {step}/{steps} loss={loss.item():.4f} {time.time() - t0:.0f}s', flush=True)
    if rank == 0:
        torch.save(model.module.state_dict(), out_path)
    dist.barrier()
    dist.destroy_process_group()


def _encode_worker(gpu, tok_dir, idx, E, ckpt, n_gpu, bs):
    dev = torch.device('cuda', gpu)
    flat, off = np.load(f'{tok_dir}/flat.npy', mmap_mode='r'), np.load(f'{tok_dir}/off.npy')
    pos = np.arange(gpu, len(idx), n_gpu)
    pos = pos[np.argsort(off[idx[pos] + 1] - off[idx[pos]], kind='stable')]   # length bucketing
    model = BiEncoder()
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.to(dev).eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(pos), bs):
            p = pos[i:i + bs]
            ids, mask = batch_ids(flat, off, idx[p])
            E[torch.from_numpy(p)] = model(ids.to(dev), mask.to(dev)).half().cpu()


def encode(tok_dir, idx, ckpt, gpus, bs=4096, dim=768):
    """Embeddings (float16, L2-normalized) for records idx, in idx order. Workers write into a shared-memory
    tensor, so nothing large touches the disk."""
    import torch.multiprocessing as mp
    E = torch.empty((len(idx), dim), dtype=torch.float16).share_memory_()
    mp.spawn(_encode_worker, args=(tok_dir, np.asarray(idx), E, ckpt, len(gpus), bs), nprocs=len(gpus))
    return E.numpy()


def knn(Eq, Es, k=10, gpu=0, chunk=2048):
    """Exact top-k cosine neighbours of each query row among Es rows (both L2-normalized)."""
    dev = torch.device('cuda', gpu)
    S = torch.from_numpy(Es).to(dev)
    I = np.empty((len(Eq), k), dtype=np.int64)
    V = np.empty((len(Eq), k), dtype=np.float32)
    for i in range(0, len(Eq), chunk):
        Q = torch.from_numpy(Eq[i:i + chunk]).to(dev)
        v, j = torch.topk(Q @ S.T, k=min(k, S.shape[0]), dim=1)
        I[i:i + chunk], V[i:i + chunk] = j.cpu().numpy(), v.float().cpu().numpy()
    return I, V
