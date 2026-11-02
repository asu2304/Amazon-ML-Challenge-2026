"""Cross-encoder reranker: multilingual-e5-base (MIT, 278M params) fine-tuned as a pair classifier.

Input per pair: <s> name | address (S2/S3 record) </s></s> name | address (S1 record) </s>, raw text
(the model reads Indic scripts and French natively). Records are tokenized once; pairs are assembled
from token ids. Training uses DDP over all visible GPUs (launch with torchrun); inference shards pairs
over GPUs with one process per GPU.
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

MODEL = 'intfloat/multilingual-e5-base'
REC_MAX = 60          # tokens kept per record
CLS, SEP, PAD = 0, 2, 1  # XLM-R special ids


def record_texts(rec):
    name, addr = rec.name.to_numpy(object), rec.addr.to_numpy(object)
    return [f'{n} | {a}' if a else n for n, a in zip(name, addr)]


def tokenize_records(texts, batch=200_000):
    """Ragged token ids per record -> (flat int32 ids, int64 offsets)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    flat, lens = [], []
    for i in range(0, len(texts), batch):
        ids = tok(texts[i:i + batch], add_special_tokens=False, truncation=True, max_length=REC_MAX)['input_ids']
        lens.extend(len(x) for x in ids)
        flat.append(np.fromiter((t for x in ids for t in x), dtype=np.int32))
    off = np.zeros(len(lens) + 1, dtype=np.int64)
    off[1:] = np.cumsum(lens)
    return np.concatenate(flat), off


def build_batch(flat, off, qa, sb):
    seqs = []
    for a, b in zip(qa, sb):
        ta, tb = flat[off[a]:off[a + 1]], flat[off[b]:off[b + 1]]
        seqs.append(np.concatenate(([CLS], ta, [SEP, SEP], tb, [SEP])))
    L = max(len(s) for s in seqs)
    ids = np.full((len(seqs), L), PAD, dtype=np.int64)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s
    ids = torch.from_numpy(ids)
    return ids, (ids != PAD).long()


class CrossEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(MODEL, add_pooling_layer=False)
        self.head = nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(h).squeeze(-1)


# ------------------------------------------------------------------ training (torchrun)

def train_ddp(tok_dir, pairs_path, out_path, epochs=1, bs=128, lr=3e-5, warmup=0.05, init=None):
    import torch.distributed as dist
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(dev)
    flat, off = np.load(f'{tok_dir}/flat.npy', mmap_mode='r'), np.load(f'{tok_dir}/off.npy')
    P = np.load(pairs_path)  # (n, 3): q, s, y
    rng = np.random.default_rng(0)
    model = CrossEncoder()
    if init:  # continue from a fine-tuned checkpoint
        model.load_state_dict(torch.load(init, map_location='cpu'))
    model = model.to(dev)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[dev.index])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * (len(P) // (bs * world))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, warmup * steps)) * max(0.0, (steps - s) / max(1, steps * (1 - warmup))))
    lossf = nn.BCEWithLogitsLoss()
    step, t0 = 0, time.time()
    model.train()
    for ep in range(epochs):
        perm = rng.permutation(len(P))
        per = len(perm) // world
        mine = perm[rank * per:(rank + 1) * per]
        for i in range(0, len(mine) - bs + 1, bs):
            b = P[mine[i:i + bs]]
            ids, mask = build_batch(flat, off, b[:, 0], b[:, 1])
            y = torch.from_numpy(b[:, 2].astype(np.float32)).to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logit = model(ids.to(dev, non_blocking=True), mask.to(dev, non_blocking=True))
            loss = lossf(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if rank == 0 and step % 200 == 0:
                print(f'step {step}/{steps} loss={loss.item():.4f} {time.time() - t0:.0f}s', flush=True)
    if rank == 0:
        torch.save(model.module.state_dict(), out_path)
    dist.barrier()
    dist.destroy_process_group()


# ------------------------------------------------------------------ inference (one process per GPU)

def _predict_worker(gpu, tok_dir, pairs_path, ckpt, out_dir, n_gpu, bs):
    dev = torch.device('cuda', gpu)
    flat, off = np.load(f'{tok_dir}/flat.npy', mmap_mode='r'), np.load(f'{tok_dir}/off.npy')
    P = np.load(pairs_path, mmap_mode='r')
    idx = np.arange(gpu, len(P), n_gpu)
    q, s = np.asarray(P[idx, 0]), np.asarray(P[idx, 1])
    ln = (off[q + 1] - off[q]) + (off[s + 1] - off[s])
    order = np.argsort(ln, kind='stable')  # length bucketing -> little padding
    model = CrossEncoder()
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.to(dev).eval()
    out = np.empty(len(idx), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(order), bs):
            o = order[i:i + bs]
            ids, mask = build_batch(flat, off, q[o], s[o])
            out[o] = torch.sigmoid(model(ids.to(dev), mask.to(dev)).float()).cpu().numpy()
            if gpu == 0 and (i // bs) % 500 == 0:
                print(f'  gpu0 {i:,}/{len(order):,} {time.time() - t0:.0f}s', flush=True)
    np.save(f'{out_dir}/ce_part{gpu}.npy', out)


def predict(tok_dir, pairs_path, ckpt, out_path, gpus, bs=2048):
    """pairs file: (n, 2+) int64 array [q, s, ...]; returns probabilities aligned with rows."""
    import torch.multiprocessing as mp
    out_dir = os.path.dirname(out_path)
    mp.spawn(_predict_worker, args=(tok_dir, pairs_path, ckpt, out_dir, len(gpus), bs), nprocs=len(gpus))
    n = len(np.load(pairs_path, mmap_mode='r'))
    res = np.empty(n, dtype=np.float32)
    for g in range(len(gpus)):
        res[np.arange(g, n, len(gpus))] = np.load(f'{out_dir}/ce_part{g}.npy')
    np.save(out_path, res)
    return res
