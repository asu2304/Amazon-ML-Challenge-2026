"""v4 step 1: learn French address normalization from the provided data (no external data).

  regions      frequent parts of S1 France addresses that are the LAST part in most of their occurrences
               (S1 ends with the region; cities are last only after reordering: 6-7% vs 86-87%)
  departments  frequent parts of S2/S3 France addresses that never occur anywhere in S1 France addresses
               (S2/S3 write the department where S1 writes the region)
  abbreviations  short address tokens of S2/S3 that stand for a longer token of the matched S1, mined from
               confident v3.1 France matches (e.g. psg -> passage)

Region and department parts are dropped from France addresses (they are redundant with the city and caused
spurious mismatches); abbreviations are expanded.

    python -u src/v4_france_maps.py      -> work_v4/france_maps.json
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from er import data, features, normalize  # noqa: E402


def is_subseq(a, b):
    it = iter(b)
    return all(ch in it for ch in a)


def main():
    W3, W4 = Path('work_v3'), Path('work_v4')
    W4.mkdir(exist_ok=True)
    rec = data.load_split('../dataset', 'test')
    fr = rec.country.to_numpy() == 'France'
    src = rec.src.to_numpy()
    addr = rec.addr.to_numpy()

    s1 = np.flatnonzero(fr & (src == 1))
    ot = np.flatnonzero(fr & (src != 1))
    s1_parts = [normalize.addr_parts(a) for a in addr[s1]]
    ot_parts = [normalize.addr_parts(a) for a in addr[ot]]
    s1_vocab = Counter(p for ps in s1_parts for p in ps)
    last = Counter(ps[-1] for ps in s1_parts if ps)
    # a region is the last part in most of its occurrences (cities are last only when parts were reordered)
    regions = sorted(p for p, c in last.items()
                     if c >= 0.01 * len(s1) and c / s1_vocab[p] >= 0.5 and not any(ch.isdigit() for ch in p))
    ot_vocab = Counter(p for ps in ot_parts for p in ps)
    departments = sorted(p for p, c in ot_vocab.items()
                         if c >= 0.005 * len(ot) and p not in s1_vocab and not any(ch.isdigit() for ch in p))

    # abbreviations from confident matches of v3.1
    P = pd.concat([pd.read_parquet('work/test_pairs.parquet', columns=['q', 's']),
                   pd.read_parquet(W3 / 'test_bi_new_pairs.parquet', columns=['q', 's'])], ignore_index=True)
    p = np.load(W3 / 'p2.npz')['te']
    q, s = P.q.to_numpy(), P.s.to_numpy()
    r, g, _, _ = features._group_rank_gap(q, p)
    conf = np.flatnonzero(fr[q] & (r == 0) & (p > 0.95) & (g > 0.5))
    cnt, tot = Counter(), Counter()
    for i in conf:
        A = set(' '.join(normalize.addr_parts(addr[q[i]])).split())
        B = set(' '.join(normalize.addr_parts(addr[s[i]])).split())
        ua, ub = A - B, B - A
        for a in ua:
            if a.isdigit():
                continue
            tot[a] += 1
            for b in ub:
                if len(a) <= 4 and len(b) >= len(a) + 2 and a[0] == b[0] and is_subseq(a, b) and not b.isdigit():
                    cnt[(a, b)] += 1
    best = defaultdict(lambda: (None, 0))
    for (a, b), c in cnt.items():
        if c > best[a][1]:
            best[a] = (b, c)
    abbr = {a: b for a, (b, c) in best.items() if c >= 50 and c / tot[a] >= 0.5}

    out = {'regions': regions, 'departments': departments, 'abbr': abbr}
    (W4 / 'france_maps.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f'S1 France {len(s1):,}, S2/S3 France {len(ot):,}, confident pairs used {len(conf):,}')
    print('regions:', regions)
    print('departments:', departments)
    print('abbreviations:', dict(sorted(abbr.items(), key=lambda kv: -cnt[(kv[0], kv[1])])))


if __name__ == '__main__':
    main()
