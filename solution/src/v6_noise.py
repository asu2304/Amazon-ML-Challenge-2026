"""v6 step 1: "noise transfer" - recreate the data generator's corruptions on new S1 records.

Every train positive pair (clean S1 -> noisy S2/S3 variant) is a recorded example of the generator. For a sample
of such pairs we detect WHICH corruptions were applied (an "edit script"), keeping their real co-occurrence, and then
replay randomly chosen scripts on target S1 records (e.g. the unlabelled France S1 records of the test set). French
vocabulary for legal forms, street abbreviations and region->department is measured from France's own unlabelled
S2/S3 records. No external data.

Output: a labelled synthetic dataset in the challenge format for one country
    <out>/test/test_source{1,2,3}.tsv  and  <out>/test_truth.tsv (source1_entity_id, matched_entity_ids)
The S1 records are split in two halves: half A is the universe (their variants are positives, clusters sized like
train), half B only donates decoys (variants of businesses absent from S1), giving ~41% decoys like test.

    python -u src/v6_noise.py --country France --out work_v6/synth_France
"""
import argparse
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent))
from er import data, normalize  # noqa: E402
from run_v2 import uhash  # noqa: E402

JUNK_RE = re.compile(r'^\s*([^\w\s]+\s*|#\s*\d+\s+|mr\.?\s+|m/s\.?\s+|@)', re.I)
SUFFIX_RE = re.compile(r'\s+#\d+$')
DOMAIN_RE = re.compile(r'\.(com|in|net|org|co|fr|biz|info)\b', re.I)
LEGAL_FR = ['SARL', 'SAS', 'SASU', 'EURL', 'SCI', 'SA']
ACCENTS = {'a': 'áàâ', 'e': 'éèêë', 'i': 'íîï', 'o': 'óôö', 'u': 'úùûü', 'c': 'ç', 'n': 'ñ'}
STREET_ABBR_FR = {'rue': ['R.', 'R', 'RUE'], 'avenue': ['AV', 'AV.', 'AVE'], 'boulevard': ['BD', 'BD.', 'BLVD'],
                  'allée': ['ALL', 'ALL.'], 'allee': ['ALL', 'ALL.'], 'impasse': ['IMP', 'IMP.'], 'place': ['PL'],
                  'passage': ['PSG', 'PSG.'], 'chemin': ['CHE', 'CH'], 'route': ['RTE'], 'quai': ['Q', 'QU'],
                  'cours': ['CRS'], 'saint': ['ST', 'ST.'], 'sainte': ['STE']}


def toks(s):
    return re.findall(r'\w+', s.lower())


# ------------------------------------------------------------------ 1. detect edit scripts on train pairs

def detect(s1n, s1a, vn, va):
    """Which corruptions turned (s1n, s1a) into (vn, va)."""
    op = {}
    op['n_upper'] = vn.isupper()
    op['n_lower'] = vn.islower()
    m = JUNK_RE.match(vn)
    op['junk'] = m.group(0) if m else ''
    ms = SUFFIX_RE.search(vn)
    op['suffix'] = ms.group(0) if ms else ''
    op['domain'] = bool(DOMAIN_RE.search(vn))
    a, b = set(toks(normalize.to_ascii(s1n))), set(toks(normalize.to_ascii(vn)))
    la, lb = {t for t in a if t in normalize.LEGAL}, {t for t in b if t in normalize.LEGAL}
    ca, cb = a - la, b - lb
    sim = fuzz.token_set_ratio(' '.join(ca), ' '.join(cb)) if ca and cb else 0
    op['trade'] = (sim < 35) and not op['domain'] and not re.search(r'[^\x00-ɏ]', vn)
    op['legal'] = ('drop' if la and not lb else 'add' if lb and not la else
                   'swap' if la and lb and la != lb else 'keep')
    op['n_drop'] = len(ca - cb) if not op['trade'] else 0
    junk_toks = set(toks(normalize.to_ascii(op['junk'] + ' ' + op['suffix'])))
    op['n_add'] = len(cb - ca - junk_toks) if not op['trade'] else 0
    wn = toks(vn)
    op['dup'] = any(x == y for x, y in zip(wn, wn[1:]))
    op['typos'] = sum(1 for t in cb - ca if any(fuzz.ratio(t, u) >= 60 for u in ca - cb)) if not op['trade'] else 0
    op['accent'] = bool(re.search(r'[À-ÿ]', vn)) and not re.search(r'[À-ÿ]', s1n)
    op['reorder'] = (ca == cb) and toks(normalize.to_ascii(s1n)) != toks(normalize.to_ascii(vn))
    # address
    op['a_empty'] = (va.strip() == '')
    op['a_upper'] = va.isupper()
    pa = [p.strip() for p in s1a.split(',') if p.strip()]
    pv = [p.strip() for p in va.split(',') if p.strip()]
    op['a_nparts_drop'] = max(0, len(pa) - len(pv)) if not op['a_empty'] else 0
    idx = [max(range(len(pa)), key=lambda j: fuzz.ratio(x.lower(), pa[j].lower())) for x in pv] if pa else []
    op['a_reorder'] = len(idx) > 1 and any(b < a for a, b in zip(idx, idx[1:]))
    op['a_numfmt'] = (re.match(r'\s*(no\.?|n°|#|\()\s*\d', va, re.I).group(1) if re.match(r'\s*(no\.?|n°|#|\()\s*\d', va, re.I) else '')
    op['a_zeropad'] = bool(re.search(r'\b0\d+', va)) and not re.search(r'\b0\d+', s1a)
    op['a_typos'] = sum(1 for t in set(toks(va)) - set(toks(s1a)) if t.isalpha() and len(t) > 3
                        and any(fuzz.ratio(t, u) >= 70 for u in set(toks(s1a)) - set(toks(va))))
    new_t, old_t = set(toks(va)) - set(toks(s1a)), set(toks(s1a)) - set(toks(va))
    op['a_abbr'] = any(len(x) <= 4 and x.isalpha() and any(len(y) > len(x) + 1 and y.startswith(x[0]) for y in old_t)
                       for x in new_t)
    return op


# ------------------------------------------------------------------ 2. replay a script on a target S1 record

def typo(w, rng):
    if len(w) < 4:
        return w
    i = rng.integers(1, len(w) - 1)
    k = rng.integers(4)
    if k == 0:
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]           # swap
    if k == 1:
        return w[:i] + w[i + 1:]                            # delete
    if k == 2:
        return w[:i] + chr(rng.integers(97, 123)) + w[i:]   # insert
    return w[:i] + chr(rng.integers(97, 123)) + w[i + 1:]   # replace


def accent(w, rng):
    idx = [i for i, ch in enumerate(w.lower()) if ch in ACCENTS]
    if not idx:
        return w
    i = idx[rng.integers(len(idx))]
    ch = ACCENTS[w[i].lower()][rng.integers(len(ACCENTS[w[i].lower()]))]
    return w[:i] + (ch.upper() if w[i].isupper() else ch) + w[i + 1:]


def apply_name(name, op, rng, ctx):
    if op['trade']:
        return ctx['trade_pool'][rng.integers(len(ctx['trade_pool']))]
    words = name.split()
    legal_idx = [i for i, w in enumerate(words) if re.sub(r'\W', '', w).lower() in normalize.LEGAL]
    core = [w for i, w in enumerate(words) if i not in legal_idx]
    legal = [words[i] for i in legal_idx]
    if op['legal'] == 'drop':
        legal = []
    elif op['legal'] == 'add' and not legal:
        legal = [ctx['legal_forms'][rng.integers(len(ctx['legal_forms']))]]
    elif op['legal'] == 'swap' and legal:
        legal = [ctx['legal_forms'][rng.integers(len(ctx['legal_forms']))]]
    for _ in range(min(op['n_drop'], max(0, len(core) - 1))):
        core.pop(rng.integers(len(core)))
    for _ in range(op['n_add']):
        core.insert(rng.integers(len(core) + 1), ctx['generic'][rng.integers(len(ctx['generic']))])
    if op['reorder'] and len(core) > 1:
        rng.shuffle(core)
    for _ in range(min(op['typos'], len(core))):
        j = rng.integers(len(core))
        core[j] = typo(core[j], rng)
    if op['accent'] and core:
        j = rng.integers(len(core))
        core[j] = accent(core[j], rng)
    if op['dup'] and core:
        j = rng.integers(len(core))
        core.insert(j, core[j])
    out = ' '.join(core + legal) if rng.random() < 0.8 else ' '.join(legal + core)
    if op['domain']:
        out = re.sub(r'\W', '', normalize.to_ascii(' '.join(core)).lower()) + ctx['tld'][rng.integers(len(ctx['tld']))]
    if op['n_upper']:
        out = out.upper()
    elif op['n_lower']:
        out = out.lower()
    return (op['junk'] + out + op['suffix']).strip()


def apply_addr(addr, op, rng, ctx):
    if op['a_empty'] or not addr:
        return ''
    parts = [p.strip() for p in addr.split(',') if p.strip()]
    if ctx['region_to_dept'] and parts:
        # France: S2/S3 write the department (or nothing) where S1 writes the region
        r = rng.random()
        if parts[-1] in ctx['region_to_dept']:
            if r < ctx['p_dept']:
                parts[-1] = ctx['region_to_dept'][parts[-1]][rng.integers(len(ctx['region_to_dept'][parts[-1]]))]
            elif r < ctx['p_dept'] + ctx['p_noregion']:
                parts = parts[:-1]
    for _ in range(min(op['a_nparts_drop'], max(0, len(parts) - 1))):
        parts.pop(rng.integers(1, len(parts)) if len(parts) > 1 else 0)
    if op['a_reorder'] and len(parts) > 1:
        rng.shuffle(parts)
    out = ', '.join(parts)
    if op['a_abbr'] and ctx['street_abbr']:
        ws = out.split(' ')
        for i, w in enumerate(ws):
            k = w.lower()
            if k in ctx['street_abbr'] and rng.random() < 0.8:
                opts = ctx['street_abbr'][k]
                ws[i] = opts[rng.integers(len(opts))]
        out = ' '.join(ws)
    ws = out.split(' ')
    alpha = [i for i, w in enumerate(ws) if w.isalpha() and len(w) > 3]
    for _ in range(min(op['a_typos'], len(alpha))):
        i = alpha[rng.integers(len(alpha))]
        ws[i] = typo(ws[i], rng)
    out = ' '.join(ws)
    if op['a_zeropad']:
        out = re.sub(r'\b(\d{1,3})\b', lambda m: '0' + m.group(1), out, count=1)
    if op['a_numfmt']:
        out = re.sub(r'^\s*(\d+)', lambda m: f"{op['a_numfmt'].upper()}{m.group(1)}" + (')' if op['a_numfmt'] == '(' else ''),
                     out, count=1)
    if op['a_upper']:
        out = out.upper()
    return out


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../dataset')
    ap.add_argument('--country', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_scripts', type=int, default=200_000)
    ap.add_argument('--max_s1', type=int, default=260_000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    (out / 'test').mkdir(parents=True, exist_ok=True)

    # --- edit scripts from train pairs (per source)
    tr = data.load_split(args.data, 'train')
    t = data.load_truth(args.data, tr)
    pos = np.flatnonzero((t >= 0) & (tr.src.to_numpy() != 1))
    pos = rng.choice(pos, min(args.n_scripts, len(pos)), replace=False)
    nm, ad, src = tr.name.to_numpy(), tr.addr.to_numpy(), tr.src.to_numpy()
    scripts = {2: [], 3: []}
    trade_pool = []
    for i in pos:
        op = detect(nm[t[i]], ad[t[i]], nm[i], ad[i])
        if re.search(r'[^\x00-ɏ]', nm[i]):          # other-script names: keep the corruption mix, not the script
            op['trade'] = False
        scripts[src[i]].append(op)
        if op['trade']:
            trade_pool.append(nm[i])
    # cluster structure from train: matches per S1 and the S2 share
    s1 = np.flatnonzero(tr.src.to_numpy() == 1)
    matched = np.flatnonzero((t >= 0) & (src != 1))
    k_per_s1 = np.bincount(t[matched], minlength=len(tr))[s1]
    p_s2 = (src[(t >= 0) & (src != 1)] == 2).mean()
    print(f'scripts: S2 {len(scripts[2]):,}, S3 {len(scripts[3]):,}; trade names {len(trade_pool):,}; '
          f'matches/S1 mean {k_per_s1.mean():.2f}; S2 share {p_s2:.3f}', flush=True)
    freq = pd.DataFrame(scripts[2] + scripts[3]).mean(numeric_only=True)
    print('operation frequencies:', {k: round(float(v), 3) for k, v in freq.items()}, flush=True)

    # --- target country context from its own unlabelled test records
    te = data.load_split(args.data, 'test')
    cm = te.country.to_numpy() == args.country
    t_s1 = te[cm & (te.src.to_numpy() == 1)].reset_index(drop=True)
    t_ot = te[cm & (te.src.to_numpy() != 1)]
    words = Counter(w for n in t_s1.name for w in n.split())
    legal_forms = [w for w, _ in words.most_common(300) if re.sub(r'\W', '', w).lower() in normalize.LEGAL][:8] or ['LLC', 'Inc']
    generic = [w for w, c in words.most_common(200) if w.isalpha() and len(w) > 3 and w not in legal_forms][:60]
    ctx = {'trade_pool': trade_pool, 'legal_forms': legal_forms, 'generic': generic,
           'tld': ['.com', '.fr'] if args.country == 'France' else ['.com', '.in'] if args.country == 'India' else ['.com'],
           'region_to_dept': {}, 'p_dept': 0.0, 'p_noregion': 0.0, 'street_abbr': {}}
    if args.country == 'France':
        import json
        fm = json.loads(Path('work_v4/france_maps.json').read_text())
        last_parts = t_s1.addr.str.split(',').str[-1].str.strip()
        regions = [r for r in last_parts.value_counts().index[:10] if normalize.addr_parts(r) and normalize.addr_parts(r)[0] in fm['regions']]
        # department of each region: departments co-occurring with the region's cities in S2/S3
        city = t_s1.addr.str.split(',').str[-2].str.strip().str.lower()
        region_of_city = pd.Series(last_parts.to_numpy(), index=city.to_numpy()).groupby(level=0).agg(lambda x: x.mode().iat[0])
        dept_counts = Counter()
        ot_parts = t_ot.addr.str.split(',')
        for ps in ot_parts:
            ps = [p.strip() for p in ps if p.strip()]
            deps = [p for p in ps if normalize.addr_parts(p) and normalize.addr_parts(p)[0] in fm['departments']]
            cities = [p.lower() for p in ps if p.lower() in region_of_city.index]
            for d in deps:
                for c in cities:
                    dept_counts[(region_of_city[c], d)] += 1
        region_names = {r for r in last_parts.unique() if normalize.addr_parts(r) and normalize.addr_parts(r)[0] in fm['regions']}
        tot_r = Counter()
        for (r, d), c in dept_counts.items():
            tot_r[r] += c
        r2d = {}
        for (r, d), c in dept_counts.items():
            if r in region_names and c >= 0.2 * tot_r[r]:
                r2d.setdefault(r, []).append(d)
        has_dept = t_ot.addr.apply(lambda a: any(normalize.addr_parts(p) and normalize.addr_parts(p)[0] in fm['departments']
                                                 for p in a.split(','))).mean()
        has_region = t_ot.addr.apply(lambda a: any(p.strip() in regions for p in a.split(','))).mean()
        ctx.update(region_to_dept=r2d, p_dept=float(has_dept), p_noregion=float(max(0.0, 1 - has_dept - has_region)),
                   street_abbr=STREET_ABBR_FR)
        print('France: region->departments', r2d, f'p_dept={has_dept:.3f} p_region={has_region:.3f}', flush=True)
    print('legal forms', legal_forms, '| generic words', generic[:15], flush=True)

    # --- build the synthetic universe
    t_s1 = t_s1.iloc[rng.permutation(len(t_s1))[:args.max_s1]].reset_index(drop=True)
    half = uhash(np.arange(len(t_s1)), 91) < 0.5
    A, B = t_s1[half].reset_index(drop=True), t_s1[~half].reset_index(drop=True)
    recs = {1: [], 2: [], 3: []}
    truth = {}
    nid = [0]

    def new_id(k):
        nid[0] += 1
        return f'S{k}-{9000000 + nid[0]}'

    def variant(row):
        k = 2 if rng.random() < p_s2 else 3
        op = scripts[k][rng.integers(len(scripts[k]))]
        return k, apply_name(row['name'], op, rng, ctx), apply_addr(row['addr'], op, rng, ctx)

    for _, row in A.iterrows():
        sid = new_id(1)
        recs[1].append((sid, row['name'], row['addr']))
        kk = int(k_per_s1[rng.integers(len(k_per_s1))])
        ids = []
        for _ in range(kk):
            k, n_, a_ = variant(row)
            rid = new_id(k)
            recs[k].append((rid, n_, a_))
            ids.append(rid)
        truth[sid] = ids
    # decoys: variants of businesses that are not in S1 (half B), ~2.1 per universe S1 like test
    n_dec = int(round(len(A) * 2.1))
    for i in rng.integers(len(B), size=n_dec):
        k, n_, a_ = variant(B.iloc[i])
        recs[k].append((new_id(k), n_, a_))
    for k in (1, 2, 3):
        pd.DataFrame(recs[k], columns=['entity_id', 'business_name', 'business_address']).assign(country=args.country) \
            .sample(frac=1, random_state=k).to_csv(out / 'test' / f'test_source{k}.tsv', sep='\t', index=False)
    pd.DataFrame({'source1_entity_id': list(truth), 'matched_entity_ids': [','.join(v) for v in truth.values()]}) \
        .to_csv(out / 'test_truth.tsv', sep='\t', index=False)
    n23 = len(recs[2]) + len(recs[3])
    print(f'synthetic {args.country}: S1 {len(recs[1]):,}, S2 {len(recs[2]):,}, S3 {len(recs[3]):,}; '
          f'records/S1 {n23 / len(recs[1]):.2f}; decoy share {n_dec / n23:.3f}', flush=True)
    for k in (2, 3):
        print(f'  examples S{k}:', [r[1] + ' | ' + r[2] for r in recs[k][:4]], flush=True)


if __name__ == '__main__':
    main()
