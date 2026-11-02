"""Name / address normalization.

Everything is transliterated to ASCII with `anyascii` (covers the 9 Indic scripts and French accents),
lower-cased and tokenized. Two small maps are *learned from the training labels* (no external data):
  - name token map: transliterated token -> English token (e.g. 'praivet' -> 'private', 'elelpi' -> 'llp')
  - address part map: whole comma-separated part -> canonical part (e.g. 'texas' -> 'tx', 'mharastr' -> 'maharashtra')
"""
import re
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
from anyascii import anyascii

LEGAL = {
    'inc': 'inc', 'incorporated': 'inc', 'llc': 'llc', 'corp': 'corp', 'corporation': 'corp',
    'co': 'co', 'company': 'co', 'ltd': 'ltd', 'limited': 'ltd', 'pvt': 'pvt', 'private': 'pvt',
    'llp': 'llp', 'pllc': 'pllc', 'pc': 'pc', 'lp': 'lp', 'plc': 'plc', 'lllp': 'lllp',
    'gmbh': 'gmbh', 'ag': 'ag', 'aktiengesellschaft': 'ag',
    'sarl': 'sarl', 'sas': 'sas', 'sasu': 'sasu', 'eurl': 'eurl', 'sci': 'sci', 'sa': 'sa',
    'ei': 'ei', 'cie': 'cie', 'snc': 'snc',
}
NAME_STOP = {'the', 'and', 'of', 'dba', 'a', 'an', 'et', 'le', 'la', 'les', 'de', 'du', 'des', 'd', 'l'}

ADDR_ABBR = {
    'street': 'st', 'saint': 'st', 'road': 'rd', 'avenue': 'ave', 'av': 'ave', 'boulevard': 'blvd', 'bd': 'blvd',
    'drive': 'dr', 'lane': 'ln', 'court': 'ct', 'place': 'pl', 'square': 'sq', 'highway': 'hwy',
    'parkway': 'pkwy', 'terrace': 'ter', 'circle': 'cir', 'north': 'n', 'south': 's', 'east': 'e', 'west': 'w',
    'suite': 'unit', 'apt': 'unit', 'apartment': 'unit', 'appartement': 'unit', 'floor': 'fl', 'building': 'bldg',
    'rue': 'r', 'allee': 'all', 'impasse': 'imp', 'route': 'rte', 'chemin': 'ch', 'near': 'nr', 'opposite': 'opp',
    'sainte': 'ste',
}
ADDR_DROP = {'no', 'ndeg', 'null', 'th', 'nd'}

DOMAIN_RE = re.compile(r'(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:co\.in|com|in|net|org|co|fr|biz|info|io|us)\b')
TOKEN_RE = re.compile(r'[a-z0-9]+')
ADDR_TOKEN_RE = re.compile(r'\d+|[a-z]+')
LEAD_JUNK_RE = re.compile(r'^[^a-z0-9#]*(?:#\s*\d+\s+)?[^a-z0-9]*')
NUM_PREFIX_RE = re.compile(r'(?:^|(?<=\s))(?:n|no|num)\.?\s*(?=\d)')
NON_LATIN_RE = r'[^\x00-ɏ -⁯]'


def to_ascii(s):
    s = s.replace('°', ' ').replace('º', ' ').replace('\x92', "'").replace('’', "'")
    return anyascii(s).lower()


def collapse_initials(toks):
    """'c i t colony' -> 'cit colony', 'l l c' -> 'llc'."""
    out, buf = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            buf.append(t)
            continue
        if buf:
            out.append(''.join(buf))
            buf = []
        out.append(t)
    if buf:
        out.append(''.join(buf))
    return out


def name_tokens(raw, tok_map):
    s = to_ascii(raw)
    s = DOMAIN_RE.sub(lambda m: ' ' + m.group(1).replace('-', '') + ' ', s)
    s = LEAD_JUNK_RE.sub('', s).replace('&', ' and ')
    toks = collapse_initials(TOKEN_RE.findall(s))
    return [tok_map.get(t, t) for t in toks]


def addr_parts(raw):
    """Normalized comma-separated parts (before the learned part map)."""
    out = []
    for p in raw.split(','):
        toks = []
        for t in ADDR_TOKEN_RE.findall(NUM_PREFIX_RE.sub(' ', to_ascii(p).strip())):
            if t in ADDR_DROP:
                continue
            if t.isdigit():
                t = t.lstrip('0') or '0'
            toks.append(ADDR_ABBR.get(t, t))
        if toks:
            out.append(' '.join(toks))
    return out


# ---------------------------------------------------------------- learned maps

def learn_name_token_map(rec, true_s1, max_pairs=1_500_000, min_count=5, min_ratio=0.5):
    """Position-aligned token pairs between transliterated non-Latin S2/S3 names and their S1 name."""
    q = np.flatnonzero((true_s1 >= 0) & (rec.src.to_numpy() != 1))
    names = rec.name.to_numpy()
    nl = pd.Series(names[q]).str.contains(NON_LATIN_RE, regex=True).to_numpy()
    q = q[nl][:max_pairs]
    cnt, tot = Counter(), Counter()
    for qi in q:
        a, b = name_tokens(names[qi], {}), name_tokens(names[true_s1[qi]], {})
        if len(a) != len(b):
            continue
        for x, y in zip(a, b):
            tot[x] += 1
            if x != y:
                cnt[(x, y)] += 1
    best = {}
    for (x, y), c in cnt.items():
        if c >= min_count and c / tot[x] >= min_ratio and c > best.get(x, (None, 0))[1]:
            best[x] = (y, c)
    return {x: y for x, (y, _) in best.items()}


def learn_addr_part_map(rec, true_s1, max_pairs=2_000_000, min_count=30, min_ratio=0.9, seed=0):
    """Map a whole address part seen on the S2/S3 side to the S1 part it (almost) always co-occurs with.
    Catches state spellings: 'texas'->'tx', native-script state names -> English. Ambiguous parts are skipped."""
    rng = np.random.default_rng(seed)
    q = np.flatnonzero((true_s1 >= 0) & (rec.src.to_numpy() != 1))
    q = rng.choice(q, size=min(max_pairs, len(q)), replace=False)
    addrs, country = rec.addr.to_numpy(), rec.country.to_numpy()
    co, tot = defaultdict(Counter), Counter()
    for qi in q:
        rp, lp = addr_parts(addrs[qi]), addr_parts(addrs[true_s1[qi]])
        ls = set(lp)
        l_un = [x for x in lp if x not in set(rp)]
        for p in rp:
            if p in ls:
                continue
            key = (country[qi], p)
            tot[key] += 1
            for x in l_un:
                co[key][x] += 1
    out = {}
    for key, c in co.items():
        if tot[key] < min_count:
            continue
        (x1, c1), *rest = c.most_common(2)
        second = rest[0][1] if rest else 0
        if c1 / tot[key] >= min_ratio and second / tot[key] < 0.5 and not any(ch.isdigit() for ch in key[1]):
            out[key] = x1
    return out


# ---------------------------------------------------------------- apply

_MAPS = {}
_NON_LATIN = re.compile(NON_LATIN_RE)


def _init(tok_map, part_map, fr_maps=None):
    _MAPS['tok'], _MAPS['part'] = tok_map, part_map
    fr_maps = fr_maps or {}
    _MAPS['fr_admin'] = set(fr_maps.get('regions', [])) | set(fr_maps.get('departments', []))
    _MAPS['fr_abbr'] = fr_maps.get('abbr', {})


def _norm_chunk(args):
    names, addrs, countries = args
    tok_map, part_map = _MAPS['tok'], _MAPS['part']
    core, legal, nospace, addr, nums = [], [], [], [], []
    for n, a, c in zip(names, addrs, countries):
        toks = name_tokens(n, tok_map if _NON_LATIN.search(n) else {})
        lg = sorted({LEGAL[t] for t in toks if t in LEGAL})
        cr = [t for t in toks if t not in LEGAL and t not in NAME_STOP] or [t for t in toks if t not in NAME_STOP] or toks
        core.append(' '.join(cr))
        legal.append(' '.join(lg))
        nospace.append(''.join(cr))
        parts = [part_map.get((c, p), p) for p in addr_parts(a)]
        if c == 'France' and (_MAPS['fr_admin'] or _MAPS['fr_abbr']):
            # v4: drop region/department parts (redundant with the city), expand learned street abbreviations
            ab = _MAPS['fr_abbr']
            parts = [' '.join(ab.get(w, w) for w in p.split()) for p in parts if p not in _MAPS['fr_admin']]
        addr.append(' '.join(parts))
        seen = []
        for t in ' '.join(parts).split():
            if t.isdigit() and t not in seen:
                seen.append(t)
        nums.append([int(t[:18]) for t in seen[:6]])
    return core, legal, nospace, addr, nums


def normalize_records(rec, tok_map, part_map, n_jobs=48, chunk=50_000, fr_maps=None):
    n = len(rec)
    names, addrs, countries = rec.name.to_numpy(), rec.addr.to_numpy(), rec.country.to_numpy()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk], countries[i:i + chunk]) for i in range(0, n, chunk)]
    with Pool(n_jobs, initializer=_init, initargs=(tok_map, part_map, fr_maps)) as pool:
        res = pool.map(_norm_chunk, jobs)
    out = pd.DataFrame({
        'name_core': [x for r in res for x in r[0]],
        'legal': [x for r in res for x in r[1]],
        'name_nospace': [x for r in res for x in r[2]],
        'addr_norm': [x for r in res for x in r[3]],
    })
    nums = np.full((n, 6), -1, dtype=np.int64)
    i = 0
    for r in res:
        for lst in r[4]:
            nums[i, :len(lst)] = lst
            i += 1
    out['nonlatin'] = rec.name.str.contains(NON_LATIN_RE, regex=True).to_numpy().astype(np.int8)
    out['is_domain'] = rec.name.str.lower().str.contains(DOMAIN_RE.pattern.replace('([a-z0-9]', '(?:[a-z0-9]'), regex=True).to_numpy().astype(np.int8)
    return out, nums
