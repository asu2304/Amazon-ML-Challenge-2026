import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 01 — Exploratory Data Analysis: Business Entity Resolution

Goal: understand the structure of the three sources, the ground truth, and the noise patterns
so we can design normalization, blocking, and the matching model.

Sections
1. Load & basic shapes
2. Missingness & country distribution
3. Ground-truth structure (cluster sizes, singletons, one-S1-per-record)
4. Scripts / transliteration
5. Name noise (legal suffixes, junk prefixes, domains)
6. Address format (components, numbers, PIN/ZIP, `null` tokens)
7. Pair-level similarity: positives vs random vs hard negatives
8. Test set & the unseen country (France)
9. Takeaways
""")

code("""
import re, unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_colwidth', 120)
pd.set_option('display.width', 200)
sns.set_theme(style='whitegrid')

DATA = Path('../dataset')

def read(path):
    return pd.read_csv(path, sep='\\t', keep_default_na=False, dtype=str, quoting=3)
""")

md("## 1. Load & basic shapes")
code("""
tr = {i: read(DATA / f'train/train_source{i}.tsv') for i in (1, 2, 3)}
te = {i: read(DATA / f'test/test_source{i}.tsv') for i in (1, 2, 3)}
gt = read(DATA / 'train/train_ground_truth.tsv')

pd.DataFrame({
    f'{split}_S{i}': [len(d[i]), d[i].entity_id.nunique()]
    for split, d in [('train', tr), ('test', te)] for i in (1, 2, 3)
}, index=['rows', 'unique_ids']).T
""")
code("""
for i in (1, 2, 3):
    print(f'--- train S{i}')
    display(tr[i].sample(5, random_state=1))
""")

md("## 2. Missingness & country distribution")
code("""
rows = []
for split, d in [('train', tr), ('test', te)]:
    for i in (1, 2, 3):
        df = d[i]
        rows.append({
            'split': split, 'source': f'S{i}',
            'empty_name': (df.business_name.str.strip() == '').mean(),
            'empty_addr': (df.business_address.str.strip() == '').mean(),
            'addr_has_null_token': df.business_address.str.contains(r'\\bnull\\b', case=False).mean(),
        })
pd.DataFrame(rows).style.format({c: '{:.2%}' for c in ['empty_name', 'empty_addr', 'addr_has_null_token']})
""")
code("""
cc = pd.concat([
    d[i].country.value_counts().rename(f'{split}_S{i}')
    for split, d in [('train', tr), ('test', te)] for i in (1, 2, 3)
], axis=1).fillna(0).astype(int)
display(cc)
(cc / cc.sum()).T.plot.barh(stacked=True, figsize=(8, 4), title='Country share per file');
""")

md("## 3. Ground-truth structure")
code("""
gt['ids'] = gt.matched_entity_ids.map(lambda s: s.split(',') if s else [])
gt['n'] = gt.ids.map(len)
gt['n_s2'] = gt.ids.map(lambda l: sum(x.startswith('S2') for x in l))
gt['n_s3'] = gt.ids.map(lambda l: sum(x.startswith('S3') for x in l))
gt = gt.merge(tr[1][['entity_id', 'country']], left_on='source1_entity_id', right_on='entity_id').drop(columns='entity_id')

print('S1 entities:', len(gt), '| in S1 file:', len(tr[1]), '| all S1 covered:', set(gt.source1_entity_id) == set(tr[1].entity_id))
print('singleton rate:', (gt.n == 0).mean().round(4))
display(gt.groupby('country').agg(entities=('n', 'size'), singleton_rate=('n', lambda x: (x == 0).mean()),
                                  mean_matches=('n', 'mean'), mean_s2=('n_s2', 'mean'), mean_s3=('n_s3', 'mean')))

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
gt.n.clip(upper=12).value_counts().sort_index().plot.bar(ax=ax[0], title='# matches per S1 (clipped at 12)')
gt.n_s2.clip(upper=8).value_counts().sort_index().plot.bar(ax=ax[1], title='# S2 matches per S1')
gt.n_s3.clip(upper=8).value_counts().sort_index().plot.bar(ax=ax[2], title='# S3 matches per S1');
""")
code("""
pairs = gt[['source1_entity_id', 'ids', 'country']].explode('ids').dropna(subset=['ids']).rename(columns={'ids': 'other_id'})
print('positive pairs:', len(pairs))

# Is every S2/S3 record linked to at most one S1?
per_rec = pairs.other_id.value_counts()
print('S2/S3 records linked to >1 S1:', int((per_rec > 1).sum()))

# Unmatched (decoy) S2/S3 records
matched = set(pairs.other_id)
for i in (2, 3):
    df = tr[i]
    m = df.entity_id.isin(matched)
    print(f'S{i}: unmatched fraction = {1 - m.mean():.2%}',
          df.assign(m=m).groupby('country').m.apply(lambda x: f'{1 - x.mean():.2%}').to_dict())
""")
code("""
# Lookup table for all train records + a 300k sample of positive pairs
rec = pd.concat([tr[1], tr[2], tr[3]]).set_index('entity_id')
P = pairs.sample(300_000, random_state=0).reset_index(drop=True)
L = rec.loc[P.source1_entity_id].reset_index()
R = rec.loc[P.other_id].reset_index()
print('matched pair same country:', (L.country.values == R.country.values).mean())

g_ids = gt.set_index('source1_entity_id').ids

def show_cluster(s1_id):
    return rec.loc[[s1_id] + g_ids[s1_id]]

for c in ['US', 'India']:
    for s in gt[(gt.country == c) & (gt.n >= 4)].sample(2, random_state=3).source1_entity_id:
        display(show_cluster(s))
""")

md("## 4. Scripts / transliteration")
code("""
def script_of(s):
    for ch in s:
        if ch.isalpha() and ord(ch) > 0x24F:
            return unicodedata.name(ch, 'UNKNOWN').split(' ')[0]
    return 'LATIN'

cols = {}
for split, d in [('train', tr), ('test', te)]:
    for i in (1, 2, 3):
        s = d[i].sample(min(200_000, len(d[i])), random_state=0)
        for col in ['business_name', 'business_address']:
            cols[f'{split}_S{i}_{col.split("_")[1]}'] = s[col].map(script_of).value_counts(normalize=True)
pd.DataFrame(cols).fillna(0).style.format('{:.2%}')
""")
code("""
# Non-Latin examples paired with their S1 (Latin) counterpart
nl = R.business_name.map(script_of) != 'LATIN'
pd.DataFrame({'s1_name': L.business_name[nl], 'other_name': R.business_name[nl],
              'other_addr': R.business_address[nl]}).sample(12, random_state=0)
""")

md("## 5. Name noise")
code("""
def toks(s):
    return re.findall(r'[a-z0-9]+', unicodedata.normalize('NFKD', s.lower()).encode('ascii', 'ignore').decode())

for c in ['US', 'India']:
    s = pd.concat([tr[i][tr[i].country == c] for i in (1, 2, 3)]).business_name.sample(300_000, random_state=0)
    print(c, 'top name tokens:', Counter(t for n in s for t in toks(n)).most_common(40), '\\n')

test_all = pd.concat([te[1], te[2], te[3]])
fr = test_all[test_all.country == 'France']
print('France top name tokens:', Counter(t for n in fr.business_name.sample(200_000, random_state=0) for t in toks(n)).most_common(50))
""")
code("""
# Junk leading characters, domains, casing
def name_flags(s):
    return pd.DataFrame({
        'junk_prefix': s.str.match(r'^\\s*[^\\w\\s]'),
        'domain': s.str.contains(r'\\.(?:com|in|net|org|co|fr|biz|info)\\b', case=False),
        'all_caps': s.str.isupper(),
        'all_lower': s.str.islower(),
        'has_&': s.str.contains('&', regex=False),
        'has_dba': s.str.contains(r'\\bdba\\b|d/b/a', case=False),
    }).mean()

pd.concat({f'{sp}_S{i}': name_flags(d[i].business_name.sample(300_000, random_state=0))
           for sp, d in [('train', tr), ('test', te)] for i in (1, 2, 3)}, axis=1).style.format('{:.2%}')
""")
code("""
s = pd.concat([tr[2], tr[3]]).business_name
print('junk-prefix examples:', s[s.str.match(r'^\\s*[^\\w\\s]')].sample(10, random_state=0).tolist())
print('domain examples:', s[s.str.contains(r'\\.(?:com|in|net|org)\\b')].sample(10, random_state=0).tolist())
""")
code("""
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
for i in (1, 2, 3):
    tr[i].business_name.str.len().clip(upper=80).plot.hist(bins=80, alpha=.5, ax=ax[0], label=f'S{i}', density=True)
    tr[i].business_address.str.len().clip(upper=160).plot.hist(bins=80, alpha=.5, ax=ax[1], label=f'S{i}', density=True)
ax[0].set_title('name length'); ax[1].set_title('address length'); ax[0].legend(); ax[1].legend();
""")

md("## 6. Address format")
code("""
def addr_stats(df):
    a = df.business_address
    return pd.DataFrame({
        'n_parts': a.str.count(',') + (a != '').astype(int),
        'has_digit': a.str.contains(r'\\d'),
        'has_pin6': a.str.contains(r'\\b\\d{6}\\b'),
        'has_zip5': a.str.contains(r'\\b\\d{5}\\b'),
        'has_near': a.str.contains(r'\\b(?:near|opp|opposite|behind)\\b', case=False),
        'upper': a.str.isupper(),
    })

rows = {}
for sp, d in [('train', tr), ('test', te)]:
    for i in (1, 2, 3):
        for c in d[i].country.unique():
            rows[(sp, f'S{i}', c)] = addr_stats(d[i][d[i].country == c].sample(100_000, random_state=0)).mean()
pd.DataFrame(rows).T.sort_index().style.format('{:.2f}')
""")
code("""
# Last comma part = usually the state; see how states are written
for c in ['US', 'India']:
    last = pd.concat([tr[i][tr[i].country == c] for i in (1, 2, 3)]).business_address.str.split(',').str[-1].str.strip()
    print(c, last.value_counts().head(30).to_dict(), '\\n')
""")
code("""
# Matched pairs: address format variation side by side
pd.DataFrame({'s1_addr': L.business_address, 'other_addr': R.business_address, 'country': L.country}).sample(15, random_state=2)
""")

md("""
## 7. Pair-level similarity: positives vs negatives

Cheap similarity measures on:
- **positives**: true pairs
- **random negatives**: a random same-country S2/S3 record
- **hard negatives**: an S2/S3 record belonging to a *different* S1 that has the *same normalized name*
""")
code("""
def norm_name(s):
    return ' '.join(toks(s))

def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))

def grams(s, n=3):
    s = f' {s} '
    return {s[i:i + n] for i in range(len(s) - n + 1)}

def pair_feats(Ld, Rd):
    ln, rn = Ld.business_name.map(norm_name), Rd.business_name.map(norm_name)
    la, ra = Ld.business_address.map(norm_name), Rd.business_address.map(norm_name)
    return pd.DataFrame({
        'name_tok_jacc': [jacc(a.split(), b.split()) for a, b in zip(ln, rn)],
        'name_3g_jacc': [jacc(grams(a), grams(b)) for a, b in zip(ln, rn)],
        'addr_tok_jacc': [jacc(a.split(), b.split()) for a, b in zip(la, ra)],
        'addr_num_overlap': [bool(set(re.findall(r'\\d+', a)) & set(re.findall(r'\\d+', b)))
                             for a, b in zip(Ld.business_address, Rd.business_address)],
        'other_addr_empty': (Rd.business_address.str.strip() == '').values,
        'other_non_latin': (Rd.business_name.map(script_of) != 'LATIN').values,
    })

N = 50_000
Lp, Rp = L.iloc[:N], R.iloc[:N]
pos = pair_feats(Lp, Rp)

others = pd.concat([tr[2], tr[3]])
Ln = pd.concat([Lp[Lp.country == c] for c in ['US', 'India']]).reset_index(drop=True)
Rn = pd.concat([others[others.country == c].sample((Lp.country == c).sum(), random_state=0)
                for c in ['US', 'India']]).reset_index(drop=True)
neg = pair_feats(Ln, Rn)
""")
code("""
# Hard negatives: S1 A paired with a true match of S1 B, where A and B share a normalized name
s1 = tr[1].copy()
s1['nn'] = s1.business_name.map(norm_name)
dup = s1[s1.nn.duplicated(keep=False)]
print('S1 rows whose normalized name is shared with another S1:', f'{len(dup) / len(s1):.2%}')
display(dup.nn.value_counts().head(10))

hard_L, hard_R = [], []
for nn, grp in dup.groupby('nn', sort=False):
    if len(hard_L) >= N:
        break
    ids = grp.entity_id.tolist()
    for a, b in zip(ids, ids[1:] + ids[:1]):
        if g_ids[b]:
            hard_L.append(a); hard_R.append(g_ids[b][0])
hard = pair_feats(rec.loc[hard_L].reset_index(), rec.loc[hard_R].reset_index())
print('hard negatives built:', len(hard))
""")
code("""
cmp = pd.concat({'positive': pos, 'random_neg': neg, 'hard_neg': hard}, names=['kind']).reset_index(level=0)
display(cmp.groupby('kind').mean().T.style.format('{:.3f}'))

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, col in zip(axes, ['name_tok_jacc', 'name_3g_jacc', 'addr_tok_jacc']):
    sns.histplot(cmp, x=col, hue='kind', bins=40, stat='density', common_norm=False, element='step', ax=ax)
    ax.set_title(col)
""")
code("""
# Positives that name-only blocking would miss
z = pos.name_tok_jacc == 0
print('positives with zero name-token overlap:', f'{z.mean():.2%}')
print('  ...of which other side is non-Latin:', f'{pos[z].other_non_latin.mean():.2%}')
print('  ...and also no shared address number:', f'{(z & ~pos.addr_num_overlap).mean():.2%}')
pd.DataFrame({'s1_name': Lp.business_name[z], 'other_name': Rp.business_name[z],
              's1_addr': Lp.business_address[z], 'other_addr': Rp.business_address[z]}).sample(15, random_state=0)
""")

md("## 8. Test set & the unseen country (France)")
code("""
for i in (1, 2, 3):
    print(f'--- test S{i} France')
    display(te[i][te[i].country == 'France'].sample(6, random_state=0))
""")
code("""
fr_addr = fr.business_address
display(fr_addr.str.split(',').str[-1].str.strip().value_counts().head(25))
print('France top address tokens:', Counter(t for a in fr_addr.sample(200_000, random_state=0) for t in toks(a)).most_common(40))
""")
code("""
# S2+S3 records per S1, train vs test
for sp, d in [('train', tr), ('test', te)]:
    for c in sorted(set(d[1].country)):
        n1 = (d[1].country == c).sum(); n23 = sum((d[i].country == c).sum() for i in (2, 3))
        print(f'{sp:5s} {c:7s} S1={n1:>9,}  S2+S3={n23:>10,}  ratio={n23 / n1:.2f}')
""")

md("""
## 9. Takeaways

- Each S2/S3 record links to **at most one** S1 → treat matching as an assignment: argmax S1 per record, or none.
- ~26% of S2/S3 records are **decoys** with no S1 match.
- Matches always share **country** → block within country (open set of labels; France appears only in test).
- Many S1 records share the same name → **address must disambiguate**; build hard negatives from same-name groups.
- ~15% of positives have zero name-token overlap (non-Latin scripts, trade names, domains) → need address-based and
  transliteration / embedding-based blocking in addition to name blocking.
- Address noise: case, `null` tokens, component reordering, state abbreviations vs full names vs native script,
  zero-padded numbers, typos in street/city names.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
nbf.write(nb, '01_eda.ipynb')
print('written')
