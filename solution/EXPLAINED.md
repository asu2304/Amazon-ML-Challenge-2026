# Business Entity Resolution — everything we did, explained from zero

This document walks through the whole solution: the problem, the data, every technique, every version
(v1 → v2 → v2.1 → v3), the bugs we hit, and the numbers. It starts simple and goes deeper section by section.
If a word is unfamiliar, check the **Glossary** at the end.

---

## Part 1 — The problem in plain words

### 1.1 What is "entity resolution"?

Imagine three phone books printed by three different companies. They all list the same shops, but:

- one writes `Payne Enterprises, 3315 Fremont Street, Peoria, IL`
- another writes `PAYNE-ENRTPRMISES, 3315 FREMONT SAINT, PEORIA, IL` (typos, capitals)
- a third writes `पेन एंटरप्राइजेज` (the same name in Hindi script) with no address at all.

**Entity resolution (ER)** = figuring out which records, across sources, describe the *same real business*.
There is no shared ID column to join on — you must decide from the (noisy) name and address text.

### 1.2 The exact task

- **Source 1 (S1)** is a clean reference list: every business appears once.
- **Source 2 (S2)** and **Source 3 (S3)** are messy lists.
- For **every S1 business**, output the list of S2/S3 records that are the same business (possibly empty).

Output file (`matching_results.tsv`), one row per S1 record:

```
source1_entity_id   matched_entity_ids
S1-00001            S2-00047,S2-00193,S3-00812
S1-00002                                         <- empty = "no match" (a singleton)
```

### 1.3 How it is scored: F0.5, and why it punishes wrong merges

For each S1 business we compare our list with the true list:

- **Precision** = of the IDs we listed, how many are correct.
- **Recall** = of the true IDs, how many we listed.
- **F0.5** = a blend that weights precision **about 2× more** than recall:
  `F0.5 = 1.25·P·R / (0.25·P + R)`.

The final score is the **average of F0.5 over all S1 businesses** (a "macro average").

Special rule for businesses with no true match ("singletons"): predicting an empty list scores **1.0**,
predicting anything scores **0.0**.

Worked example with 4 true matches:

| What we predict | Precision | Recall | F0.5 |
|---|---|---|---|
| all 4 correct | 1.0 | 1.0 | **1.000** |
| 4 correct + 1 wrong | 0.80 | 1.0 | **0.833** |
| 3 correct, 1 missed | 1.0 | 0.75 | **0.938** |

A wrong extra ID costs ~2.7× more than a missed one. **Lesson that shaped everything: when in doubt, leave it out.**

### 1.4 The "classification" way to think about it (your framing)

- Data points = every S2/S3 record (~10 million in test).
- Classes = every S1 business, **plus one extra class "∅ = matches nothing"**.
- Each record gets exactly one class.

The score is then the per-class F0.5, averaged over S1 classes. Two twists vs a textbook classifier:
there are ~1.7 million classes, and the **test classes are new businesses never seen in training** — so you
cannot train a normal "softmax over classes". Instead we learn a **scoring function**
`score(record, S1 candidate)` and pick the best candidate (or ∅) per record.

---

## Part 2 — What the data taught us (EDA)

EDA = exploratory data analysis: looking at the data before modelling. Notebook: `01_eda.ipynb`.

| Fact | Number | Why it matters |
|---|---|---|
| Train size | S1 2.2M, S2 5.0M, S3 5.3M | Big: every method must scale to millions |
| Test size | S1 1.73M, S2 4.9M, S3 5.1M | Same scale |
| Countries | train: US, India. **Test adds France** | The model must work on a country it never saw |
| S1 with no match (singletons) | 5.6% | Each worth a full 1.0 if we predict "nothing" |
| S2/S3 records matching **no** S1 ("decoys") | 26% in train | We need a "∅ / none" option |
| S2/S3 records matching **more than one** S1 | **0** out of 7.6M | Hard rule: each record → at most one S1 |
| Matches in different countries | 0% | Only compare within the same country |
| S1 names shared with another S1 | 39% | Name alone can't decide; address must |
| True matches sharing **zero** name words | 14.5% | Other scripts, trade names, domains |
| Non-Latin scripts | 9 Indian scripts (Hindi/Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam, Odia, Punjabi), only in India S2/S3 | Need transliteration or a multilingual model |
| France | French in Latin script with accents; **no postcodes**; region vs department names | Different address habits |

Typical noise (all deliberately injected by the data generator): typos (`Payne Etrepndiels`), random
capitals, junk prefixes (`-- `, `# 8 `), fake accents (`Énterprises`), names written as web domains
(`maurewilliamscolombier.com`), shuffled address parts, `null` tokens, `TX` vs `Texas`, zero-padded numbers
(`AF-0684`).

---

## Part 3 — The general recipe every serious ER system follows

```
all records ──► 1. NORMALIZE ──► 2. BLOCK (find a few candidates per record)
            ──► 3. SCORE each (record, candidate) pair with a model
            ──► 4. DECIDE which links to output (thresholds + rules)
```

Why "block" first? Comparing every S2/S3 record with every S1 record is 10M × 1.7M ≈ 17 trillion pairs —
impossible. Blocking cheaply shortlists ~15 plausible S1 candidates per record, so the model only scores
~150M pairs. **Blocking recall** (the share of true matches that survive into the shortlist) is the ceiling:
a match that is not shortlisted can never be found.

---

## Part 4 — v1: the low-compute baseline (CPU only) → leaderboard **0.967**

Code: `src/run.py` and the package `src/er/`.

### 4.1 Normalization (`er/normalize.py`)

Goal: make two spellings of the same thing look identical before comparing.

- **Transliteration** with the `anyascii` library: turns any script into Latin letters.
  `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்` → `raj investments`, `Fédération` → `Federation` (this also removes accents,
  called *accent folding*).
- Lowercase, strip junk prefixes, turn domains into words (`bluennov.com` → `bluennov`), join initials
  (`C.I.T.` → `cit`).
- Split off **legal forms** (Inc, LLC, Pvt, Ltd, SARL, SAS…) into their own field, so `Payne Enterprises`
  and `Payne Enterprises LLC` have the same "core name".
- Addresses: expand/unify abbreviations (street/st, avenue/av/ave, rue/r), drop `null`, strip zero padding,
  extract house numbers.
- Two small **dictionaries learned from the training labels** (no outside data):
  - transliterated word → English word (`praivet` → `private`, `phud` → `food`), 517 entries;
  - state spellings → one form (`texas` → `tx`, Tamil-script Tamil Nadu → `tamil nadu`), 310 entries.

### 4.2 Blocking with TF-IDF (`er/blocking.py`)

**TF-IDF** turns text into a vector of numbers where each dimension is a small piece of text (a word, or a
3-letter chunk like `pay`, `ayn`, `yne`), weighted so that **rare pieces count more** than common ones
("pediatric" counts less than "fremont"). **Cosine similarity** between two vectors measures how much text
they share (1 = identical, 0 = nothing in common).

For every S2/S3 record we took the **top 10 most similar S1 records of the same country** from three views:

| Channel | Text used | Catches |
|---|---|---|
| name | 3-letter chunks of the name | typos, abbreviations |
| addr | 3-letter chunks of the address | same address, different/transliterated name |
| combo | whole words of name + address | separates the 250 different "Pediatric Group" S1s by street |

The union, capped at the **15 best per record**, is the candidate list (this is `candidate_pairs.tsv`).
A fast sparse library (`sparse_dot_topn`) does the similarity search. Full-scale **blocking recall: 98.55%**
(India 97.7%, US 99.1%). Most misses are genuinely ambiguous (empty address + a generic name shared by many S1s).

### 4.3 Features (`er/features.py`)

For every candidate pair we compute ~50 numbers ("features") describing how similar the two records are:

- **String similarities** with the `rapidfuzz` library: edit-distance ratio, token-set ratio (ignores word
  order), partial ratio (one inside the other), Jaro-Winkler.
- **Address numbers**: how many house numbers overlap, is the first number equal.
- **Legal form**: equal, conflicting, or missing.
- **TF-IDF cosines** from the 3 channels.
- **Ambiguity**: how many S1 records share this name (a match on "Pediatric Group" means little).
- **Context / "listwise" features** — the most important ones: for this record, is this candidate the best
  one, and **how far ahead of the second best** (rank and gap)?
- Country is **never** a feature (so the model can't learn US/India-only habits that fail on France).

### 4.4 The model: LightGBM (`er/model.py`)

**LightGBM** is a *gradient-boosted decision tree* model: it builds hundreds of small decision trees, each
fixing the previous trees' mistakes. It is the standard strongest choice for **tabular data** (a table of
numeric features), handles missing values natively, and trains fast on CPU.

- One training row = one candidate pair; label 1 if it is the true match, else 0.
- **One single model** is trained on all pairs pooled together (not one model per record).
- **Out-of-fold (OOF) predictions**: we split businesses into 2 folds; the model trained on fold A predicts
  fold B and vice versa. Every business in all its records stays in one fold, so the scores we report are honest
  (the model never scores data it trained on).
- **Two stages**: stage 1 gives a probability `p1`; stage 2 adds features computed from other rows' `p1`
  (e.g. "does another record claim this S1 more confidently?") and re-predicts.

### 4.5 The decision layer (`er/decide.py`) — where F0.5 is won

1. **Assignment**: each S2/S3 record keeps **only its single highest-probability S1**, and only if the
   probability ≥ τ (a threshold). Otherwise it is assigned to nothing. This enforces "at most one S1".
2. **Per S1**: sort its assigned records by probability and keep the top-k that gives the highest
   **expected F0.5** (k = 0, "nothing", is allowed — this protects singletons).
3. τ is tuned by grid search on the OOF predictions using the exact metric (`er/metric.py`).

### 4.6 v1 results

- OOF F0.5 on train: **0.9797**. Leaderboard: **0.967**.
- Gap analysis pointed at two things: **France** (unseen) and **test has far more decoys** than train.

---

## Part 5 — v2: test-like validation + a transformer → leaderboard **0.984** (+1.7)

Code: `src/run_v2.py`, `src/er/crossenc.py`, `src/train_ce.py`.

### 5.1 Making validation look like test ("test-like universes")

We measured that test has **~2× more decoys per S1** than train (≈41% vs 26% of S2/S3 records), with the same
cluster sizes. A model tuned on train therefore over-matches on test.

Fix: split train businesses into two halves. Each half becomes its own "universe": half the S1 businesses,
their true matches, and **all** the decoys → **41.3% decoys**, like test. We train, validate and tune
thresholds inside these universes. (Technical trick: record *i* in universe *u* gets index *i + u·N*, so ranks,
groups and the metric never mix universes.)

### 5.2 The cross-encoder (the big win)

A **transformer** is a neural network that reads text as tokens and lets every token look at every other
token ("attention"). **multilingual-e5-base** is a pretrained transformer (278M parameters, MIT licence) that
already understands ~100 languages, including all the Indian scripts and French.

A **cross-encoder** feeds **both records together as one text**:

```
<s> Payne Énterprises | 3315 FREMONT ST, PEORIA, IL </s></s> Payne Enterprises | 3315 Fremont Street, Peoria, IL </s>
      └──────── 12 transformer layers, every token attends to both records ────────┘ → head → P(match)
```

so it can directly compare `Énterprises`↔`Enterprises`, `ST`↔`Street`, `ராஜ்`↔`Raj`.

- **Fine-tuning** = continuing training of the pretrained model on our labelled pairs. All 278M parameters
  plus a new output "head" are updated.
- Training data: 6M hard pairs (the top-5 candidates per record, so the wrong ones look similar) from
  **part A** = 25% of train businesses. The other 75% (**part B**) are never seen by the cross-encoder, so its
  scores on part B are honest.
- Settings: AdamW optimizer, learning rate 3e-5, 1 epoch, bf16 mixed precision, 2 GPUs.
- It scores only the **top-5 stage-1 candidates** of each record (99.9% of true matches are in the top 5).

### 5.3 Stacking + sibling features

**Stacking** = feeding one model's output into another model as a feature. The cross-encoder's probability
(and its rank/gap among the record's candidates) becomes a column for stage-2 LightGBM. The transformer's
internal 768-number vector is used *inside* the cross-encoder; LightGBM only sees the final probability.

**Sibling features**: if 3 other records already confidently matched an S1 and this record's address agrees
with theirs, that is evidence too — useful for records with no name or a trade name.

### 5.4 Bugs we found and fixed along the way

- **Correlated random splits**: my first hash `(x·a + salt) mod 2³²` barely changed with the salt, so "random"
  folds, universes and the part-A split were 82% correlated. Replaced with a proper hash (splitmix64).
- **Count features distorted in the universes** (each record keeps ~half its candidates there) → dropped
  `n_cand_q`, `n_cand_s`.
- Practical: other users on the shared GPUs → switched to the free ones; LightGBM thread over-subscription;
  output buffering; disk nearly full.

### 5.5 v2 results — the key comparison

Same records, test-like validation:

| Model | F0.5 |
|---|---|
| Stage-1 LightGBM (hand-made features) | 0.9821 |
| **Cross-encoder alone** | **0.9872** |
| Stage-2 stacked (LightGBM + cross-encoder + siblings + context) | **0.9903** |

The text model alone beat all the hand-made features (+0.5, and +0.85 on India where scripts are the hard
part). Stacking added +0.3 more. **Leaderboard: 0.984.** France predictions also started to look like the
other countries (fewer extra links).

### 5.6 Where the remaining errors are (error analysis on v2)

| Error type (part-B records) | India | US |
|---|---|---|
| decoy wrongly linked | 0.04% | 0.07% |
| linked to the wrong S1 | 0.04% | 0.04% |
| missed — true S1 never shortlisted (blocking) | **1.33%** | 0.51% |
| missed — shortlisted but rejected by the model | 0.54% | **0.85%** |

Precision is almost perfect; the remaining losses are **missed matches**, mostly India blocking misses.
Scoring more candidates would not help (99.9% of true matches are already in the top 5).

---

## Part 6 — v2.1: teaching the model French without labels (self-training)

Code: `src/adapt_france.py`, `--pseudo_france` in `run_v2.py`.

Leaderboard 0.984 vs validation 0.990 → the missing ~0.6 is most likely **France** (≈15% of test, never seen
in training; estimated ≈0.95).

**Self-training / pseudo-labelling** = using the model's own *very confident* predictions on unlabelled data
as if they were labels, then training on them. No external data is involved.

1. From v2's test predictions for France, take records the model is very sure about:
   600k confident matches (p > 0.97, big gap to the runner-up) and 1.8M confident non-matches.
2. Continue fine-tuning the cross-encoder on these, mixed 50/50 with original training pairs (so it does not
   forget US/India), at a lower learning rate (1e-5).
3. Re-score France test pairs, re-run stage 2, and optionally refit stage 2 with France pseudo-labels.

Outputs: `output_v2_1/` (adapted cross-encoder) and `output_v2_1_pseudo/` (plus the pseudo-label refit).
Risk: the model can reinforce its own mistakes, so only very confident pairs are used; the leaderboard decides.

**Result: leaderboard 0.984 — no change.** The adapted model changed 12% of France's lists (more links: 3.35 → 3.43
per S1) but the net effect was ~0; it mostly re-learned what it already believed. The pseudo-label LightGBM refit was
stopped (it needed 344 GB of RAM on a shared machine and had no expected gain). Lesson: self-training is not a lever
here; the remaining loss is elsewhere (see the error analysis in 5.6 → missed matches).

---

## Part 7 — v3: your Siamese idea becomes a second "search engine" for candidates

Code: `src/er/biencoder.py`, `src/train_bi.py`, `src/v3_retrieve.py`, and `--extra` in `src/run_v2.py`.

### 7.1 Why v3 exists (the problem it attacks)

The v2 error analysis (5.6) showed that almost all remaining errors are **missed matches**, and the single biggest
bucket was **India records whose true S1 was never in the shortlist** (1.33% of India records). TF-IDF blocking
compares *spelling* (3-letter chunks). It cannot connect:

- `तिरुपति लॉजिस्टिक्स` (Hindi) with `Tirupati Logistics` when transliteration is imperfect,
- a truncated address `106, MUMBAI, MH` with `3106, Marathon Futurex, N.M. Joshi Marg, Lower Parel, Mumbai`,
- trade names or domains with the registered name.

If the right S1 is not in the shortlist, no model downstream can ever pick it. So v3 changes **only candidate
generation** and keeps everything else from v2, so the effect is measurable.

### 7.2 The bi-encoder (Siamese network), intuitively

Think of it as a **learned "meaning" map**: every record becomes a point (a 768-number vector) such that
records of the same business land close together, whatever the script or spelling.

```
   "Tirupati Logistics Ltd | 106, Mumbai, MH"      ─┐
                                                   ├─►  [ same encoder: multilingual-e5-base ]  ─►  vector u  ─┐
   "तिरुपति लॉजिस्टिक्स | 3106 Marathon ..., Mumbai" ─┘                                                  vector v  ─┴─► cosine(u, v) ≈ 0.95
        (Siamese = the SAME network, same weights, applied to each record separately)
```

- **Model**: `intfloat/multilingual-e5-base` (278M parameters, MIT licence), mean-pooled and L2-normalised, so
  **cosine = dot product**.
- **Input text**: the raw `name | address` of each record (the model reads Indic scripts and French natively).
- **Training data**: **1.91M true pairs** from **part A** (25% of train businesses; part B stays unseen, so its
  scores remain honest).
- **Training objective (contrastive / InfoNCE with in-batch negatives)**: take a batch of 512 true pairs, embed the
  512 records and 512 S1s, compute the **512 × 512 cosine table**. The diagonal = true pairs, the other 261,632 cells =
  non-matches. Loss = "in every row (and column) the true S1 must win". Duplicate S1s inside a batch are masked so
  they are not wrongly treated as negatives.

```
                 S1_1   S1_2   S1_3  ...  S1_512
   record_1     [0.93]  0.12   0.40        0.05      <- row softmax: push 0.93 up, the rest down
   record_2      0.10  [0.88]  0.07        0.31
   record_3      0.35   0.02  [0.91]       0.11
   ...
```
  1,863 steps × 2 GPUs, one epoch, ~11 minutes, bf16, lr 2e-5, temperature 0.05. It sees ~1 billion random
  non-matches — not the trillions of possible pairs (not needed: almost all are trivially easy).

### 7.3 Finding candidates with it (exact nearest neighbours on GPU)

Dense vectors have no zeros, so the TF-IDF "inverted index" trick does not apply. Instead: **brute force, but on a
GPU**.

```
 per country:
   S1 embeddings  ── one matrix on the GPU  (India: 883K × 768, fp16 ≈ 1.3 GB)
   records        ── in chunks of 2,048
        chunk (2048 × 768)  ·  S1ᵀ (768 × 883K)   = 1.8 billion cosines at once
        torch.topk(…, k=10) per row               = each record's 10 closest S1s
```
- Every pair is really compared: India ≈ 4.1M × 883K × 768 ≈ 2.8 × 10¹⁵ multiply-adds, done in **~95 s**; US ~5 min.
- A first attempt crashed with *CUDA out of memory* (chunks of 8,192 converted to float32 needed ~40 GB); fixed by
  2,048-row chunks and top-k directly in fp16.
- Embeddings are written by the GPU workers into a **shared-memory** array (no 37 GB file on the nearly full disk).

### 7.4 How much it helps the shortlist (measured on part B)

| Shortlist recall | India | US | Overall |
|---|---|---|---|
| TF-IDF only (v1/v2) | 97.70% | 99.12% | 98.55% |
| **TF-IDF + bi-encoder (v3)** | **99.45%** | **99.42%** | **99.43%** |

India's missing share fell from 2.30% to 0.55% (about 3/4 of its blocking misses recovered).
Cost: the bi-encoder adds **~8 new candidates per record** (80.9M new train pairs, 79.2M new test pairs), so the
candidate lists grow from ~15 to **~23** per record (train 236M pairs, test 229M).

### 7.5 The full v3 pipeline

```
                          ┌───────────────────── BLOCKING (candidate generation) ─────────────────────┐
 S2/S3 record ──► TF-IDF name / addr / combo  → top-10 each → union → best 15   (same as v2)            │
             └──► bi-encoder embedding        → top-10 by cosine → keep the ones TF-IDF missed (~8) ◄── NEW
                          └──────────────────────────────── ~23 candidates per record ─────────────────┘
                                                   │
   FEATURES per pair (one row of scalars)          ▼
     TF-IDF cosines (recomputed for new pairs with the SAME fitted vectorizers)
     ~50 hand-built similarities (rapidfuzz, house numbers, legal form, ambiguity, …)
     cos_bi  = bi-encoder cosine        ◄── NEW (for every pair)
     bi_only = found only by bi-encoder ◄── NEW
     rank / gap context among the record's (and the S1's) candidates
                                                   │
   STAGE 1  LightGBM (out-of-fold, test-like universes) ─► p1
                                                   │
   CROSS-ENCODER on each record's top-5 by p1
     • same v2 model (ce.pt)
     • scores looked up by (record, S1) key in v2's cache  ◄── NEW (reuse)
     • only never-scored pairs go to the GPU (28.4M train, 25.2M test)
                                                   │
   STAGE 2  LightGBM (+ p1 rank/gap, ce + rank/gap, sibling agreement) ─► p2
                                                   │
   DECISION  each record → its best S1 if p2 ≥ τ  │  per S1 → best expected-F0.5 prefix
                                                   ▼
                                     output_v3/matching_results.tsv
```

Intricate details worth knowing:
- **Nothing from v2 is dropped** — v3's candidates are a superset (TF-IDF's 15 + bi-encoder's new ones).
- New pairs get **exactly the same features** as old ones: the TF-IDF vectorizers are re-fitted on the same texts in
  the same order, so their cosines are identical to what blocking would have produced (`blocking.pair_cosines`).
- The **test-like universes** (41.3% decoys, same random split as v2) are rebuilt on the enlarged set: 152.5M
  universe rows (part A 38.1M, part B 114.4M).
- Fewer positives per row (5.0% vs 7.4% in v2) because each record now has more wrong candidates.
- `cos_bi` gets **no rank/gap context** in v3 (added in v3.1).
- France keeps v2's original cross-encoder scores (the adaptation of v2.1 showed no gain).

### 7.6 v3 results so far

| Stage 1, test-like validation (part B) | Overall | India | US |
|---|---|---|---|
| v2 | 0.9821 | 0.9774 | 0.9853 |
| **v3** | **0.9857** | **0.9847** | **0.9863** |

+0.36 overall, +0.73 on India — exactly where the recall was recovered. Stage 2 and the leaderboard score:
pending (`output_v3/`).

---

## Part 8 — v3.1 ("v4"): make the scorers fit the new candidates, sharpen the decision

Code: `src/v31_ce.py` and `--ce2`, `--per_country_tau` in `src/run_v2.py`.

### 8.1 What is still sub-optimal in v3

1. **The cross-encoder never trained on bi-encoder candidates.** v2's cross-encoder learned from TF-IDF candidates
   only, yet in v3 **28M train / 25M test top-5 pairs are bi-encoder candidates** — a different kind of "hard wrong
   answer" (semantically close, spelled differently).
2. **`cos_bi` has no context.** Rank/gap features ("best of the record's candidates, by how much?") were among the
   most useful for every other similarity, but not computed for `cos_bi`.
3. **One threshold for everyone.** τ and the decision mode are tuned on the pooled US+India validation; the two
   countries have different ambiguity levels.

### 8.2 The three changes

```
 (1) CROSS-ENCODER 2 — trained on exactly the decisions it will face
     part-A records ─► their top-5 candidates ranked by v3's STAGE-1 model (incl. bi-encoder ones) + true S1
                     ─► 4M pairs (11.7% positive)
     start from v2's ce.pt ─► continue fine-tuning (lr 2e-5, 1 epoch, GPU 0) ─► ce2.pt
     score the same top-5 rows of part B and test with ce2
     ce2, ce2_rank_q, ce2_gap_q, ce2_max_q  ─► NEW stage-2 features (the old ce stays too → ensemble)

 (2) cos_bi CONTEXT
     cos_bi_rank_q / cos_bi_gap_q (per record) and cos_bi_rank_s / cos_bi_gap_s (per S1) ─► stage-2 features

 (3) PER-COUNTRY DECISION
     tune (mode, τ) separately on US and on India OOF predictions;
     France (no labels) uses the pooled setting
```

Why keep **both** cross-encoders as features: they make partly different mistakes (different training data), and
LightGBM learns how much to trust each — a cheap ensemble.

### 8.3 How v3.1 runs (reusing everything)

```
 v3 caches (candidates, features, p1, cross-encoder scores) ──┐
 ce2.pt ──► score top-5 rows (GPUs 0+3) ──────────────────────┼─► stage-2 features (+ce2, +cos_bi context)
                                                              └─► stage-2 LightGBM ─► per-country decision ─► output_v3_1/
```
Stage 1 is not retrained (its predictions are cached); only the parts that change are recomputed.

Expected effect: small but real — better scoring of the new bi-encoder candidates (where v3 gained recall) and a
tighter decision per country. Results: pending.

---

## Part 9 — Scoreboard

| Version | What changed | Validation F0.5 (test-like) | Leaderboard |
|---|---|---|---|
| v1 | normalize + TF-IDF blocking + LightGBM + F0.5 decision (CPU) | 0.9821 | **0.967** |
| v2 | test-like universes + cross-encoder + stacking + siblings | 0.9903 | **0.984** |
| v2.1 | France self-training of the cross-encoder | 0.9903 (France not in validation) | **0.984** |
| v3 | + bi-encoder retrieval (recall 98.55% → 99.43%), `cos_bi` | stage 1: 0.9857 (v2: 0.9821); final pending | pending |
| v3.1 | + cross-encoder 2, `cos_bi` context, per-country τ | pending | pending |

---

## Part 10 — Map of the files

| Path | What it is |
|---|---|
| `01_eda.ipynb` | exploratory analysis |
| `src/run.py` | v1 pipeline (normalize → block → features → LightGBM → decide → outputs) |
| `src/run_v2.py` | v2 / v2.1 / v3 / v3.1 pipeline on top of v1 caches (`--extra`, `--ce_from`, `--ce2`, `--per_country_tau`) |
| `src/er/data.py` | loading TSVs, truth labels |
| `src/er/normalize.py` | text cleaning, transliteration, learned maps |
| `src/er/blocking.py` | TF-IDF candidate generation; `pair_cosines` for arbitrary pairs |
| `src/er/features.py` | pair and context features |
| `src/er/model.py` | LightGBM training with out-of-fold predictions |
| `src/er/decide.py` | assignment + expected-F0.5 decision |
| `src/er/metric.py` | exact macro F0.5 |
| `src/er/crossenc.py`, `src/train_ce.py` | cross-encoder (train / predict) |
| `src/adapt_france.py` | France self-training (v2.1) |
| `src/er/biencoder.py`, `src/train_bi.py`, `src/v3_retrieve.py` | bi-encoder retrieval (v3) |
| `src/v31_ce.py` | cross-encoder 2 training (v3.1) |
| `work/`, `work_v2/`, `work_v3/`, `work_v31/` | cached intermediate results |
| `output/`, `output_v2/`, `output_v2_1/`, `output_v3/`, `output_v3_1/` | submission files per version |

---

## Glossary

- **Accent folding** — removing accents: é → e.
- **Bi-encoder / Siamese network** — encodes each item separately into a vector; compare vectors.
- **Blocking** — cheap shortlisting of candidates so the expensive model scores few pairs.
- **Blocking recall** — share of true matches that survive blocking.
- **Contrastive learning** — training vectors so true pairs are close and others are far.
- **Cosine similarity** — how aligned two vectors are (1 same, 0 unrelated).
- **Cross-encoder** — reads both items together in one transformer pass; very accurate, costly.
- **Decoy** — an S2/S3 record that matches no S1.
- **Embedding** — a vector of numbers representing an item.
- **F0.5** — score weighting precision ~2× more than recall.
- **Fine-tuning** — further training a pretrained model on your task.
- **fp16 / bf16** — 16-bit number formats; halve memory and speed up GPU maths.
- **In-batch negatives** — in contrastive training, the other items of the same batch serve as the wrong answers.
- **InfoNCE loss** — "pick the right partner out of the batch" loss used to train the bi-encoder.
- **Gradient-boosted trees / LightGBM** — many small decision trees added up; best for tabular features.
- **L2-normalised** — vector scaled to length 1, so cosine similarity equals the dot product.
- **Macro average** — compute the score per S1, then average.
- **Nearest-neighbour (kNN) search** — for each item, find the k most similar items.
- **Mean pooling** — averaging the transformer's token vectors into one vector for the whole text.
- **Out-of-fold (OOF)** — predictions made by a model that never trained on those rows.
- **Pseudo-labelling / self-training** — training on your own confident predictions.
- **Singleton** — an S1 with no true match.
- **Shortlist / candidate recall** — share of true matches present in the candidate lists.
- **Stacking** — using one model's output as a feature for another.
- **TF-IDF** — text → vector, rare pieces weigh more.
- **Transliteration** — writing one script in another's letters (Hindi → Latin).
- **Threshold τ** — minimum probability to accept a link.
