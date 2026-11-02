# Amazon ML Challenge 2026 — Business Entity Resolution

Solution for the **Amazon ML Challenge 2026**: given business records from three independent, noisy
sources (US, India and an unseen-at-train-time France), find which Source 2 / Source 3 records refer to
the same real-world business as each Source 1 record. Scored with macro **F0.5** (precision-heavy,
singletons included).

**Leaderboard: 0.984 F0.5** · validation (test-like split): up to **0.9928 F0.5**

## Dataset

The dataset is not included in this repo. Download it from Kaggle (private):

**https://www.kaggle.com/datasets/asu2304/amazon-ml-challange-2026**

Place it so the layout is:

```
dataset/
├── train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
└── test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
```

Scale: train S1 2.2M / S2 5.0M / S3 5.3M records; test S1 1.73M / S2 4.9M / S3 5.1M.

## Approach

```
raw records ─► normalize ─► block (TF-IDF + bi-encoder) ─► pair features
            ─► LightGBM stage 1 ─► cross-encoders on top-5 ─► LightGBM stage 2 ─► F0.5-optimal decision
```

1. **Normalization** — `anyascii` transliteration of 9 Indic scripts and accent folding, junk-prefix and
   domain cleanup, legal forms (Inc, Pvt Ltd, SARL…) split into their own field, address abbreviation
   unification, plus two small dictionaries learned only from the training labels.
2. **Blocking** — per country, top-10 S1 candidates from three TF-IDF channels (name char-3grams,
   address char-3grams, name+address words) via `sparse_dot_topn`, unioned with a fine-tuned
   **bi-encoder** (Siamese, multilingual-e5) retrieval channel. Blocking recall 98.55% → 99.43%.
3. **Features** — rapidfuzz name/address similarities, house-number overlap, legal-form agreement,
   TF-IDF / bi-encoder cosines, name ambiguity counts and listwise rank/gap context. Country is never a
   feature, so the model transfers to France.
4. **Models** — 2-stage LightGBM with out-of-fold training grouped by true cluster; two fine-tuned
   **multilingual-e5-base cross-encoders** (MIT, 278M params) score each record's top-5 candidates and
   are stacked into stage 2, along with "sibling" agreement features.
5. **Decision** — each S2/S3 record links to at most its single best S1 if p ≥ τ; per S1 keep the
   prefix with maximal expected F0.5 (empty allowed, protecting singletons). τ tuned on **test-like
   universes** that reproduce test's ~2× higher decoy rate.
6. **Later experiments** — France self-training (v2.1), sibling propagation (v5) and a synthetic
   "noise transfer" evaluation set for France (v6).

| Version | What changed | Validation F0.5 | Leaderboard |
|---|---|---|---|
| v1 | normalize + TF-IDF blocking + LightGBM + F0.5 decision (CPU only) | 0.9821 | 0.967 |
| v2 | test-like universes + cross-encoder + stacking + siblings | 0.9903 | 0.984 |
| v2.1 | France self-training of the cross-encoder | 0.9903 | 0.984 |
| v3 / v4 | bi-encoder retrieval, second cross-encoder, per-country τ | 0.9928 | — |

A full from-zero walkthrough of every version, bug and number is in
[solution/EXPLAINED.md](solution/EXPLAINED.md).

## Repository layout

| Path | Contents |
|---|---|
| [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md) | Official challenge description, formats and rules |
| [Documentation_template.md](Documentation_template.md) | Methodology template for the final submission |
| [solution/README.md](solution/README.md) | Setup and run commands for the pipeline |
| [solution/EXPLAINED.md](solution/EXPLAINED.md) | Detailed explanation of the whole solution |
| [solution/01_eda.ipynb](solution/01_eda.ipynb) | Exploratory data analysis |
| [solution/src/er/](solution/src/er) | Core package: data, normalize, blocking, features, model, crossenc, biencoder, decide, metric |
| [solution/src/run.py](solution/src/run.py) | v1 end-to-end pipeline (CPU) |
| [solution/src/run_v2.py](solution/src/run_v2.py) | v2 → v3.1 pipeline on top of v1 caches |
| `solution/src/v*_*.py` | v3–v6 stages (retrieval, France features, propagation, noise transfer, eval) |
| `solution/*.log` | Training / evaluation logs of each run |
| [utils/validate_submission.py](utils/validate_submission.py) | Official submission format validator |

Cached stages (`solution/work*/`), model weights and generated outputs (`solution/output*/`) are
git-ignored because of their size; the pipeline regenerates them.

## Quick start

```bash
cd solution
conda create -n er python=3.11 -y && conda activate er
pip install -r requirements.txt

# v1 full run (CPU): trains, scores out-of-fold, writes the test submission
python -u src/run.py --data ../dataset --work work --out output

# validate the submission format
python ../utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir ../dataset/test
```

The transformer stages (v2+) additionally need PyTorch with CUDA and `transformers`; see
[solution/EXPLAINED.md](solution/EXPLAINED.md) for the per-version commands.
