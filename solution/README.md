# Business Entity Resolution — low-compute pipeline

CPU-only pipeline (no GPU, no pretrained model, no external data):
normalize → TF-IDF blocking → pair features → 2-stage LightGBM → F0.5-optimal decision.

## Setup

```bash
conda create -n er python=3.11 -y
conda activate er
pip install -r requirements.txt
```

## Run

```bash
# full run: trains on all of train, scores it out-of-fold, writes the test submission
python -u src/run.py --data ../dataset --work work --out output

# quick development run on 10% of the train universe (no test outputs)
python -u src/run.py --data ../dataset --work work_dev --dev_frac 0.1 --skip_test

# validate
python ../utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir ../dataset/test
```

`--data` must contain `train/` and `test/` as distributed. Every stage is cached in `--work`
(delete a file there to recompute that stage). Main options: `--k` (top-K per blocking channel, 10),
`--max_cand` (candidates kept per record, 15), `--fit_frac` (share of train records used to fit each
fold, 0.3), `--jobs` (CPU threads, 40).

## Pipeline

| Step | File | What it does |
|---|---|---|
| Load | `src/er/data.py` | All sources of a split in one table; `true_s1` label per record |
| Normalize | `src/er/normalize.py` | `anyascii` transliteration (9 Indic scripts, accents), junk prefixes, domains, initials, legal forms split out, address abbreviations, number extraction. Two maps learned from train labels: transliterated token → English token, state spelling → canonical |
| Block | `src/er/blocking.py` | Per country, per S2/S3 record: top-10 S1 from three TF-IDF channels (name char-3grams, address char-3grams, name+address words); union capped at 15 per record. This set is `candidate_pairs.tsv` |
| Features | `src/er/features.py` | rapidfuzz name/address similarities, address-number overlap, legal-form agreement, TF-IDF cosines, name ambiguity counts, listwise rank/gap context per record and per S1. Country is never a feature |
| Model | `src/er/model.py` | LightGBM binary classifier per candidate pair, 2 folds grouped by true cluster (out-of-fold). Stage 2 adds rank/gap/sum features of stage-1 probabilities |
| Decide | `src/er/decide.py` | Each S2/S3 record → its single best S1 if p ≥ τ; per S1 keep the prefix with maximal expected F0.5 (empty allowed). τ and mode tuned on out-of-fold F0.5 |
| Metric | `src/er/metric.py` | Exact macro F0.5 incl. singleton rule |

Outputs: `output/matching_results.tsv`, `output/candidate_pairs.tsv`; run summary in `work/run_summary.json`.
