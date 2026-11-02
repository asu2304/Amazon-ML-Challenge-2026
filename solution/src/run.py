"""End-to-end pipeline: normalize -> block -> features -> 2-stage LightGBM -> F0.5 decision -> outputs.

    python src/run.py --data ../dataset --work work --out output
    python src/run.py --data ../dataset --work work_dev --out output_dev --dev_frac 0.1   # quick dev run

Every stage is cached under --work; delete a file there to recompute that stage.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from er import blocking, data, decide, features, metric, model, normalize  # noqa: E402

T0 = time.time()


def log(msg):
    print(f'[{time.time() - T0:7.0f}s] {msg}', flush=True)


def cached(path, build, save, load):
    if path.exists():
        log(f'load {path.name}')
        return load(path)
    obj = build()
    save(obj, path)
    return obj


def pq_save(df, p):
    df.to_parquet(p, index=False)


def stage_prepare(args, work):
    rec_tr = data.load_split(args.data, 'train')
    true_s1 = data.load_truth(args.data, rec_tr)
    if args.dev_frac < 1:
        rec_tr, true_s1 = data.subsample_universe(rec_tr, true_s1, args.dev_frac)
        log(f'dev universe: {len(rec_tr):,} train records')
    rec_te = data.load_split(args.data, 'test') if not args.skip_test else rec_tr.iloc[:0]
    maps_p = work / 'maps.json'
    if maps_p.exists():
        m = json.loads(maps_p.read_text())
        tok_map = m['tok']
        part_map = {tuple(k.split('\t')): v for k, v in m['part'].items()}
    else:
        log('learning name-token map and address-part map from train labels')
        tok_map = normalize.learn_name_token_map(rec_tr, true_s1)
        part_map = normalize.learn_addr_part_map(rec_tr, true_s1)
        maps_p.write_text(json.dumps({'tok': tok_map, 'part': {'\t'.join(k): v for k, v in part_map.items()}},
                                     ensure_ascii=False, indent=0))
    log(f'token map: {len(tok_map)} entries, part map: {len(part_map)} entries')
    out = {}
    for split, rec in [('train', rec_tr), ('test', rec_te)][:1 if args.skip_test else 2]:
        def build(rec=rec):
            log(f'normalizing {split} ({len(rec):,} records)')
            norm, nums = normalize.normalize_records(rec, tok_map, part_map, n_jobs=args.jobs)
            np.save(work / f'{split}_nums.npy', nums)
            return norm
        norm = cached(work / f'{split}_norm.parquet', build, pq_save, pd.read_parquet)
        nums = np.load(work / f'{split}_nums.npy')
        out[split] = (rec, norm, nums)
    return out, true_s1


def stage_block(args, work, split, rec, norm):
    def build():
        log(f'blocking {split}')
        return blocking.block(rec, norm, k=args.k, max_df=args.max_df, max_cand=args.max_cand, n_threads=args.jobs, log=log)
    return cached(work / f'{split}_pairs.parquet', build, pq_save, pd.read_parquet)


def stage_features(args, work, split, rec, norm, nums, pairs):
    def build():
        log(f'features {split} ({len(pairs):,} pairs)')
        return features.pair_features(rec, norm, nums, pairs, log=log)
    return cached(work / f'{split}_feats.parquet', build, pq_save, pd.read_parquet)


def tune_decision(q, s, p, true_s1, rec):
    """Coarse grid over tau for both modes, then refine around the best."""
    def score(mode, tau):
        pq_, ps_ = decide.decide(q, s, p, tau, mode)
        return metric.fbeta_macro(pq_, ps_, true_s1, rec), mode, tau

    best = max(score(m, t) for m in ['assign', 'expf'] for t in [0.3, 0.5, 0.7, 0.9])
    fine = [round(best[2] + d, 3) for d in (-0.15, -0.1, -0.05, 0.05, 0.1) if 0.05 <= best[2] + d <= 0.98]
    return max([best] + [score(best[1], t) for t in fine])


def write_outputs(out_dir, rec, pairs, pred_q, pred_s):
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = rec.id.to_numpy()
    s1 = np.flatnonzero(rec.src.to_numpy() == 1)

    def lists(q, s):
        df = pd.DataFrame({'s': s, 'id': ids[q]}).sort_values(['s', 'id'])
        agg = df.groupby('s').id.agg(','.join)
        return agg.reindex(s1).fillna('').to_numpy()

    pd.DataFrame({'source1_entity_id': ids[s1], 'matched_entity_ids': lists(pred_q, pred_s)}) \
        .to_csv(out_dir / 'matching_results.tsv', sep='\t', index=False)
    pd.DataFrame({'source1_entity_id': ids[s1],
                  'candidate_entity_ids': lists(pairs.q.to_numpy(), pairs.s.to_numpy())}) \
        .to_csv(out_dir / 'candidate_pairs.tsv', sep='\t', index=False)
    log(f'wrote {out_dir}/matching_results.tsv and candidate_pairs.tsv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../dataset')
    ap.add_argument('--work', default='work')
    ap.add_argument('--out', default='output')
    ap.add_argument('--dev_frac', type=float, default=1.0, help='<1: subsample train universe for quick runs')
    ap.add_argument('--k', type=int, default=10, help='top-K per blocking channel')
    ap.add_argument('--max_df', type=float, default=0.05)
    ap.add_argument('--max_cand', type=int, default=15, help='max candidates kept per S2/S3 record')
    ap.add_argument('--fit_frac', type=float, default=0.3, help='fraction of train queries used to fit each fold')
    ap.add_argument('--jobs', type=int, default=40)
    ap.add_argument('--skip_test', action='store_true', help='dev only: evaluate on train OOF, no test outputs')
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    model.PARAMS['num_threads'] = args.jobs

    splits, true_s1 = stage_prepare(args, work)
    rec_tr, norm_tr, nums_tr = splits['train']
    rec_te, norm_te, nums_te = splits.get('test', splits['train'])

    pairs_tr = stage_block(args, work, 'train', rec_tr, norm_tr)
    r, by = blocking.blocking_recall(pairs_tr, rec_tr, true_s1)
    log(f'train blocking recall={r:.4f} {by}  pairs/query={len(pairs_tr) / (rec_tr.src != 1).sum():.1f}')
    F_tr = stage_features(args, work, 'train', rec_tr, norm_tr, nums_tr, pairs_tr)
    if args.skip_test:  # reuse a small slice of train as a stand-in so the code path stays identical
        pairs_te, F_te = pairs_tr.iloc[:1000], F_tr.iloc[:1000]
    else:
        pairs_te = stage_block(args, work, 'test', rec_te, norm_te)
        F_te = stage_features(args, work, 'test', rec_te, norm_te, nums_te, pairs_te)

    q_tr, s_tr = pairs_tr.q.to_numpy(), pairs_tr.s.to_numpy()
    q_te, s_te = pairs_te.q.to_numpy(), pairs_te.s.to_numpy()
    y = (true_s1[q_tr] == s_tr).astype(np.int8)
    folds = model.query_folds(q_tr, true_s1)

    log('stage 1')
    p1_tr, p1_te = model.oof_and_test(F_tr, y, folds, F_te, q_tr, fit_frac=args.fit_frac, log=log)
    f, mode, tau = tune_decision(q_tr, s_tr, p1_tr, true_s1, rec_tr)
    log(f'stage 1 OOF F0.5={f:.4f} (mode={mode}, tau={tau})')

    log('stage 2 (listwise features from stage-1 probabilities)')
    F2_tr = features.add_stage2(F_tr, q_tr, s_tr, p1_tr)
    F2_te = features.add_stage2(F_te, q_te, s_te, p1_te)
    p2_tr, p2_te = model.oof_and_test(F2_tr, y, folds, F2_te, q_tr, fit_frac=args.fit_frac, seed=1, log=log)
    f, mode, tau = tune_decision(q_tr, s_tr, p2_tr, true_s1, rec_tr)
    pq_, ps_ = decide.decide(q_tr, s_tr, p2_tr, tau, mode)
    f, by = metric.fbeta_macro(pq_, ps_, true_s1, rec_tr, by_country=True)
    log(f'stage 2 OOF F0.5={f:.4f} {by} (mode={mode}, tau={tau})')

    (work / 'run_summary.json').write_text(json.dumps(
        {'oof_f05': float(f), 'oof_by_country': by, 'mode': mode, 'tau': float(tau),
         'blocking_recall_train': float(r), 'args': vars(args)}, indent=2))
    if args.skip_test:
        return
    np.save(work / 'test_p2.npy', p2_te)
    pred_q, pred_s = decide.decide(q_te, s_te, p2_te, tau, mode)
    write_outputs(Path(args.out), rec_te, pairs_te, pred_q, pred_s)


if __name__ == '__main__':
    main()
