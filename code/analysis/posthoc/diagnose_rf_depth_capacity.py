"""Diagnostic: does the searched RF `max_depth` bind, and what does it cost?

Fits candidate RF configs on the TRAINING split only (pooled and per cancer type)
and measures generalisation out-of-bag, so the held-out test set is never used.
Caveats: pooled AUCs are not comparable to per-cancer AUCs; OOB ignores the
SUBJECT_CENTER grouping and is mildly optimistic; single seed, no CIs.

Usage:
  python diagnose_rf_depth_capacity.py --db-path <db.sqlite> \
      --split-file runs/<run>/tier2_data_setup/train_test_split/train_test_split.csv
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'code' / 'modeling'))
sys.path.insert(0, str(REPO / 'code' / 'tools'))

CANCER_TYPES = ['LUAD', 'IDC', 'BRCA', 'MEL', 'GIST', 'COAD', 'THPA', 'ILC', 'PRAD',
                'AML', 'BLCA', 'SKCM', 'NSCLC', 'ODG', 'HGSOC', 'ASTR', 'MDLC',
                'BRCNOS', 'BRCANOS', 'COADREAD', 'IHCH']

# Kept in sync with run_final_model_training.DEFAULT_RF_PARAMS and the TIER 3 consensus.
# Capacity knobs only; random_state / n_jobs / bootstrap are set below.
CONFIGS = {
    'hand-tuned (shipped)': dict(n_estimators=200, max_depth=5, max_features=0.3,
                                 min_samples_leaf=10, min_samples_split=20,
                                 class_weight='balanced_subsample'),
    'searched (TIER 3)': dict(n_estimators=100, max_depth=15, max_features=0.3,
                              min_samples_leaf=5, min_samples_split=100,
                              class_weight='balanced_subsample'),
    'searched, depth uncapped': dict(n_estimators=100, max_depth=None, max_features=0.3,
                                     min_samples_leaf=5, min_samples_split=100,
                                     class_weight='balanced_subsample'),
}


# Pre-specified candidate set and selection rule, fixed before any candidate was scored.
# Only capacity varies; max_features and class_weight agreed across all outer folds of the search.
# Selection rule:
#   primary     = highest mean per-cancer OOB AUC over all codes, 3 seeds
#   sensitivity = same mean excluding the unstable codes (ODG, MDLC)
#   tiebreak    = smaller mean (train - OOB) gap
# A candidate wins only if it leads on both primary and sensitivity; ties within 1 seed-SD (~0.003) are not leads.
# 'shipped' was tuned with reference to the held-out test set, so it is a baseline only and not adoptable.
CANDIDATE_CONFIGS = {
    # label: (params, provenance)
    'C1 shipped (d5)': (dict(n_estimators=200, max_depth=5, max_features=0.3,
                             min_samples_leaf=10, min_samples_split=20,
                             class_weight='balanced_subsample'),
                        'hand-tuned vs TEST SET (D-f) -- NOT adoptable'),
    'C2 searched-2026 (d15)': (dict(n_estimators=100, max_depth=15, max_features=0.3,
                                    min_samples_leaf=5, min_samples_split=100,
                                    class_weight='balanced_subsample'),
                               'nested CV 2026-08-26, pooled'),
    'C3 searched-Jan (d10)': (dict(n_estimators=100, max_depth=10, max_features=0.3,
                                   min_samples_leaf=4, min_samples_split=2,
                                   class_weight='balanced_subsample'),
                              'nested CV 2026-01, stale cohort'),
    'C4 searched-2026 @ d10': (dict(n_estimators=100, max_depth=10, max_features=0.3,
                                    min_samples_leaf=5, min_samples_split=100,
                                    class_weight='balanced_subsample'),
                               'C2 with depth stepped down within grid'),
    'C5 searched-2026 @ d7': (dict(n_estimators=100, max_depth=7, max_features=0.3,
                                   min_samples_leaf=5, min_samples_split=100,
                                   class_weight='balanced_subsample'),
                              'C2 with depth stepped down within grid'),
}

UNSTABLE_CODES = ['ODG', 'MDLC']  # fixed in advance; see selection rule above

# Extension pass, run after the pre-specified sweep and kept separate so that sweep stays intact.
# The best candidate sat at the edge of the tested depth range; this checks whether the trend continues. Enable with --extend.
EXTENSION_CONFIGS = {
    'C6 searched-2026 @ d5': (dict(n_estimators=100, max_depth=5, max_features=0.3,
                                   min_samples_leaf=5, min_samples_split=100,
                                   class_weight='balanced_subsample'),
                              'C2 with depth stepped down within grid'),
    'C7 searched-2026 @ d3': (dict(n_estimators=100, max_depth=3, max_features=0.3,
                                   min_samples_leaf=5, min_samples_split=100,
                                   class_weight='balanced_subsample'),
                              'C2 at the grid floor'),
}


def run_candidates(args):
    """Score the pre-specified candidate set per cancer type, training data only."""
    from oncokb_analyzer.preprocessing import create_preprocessor

    configs = dict(CANDIDATE_CONFIGS)
    if getattr(args, 'extend', False):
        configs.update(EXTENSION_CONFIGS)
    d = load_per_cancer_training(args.db_path, args.split_file, args.feature_set)
    seeds = [args.seed + i for i in range(args.n_seeds)]
    print(f"E2b candidate sweep: {len(configs)} configs x {len(d)} codes "
          f"x {args.n_seeds} seeds, alpha={args.institution_weight_alpha}\n")

    per_cancer = {}
    for code, val in d.items():
        X, y = val[0], np.asarray(val[1])
        if len(np.unique(y)) < 2:
            continue
        Xt = create_preprocessor(available_columns=X.columns.tolist()).fit_transform(X)
        if hasattr(Xt, 'toarray'):
            Xt = Xt.toarray()
        sw = institution_weights(val[2] if len(val) > 2 else None,
                                 args.institution_weight_alpha)
        row = {'n': len(y)}
        for label, (params, _) in configs.items():
            res = [fit_and_score(Xt, y, params, sd, sample_weight=sw) for sd in seeds]
            row[label] = float(np.mean([r[3] for r in res]))
            row[label + '__gap'] = float(np.mean([r[2] - r[3] for r in res]))
        per_cancer[code] = row
        print(f"  {code:9s} n={len(y):6,}  " +
              "  ".join(f"{lab.split()[0]} {row[lab]:.4f}" for lab in configs))

    df = pd.DataFrame(per_cancer).T
    keep = [c for c in df.index if c not in UNSTABLE_CODES]
    print("\n" + "=" * 78)
    print(f"{'candidate':26s} {'mean OOB':>9s} {'excl-2':>9s} {'median':>8s} {'gap':>7s}  provenance")
    print("-" * 78)
    summary = []
    for label, (_, prov) in configs.items():
        m, m2 = df[label].mean(), df.loc[keep, label].mean()
        summary.append(dict(candidate=label, mean_oob=round(m, 4),
                            mean_oob_excl_unstable=round(m2, 4),
                            median_oob=round(df[label].median(), 4),
                            mean_gap=round(df[label + '__gap'].mean(), 4), provenance=prov))
        print(f"{label:26s} {m:9.4f} {m2:9.4f} {df[label].median():8.4f} "
              f"{df[label + '__gap'].mean():7.4f}  {prov}")
    sdf = pd.DataFrame(summary)
    adoptable = sdf[~sdf.provenance.str.contains('NOT adoptable')]
    best_p = adoptable.loc[adoptable.mean_oob.idxmax()]
    best_s = adoptable.loc[adoptable.mean_oob_excl_unstable.idxmax()]
    print("-" * 78)
    print(f"best on primary (adoptable only)     : {best_p.candidate}")
    print(f"best on sensitivity (adoptable only) : {best_s.candidate}")
    if best_p.candidate == best_s.candidate:
        print(f"=> RULE SATISFIED: {best_p.candidate} leads on both.")
    else:
        print("=> INCONCLUSIVE by the pre-specified rule; fall back to provenance.")
    print(f"(incumbent C1 shipped, NOT adoptable: {sdf.iloc[0].mean_oob:.4f} / "
          f"{sdf.iloc[0].mean_oob_excl_unstable:.4f})")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        sdf.to_csv(args.output, index=False)
        df.to_csv(str(args.output).replace('.csv', '_per_cancer.csv'))
        print(f"\nWrote {args.output}")


def load_pooled_training(db_path, split_file, feature_set):
    """Pool the training split across the OncoTree codes, as TIER 3 does."""
    from oncokb_analyzer.data_loader import DataLoader

    split = pd.read_csv(split_file)
    tcol = next(c for c in split.columns if c.lower() in ('split', 'set', 'partition'))
    pcol = next(c for c in split.columns if 'PATIENT' in c.upper())
    train_ids = split.loc[split[tcol].str.lower() == 'train', pcol].unique().tolist()

    d = DataLoader(db_path).prepare_cancer_data_dict(
        cancer_types=CANCER_TYPES, feature_set=feature_set,
        outcome_col='isOncoKbLevel1', group_column='SUBJECT_CENTER',
        patient_ids_filter=train_ids)
    X = pd.concat([v[0] for v in d.values()], ignore_index=True)
    y = pd.concat([pd.Series(v[1]) for v in d.values()], ignore_index=True).values
    return X, y, len(train_ids)


def load_per_cancer_training(db_path, split_file, feature_set):
    """Same load, but kept split by cancer type (TIER 5 fits one model per code)."""
    from oncokb_analyzer.data_loader import DataLoader

    split = pd.read_csv(split_file)
    tcol = next(c for c in split.columns if c.lower() in ('split', 'set', 'partition'))
    pcol = next(c for c in split.columns if 'PATIENT' in c.upper())
    train_ids = split.loc[split[tcol].str.lower() == 'train', pcol].unique().tolist()

    return DataLoader(db_path).prepare_cancer_data_dict(
        cancer_types=CANCER_TYPES, feature_set=feature_set,
        outcome_col='isOncoKbLevel1', group_column='SUBJECT_CENTER',
        patient_ids_filter=train_ids)


def institution_weights(groups, alpha):
    """weight_i = (1 / count_i) ** alpha, normalised to mean 1.

    Matches the weighting used in TIER 3 and TIER 5.
    """
    if alpha <= 0 or groups is None:
        return None
    g = pd.Series(groups).reset_index(drop=True)
    raw = (1.0 / g.map(g.value_counts())) ** alpha
    return (raw / raw.sum() * len(raw)).values


def fit_and_score(Xt, y, params, seed, sample_weight=None):
    """Fit one RF; return (realized depths, leaves, train AUC, OOB AUC)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    rf = RandomForestClassifier(random_state=seed, n_jobs=-1, bootstrap=True,
                                oob_score=True, **params).fit(Xt, y,
                                                              sample_weight=sample_weight)
    depths = np.array([e.get_depth() for e in rf.estimators_])
    leaves = np.array([e.get_n_leaves() for e in rf.estimators_])
    train_auc = roc_auc_score(y, rf.predict_proba(Xt)[:, 1])
    oob = rf.oob_decision_function_[:, 1]
    ok = ~np.isnan(oob)
    oob_auc = roc_auc_score(y[ok], oob[ok]) if len(np.unique(y[ok])) > 1 else float('nan')
    return depths, leaves, train_auc, oob_auc


def run_per_cancer(args):
    """Per-cancer-type comparison (the setting TIER 5 uses)."""
    from oncokb_analyzer.preprocessing import create_preprocessor

    d = load_per_cancer_training(args.db_path, args.split_file, args.feature_set)
    seeds = [args.seed + i for i in range(args.n_seeds)]
    print(f"per-cancer mode: {len(d)} codes, {args.n_seeds} seed(s) averaged, "
          f"institution-weight alpha={args.institution_weight_alpha}\n")

    rows = []
    for code, val in d.items():
        X, y = val[0], np.asarray(val[1])
        if len(np.unique(y)) < 2:
            print(f"  {code}: single-class outcome, skipped")
            continue
        Xt = create_preprocessor(available_columns=X.columns.tolist()).fit_transform(X)
        if hasattr(Xt, 'toarray'):
            Xt = Xt.toarray()
        sw = institution_weights(val[2] if len(val) > 2 else None, args.institution_weight_alpha)
        rec = dict(cancer=code, n=len(y), n_actionable=int(y.sum()),
                   n_minority=int(min(y.sum(), len(y) - y.sum())))
        for label, key in (('hand-tuned (shipped)', 'hand'), ('searched (TIER 3)', 'srch')):
            res = [fit_and_score(Xt, y, CONFIGS[label], s, sample_weight=sw) for s in seeds]
            rec[f'{key}_depth'] = round(float(np.mean([r[0].mean() for r in res])), 2)
            rec[f'{key}_train'] = round(float(np.mean([r[2] for r in res])), 4)
            rec[f'{key}_oob'] = round(float(np.mean([r[3] for r in res])), 4)
            rec[f'{key}_oob_sd'] = round(float(np.std([r[3] for r in res])), 4)
        rec['d_oob'] = round(rec['srch_oob'] - rec['hand_oob'], 4)
        rec['d_gap'] = round((rec['srch_train'] - rec['srch_oob'])
                             - (rec['hand_train'] - rec['hand_oob']), 4)
        rows.append(rec)

    df = pd.DataFrame(rows).sort_values('n', ascending=False)
    pd.set_option('display.width', 200)
    print(df[['cancer', 'n', 'n_minority', 'hand_depth', 'srch_depth',
              'hand_oob', 'srch_oob', 'oob_sd' if 'oob_sd' in df else 'srch_oob_sd',
              'd_oob', 'd_gap']].to_string(index=False))

    better = int((df.d_oob > 0).sum())
    print(f"\nsearched better on OOB in {better}/{len(df)} cancer types "
          f"(mean delta {df.d_oob.mean():+.4f}, median {df.d_oob.median():+.4f})")
    print(f"mean seed-to-seed SD of OOB AUC: {df.srch_oob_sd.mean():.4f} "
          f"-- deltas smaller than ~2x this are noise")
    big = df[df.n >= 3000]
    small = df[df.n < 1200]
    print(f"  large codes (n>=3000, {len(big)}): mean delta {big.d_oob.mean():+.4f}, "
          f"depth {big.hand_depth.mean():.1f} -> {big.srch_depth.mean():.1f}")
    print(f"  small codes (n<1200, {len(small)}): mean delta {small.d_oob.mean():+.4f}, "
          f"depth {small.hand_depth.mean():.1f} -> {small.srch_depth.mean():.1f}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db-path', required=True,
                    help='Collapsed patient-unit DB. Pass explicitly -- the default '
                         'resolves to the unpatched DB and reintroduces the A1 age bias.')
    ap.add_argument('--split-file', required=True,
                    help='TIER 2 train_test_split.csv; only the TRAIN rows are used.')
    ap.add_argument('--feature-set', default='both')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output', default=None, help='Optional CSV path for the table.')
    ap.add_argument('--extend', action='store_true',
                    help='Add the post-hoc EXTENSION_CONFIGS depth ladder (d5, d3). '
                         'Labelled post-hoc on purpose -- see the comment on that dict.')
    ap.add_argument('--candidates', action='store_true',
                    help='E2b: score the PRE-SPECIFIED candidate set per cancer type '
                         'under the selection rule fixed at the top of this file.')
    ap.add_argument('--per-cancer', action='store_true',
                    help='Compare per cancer type -- the setting TIER 5 actually uses. '
                         'TIER 3 searches on POOLED data, so this tests whether the '
                         'pooled choice transfers to 21 separate models.')
    ap.add_argument('--institution-weight-alpha', type=float, default=0.5,
                    help='Match TIER 3/TIER 5 (default 0.5). 0 disables. Omitting these '
                         'weights does not reproduce what the pipeline fits.')
    ap.add_argument('--n-seeds', type=int, default=1,
                    help='Average over this many seeds. Small codes need >1.')
    args = ap.parse_args()

    warnings.filterwarnings('ignore')
    from oncokb_analyzer.preprocessing import create_preprocessor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    if args.candidates:
        run_candidates(args)
        return

    if args.per_cancer:
        run_per_cancer(args)
        return

    X, y, n_train = load_pooled_training(args.db_path, args.split_file, args.feature_set)
    print(f"train patients : {n_train:,}")
    print(f"pooled         : {X.shape[0]:,} rows x {X.shape[1]} raw features, "
          f"{y.mean():.1%} actionable")

    Xt = create_preprocessor(available_columns=X.columns.tolist()).fit_transform(X)
    if hasattr(Xt, 'toarray'):
        Xt = Xt.toarray()
    print(f"after one-hot  : {Xt.shape[1]} columns\n")

    rows = []
    for name, params in CONFIGS.items():
        rf = RandomForestClassifier(random_state=args.seed, n_jobs=-1, bootstrap=True,
                                    oob_score=True, **params).fit(Xt, y)
        depths = np.array([e.get_depth() for e in rf.estimators_])
        leaves = np.array([e.get_n_leaves() for e in rf.estimators_])
        train_auc = roc_auc_score(y, rf.predict_proba(Xt)[:, 1])
        oob_auc = roc_auc_score(y, rf.oob_decision_function_[:, 1])
        cap = params['max_depth']
        at_cap = float((depths == cap).mean()) if cap else float('nan')
        rows.append(dict(config=name, max_depth_cap=cap, depth_mean=round(depths.mean(), 2),
                         depth_max=int(depths.max()), frac_trees_at_cap=at_cap,
                         leaves_per_tree=round(leaves.mean()), train_auc=round(train_auc, 4),
                         oob_auc=round(oob_auc, 4), gap=round(train_auc - oob_auc, 4)))
        cap_txt = f"cap {cap}, {at_cap:.0%} of trees at cap" if cap else "no cap"
        print(f"{name}")
        print(f"  realized depth : mean {depths.mean():5.2f}  max {depths.max():3d}  ({cap_txt})")
        print(f"  leaves / tree  : {leaves.mean():,.0f}")
        print(f"  train AUC {train_auc:.4f}   OOB AUC {oob_auc:.4f}   gap {train_auc - oob_auc:.4f}\n")

    df = pd.DataFrame(rows)
    print("Caveats: pooled (not comparable to per-cancer TIER 5 AUCs); OOB ignores the")
    print("SUBJECT_CENTER grouping nested CV respected; single seed. The test set was")
    print("not read by this script.")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")


if __name__ == '__main__':
    main()
