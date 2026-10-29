#!/usr/bin/env python3
"""Build two supplement tables: model hyperparameters, and the `Unknown`-category comparison.

Both are emits of numbers already in a run directory or module constant; nothing is refit.

  Hyperparameters: the validated RF params JSON (two-step selection) and the LR defaults,
  documenting the deliberate RF/LR asymmetry.
  Unknown-category comparison: each gene-stratified ancestry/ethnicity test run with the
  `Unknown` sentinel retained (legacy) vs excluded (current), for both pathways. The legacy
  runs must come from the same fitted models; the script compares TIER 5 test AUCs and
  refuses to write if they differ.

Usage:
  python build_supplement_tables.py --rf-run runs/<rf_run> --lr-run runs/<lr_run> \
      --rf-legacy-run runs/<rf_legacy_run> --lr-legacy-run runs/<lr_legacy_run> --output-dir <dir>
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / 'code' / 'modeling'))

# Families whose tests the sentinel change can touch; mirrors
# analyze_gene_stratified_associations.SENTINEL_GATED_FEATURES (from preprocessing.MISSINGNESS_INDICATOR_COLS).
SENTINEL_FAMILIES = {'race': 'ANCESTRY', 'ethnicity': 'ETHNICITY'}

GENE_STRAT = 'tier7_posthoc_statistics/gene_stratified'
FAMILY_FILE = {'race': 'race_by_gene/race_gene_results.csv',
               'ethnicity': 'ethnicity_by_gene/ethnicity_gene_results.csv'}


def _perf(run: Path) -> pd.DataFrame:
    """TIER 5 per-cancer performance summary for a run (used for the guard)."""
    hits = sorted(run.glob('tier5_final_models/*/performance_summary.csv'))
    if not hits:
        return pd.DataFrame()
    return pd.read_csv(hits[-1])


def assert_same_models(a: Path, b: Path, tol: float = 1e-9) -> str:
    """Fail closed if the two runs were not built from the same fitted models.

    The paired table is only valid if sentinel handling is the sole difference.
    """
    pa, pb = _perf(a), _perf(b)
    if pa.empty or pb.empty:
        return f'SKIPPED (no TIER 5 performance_summary.csv in {"a" if pa.empty else "b"})'
    key = 'cancer_type'
    col = 'test_auc' if 'test_auc' in pa.columns else 'test_roc_auc'
    m = pa[[key, col]].merge(pb[[key, col]], on=key, suffixes=('_a', '_b'))
    if m.empty:
        raise SystemExit(f'ERROR: no shared cancer types between {a} and {b}')
    delta = (m[f'{col}_a'] - m[f'{col}_b']).abs().max()
    if delta > tol:
        raise SystemExit(
            f'ERROR: {a.name} and {b.name} do not share fitted models '
            f'(max |delta test AUC| = {delta:.6f} > {tol}). Table S10 would '
            f'confound the sentinel change with a model change. Refusing to write.')
    return f'OK (max |delta test AUC| = {delta:.2e} over {len(m)} cancer types)'


# Hyperparameters table
def build_s4(rf_params_json: Path, out_dir: Path) -> pd.DataFrame:
    from run_final_model_training import DEFAULT_LR_PARAMS

    blob = json.loads(rf_params_json.read_text())
    rf = blob['best_params_consensus']
    step1 = blob.get('step1_pooled_nested_cv', {})
    step2 = blob.get('step2_per_cancer_validation', {})

    rows = []
    for k, v in rf.items():
        rows.append({'model': 'Random forest', 'parameter': k, 'value': v,
                     'source': 'Nested CV consensus, then per-cancer depth validation (two steps)'})
    rows.append({'model': 'Random forest', 'parameter': 'random_state', 'value': 42,
                 'source': 'Module default, merged onto the searched keys for reproducibility'})
    for k, v in DEFAULT_LR_PARAMS.items():
        note = 'Module default'
        if k == 'C':
            note = ('Module default. The searched value was NOT adopted: scikit-learn ignores C '
                    'when penalty=None, and both folds voting for the modal C=0.001 were exactly '
                    'the penalty=None folds.')
        elif k == 'max_iter':
            note = 'Module default; a convergence budget, not a capacity parameter'
        rows.append({'model': 'Logistic regression', 'parameter': k, 'value': v, 'source': note})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'table_S4_hyperparameters.csv', index=False)

    notes = out_dir / 'table_S4_hyperparameters_notes.md'
    notes.write_text(f"""# Table S4 — Model hyperparameters and the selection procedure

## Random forest — selected in TWO steps, both on training data only

**Step 1 — pooled nested cross-validation.** {step1.get('design', 'n/a')}.
Mean nested-CV score {step1.get('mean_score', float('nan')):.4f}
(SD {step1.get('std_score', float('nan')):.4f}); selected `max_depth` =
{step1.get('selected_max_depth', 'n/a')}.

⚠️ {step1.get('caveat', '')}

**Step 2 — per-cancer depth validation.** {step2.get('design', 'n/a')}.
Selection rule: {step2.get('rule', 'n/a')}.
Selected: {step2.get('selected', 'n/a')}.
Mean out-of-bag AUC {step2.get('mean_oob_auc', float('nan'))}
(excluding unstable codes {step2.get('mean_oob_auc_excl_unstable', float('nan'))}),
mean train-OOB gap {step2.get('mean_train_oob_gap', float('nan'))}.

**Honest effect size.** {step2.get('honest_effect_size', 'n/a')}

Neither step used the held-out test set at any point. This is the distinction
from the hand-tuned constants this configuration replaces.

## Logistic regression — module defaults, deliberately

The RF/LR asymmetry is intentional. The TIER 3 search returned an invalid `C`
for logistic regression: scikit-learn ignores `C` when `penalty=None`, and the
folds that produced the modal `C` were exactly the unpenalized folds. Every
other searched LR value reproduces the module default, so the search supplies no
evidence to move off them.

⚠️ The LR search was also **exhaustive, not random** — the grid holds 192
combinations and all 192 were evaluated, so the nominal `n_iter=500` does not
describe the LR arm.

## Cohort caveat

⚠️ The hyperparameter search ran on the **21-cancer-type** pooled grid, before
oligodendroglioma was removed from the cohort. It was not re-run for the final
20-type cohort; ODG contributed 0.73% of the pooled search rows. This is a
stated limitation, not an oversight.

Source: `{rf_params_json}`
""")
    return df


# Unknown-category comparison table
def _family(run: Path, fam: str) -> pd.DataFrame:
    f = run / GENE_STRAT / FAMILY_FILE[fam]
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f)
    keep = [c for c in ['ONCOTREE_CODE', 'Hugo_Symbol', 'p_value', 'p_fdr',
                        'cramers_v', 'dof', 'significant_fdr'] if c in d.columns]
    return d[keep]


def build_s10(runs: dict, out_dir: Path) -> pd.DataFrame:
    frames = []
    for pathway in ('RF', 'LR'):
        for fam, feature in SENTINEL_FAMILIES.items():
            cur = _family(runs[f'{pathway}_current'], fam)
            leg = _family(runs[f'{pathway}_legacy'], fam)
            if cur.empty or leg.empty:
                continue
            m = leg.merge(cur, on=['ONCOTREE_CODE', 'Hugo_Symbol'],
                          suffixes=('_unknown_retained', '_unknown_excluded'), how='outer')
            m.insert(0, 'feature', feature)
            m.insert(0, 'pathway', pathway)
            frames.append(m)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return df

    df['call_change'] = df.apply(
        lambda r: 'unchanged' if bool(r.get('significant_fdr_unknown_retained')) ==
                                 bool(r.get('significant_fdr_unknown_excluded'))
        else ('lost with exclusion' if r.get('significant_fdr_unknown_retained')
              else 'gained with exclusion'), axis=1)
    df = df.sort_values(['pathway', 'feature', 'ONCOTREE_CODE', 'Hugo_Symbol'])
    df.to_csv(out_dir / 'table_S10_unknown_sensitivity.csv', index=False)
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rf-run', type=Path, required=True)
    p.add_argument('--lr-run', type=Path, required=True)
    p.add_argument('--rf-legacy-run', type=Path, required=True,
                   help='RF run with --keep-unknown-category, SAME fitted models')
    p.add_argument('--lr-legacy-run', type=Path, required=True,
                   help='LR run with legacy Unknown handling, SAME fitted models')
    p.add_argument('--rf-params-json', type=Path,
                   default=REPO_ROOT / 'code/config/hyperparameters'
                                       '/random_forest_universal_params.json')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()

    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print('Guard — the legacy runs must share fitted models with the current ones:')
    print(f'  RF: {assert_same_models(a.rf_run, a.rf_legacy_run)}')
    print(f'  LR: {assert_same_models(a.lr_run, a.lr_legacy_run)}')

    s4 = build_s4(a.rf_params_json, out)
    print(f'\nTable S4  {len(s4)} parameter rows '
          f'({s4.model.value_counts().to_dict()}) -> table_S4_hyperparameters.csv')

    s10 = build_s10({'RF_current': a.rf_run, 'RF_legacy': a.rf_legacy_run,
                     'LR_current': a.lr_run, 'LR_legacy': a.lr_legacy_run}, out)
    if s10.empty:
        print('Table S10  no ancestry/ethnicity result files found — nothing written')
    else:
        print(f'Table S10 {len(s10)} paired tests -> table_S10_unknown_sensitivity.csv')
        print('  call changes: ' + str(s10.call_change.value_counts().to_dict()))

    print(f'\nOutputs -> {out}')


if __name__ == '__main__':
    main()
