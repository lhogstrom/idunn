#!/usr/bin/env python3
"""Define a reproducible 80/20 train/test split for ML modeling.

Stratified by cancer type; defined before any tuning so the test set is held out
from all model development. The default 'patient' strategy assigns each PATIENT_ID
to exactly one split (the modeling table is one row per sample, so a row-level
shuffle would place the same patient in both train and test).

Usage:
    python define_train_test_split.py
    python define_train_test_split.py --test-fraction 0.2 --random-seed 42
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import logging

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
from db_config import get_default_db_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Eligible cancer types (at least 250 actionable events)
TIER1_CANCER_TYPES = [
    'LUAD', 'IDC', 'BRCA', 'MEL', 'GIST', 'COAD', 'THPA', 'ILC', 'PRAD', 'AML',
    'BLCA', 'SKCM', 'NSCLC', 'ODG', 'HGSOC', 'ASTR', 'MDLC', 'BRCNOS', 'BRCANOS',
    'COADREAD', 'IHCH'
]


def assign_patients_grouped(
    tier1_data: pd.DataFrame,
    test_fraction: float,
    random_seed: int
) -> pd.DataFrame:
    """Assign each PATIENT_ID to exactly one split, stratified at the patient level."""
    patients = (
        tier1_data.groupby('PATIENT_ID')
        .agg(
            strat_cancer=('ONCOTREE_CODE', lambda s: sorted(s.unique())[0]),
            any_actionable=('isOncoKbLevel1', 'max'),
            n_rows=('ONCOTREE_CODE', 'size'),
            n_cancer_types=('ONCOTREE_CODE', 'nunique'),
        )
        .reset_index()
    )

    logger.info(
        f"Patient-grouped assignment over {len(patients):,} unique patients "
        f"({len(tier1_data):,} sample rows); "
        f"{int((patients['n_rows'] > 1).sum()):,} patients contribute >1 row, "
        f"{int((patients['n_cancer_types'] > 1).sum()):,} carry >1 cancer type"
    )

    rng = np.random.RandomState(random_seed)
    patients['split'] = 'train'

    # Stratify over patients within (representative cancer type x any-actionable).
    for _, grp in patients.groupby(['strat_cancer', 'any_actionable'], sort=True):
        idx = grp.index.values.copy()
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_fraction))
        patients.loc[idx[:n_test], 'split'] = 'test'

    return patients


def audit_patient_leakage(split_df: pd.DataFrame) -> pd.DataFrame:
    """Quantify how many patients (and test rows) straddle the train/test boundary."""
    per_patient = split_df.groupby('PATIENT_ID')['split'].nunique()
    straddlers = set(per_patient[per_patient > 1].index)

    test_rows = split_df[split_df['split'] == 'test']
    leaked_test_rows = int(test_rows['PATIENT_ID'].isin(straddlers).sum())
    n_test_rows = len(test_rows)

    audit = pd.DataFrame([{
        'n_patients_total': split_df['PATIENT_ID'].nunique(),
        'n_patients_in_both_splits': len(straddlers),
        'n_test_rows': n_test_rows,
        'n_test_rows_from_straddling_patients': leaked_test_rows,
        'pct_test_rows_leaked': (100.0 * leaked_test_rows / n_test_rows) if n_test_rows else 0.0,
    }])

    if straddlers:
        logger.warning(
            f"LEAKAGE: {len(straddlers):,} patients appear in BOTH splits, contributing "
            f"{leaked_test_rows:,}/{n_test_rows:,} test rows "
            f"({100.0 * leaked_test_rows / n_test_rows:.1f}%)"
        )
    else:
        logger.info("No patient appears in both splits (0 leaked test rows)")

    return audit


def create_stratified_split(
    db_path: str,
    output_dir: Path,
    test_fraction: float = 0.2,
    random_seed: int = 42,
    split_strategy: str = 'patient'
) -> pd.DataFrame:
    """Create a stratified train/test split for all eligible cancer types."""
    from sqlalchemy import create_engine

    logger.info(f"Creating {test_fraction:.0%} held-out test split")
    logger.info(f"Split strategy: {split_strategy}")
    logger.info(f"Random seed: {random_seed}")
    logger.info(f"Database: {db_path}")

    # Load data
    engine = create_engine(f"sqlite:///{db_path}")
    data = pd.read_sql_table('perPatientOncoKBFilteredForModeling', con=engine)
    # Coerce sqlalchemy `quoted_name` column labels to plain str (sklearn rejects mixed types).
    data.columns = [str(c) for c in data.columns]

    logger.info(f"Loaded {len(data):,} sample rows ({data['PATIENT_ID'].nunique():,} unique patients)")

    # Keep eligible cancer types only
    tier1_data = data[data['ONCOTREE_CODE'].isin(TIER1_CANCER_TYPES)].copy()
    logger.info(
        f"Filtered to {len(tier1_data):,} sample rows "
        f"({tier1_data['PATIENT_ID'].nunique():,} unique patients) "
        f"in {len(TIER1_CANCER_TYPES)} tier 1 cancer types"
    )

    # Set random seed
    np.random.seed(random_seed)

    summary_stats = []

    if split_strategy == 'patient':
        patients = assign_patients_grouped(tier1_data, test_fraction, random_seed)
        tier1_data = tier1_data.merge(patients[['PATIENT_ID', 'split']], on='PATIENT_ID', how='left')
        if tier1_data['split'].isna().any():
            raise RuntimeError(
                f"{int(tier1_data['split'].isna().sum())} rows failed to receive a split "
                f"assignment -- PATIENT_ID join is incomplete."
            )
        split_df = tier1_data[['PATIENT_ID', 'ONCOTREE_CODE', 'split', 'isOncoKbLevel1']].copy()

        for cancer_type in TIER1_CANCER_TYPES:
            ct_data = split_df[split_df['ONCOTREE_CODE'] == cancer_type]
            if len(ct_data) == 0:
                logger.warning(f"No rows found for {cancer_type}")
                continue
            train_data = ct_data[ct_data['split'] == 'train']
            test_data = ct_data[ct_data['split'] == 'test']
            summary_stats.append({
                'ONCOTREE_CODE': cancer_type,
                'n_total': len(ct_data),
                'n_train': len(train_data),
                'n_test': len(test_data),
                'n_actionable_total': ct_data['isOncoKbLevel1'].sum(),
                'n_actionable_train': train_data['isOncoKbLevel1'].sum(),
                'n_actionable_test': test_data['isOncoKbLevel1'].sum(),
                'pct_actionable_train': train_data['isOncoKbLevel1'].mean() * 100,
                'pct_actionable_test': test_data['isOncoKbLevel1'].mean() * 100,
                'actual_test_fraction': len(test_data) / len(ct_data),
            })
            logger.info(
                f"  {cancer_type}: {len(train_data)} train, {len(test_data)} test "
                f"(test frac {len(test_data)/len(ct_data):.1%}; actionable: "
                f"{train_data['isOncoKbLevel1'].mean():.1%} train, "
                f"{test_data['isOncoKbLevel1'].mean():.1%} test)"
            )

        summary_df = pd.DataFrame(summary_stats)
        drift = (summary_df['actual_test_fraction'] - test_fraction).abs()
        logger.info(
            f"Per-cancer test-fraction drift from {test_fraction:.0%}: "
            f"max {drift.max():.2%} ({summary_df.loc[drift.idxmax(), 'ONCOTREE_CODE']}), "
            f"mean {drift.mean():.2%}"
        )
        return _finalize_split(split_df, summary_df, output_dir, test_fraction,
                               random_seed, split_strategy)

    logger.warning(
        "split_strategy='row' shuffles SAMPLE ROWS, not patients. "
        "perPatientOncoKBFilteredForModeling is sample-level, so patients with multiple "
        "samples will appear in BOTH train and test. Use only to reproduce runs made "
        "before 2026-08-13."
    )
    split_assignments = []

    for cancer_type in TIER1_CANCER_TYPES:
        ct_data = tier1_data[tier1_data['ONCOTREE_CODE'] == cancer_type].copy()

        if len(ct_data) == 0:
            logger.warning(f"No patients found for {cancer_type}")
            continue

        n_patients = len(ct_data)
        n_actionable = ct_data['isOncoKbLevel1'].sum()

        # Shuffle and split
        indices = ct_data.index.values.copy()
        np.random.shuffle(indices)

        n_test = int(n_patients * test_fraction)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        # Assign splits
        ct_data.loc[test_indices, 'split'] = 'test'
        ct_data.loc[train_indices, 'split'] = 'train'

        # Calculate split statistics
        train_data = ct_data[ct_data['split'] == 'train']
        test_data = ct_data[ct_data['split'] == 'test']

        summary_stats.append({
            'ONCOTREE_CODE': cancer_type,
            'n_total': n_patients,
            'n_train': len(train_data),
            'n_test': len(test_data),
            'n_actionable_total': n_actionable,
            'n_actionable_train': train_data['isOncoKbLevel1'].sum(),
            'n_actionable_test': test_data['isOncoKbLevel1'].sum(),
            'pct_actionable_train': train_data['isOncoKbLevel1'].mean() * 100,
            'pct_actionable_test': test_data['isOncoKbLevel1'].mean() * 100,
            'actual_test_fraction': len(test_data) / n_patients
        })

        # Keep only needed columns for split file
        split_assignments.append(
            ct_data[['PATIENT_ID', 'ONCOTREE_CODE', 'split', 'isOncoKbLevel1']]
        )

        logger.info(
            f"  {cancer_type}: {len(train_data)} train, {len(test_data)} test "
            f"(actionable: {train_data['isOncoKbLevel1'].mean():.1%} train, "
            f"{test_data['isOncoKbLevel1'].mean():.1%} test)"
        )

    # Combine all splits
    split_df = pd.concat(split_assignments, ignore_index=True)
    summary_df = pd.DataFrame(summary_stats)

    return _finalize_split(split_df, summary_df, output_dir, test_fraction,
                           random_seed, split_strategy)


def _finalize_split(
    split_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    test_fraction: float,
    random_seed: int,
    split_strategy: str,
) -> pd.DataFrame:
    """Write outputs, audit patient leakage, and print the summary block."""
    output_dir.mkdir(parents=True, exist_ok=True)

    split_file = output_dir / 'train_test_split.csv'
    summary_file = output_dir / 'train_test_split_summary.csv'
    leakage_file = output_dir / 'train_test_split_leakage.csv'

    split_df.to_csv(split_file, index=False)
    summary_df.to_csv(summary_file, index=False)

    audit = audit_patient_leakage(split_df)
    audit.insert(0, 'split_strategy', split_strategy)
    audit.to_csv(leakage_file, index=False)

    logger.info(f"\nSaved split assignments to: {split_file}")
    logger.info(f"Saved summary statistics to: {summary_file}")
    logger.info(f"Saved leakage audit to: {leakage_file}")

    n_train = (split_df['split'] == 'train').sum()
    n_test = (split_df['split'] == 'test').sum()
    n_patients = split_df['PATIENT_ID'].nunique()

    print(f"\n{'='*60}")
    print("TRAIN/TEST SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"Split strategy: {split_strategy}")
    print(f"Total sample rows: {len(split_df):,}  (unique patients: {n_patients:,})")
    print(f"Training set: {n_train:,} ({n_train/len(split_df):.1%})")
    print(f"Test set: {n_test:,} ({n_test/len(split_df):.1%})")
    print(f"Cancer types: {len(summary_df)}")
    print(f"Random seed: {random_seed}")
    print(f"\nPatient leakage:")
    print(f"  Patients in both splits: {int(audit['n_patients_in_both_splits'].iloc[0]):,}")
    print(f"  Test rows from those patients: "
          f"{int(audit['n_test_rows_from_straddling_patients'].iloc[0]):,} "
          f"({audit['pct_test_rows_leaked'].iloc[0]:.1f}%)")
    print(f"\nOutput files:")
    print(f"  - {split_file}")
    print(f"  - {summary_file}")
    print(f"  - {leakage_file}")
    print(f"{'='*60}\n")

    # Verify stratification worked
    train_actionable_rate = split_df[split_df['split'] == 'train']['isOncoKbLevel1'].mean()
    test_actionable_rate = split_df[split_df['split'] == 'test']['isOncoKbLevel1'].mean()

    print("Actionability rates (should be similar):")
    print(f"  Training: {train_actionable_rate:.2%}")
    print(f"  Test: {test_actionable_rate:.2%}")
    print(f"  Difference: {abs(train_actionable_rate - test_actionable_rate):.2%}")

    if abs(train_actionable_rate - test_actionable_rate) > 0.02:
        logger.warning("Actionability rates differ by >2% between train/test - check stratification")

    return split_df


def main():
    parser = argparse.ArgumentParser(
        description='Create stratified train/test split for ML modeling'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=get_default_db_path(),
        help='Path to SQLite database'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../../analysis_output/train_test_split',
        help='Output directory for split files'
    )
    parser.add_argument(
        '--test-fraction',
        type=float,
        default=0.2,
        help='Fraction of data for test set (default: 0.2 = 20%%)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--split-strategy',
        type=str,
        choices=['patient', 'row'],
        default='patient',
        help=(
            "'patient' (default): one split assignment per PATIENT_ID, no leakage. "
            "'row': legacy pre-2026-08-13 row shuffle -- leaks multi-sample patients "
            "across train/test; use only to reproduce older runs."
        )
    )

    args = parser.parse_args()

    # Validate inputs
    if not (0 < args.test_fraction < 1):
        logger.error(f"Invalid test fraction: {args.test_fraction}. Must be between 0 and 1.")
        sys.exit(1)

    if not Path(args.db_path).exists():
        logger.error(f"Database not found: {args.db_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    # Create split
    create_stratified_split(
        db_path=args.db_path,
        output_dir=output_dir,
        test_fraction=args.test_fraction,
        random_seed=args.random_seed,
        split_strategy=args.split_strategy
    )


if __name__ == '__main__':
    main()
