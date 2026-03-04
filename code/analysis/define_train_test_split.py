#!/usr/bin/env python3
"""
Define Train/Test Split for ML Modeling

Creates a reproducible 80/20 train/test split stratified by tier 1 cancer types.
This split MUST be defined BEFORE any hyperparameter tuning or model selection
to prevent data leakage.

Scientific Rationale:
- Test set must be held out from ALL model development activities
- Hyperparameter optimization uses only training data (with cross-validation)
- Feature set comparison uses only training data (with cross-validation)
- Test set is touched ONLY ONCE for final model evaluation

Output:
- train_test_split.csv: Patient-level assignments (PATIENT_ID, ONCOTREE_CODE, split)
- train_test_split_summary.csv: Per-cancer-type split statistics

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

# 21 Tier 1 eligible cancer types (≥250 actionable events)
TIER1_CANCER_TYPES = [
    'LUAD', 'IDC', 'BRCA', 'MEL', 'GIST', 'COAD', 'THPA', 'ILC', 'PRAD', 'AML',
    'BLCA', 'SKCM', 'NSCLC', 'ODG', 'HGSOC', 'ASTR', 'MDLC', 'BRCNOS', 'BRCANOS',
    'COADREAD', 'IHCH'
]


def create_stratified_split(
    db_path: str,
    output_dir: Path,
    test_fraction: float = 0.2,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Create stratified train/test split for all tier 1 cancer types.

    Parameters
    ----------
    db_path : str
        Path to SQLite database
    output_dir : Path
        Directory to save output files
    test_fraction : float
        Fraction of data to hold out for testing (default: 0.2 = 20%)
    random_seed : int
        Random seed for reproducibility (default: 42)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: PATIENT_ID, ONCOTREE_CODE, split, isOncoKbLevel1
    """
    from sqlalchemy import create_engine

    logger.info(f"Creating {test_fraction:.0%} held-out test split")
    logger.info(f"Random seed: {random_seed}")
    logger.info(f"Database: {db_path}")

    # Load data
    engine = create_engine(f"sqlite:///{db_path}")
    data = pd.read_sql_table('perPatientOncoKBFilteredForModeling', con=engine)

    logger.info(f"Loaded {len(data)} patient records")

    # Filter to tier 1 cancer types only
    tier1_data = data[data['ONCOTREE_CODE'].isin(TIER1_CANCER_TYPES)].copy()
    logger.info(f"Filtered to {len(tier1_data)} patients in {len(TIER1_CANCER_TYPES)} tier 1 cancer types")

    # Set random seed
    np.random.seed(random_seed)

    # Create split for each cancer type (stratified)
    split_assignments = []
    summary_stats = []

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

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    split_file = output_dir / 'train_test_split.csv'
    summary_file = output_dir / 'train_test_split_summary.csv'

    split_df.to_csv(split_file, index=False)
    summary_df.to_csv(summary_file, index=False)

    logger.info(f"\nSaved split assignments to: {split_file}")
    logger.info(f"Saved summary statistics to: {summary_file}")

    # Print overall summary
    n_train = (split_df['split'] == 'train').sum()
    n_test = (split_df['split'] == 'test').sum()

    print(f"\n{'='*60}")
    print("TRAIN/TEST SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"Total patients: {len(split_df):,}")
    print(f"Training set: {n_train:,} ({n_train/len(split_df):.1%})")
    print(f"Test set: {n_test:,} ({n_test/len(split_df):.1%})")
    print(f"Cancer types: {len(summary_df)}")
    print(f"Random seed: {random_seed}")
    print(f"\nOutput files:")
    print(f"  - {split_file}")
    print(f"  - {summary_file}")
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
        random_seed=args.random_seed
    )


if __name__ == '__main__':
    main()
