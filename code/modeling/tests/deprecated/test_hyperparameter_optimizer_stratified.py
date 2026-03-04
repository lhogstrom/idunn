"""
Test script to validate StratifiedGroupKFold integration in HyperparameterOptimizer

This script tests that the hyperparameter optimizer correctly uses StratifiedGroupKFold
when cv_strategy='stratified_group' is specified.

"""

import sys
import os

# Add tools to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))
from db_config import get_default_db_path
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oncokb_analyzer.config import Config
from oncokb_analyzer.data_loader import DataLoader
from oncokb_analyzer.hyperparameter_optimizer import NestedCVOptimizer
from oncokb_analyzer.param_spaces import RF_PARAM_DISTRIBUTIONS_QUICK
from oncokb_analyzer.preprocessing import create_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def test_optimizer_with_stratified_groupkfold():
    """Test NestedCVOptimizer with cv_strategy='stratified_group'"""

    print("="*80)
    print("TESTING HYPERPARAMETER OPTIMIZER WITH STRATIFIED GROUPKFOLD")
    print("="*80)

    # Database path
    db_path = get_default_db_path()

    if not os.path.exists(db_path):
        print(f"ERROR: Database not found at {db_path}")
        return

    # Create config
    config = Config(
        db_path=db_path,
        cancer_types=['BRCA']  # Use BRCA as test case (known to have fold imbalance)
    )

    # Load data
    print("\nLoading data...")
    loader = DataLoader(config.db_path)
    data = loader.load_cancer_type_data('BRCA')

    if data is None or len(data) < 1000:
        print(f"ERROR: Insufficient data ({len(data) if data is not None else 0} samples)")
        return

    print(f"Loaded {len(data)} samples for BRCA")

    # Prepare features (demographic only)
    from oncokb_analyzer.trainer import ModelTrainer
    trainer = ModelTrainer(
        algorithm="random_forest",
        cv_strategy="group",
        group_column="SUBJECT_CENTER",
        random_state=42
    )

    X, y, groups = trainer.prepare_features(
        data,
        outcome_col='isOncoKbLevel1',
        feature_set='demographic'
    )

    print(f"Features: {X.shape[1]} features, {len(X)} samples")
    print(f"Groups: {groups.nunique()} unique institutions")
    print(f"Class balance: {y.value_counts().to_dict()}")

    # Test 1: Standard GroupKFold (baseline)
    print("\n" + "="*80)
    print("TEST 1: Standard GroupKFold (cv_strategy='group')")
    print("="*80)

    optimizer_group = NestedCVOptimizer(
        param_distributions=RF_PARAM_DISTRIBUTIONS_QUICK,
        search_method='random',
        n_iter=5,  # Small number for quick test
        random_state=42,
        verbose=1,
        cv_strategy='group'
    )

    results_group = optimizer_group.optimize(
        X=X.values,
        y=y.values,
        groups=groups.values,
        algorithm='random_forest'
    )

    print(f"\nGroupKFold Results:")
    print(f"  Mean AUC: {results_group['mean_score']:.4f} ± {results_group['std_score']:.4f}")
    print(f"  Best params: {results_group['best_params_consensus']}")

    # Test 2: StratifiedGroupKFold (new)
    print("\n" + "="*80)
    print("TEST 2: StratifiedGroupKFold (cv_strategy='stratified_group')")
    print("="*80)

    optimizer_stratified = NestedCVOptimizer(
        param_distributions=RF_PARAM_DISTRIBUTIONS_QUICK,
        search_method='random',
        n_iter=5,  # Small number for quick test
        random_state=42,
        verbose=1,
        cv_strategy='stratified_group'
    )

    results_stratified = optimizer_stratified.optimize(
        X=X.values,
        y=y.values,
        groups=groups.values,
        algorithm='random_forest'
    )

    print(f"\nStratifiedGroupKFold Results:")
    print(f"  Mean AUC: {results_stratified['mean_score']:.4f} ± {results_stratified['std_score']:.4f}")
    print(f"  Best params: {results_stratified['best_params_consensus']}")

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"GroupKFold AUC:            {results_group['mean_score']:.4f} ± {results_group['std_score']:.4f}")
    print(f"StratifiedGroupKFold AUC:  {results_stratified['mean_score']:.4f} ± {results_stratified['std_score']:.4f}")
    print(f"Difference:                {results_stratified['mean_score'] - results_group['mean_score']:.4f}")

    if abs(results_stratified['mean_score'] - results_group['mean_score']) < 0.02:
        print("\n✅ SUCCESS: StratifiedGroupKFold shows similar performance (difference < 0.02)")
    else:
        print(f"\n⚠️  WARNING: Large performance difference ({abs(results_stratified['mean_score'] - results_group['mean_score']):.4f})")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    test_optimizer_with_stratified_groupkfold()
