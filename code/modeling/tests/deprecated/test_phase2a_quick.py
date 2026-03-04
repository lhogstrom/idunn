"""
Quick validation test for Phase 2a pilot script.

This script runs a minimal version of the Phase 2a pilot to ensure:
1. Data loading works correctly
2. Nested CV optimizer integrates properly
3. Results are saved correctly
4. No crashes or errors

Uses:
- Single cancer type (LUAD only)
- Quick search space (reduced combinations)
- Minimal CV folds (3 outer, 3 inner)
- Reduced iterations (n_iter=5)

Expected runtime: ~5 minutes
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from run_ml_model_selection_pilot import Phase2aPilot
import logging

# Add tools to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))
from db_config import get_default_db_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_phase2a_minimal():
    """Test Phase 2a with minimal configuration"""
    print("\n" + "="*80)
    print("PHASE 2A QUICK VALIDATION TEST")
    print("="*80)
    print("\nThis will test the Phase 2a pipeline with:")
    print("  - Single cancer type (LUAD)")
    print("  - Quick search space")
    print("  - Minimal CV (3 folds)")
    print("  - Random Forest only")
    print("  - Expected runtime: ~5 minutes")
    print("\n" + "="*80 + "\n")

    # Create pilot with quick mode
    pilot = Phase2aPilot(
        db_path='/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20250522/balder-harmonized-biomarker-data-v20250522.sqlite',
        output_dir='ml_experiments/phase2a_validation_test',
        quick_mode=True
    )

    # Override to use only LUAD for faster testing
    pilot.cancer_types = ['LUAD']
    pilot.n_iter = 5  # Even fewer iterations for validation

    print("Running Phase 2a pilot with LUAD only...")

    try:
        # Run only Random Forest
        results = pilot.run_comparison(algorithm='random_forest')

        print("\n" + "="*80)
        print("✅ VALIDATION TEST PASSED")
        print("="*80)
        print(f"\nResults saved to: {pilot.output_dir}")
        print(f"\nDecision: {results['decision']}")
        print(f"Mean delta AUC: {results['mean_delta_auc']:.4f}")

        return results

    except Exception as e:
        print("\n" + "="*80)
        print("❌ VALIDATION TEST FAILED")
        print("="*80)
        print(f"\nError: {str(e)}")
        raise


if __name__ == '__main__':
    test_phase2a_minimal()
