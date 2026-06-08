"""Smoke tests for run_all_analyses orchestrator: pathway gating and TIER wiring.

These tests construct MasterAnalysis instances in dry-run mode and assert on the
generated script list. They do NOT execute any tier — they only verify that the
right ScriptConfigs are emitted under different flag combinations.
"""

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / 'code'))
sys.path.insert(0, str(REPO_ROOT / 'code' / 'tools'))

from run_all_analyses import MasterAnalysis  # noqa: E402


@pytest.fixture
def cleanup_run_dirs():
    """Track and remove run directories created by MasterAnalysis instances."""
    created = []

    def track(runner):
        created.append(runner.run_dir)
        return runner

    yield track

    for run_dir in created:
        shutil.rmtree(run_dir, ignore_errors=True)


@pytest.fixture
def fake_rf_run(tmp_path):
    """A minimal RF run-dir with the artifacts the cross-pathway gates check."""
    rf = tmp_path / 'fake_rf_run'

    (rf / 'tier2_data_setup' / 'train_test_split').mkdir(parents=True)
    (rf / 'tier2_data_setup' / 'train_test_split' / 'train_test_split.csv').write_text(
        'PATIENT_ID,split\n1,train\n2,test\n'
    )

    shap_dir = rf / 'tier6_shap_analysis' / 'primary_values'
    shap_dir.mkdir(parents=True)
    (shap_dir / 'combined_feature_importance.csv').write_text(
        'feature,mean_abs_shap\nAGE_AT_SEQ_REPORT,0.10\n'
    )

    tier5 = rf / 'tier5_final_models'
    tier5.mkdir(parents=True)
    (tier5 / 'performance_summary.csv').write_text(
        'cancer_type,test_auc,train_auc\nLUAD,0.65,0.72\n'
    )

    gene_dir = rf / 'tier7_posthoc_statistics' / 'gene_stratified' / 'sex_by_gene'
    gene_dir.mkdir(parents=True)
    (gene_dir / 'sex_gene_significant.csv').write_text('gene,fdr\nTP53,0.01\n')

    return rf


def _paths(scripts):
    return [s.path for s in scripts]


def test_lr_pass_with_base_run_dir_emits_cross_pathway_plots(fake_rf_run, cleanup_run_dirs):
    """LR + base_run_dir should fire all three cross-pathway plots."""
    runner = cleanup_run_dirs(MasterAnalysis(
        mode='post-model-selection',
        algorithm='logistic',
        base_run_dir=str(fake_rf_run),
        dry_run=True,
    ))
    paths = _paths(runner.scripts)
    assert 'code/analysis/posthoc/plot_rf_shap_lr_or_feature_ranks.py' in paths
    assert 'code/analysis/posthoc/visualize_rf_lr_hit_venn.py' in paths
    assert 'code/analysis/posthoc/plot_rf_vs_lr_auc_per_cancer.py' in paths


def test_lr_without_base_run_dir_omits_cross_pathway(cleanup_run_dirs):
    """No base_run_dir → no cross-pathway plots."""
    runner = cleanup_run_dirs(MasterAnalysis(
        mode='full',
        algorithm='logistic',
        dry_run=True,
    ))
    paths = _paths(runner.scripts)
    assert 'code/analysis/posthoc/plot_rf_shap_lr_or_feature_ranks.py' not in paths
    assert 'code/analysis/posthoc/visualize_rf_lr_hit_venn.py' not in paths
    assert 'code/analysis/posthoc/plot_rf_vs_lr_auc_per_cancer.py' not in paths


def test_rf_pass_omits_cross_pathway_plots(cleanup_run_dirs):
    """RF pathway never emits the LR-side cross-pathway plots."""
    runner = cleanup_run_dirs(MasterAnalysis(
        mode='full',
        algorithm='random_forest',
        dry_run=True,
    ))
    paths = _paths(runner.scripts)
    assert 'code/analysis/posthoc/plot_rf_shap_lr_or_feature_ranks.py' not in paths
    assert 'code/analysis/posthoc/visualize_rf_lr_hit_venn.py' not in paths


def test_tier1_promotes_cv_forest_plots(cleanup_run_dirs):
    """TIER 1 includes the promoted CV-and-forest publication script."""
    runner = cleanup_run_dirs(MasterAnalysis(mode='full', algorithm='random_forest', dry_run=True))
    paths = _paths(runner.scripts)
    assert 'code/analysis/institutional/plot_cv_and_forest_simple.py' in paths


def test_base_run_dir_skips_tier1_2_3(fake_rf_run, cleanup_run_dirs):
    """When base_run_dir is set, TIER 1, 2, 3 are skipped (their outputs come from the base run)."""
    runner = cleanup_run_dirs(MasterAnalysis(
        mode='full',
        algorithm='logistic',
        base_run_dir=str(fake_rf_run),
        dry_run=True,
    ))
    tiers = {s.tier for s in runner.scripts}
    assert 1 not in tiers
    assert 2 not in tiers
    assert 3 not in tiers
    assert 4 in tiers
    assert 5 in tiers


def test_tier3_default_omits_xgboost(cleanup_run_dirs):
    """Default --tier3-algorithms is 'rf,lr' — XGBoost off by default."""
    runner = cleanup_run_dirs(MasterAnalysis(mode='full', algorithm='random_forest', dry_run=True))
    tier3 = [s for s in runner.scripts if s.tier == 3]
    assert tier3, "TIER 3 must be present in 'full' mode"
    algo_arg = ' '.join(tier3[0].args)
    assert 'rf,lr' in algo_arg
    assert 'xgb' not in algo_arg


def test_tier3_xgboost_via_flag(cleanup_run_dirs):
    """tier3_algorithms='rf,lr,xgb' restores XGBoost in TIER 3 search."""
    runner = cleanup_run_dirs(MasterAnalysis(
        mode='full',
        algorithm='random_forest',
        tier3_algorithms='rf,lr,xgb',
        dry_run=True,
    ))
    tier3 = [s for s in runner.scripts if s.tier == 3]
    algo_arg = ' '.join(tier3[0].args)
    assert 'rf,lr,xgb' in algo_arg


def test_quick_rf_passes_no_shap_filter(cleanup_run_dirs):
    """--quick + RF skips SHAP, so the gene-strat script must get --no-shap-filter
    rather than auto-search and fail with 'No v2.0 SHAP file found'."""
    runner = cleanup_run_dirs(MasterAnalysis(
        mode='quick',
        algorithm='random_forest',
        skip_shap=True,  # --quick + RF
        dry_run=True,
    ))
    gene_strat = next(
        s for s in runner.scripts
        if s.path == 'code/analysis/posthoc/analyze_gene_stratified_associations.py'
    )
    assert '--no-shap-filter' in gene_strat.args
    assert '--shap-file' not in gene_strat.args


def test_lr_pass_with_rankings_omits_no_shap_filter(fake_rf_run, cleanup_run_dirs):
    """LR pathway with its own rankings file passes --shap-file, not --no-shap-filter."""
    runner = cleanup_run_dirs(MasterAnalysis(
        mode='post-model-selection',
        algorithm='logistic',
        base_run_dir=str(fake_rf_run),
        dry_run=True,
    ))
    gene_strat = next(
        s for s in runner.scripts
        if s.path == 'code/analysis/posthoc/analyze_gene_stratified_associations.py'
    )
    assert '--shap-file' in gene_strat.args
    assert '--no-shap-filter' not in gene_strat.args
