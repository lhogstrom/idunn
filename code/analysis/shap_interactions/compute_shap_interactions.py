#!/usr/bin/env python3
"""
SHAP Interaction Analysis Script

Computes pairwise feature interactions for Random Forest models
to determine if models rely on feature combinations vs individual features.

This standalone script:
1. Loads data and trains Random Forest model
2. Calculates SHAP interaction values (3D matrix)
3. Decomposes predictions into main effects vs interactions
4. Identifies strongest interaction pairs
5. Generates comprehensive visualizations
6. Saves results for further analysis

Usage:
    # Quick test with LUAD
    python compute_shap_interactions.py --cancer-types LUAD

    # Analyze LUAD + BRCA
    python compute_shap_interactions.py \\
        --cancer-types LUAD BRCA \\
        --feature-set demographic \\
        --top-n-interactions 20

"""

import sys
import os

# Add tools to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))
from db_config import get_default_db_path
import argparse
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add paths for imports
sys.path.append('../../modeling')
from oncokb_analyzer.data_loader import DataLoader
from oncokb_analyzer.trainer import ModelTrainer
from oncokb_analyzer.config import Config

# Import local modules
from interaction_analyzer import SHAPInteractionAnalyzer
from interaction_visualizations import *
from feature_name_mapper import FeatureNameMapper, create_mapper_from_data_loader


def setup_logging(output_dir, verbose=False):
    """Configure logging to file and console."""
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )

    # File handler
    log_file = output_dir / 'analysis.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_feature_types(feature_set):
    """
    Define feature types for categorization.

    Returns dict mapping original feature names to types.
    """
    demographic_features = [
        'AGE_AT_SEQ_REPORT', 'SEX', 'RACE', 'ETHNICITY',
        'SAMPLE_TYPE', 'SAMPLE_TYPE_DETAILED'
    ]

    technical_features = [
        'SUBJECT_CENTER', 'SEQ_ASSAY_ID', 'SEQ_YEAR',
        'library_selection', 'calling_strategy'
    ]

    # Create mapping
    feature_types = {}
    for feat in demographic_features:
        feature_types[feat] = 'demographic'
    for feat in technical_features:
        feature_types[feat] = 'technical'

    return feature_types


def write_summary_report(output_path, cancer_type, feature_set, results_dict):
    """Write human-readable summary report."""
    main_pct = results_dict['main_pct']
    interaction_pct = results_dict['interaction_pct']
    n_samples = results_dict['n_samples']
    n_features = results_dict['n_features']
    test_auc = results_dict.get('test_auc', 'N/A')
    top_interaction = results_dict.get('top_interaction', 'N/A')

    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SHAP INTERACTION ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write("Analysis Details:\n")
        f.write(f"  Cancer Type: {cancer_type}\n")
        f.write(f"  Feature Set: {feature_set}\n")
        f.write(f"  Number of samples: {n_samples}\n")
        f.write(f"  Number of features: {n_features}\n")
        f.write(f"  Model test AUC: {test_auc}\n\n")

        f.write("-" * 70 + "\n")
        f.write("CONTRIBUTION BREAKDOWN:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Main effects:  {main_pct:5.2f}%  (individual features)\n")
        f.write(f"  Interactions:  {interaction_pct:5.2f}%  (feature combinations)\n\n")

        f.write("-" * 70 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("-" * 70 + "\n")

        if interaction_pct < 10:
            f.write("  → Model relies primarily on MAIN EFFECTS (additive)\n\n")
            f.write("  Scientific Implication:\n")
            f.write("    - Each demographic feature contributes independently\n")
            f.write("    - Demographic associations are separate, not combined\n\n")
            f.write("  Modeling Implication:\n")
            f.write("    - Logistic regression should perform similarly to Random Forest\n")
            f.write("    - Simpler model preferred (better interpretability, faster)\n")
            f.write("    - No need for explicit interaction terms\n")
        elif interaction_pct < 30:
            f.write("  → Model uses MODERATE interaction effects\n\n")
            f.write("  Scientific Implication:\n")
            f.write("    - Some feature combinations matter\n")
            f.write("    - Worth investigating specific interactions\n\n")
            f.write("  Modeling Implication:\n")
            f.write("    - Random Forest has advantage over logistic regression\n")
            f.write("    - Consider adding key interaction terms to LR if needed\n")
        else:
            f.write("  → Model relies heavily on INTERACTIONS\n\n")
            f.write("  Scientific Implication:\n")
            f.write("    - Feature combinations drive predictions\n")
            f.write("    - Demographic associations depend on other factors\n")
            f.write("    - Example: Age effect may differ by sex\n\n")
            f.write("  Modeling Implication:\n")
            f.write("    - Random Forest captures critical biology\n")
            f.write("    - Logistic regression would underperform\n")
            f.write("    - Need explicit interaction terms if using LR\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write(f"TOP INTERACTION:\n")
        f.write("-" * 70 + "\n")
        if top_interaction != 'N/A':
            f.write(f"  {top_interaction}\n")

        f.write("\n" + "=" * 70 + "\n")


def main():
    """Main analysis workflow."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='SHAP Interaction Analysis for Cancer Genomics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with LUAD
  python compute_shap_interactions.py --cancer-types LUAD

  # Analyze LUAD + BRCA
  python compute_shap_interactions.py --cancer-types LUAD BRCA

  # Custom configuration
  python compute_shap_interactions.py \\
      --cancer-types LUAD BRCA MEL \\
      --feature-set demographic \\
      --top-n-interactions 20 \\
      --output-dir ../../../analysis_output/shap_interactions/
        """
    )

    parser.add_argument('--cancer-types', nargs='+', default=['LUAD', 'BRCA'],
                        help='Cancer types to analyze (default: LUAD BRCA)')
    parser.add_argument('--feature-set', default='demographic',
                        choices=['demographic', 'technical', 'both'],
                        help='Feature set to use (default: demographic)')
    parser.add_argument('--top-n-interactions', type=int, default=20,
                        help='Number of top interactions to report (default: 20)')
    parser.add_argument('--output-dir',
                        default='../../../analysis_output/shap_interactions/',
                        help='Output directory (default: ../../../analysis_output/shap_interactions/)')
    parser.add_argument('--db-path',
                        default='/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20250522/balder-harmonized-biomarker-data-v20250522.sqlite',
                        help='Path to database')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_base, verbose=args.verbose)

    logger.info("=" * 70)
    logger.info("SHAP INTERACTION ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Cancer types: {', '.join(args.cancer_types)}")
    logger.info(f"Feature set: {args.feature_set}")
    logger.info(f"Output directory: {output_base}")
    logger.info("")

    # Get feature types
    feature_types = get_feature_types(args.feature_set)

    # Main analysis loop
    all_results = []
    all_top_interactions = []

    for cancer_type in args.cancer_types:
        logger.info("=" * 70)
        logger.info(f"ANALYZING {cancer_type}")
        logger.info("=" * 70)

        try:
            # 1. Load data
            logger.info("Step 1/7: Loading data...")
            data_loader = DataLoader(db_path=args.db_path)

            # Load full patient data
            data = data_loader.load_patient_data()

            # Filter for cancer type
            cancer_data = data_loader.filter_for_cancer_type(data, cancer_type)

            # Prepare features and outcome
            outcome_col = 'isOncoKbLevel1'
            y = cancer_data[outcome_col]

            # Select features based on feature_set
            demographic_features = ['AGE_AT_SEQ_REPORT', 'SEX', 'RACE', 'ETHNICITY',
                                   'SAMPLE_TYPE', 'SAMPLE_TYPE_DETAILED']
            technical_features = ['SUBJECT_CENTER', 'SEQ_ASSAY_ID', 'SEQ_YEAR',
                                 'library_selection', 'calling_strategy']

            if args.feature_set == 'demographic':
                feature_cols = [col for col in demographic_features if col in cancer_data.columns]
            elif args.feature_set == 'technical':
                feature_cols = [col for col in technical_features if col in cancer_data.columns]
            else:  # 'both'
                feature_cols = [col for col in demographic_features + technical_features
                               if col in cancer_data.columns]

            X = cancer_data[feature_cols]

            logger.info(f"  Loaded {len(X)} samples with {X.shape[1]} features")
            logger.info(f"  Actionable: {y.sum()} ({y.mean()*100:.1f}%)")

            # 2. Train Random Forest model
            logger.info("Step 2/7: Training Random Forest model...")

            # Create train/test split first (needed for SHAP)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.3,
                random_state=args.random_seed,
                stratify=y
            )

            trainer = ModelTrainer(
                algorithm='random_forest',
                random_state=args.random_seed,
                cv_folds=5
            )

            # Train using full data for cross-validation, but we'll use X_train for SHAP
            results = trainer.train_and_evaluate(X, y)

            # Extract the trained model (key is 'model', not 'pipeline')
            pipeline = results['model']
            test_auc = results.get('test_auc', 0.0)
            cv_auc = results.get('cv_auc_mean', 0.0)

            logger.info(f"  Model trained. CV AUC: {cv_auc:.3f}, Test AUC: {test_auc:.3f}")

            # 2b. Create feature name mapper
            logger.info("Step 2b/7: Creating feature name mapper...")
            feature_mapper = create_mapper_from_data_loader(feature_cols)
            logger.info("  Feature name mapper created")

            # 3. Calculate SHAP interaction values
            logger.info("Step 3/7: Calculating SHAP interaction values...")
            logger.info("  (This may take 5-10 minutes depending on data size)")

            analyzer = SHAPInteractionAnalyzer(pipeline, X_train, feature_name_mapper=feature_mapper)
            interaction_values = analyzer.calculate_interaction_values(X_test)

            logger.info(f"  Interaction values calculated: shape {interaction_values.shape}")

            # 4. Analyze main vs interaction contributions
            logger.info("Step 4/7: Analyzing contribution breakdown...")
            main_vs_interaction = analyzer.compare_main_vs_interaction_contributions(
                interaction_values
            )

            # 5. Identify strongest interactions
            logger.info(f"Step 5/7: Identifying top {args.top_n_interactions} interactions...")
            top_interactions = analyzer.get_strongest_interactions(
                interaction_values,
                top_n=args.top_n_interactions,
                feature_types=feature_types
            )
            top_interactions['cancer_type'] = cancer_type

            # 6. Create output directory for this cancer type
            cancer_output_dir = output_base / f"{cancer_type}_{args.feature_set}_{timestamp}"
            cancer_output_dir.mkdir(exist_ok=True)

            logger.info(f"Step 6/7: Generating visualizations...")

            # Interaction heatmap (use readable names)
            plot_interaction_heatmap(
                interaction_values,
                analyzer.feature_names_readable,
                cancer_output_dir / 'interaction_heatmap.pdf'
            )

            # Main vs interaction contribution
            plot_main_vs_interaction_pie(
                main_vs_interaction,
                cancer_output_dir / 'main_vs_interaction_contribution.pdf'
            )

            # Top interactions bar plot
            plot_top_interactions_bar(
                top_interactions,
                cancer_output_dir / 'top_interactions_bar.pdf',
                top_n=args.top_n_interactions
            )

            # Demographic interaction matrix (if using demographic features)
            if args.feature_set in ['demographic', 'both']:
                # Find demographic features in transformed names
                demographic_original = ['AGE_AT_SEQ_REPORT', 'SEX', 'RACE', 'ETHNICITY']
                demo_indices = []
                demo_labels = []

                for orig_feat in demographic_original:
                    for idx, feat_name in enumerate(analyzer.feature_names):
                        if orig_feat in feat_name:
                            if idx not in demo_indices:  # Avoid duplicates
                                demo_indices.append(idx)
                                # Use mapper to get readable name
                                readable_name = feature_mapper.map_feature_name(
                                    feat_name, include_value=False, short_form=True
                                )
                                demo_labels.append(readable_name)
                            break

                if len(demo_indices) > 1:
                    plot_demographic_interaction_matrix(
                        interaction_values,
                        analyzer.feature_names_readable,
                        demo_indices,
                        demo_labels,
                        cancer_output_dir / 'demographic_interaction_matrix.pdf'
                    )

            # Interaction dependence plots for top 3 interactions
            for idx, row in top_interactions.head(3).iterrows():
                # Find feature indices using raw feature names
                feat_i_idx = list(analyzer.feature_names).index(row['feature_i_raw'])
                feat_j_idx = list(analyzer.feature_names).index(row['feature_j_raw'])

                plot_interaction_dependence(
                    interaction_values,
                    X_test,
                    feat_i_idx,
                    feat_j_idx,
                    analyzer.feature_names_readable,
                    cancer_output_dir / f"interaction_dependence_{idx+1}.pdf"
                )

            # 7. Save results
            logger.info("Step 7/7: Saving results...")

            # Summary report
            results_dict = {
                **main_vs_interaction,
                'test_auc': f"{test_auc:.3f}",
                'top_interaction': f"{top_interactions.iloc[0]['feature_i']} × {top_interactions.iloc[0]['feature_j']} (strength: {top_interactions.iloc[0]['mean_abs_interaction']:.4f})"
            }

            write_summary_report(
                cancer_output_dir / 'main_vs_interaction_summary.txt',
                cancer_type,
                args.feature_set,
                results_dict
            )

            # Top interactions CSV
            top_interactions.to_csv(
                cancer_output_dir / 'top_interactions.csv',
                index=False
            )

            # Raw interaction values (compressed)
            np.savez_compressed(
                cancer_output_dir / 'interaction_values.npz',
                interaction_values=interaction_values,
                feature_names=analyzer.feature_names,
                cancer_type=cancer_type,
                feature_set=args.feature_set
            )

            # Collect for cross-cancer comparison
            all_results.append({
                'cancer_type': cancer_type,
                'n_samples': main_vs_interaction['n_samples'],
                'n_features': main_vs_interaction['n_features'],
                'main_pct': main_vs_interaction['main_pct'],
                'interaction_pct': main_vs_interaction['interaction_pct'],
                'test_auc': test_auc,
                'top_interaction_pair': f"{top_interactions.iloc[0]['feature_i']} × {top_interactions.iloc[0]['feature_j']}",
                'top_interaction_strength': top_interactions.iloc[0]['mean_abs_interaction']
            })

            all_top_interactions.append(top_interactions)

            logger.info(f"✓ Analysis complete for {cancer_type}")
            logger.info(f"  Results saved to: {cancer_output_dir}/")
            logger.info("")

        except Exception as e:
            logger.error(f"✗ Analysis failed for {cancer_type}: {str(e)}")
            logger.exception("Full traceback:")
            continue

    # Create cross-cancer comparison if multiple cancer types
    if len(args.cancer_types) > 1 and len(all_results) > 0:
        logger.info("=" * 70)
        logger.info("CREATING CROSS-CANCER COMPARISON")
        logger.info("=" * 70)

        # Summary table
        comparison_df = pd.DataFrame(all_results)
        comparison_df.to_csv(output_base / f'cross_cancer_comparison_{timestamp}.csv',
                            index=False)

        # Comparison plot
        plot_cross_cancer_comparison(
            comparison_df,
            output_base / f'cross_cancer_contribution_comparison_{timestamp}.pdf'
        )

        # Combined top interactions
        all_top_combined = pd.concat(all_top_interactions, ignore_index=True)
        all_top_combined.to_csv(
            output_base / f'all_top_interactions_{timestamp}.csv',
            index=False
        )

        logger.info(f"✓ Cross-cancer comparison saved to: {output_base}/")

    logger.info("")
    logger.info("=" * 70)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"All results saved to: {output_base}/")
    logger.info("")

    # Print summary
    if all_results:
        logger.info("Summary:")
        for result in all_results:
            logger.info(f"  {result['cancer_type']:8s}: Main {result['main_pct']:5.1f}%  |  Interaction {result['interaction_pct']:5.1f}%  |  AUC {result['test_auc']:.3f}")


if __name__ == '__main__':
    main()
