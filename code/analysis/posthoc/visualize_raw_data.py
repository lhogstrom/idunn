#!/usr/bin/env python3
"""
Visualize Raw Data for Significant Post-Hoc Associations

Creates detailed plots showing the underlying raw data for each significant
demographic and technical feature association identified by SHAP analysis.

Visualizations:
- Age distributions: Violin plots, box plots, histogram + KDE
- Sex contingency: Grouped bar charts, stacked percentage bars
- Race/Ethnicity contingency: Grouped bar charts, actionability rates with CI
- Technical features: Similar contingency visualizations

Key behavior:
- Only generates plots for SIGNIFICANT associations (FDR < 0.05)
- Includes both demographic AND technical features
- Associations must be identified as important by SHAP (top 5 rankings)

Usage:
    # Auto-detect input from latest run
    python visualize_raw_data.py

    # Specify input and output directories
    python visualize_raw_data.py --input-dir runs/run_20260127/tier7_posthoc_statistics --output-dir runs/run_20260127/tier7_posthoc_statistics/raw_data_plots

"""

import argparse
import pandas as pd
from pathlib import Path
import sys

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))
from db_config import get_default_db_path

# Import shared plotting utilities
from raw_data_plot_utils import (
    plot_age_distribution,
    plot_sex_contingency,
    plot_categorical_contingency
)

# Repository root for default paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Database configuration
DB_PATH = get_default_db_path()


def get_default_input_dir():
    """Find latest post-hoc results directory."""
    runs_dir = REPO_ROOT / "runs"
    if runs_dir.exists():
        runs = sorted(runs_dir.glob("run_*"), reverse=True)
        for run_dir in runs:
            posthoc_dir = run_dir / "tier7_posthoc_statistics"
            if posthoc_dir.exists():
                return posthoc_dir

    legacy_dir = REPO_ROOT / "analysis_output" / "posthoc_statistics"
    if legacy_dir.exists():
        return legacy_dir

    return None


def load_significant_results(input_dir):
    """
    Load only significant post-hoc results (FDR < 0.05).

    Returns dict with significant associations for each category.
    """
    input_dir = Path(input_dir)
    data = {}

    # Try subdirectory structure first (unified runs)
    age_dir = input_dir / "age_actionability"
    shap_dir = input_dir / "shap_associations"

    # Age results - filter to significant
    age_file = age_dir / "age_actionability_results.csv" if age_dir.exists() else input_dir / "age_actionability_results.csv"
    if age_file.exists():
        age_df = pd.read_csv(age_file)
        data['age'] = age_df[age_df['mann_whitney_significant_fdr'] == True].copy()
        print(f"Loaded {len(data['age'])} significant age associations (from {len(age_df)} tested)")

    # Sex associations - filter to significant
    sex_file = shap_dir / "sex_associations.csv" if shap_dir.exists() else input_dir / "sex_associations.csv"
    if sex_file.exists():
        sex_df = pd.read_csv(sex_file)
        data['sex'] = sex_df[sex_df['significant_fdr'] == True].copy()
        print(f"Loaded {len(data['sex'])} significant sex associations (from {len(sex_df)} tested)")

    # Race/ethnicity associations - filter to significant
    race_file = shap_dir / "race_ethnicity_associations.csv" if shap_dir.exists() else input_dir / "race_ethnicity_associations.csv"
    if race_file.exists():
        race_df = pd.read_csv(race_file)
        data['race_eth'] = race_df[race_df['significant_fdr'] == True].copy()
        print(f"Loaded {len(data['race_eth'])} significant race/ethnicity associations (from {len(race_df)} tested)")

    # Sample type associations - filter to significant
    sample_type_file = shap_dir / "sample_type_associations.csv" if shap_dir.exists() else input_dir / "sample_type_associations.csv"
    if sample_type_file.exists():
        sample_df = pd.read_csv(sample_type_file)
        data['sample_type'] = sample_df[sample_df['significant_fdr'] == True].copy()
        print(f"Loaded {len(data['sample_type'])} significant sample type associations (from {len(sample_df)} tested)")

    # Technical feature associations - filter to significant
    tech_file = shap_dir / "technical_feature_associations.csv" if shap_dir.exists() else input_dir / "technical_feature_associations.csv"
    if tech_file.exists():
        tech_df = pd.read_csv(tech_file)
        data['technical'] = tech_df[tech_df['significant_fdr'] == True].copy()
        print(f"Loaded {len(data['technical'])} significant technical associations (from {len(tech_df)} tested)")

    return data


def main():
    parser = argparse.ArgumentParser(
        description='Visualize raw data for significant post-hoc associations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-detect input from latest run
    python visualize_raw_data.py

    # Specify input and output directories
    python visualize_raw_data.py --input-dir runs/run_20260127/tier7_posthoc_statistics --output-dir runs/run_20260127/tier7_posthoc_statistics/raw_data_plots
        """
    )

    parser.add_argument('--input-dir', type=str, default=None,
                       help='Directory containing post-hoc analysis results')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for raw data plots')
    parser.add_argument('--db-path', type=str, default=None,
                       help='Path to SQLite database (uses default if not specified)')

    args = parser.parse_args()

    # Determine input directory
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = get_default_input_dir()
        if input_dir is None:
            print("ERROR: No post-hoc results found. Run analyze_shap_feature_associations.py first.")
            return 1

    print(f"Input directory: {input_dir}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir / "raw_data_plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Database path
    db_path = args.db_path or DB_PATH
    print(f"Database: {db_path}")

    # Load SIGNIFICANT results only
    print("\n" + "=" * 80)
    print("Loading SIGNIFICANT post-hoc results (FDR < 0.05)...")
    print("=" * 80)

    data = load_significant_results(input_dir)

    if not data:
        print("ERROR: No post-hoc results found")
        return 1

    total_plots = 0

    # SECTION 1: Age distributions (significant only)
    if 'age' in data and len(data['age']) > 0:
        print("\n" + "=" * 80)
        print(f"SECTION 1: Age Distributions ({len(data['age'])} significant associations)")
        print("=" * 80)

        for _, row in data['age'].iterrows():
            cancer = row['ONCOTREE_CODE']
            print(f"\nPlotting {cancer}...")
            plot_age_distribution(
                cancer,
                row['cohens_d'],
                row['mann_whitney_p_fdr'],
                row['mean_age_difference'],
                output_dir,
                db_path=db_path
            )
            total_plots += 1
    else:
        print("\nNo significant age associations to visualize")

    # SECTION 2: Sex associations (significant only)
    if 'sex' in data and len(data['sex']) > 0:
        print("\n" + "=" * 80)
        print(f"SECTION 2: Sex Associations ({len(data['sex'])} significant associations)")
        print("=" * 80)

        for _, row in data['sex'].iterrows():
            cancer = row['ONCOTREE_CODE']
            print(f"\nPlotting {cancer}...")
            plot_sex_contingency(
                cancer,
                row['odds_ratio'],
                row['p_fdr'],
                output_dir,
                db_path=db_path
            )
            total_plots += 1
    else:
        print("\nNo significant sex associations to visualize")

    # SECTION 3: Race/Ethnicity associations (significant only)
    if 'race_eth' in data and len(data['race_eth']) > 0:
        print("\n" + "=" * 80)
        print(f"SECTION 3: Race/Ethnicity Associations ({len(data['race_eth'])} significant associations)")
        print("=" * 80)

        for _, row in data['race_eth'].iterrows():
            cancer = row['ONCOTREE_CODE']
            column = row['column']
            print(f"\nPlotting {cancer} - {column}...")
            plot_categorical_contingency(
                cancer, column,
                row['cramers_v'],
                row['p_fdr'],
                output_dir,
                feature_type='demographic',
                db_path=db_path
            )
            total_plots += 1
    else:
        print("\nNo significant race/ethnicity associations to visualize")

    # SECTION 4: Technical feature associations (significant only)
    # Section 4: Sample Type
    if 'sample_type' in data and len(data['sample_type']) > 0:
        print("\n" + "=" * 80)
        print(f"SECTION 4: Sample Type Associations ({len(data['sample_type'])} significant associations)")
        print("=" * 80)

        for _, row in data['sample_type'].iterrows():
            cancer = row['ONCOTREE_CODE']
            column = row['column']
            print(f"\nPlotting {cancer} - {column}...")
            plot_categorical_contingency(
                cancer, column,
                row['cramers_v'],
                row['p_fdr'],
                output_dir,
                feature_type='demographic',
                db_path=db_path
            )
            total_plots += 1
    else:
        print("\nNo significant sample type associations to visualize")

    # Section 5: Technical Features
    if 'technical' in data and len(data['technical']) > 0:
        print("\n" + "=" * 80)
        print(f"SECTION 5: Technical Feature Associations ({len(data['technical'])} significant associations)")
        print("=" * 80)

        for _, row in data['technical'].iterrows():
            cancer = row['ONCOTREE_CODE']
            column = row['column']
            print(f"\nPlotting {cancer} - {column}...")
            plot_categorical_contingency(
                cancer, column,
                row['cramers_v'],
                row['p_fdr'],
                output_dir,
                feature_type='technical',
                db_path=db_path
            )
            total_plots += 1
    else:
        print("\nNo significant technical feature associations to visualize")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nGenerated raw data visualizations for SIGNIFICANT associations only:")
    print(f"  Age distributions: {len(data.get('age', []))} cancer types")
    print(f"  Sex contingency: {len(data.get('sex', []))} cancer types")
    print(f"  Race/Ethnicity contingency: {len(data.get('race_eth', []))} tests")
    print(f"  Sample Type contingency: {len(data.get('sample_type', []))} tests")
    print(f"  Technical feature contingency: {len(data.get('technical', []))} tests")
    print(f"\nTotal figures generated: {total_plots}")
    print(f"\nAll visualizations saved to: {output_dir}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
