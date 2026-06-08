#!/usr/bin/env python3
"""
Feature Engineering Analysis for Balder Project

This script performs comprehensive analysis of all features in the database:
1. Feature summary statistics (type, missingness, cardinality, distributions)
2. Feature distributions stratified by actionability status
3. Missing data patterns visualization
4. Feature correlation heatmaps
5. Feature associations with outcome (chi-square for categorical, t-test for numeric)

Output: analysis_output/feature_engineering_analysis/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

# Add tools to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
from db_config import get_default_db_path
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class FeatureAnalyzer:
    """Comprehensive feature analysis for cancer genomics data"""

    def __init__(self, db_path: str, output_dir: str = None):
        """
        Initialize feature analyzer.

        Args:
            db_path: Path to SQLite database
            output_dir: Output directory for results (defaults to repo_root/analysis_output/feature_engineering_analysis)
        """
        self.db_path = Path(db_path)

        # If output_dir not specified, use repo root
        if output_dir is None:
            # Get repository root (2 levels up from this script)
            script_dir = Path(__file__).parent  # code/analysis/
            repo_root = script_dir.parent.parent  # idunn_dev/
            self.output_dir = repo_root / "analysis_output" / "feature_engineering_analysis"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create engine
        self.engine = create_engine(f"sqlite:///{self.db_path}")

        # Load data
        print(f"Loading data from {self.db_path}...")
        self.data = pd.read_sql_table('perPatientOncoKBFilteredForModeling', con=self.engine)
        print(f"Loaded {len(self.data)} patients with {self.data.shape[1]} features")

        # Define feature categories (based on CLAUDE.md documentation)
        self.demographic_features = [
            'AGE_AT_SEQ_REPORT', 'SEX', 'RACE', 'ETHNICITY',
            'SAMPLE_TYPE', 'STAGE', 'TUMOR_PURITY'
        ]

        self.technical_features = [
            'SUBJECT_CENTER', 'SEQ_ASSAY_ID', 'library_selection',
            'calling_strategy', 'COVERAGE', 'SEQ_PLATFORM'
        ]

        # Outcome variable
        self.outcome = 'isOncoKbLevel1'

    def identify_feature_types(self) -> pd.DataFrame:
        """
        Identify data types for all features.

        Returns:
            DataFrame with feature metadata
        """
        print("\nAnalyzing feature types...")

        feature_info = []

        for col in self.data.columns:
            if col == self.outcome:
                continue  # Skip outcome variable

            info = {
                'feature': col,
                'dtype': str(self.data[col].dtype),
                'n_unique': self.data[col].nunique(),
                'n_missing': self.data[col].isnull().sum(),
                'pct_missing': 100 * self.data[col].isnull().sum() / len(self.data),
                'n_total': len(self.data)
            }

            # Determine feature category
            if col in self.demographic_features:
                info['category'] = 'demographic'
            elif col in self.technical_features:
                info['category'] = 'technical'
            else:
                info['category'] = 'other'

            # Infer semantic type
            if self.data[col].dtype in ['int64', 'float64']:
                if info['n_unique'] <= 10:
                    info['semantic_type'] = 'categorical_numeric'
                else:
                    info['semantic_type'] = 'continuous'
                    info['mean'] = self.data[col].mean()
                    info['median'] = self.data[col].median()
                    info['std'] = self.data[col].std()
                    info['min'] = self.data[col].min()
                    info['max'] = self.data[col].max()
            else:
                info['semantic_type'] = 'categorical'
                # Get top categories
                top_cats = self.data[col].value_counts().head(5)
                info['top_categories'] = ', '.join([f"{cat}({count})" for cat, count in top_cats.items()])

            feature_info.append(info)

        feature_df = pd.DataFrame(feature_info)

        # Sort by category and missingness
        feature_df = feature_df.sort_values(['category', 'pct_missing'], ascending=[True, False])

        return feature_df

    def test_feature_association(self, feature: str) -> dict:
        """
        Test association between feature and outcome.

        Args:
            feature: Feature name

        Returns:
            Dictionary with test results
        """
        # Remove missing values
        data_clean = self.data[[feature, self.outcome]].dropna()

        if len(data_clean) < 50:
            return {'test': 'insufficient_data', 'p_value': np.nan, 'statistic': np.nan}

        result = {'feature': feature}

        # Determine test based on feature type
        if self.data[feature].dtype in ['int64', 'float64'] and self.data[feature].nunique() > 10:
            # Continuous variable - use Mann-Whitney U test
            actionable = data_clean[data_clean[self.outcome] == 1][feature]
            non_actionable = data_clean[data_clean[self.outcome] == 0][feature]

            try:
                statistic, p_value = mannwhitneyu(actionable, non_actionable, alternative='two-sided')
                result['test'] = 'mann_whitney'
                result['statistic'] = statistic
                result['p_value'] = p_value
                result['mean_actionable'] = actionable.mean()
                result['mean_non_actionable'] = non_actionable.mean()
                result['effect_size'] = (actionable.mean() - non_actionable.mean()) / data_clean[feature].std()
            except Exception as e:
                result['test'] = 'error'
                result['error'] = str(e)
                result['p_value'] = np.nan
        else:
            # Categorical variable - use chi-square test
            try:
                contingency = pd.crosstab(data_clean[feature], data_clean[self.outcome])
                chi2, p_value, dof, expected = chi2_contingency(contingency)

                result['test'] = 'chi_square'
                result['statistic'] = chi2
                result['p_value'] = p_value
                result['dof'] = dof

                # Calculate Cramér's V
                n = contingency.sum().sum()
                min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
                result['cramers_v'] = np.sqrt(chi2 / (n * min_dim))
            except Exception as e:
                result['test'] = 'error'
                result['error'] = str(e)
                result['p_value'] = np.nan

        return result

    def generate_feature_summary(self) -> pd.DataFrame:
        """
        Generate comprehensive feature summary table.

        Returns:
            DataFrame with feature summary statistics and associations
        """
        print("\nGenerating comprehensive feature summary...")

        # Get basic feature info
        feature_df = self.identify_feature_types()

        # Test associations with outcome
        print("Testing feature associations with outcome...")
        associations = []
        for feature in feature_df['feature']:
            assoc = self.test_feature_association(feature)
            associations.append(assoc)

        assoc_df = pd.DataFrame(associations)

        # Merge with feature info
        summary = feature_df.merge(assoc_df, on='feature', how='left')

        # Apply FDR correction to p-values
        valid_pvals = summary['p_value'].notna()
        if valid_pvals.sum() > 0:
            from statsmodels.stats.multitest import multipletests
            _, pvals_corrected, _, _ = multipletests(
                summary.loc[valid_pvals, 'p_value'],
                method='fdr_bh'
            )
            summary.loc[valid_pvals, 'p_value_fdr'] = pvals_corrected
            summary.loc[valid_pvals, 'significant_fdr'] = pvals_corrected < 0.05

        # Save summary table
        output_file = self.output_dir / 'feature_summary_table.csv'
        summary.to_csv(output_file, index=False)
        print(f"Saved feature summary table to {output_file}")

        return summary

    def visualize_missing_patterns(self):
        """Generate missing data patterns visualization"""
        print("\nVisualizing missing data patterns...")

        # Calculate missingness by feature
        missing_pct = (self.data.isnull().sum() / len(self.data) * 100).sort_values(ascending=False)

        # Filter to features with >0% missing
        missing_pct = missing_pct[missing_pct > 0]

        if len(missing_pct) == 0:
            print("No missing data found")
            return

        # Create visualization
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Bar plot of missingness
        ax1 = axes[0]
        missing_pct.plot(kind='barh', ax=ax1, color='steelblue')
        ax1.set_xlabel('Percentage Missing (%)')
        ax1.set_ylabel('Feature')
        ax1.set_title('Missing Data by Feature', fontsize=14, fontweight='bold')
        ax1.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        ax1.legend()

        # Heatmap of missingness patterns
        ax2 = axes[1]

        # Select top 30 features with most missingness for heatmap
        top_missing = missing_pct.head(30).index.tolist()
        missing_matrix = self.data[top_missing].isnull().astype(int)

        # Sample patients for visualization if too many
        if len(missing_matrix) > 1000:
            missing_matrix = missing_matrix.sample(n=1000, random_state=42)

        sns.heatmap(missing_matrix.T, cmap='RdYlGn_r', cbar_kws={'label': 'Missing'},
                   yticklabels=True, xticklabels=False, ax=ax2)
        ax2.set_title('Missing Data Pattern (Top 30 Features, Sample of Patients)',
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Patients (sample)')
        ax2.set_ylabel('Features')

        plt.tight_layout()
        output_file = self.output_dir / 'missing_data_patterns.pdf'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved missing data visualization to {output_file}")

    def visualize_feature_distributions(self, summary_df: pd.DataFrame):
        """
        Create feature distribution visualizations stratified by actionability.

        Args:
            summary_df: Feature summary DataFrame
        """
        print("\nCreating feature distribution visualizations...")

        # Focus on significant features with low missingness
        sig_features = summary_df[
            (summary_df['significant_fdr'] == True) &
            (summary_df['pct_missing'] < 20)
        ]['feature'].tolist()

        if len(sig_features) == 0:
            print("No significant features with <20% missingness found")
            return

        print(f"Visualizing {len(sig_features)} significant features...")

        # Create visualizations for top 20 most significant features
        sig_features_sorted = summary_df[
            summary_df['feature'].isin(sig_features)
        ].sort_values('p_value_fdr').head(20)['feature'].tolist()

        # Split into continuous and categorical
        continuous_features = []
        categorical_features = []

        for feat in sig_features_sorted:
            if self.data[feat].dtype in ['int64', 'float64'] and self.data[feat].nunique() > 10:
                continuous_features.append(feat)
            else:
                categorical_features.append(feat)

        # Visualize continuous features
        if continuous_features:
            n_features = len(continuous_features)
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            axes = axes.flatten() if n_features > 1 else [axes]

            for idx, feat in enumerate(continuous_features):
                ax = axes[idx]
                data_clean = self.data[[feat, self.outcome]].dropna()

                # Create violin plot
                sns.violinplot(data=data_clean, x=self.outcome, y=feat, ax=ax)
                ax.set_xlabel('Actionable Event')
                ax.set_ylabel(feat)
                ax.set_title(f'{feat} by Actionability', fontsize=10)
                ax.set_xticklabels(['No', 'Yes'])

            # Hide empty subplots
            for idx in range(len(continuous_features), len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()
            output_file = self.output_dir / 'continuous_features_distributions.pdf'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved continuous feature distributions to {output_file}")

        # Visualize categorical features
        if categorical_features:
            n_features = min(len(categorical_features), 12)  # Limit to 12 for readability
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            axes = axes.flatten() if n_features > 1 else [axes]

            for idx, feat in enumerate(categorical_features[:12]):
                ax = axes[idx]
                data_clean = self.data[[feat, self.outcome]].dropna()

                # Create grouped bar plot
                ct = pd.crosstab(data_clean[feat], data_clean[self.outcome], normalize='index') * 100
                ct.plot(kind='bar', ax=ax, legend=False)
                ax.set_xlabel(feat)
                ax.set_ylabel('Percentage (%)')
                ax.set_title(f'{feat} by Actionability', fontsize=10)
                ax.tick_params(axis='x', rotation=45)

                if idx == 1:  # Add legend to second plot
                    ax.legend(['Non-Actionable', 'Actionable'], loc='upper right')

            # Hide empty subplots
            for idx in range(n_features, len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()
            output_file = self.output_dir / 'categorical_features_distributions.pdf'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved categorical feature distributions to {output_file}")

    def create_correlation_heatmap(self):
        """Create correlation heatmap for numeric features"""
        print("\nCreating feature correlation heatmap...")

        # Select numeric features
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()

        # Remove outcome variable and features with >50% missing
        numeric_cols = [col for col in numeric_cols
                       if col != self.outcome and
                       self.data[col].isnull().sum() / len(self.data) < 0.5]

        if len(numeric_cols) < 2:
            print("Insufficient numeric features for correlation analysis")
            return

        # Calculate correlation matrix
        corr_matrix = self.data[numeric_cols].corr()

        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 12))

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm',
                   center=0, square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8}, ax=ax)

        ax.set_title('Feature Correlation Heatmap (Numeric Features)',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        output_file = self.output_dir / 'feature_correlation_heatmap.pdf'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved correlation heatmap to {output_file}")

        # Save correlation matrix
        corr_matrix.to_csv(self.output_dir / 'feature_correlations.csv')

    def generate_summary_report(self, summary_df: pd.DataFrame):
        """
        Generate text summary report.

        Args:
            summary_df: Feature summary DataFrame
        """
        print("\nGenerating summary report...")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FEATURE ENGINEERING ANALYSIS REPORT")
        report_lines.append("Balder Cancer Genomics Project")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Dataset overview
        report_lines.append("DATASET OVERVIEW")
        report_lines.append("-" * 80)
        report_lines.append(f"Total patients: {len(self.data):,}")
        report_lines.append(f"Total features: {self.data.shape[1]}")
        report_lines.append(f"Actionable events: {self.data[self.outcome].sum():,} ({100*self.data[self.outcome].mean():.1f}%)")
        report_lines.append(f"Cancer types: {self.data['ONCOTREE_CODE'].nunique()}")
        report_lines.append("")

        # Feature categories
        report_lines.append("FEATURE CATEGORIES")
        report_lines.append("-" * 80)
        cat_counts = summary_df['category'].value_counts()
        for cat, count in cat_counts.items():
            report_lines.append(f"  {cat}: {count} features")
        report_lines.append("")

        # Missingness summary
        report_lines.append("MISSING DATA SUMMARY")
        report_lines.append("-" * 80)
        high_missing = summary_df[summary_df['pct_missing'] > 50]
        report_lines.append(f"Features with >50% missing: {len(high_missing)}")
        if len(high_missing) > 0:
            for _, row in high_missing.iterrows():
                report_lines.append(f"  - {row['feature']}: {row['pct_missing']:.1f}% missing")
        report_lines.append("")

        # Significant associations
        report_lines.append("SIGNIFICANT FEATURE ASSOCIATIONS (FDR < 0.05)")
        report_lines.append("-" * 80)
        sig_features = summary_df[summary_df['significant_fdr'] == True].sort_values('p_value_fdr')
        report_lines.append(f"Total significant features: {len(sig_features)}")
        report_lines.append("")

        # By category
        for cat in ['demographic', 'technical', 'other']:
            cat_sig = sig_features[sig_features['category'] == cat]
            if len(cat_sig) > 0:
                report_lines.append(f"{cat.upper()} features ({len(cat_sig)}):")
                for _, row in cat_sig.iterrows():
                    report_lines.append(f"  - {row['feature']}: p={row['p_value_fdr']:.2e}")
                    if 'cramers_v' in row and not pd.isna(row['cramers_v']):
                        report_lines.append(f"    Cramér's V = {row['cramers_v']:.3f}")
                    elif 'effect_size' in row and not pd.isna(row['effect_size']):
                        report_lines.append(f"    Effect size = {row['effect_size']:.3f}")
                report_lines.append("")

        # Save report
        report_text = "\n".join(report_lines)
        output_file = self.output_dir / 'feature_analysis_report.txt'
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Saved summary report to {output_file}")

        # Also print to console
        print("\n" + report_text)

    def run_full_analysis(self):
        """Run complete feature engineering analysis"""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE FEATURE ENGINEERING ANALYSIS")
        print("="*80)

        # 1. Generate feature summary
        summary_df = self.generate_feature_summary()

        # 2. Visualize missing patterns
        self.visualize_missing_patterns()

        # 3. Visualize feature distributions
        self.visualize_feature_distributions(summary_df)

        # 4. Create correlation heatmap
        self.create_correlation_heatmap()

        # 5. Generate summary report
        self.generate_summary_report(summary_df)

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print(f"All outputs saved to: {self.output_dir}")
        print("="*80)


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Feature engineering analysis')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    args = parser.parse_args()

    # Database path
    db_path = get_default_db_path()

    # Create analyzer
    analyzer = FeatureAnalyzer(db_path, output_dir=args.output_dir)

    # Run analysis
    analyzer.run_full_analysis()
