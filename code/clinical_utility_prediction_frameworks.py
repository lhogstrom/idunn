"""
Clinical Utility Prediction Frameworks

This script implements clinically meaningful prediction tasks that address real clinical decisions,
moving beyond the binary presence/absence classification to more actionable clinical questions:

1. Optimal Testing Strategy: Predict which patients benefit from expanded panel testing
2. Actionability Risk Stratification: Multi-class prediction for different actionability levels
3. Time-to-Actionable-Event: Survival analysis for longitudinal monitoring
4. Clinical Decision Support: Resource allocation and cost-effectiveness modeling

These frameworks provide more direct clinical utility compared to simple binary classification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import os
import logging
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

# Survival analysis
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
    SURVIVAL_AVAILABLE = True
except ImportError:
    SURVIVAL_AVAILABLE = False
    logging.warning("lifelines not available. Survival analysis will be skipped.")

# Clinical utility metrics
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClinicalUtilityFrameworks:
    """
    Advanced prediction frameworks focused on clinical utility
    """

    def __init__(self, db_path, output_dir):
        self.db_path = db_path
        self.output_dir = output_dir
        self.engine = create_engine(f"sqlite:///{db_path}")

        # Demographics features (no technical features to avoid data leakage)
        self.demographic_features = [
            'SAMPLE_TYPE',
            'SAMPLE_TYPE_DETAILED',
            'STAGE',
            'SEX',
            'RACE',
            'ETHNICITY',
            'AGE_AT_SEQ_REPORT',
            'TUMOR_PURITY',
            'TUMOR_PLOIDY'
        ]

    def load_comprehensive_data(self):
        """Load all relevant data for clinical utility modeling"""

        # Load main patient data
        patient_data = pd.read_sql_table('perPatientOncoKBLevelSorted', con=self.engine)

        # Load genomic position data for panel coverage analysis
        try:
            genomic_data = pd.read_sql_table('genomicPositionLookup', con=self.engine)
        except:
            genomic_data = None
            logger.warning("genomicPositionLookup table not available")

        # Load OncoKB annotation data
        try:
            oncokb_data = pd.read_sql_table('oncoKBAnnotatedVariantTable', con=self.engine)
        except:
            oncokb_data = None
            logger.warning("oncoKBAnnotatedVariantTable not available")

        return patient_data, genomic_data, oncokb_data

    def create_optimal_testing_strategy_targets(self, patient_data, genomic_data=None):
        """
        Create targets for optimal testing strategy prediction

        Clinical Question: Which patients would benefit from expanded panel testing?

        Target Categories:
        - High Yield: Patients likely to have actionable variants with expanded testing
        - Standard Yield: Patients adequately served by standard panels
        - Low Yield: Patients unlikely to benefit from expanded testing
        """

        # Create testing strategy categories based on available data
        patient_data = patient_data.copy()

        # Initialize testing strategy recommendation
        patient_data['testing_strategy'] = 'standard'

        # High yield criteria (patients who would benefit from expanded testing)
        high_yield_criteria = (
            # Younger patients (more likely to have hereditary variants)
            (patient_data['AGE_AT_SEQ_REPORT'] < 50) |
            # Advanced stage cancers (need comprehensive profiling)
            (patient_data['STAGE'].isin(['Stage III', 'Stage IV'])) |
            # Metastatic samples (need therapy guidance)
            (patient_data['SAMPLE_TYPE'] == 'Metastasis') |
            # Cancer types with high actionability rates
            (patient_data['ONCOTREE_CODE'].isin(['LUAD', 'BRCA', 'CRC', 'BLCA']))
        )

        # Low yield criteria (patients unlikely to benefit)
        low_yield_criteria = (
            # Very elderly patients
            (patient_data['AGE_AT_SEQ_REPORT'] > 80) |
            # Early stage with good prognosis
            ((patient_data['STAGE'].isin(['Stage I', 'Stage II'])) &
             (patient_data['SAMPLE_TYPE'] == 'Primary'))
        )

        patient_data.loc[high_yield_criteria, 'testing_strategy'] = 'expanded'
        patient_data.loc[low_yield_criteria, 'testing_strategy'] = 'minimal'

        # If genomic data available, refine based on panel coverage
        if genomic_data is not None:
            # Patients in cancer types with low panel coverage benefit from expanded testing
            low_coverage_types = genomic_data.groupby('ONCOTREE_CODE')['perc_of_patients_with_coverage'].mean()
            low_coverage_types = low_coverage_types[low_coverage_types < 70].index

            high_yield_coverage = patient_data['ONCOTREE_CODE'].isin(low_coverage_types)
            patient_data.loc[high_yield_coverage, 'testing_strategy'] = 'expanded'

        # Create binary versions for different comparisons
        patient_data['needs_expanded_testing'] = (patient_data['testing_strategy'] == 'expanded').astype(int)
        patient_data['minimal_testing_sufficient'] = (patient_data['testing_strategy'] == 'minimal').astype(int)

        logger.info(f"Testing strategy distribution:")
        logger.info(patient_data['testing_strategy'].value_counts())

        return patient_data

    def create_actionability_risk_stratification(self, patient_data):
        """
        Create multi-level actionability risk stratification

        Clinical Question: What is the patient's risk level for having actionable variants?

        Risk Levels:
        - High Risk: >50% probability of actionable variants
        - Moderate Risk: 20-50% probability
        - Low Risk: <20% probability
        """

        patient_data = patient_data.copy()

        # Calculate actionability rates by relevant factors
        age_actionability = patient_data.groupby(pd.cut(patient_data['AGE_AT_SEQ_REPORT'],
                                                       bins=[0, 40, 60, 80, 100]))['isOncoKbLevel1'].mean()

        cancer_actionability = patient_data.groupby('ONCOTREE_CODE')['isOncoKbLevel1'].mean()

        stage_actionability = patient_data.groupby('STAGE')['isOncoKbLevel1'].mean()

        # Create risk score based on multiple factors
        patient_data['actionability_risk_score'] = 0.0

        # Age-based risk (younger patients often have higher actionability in some cancers)
        age_bins = pd.cut(patient_data['AGE_AT_SEQ_REPORT'], bins=[0, 40, 60, 80, 100])
        for age_bin in age_bins.cat.categories:
            mask = age_bins == age_bin
            if age_bin in age_actionability.index:
                patient_data.loc[mask, 'actionability_risk_score'] += age_actionability[age_bin] * 0.3

        # Cancer type-based risk
        for cancer_type in cancer_actionability.index:
            mask = patient_data['ONCOTREE_CODE'] == cancer_type
            patient_data.loc[mask, 'actionability_risk_score'] += cancer_actionability[cancer_type] * 0.5

        # Stage-based risk
        for stage in stage_actionability.index:
            mask = patient_data['STAGE'] == stage
            patient_data.loc[mask, 'actionability_risk_score'] += stage_actionability[stage] * 0.2

        # Create risk categories
        patient_data['actionability_risk_category'] = pd.cut(
            patient_data['actionability_risk_score'],
            bins=[0, 0.2, 0.5, 1.0],
            labels=['Low', 'Moderate', 'High'],
            include_lowest=True
        )

        # Create binary high-risk indicator
        patient_data['high_actionability_risk'] = (patient_data['actionability_risk_category'] == 'High').astype(int)

        logger.info(f"Actionability risk distribution:")
        logger.info(patient_data['actionability_risk_category'].value_counts())

        return patient_data

    def prepare_survival_analysis_data(self, patient_data):
        """
        Prepare data for time-to-actionable-event survival analysis

        Clinical Question: When will actionable variants be detected in longitudinal monitoring?
        """

        if not SURVIVAL_AVAILABLE:
            logger.warning("Survival analysis not available - lifelines package required")
            return None

        # For this analysis, we need to simulate time-to-event data since GENIE is cross-sectional
        # In practice, this would use longitudinal data with multiple time points

        patient_data = patient_data.copy()

        # Simulate time to actionable variant detection (months from first sequencing)
        # This is a placeholder - real implementation would use actual longitudinal data
        np.random.seed(42)

        # Base time influenced by patient characteristics
        base_time = np.random.exponential(scale=12, size=len(patient_data))  # 12 month average

        # Adjust based on patient characteristics
        # Younger patients: faster detection (more aggressive cancers)
        age_factor = np.where(patient_data['AGE_AT_SEQ_REPORT'] < 50, 0.7, 1.2)

        # Advanced stage: faster detection
        stage_factor = np.where(patient_data['STAGE'].isin(['Stage III', 'Stage IV']), 0.8, 1.3)

        # Metastatic samples: faster detection
        sample_factor = np.where(patient_data['SAMPLE_TYPE'] == 'Metastasis', 0.6, 1.1)

        patient_data['time_to_actionable'] = base_time * age_factor * stage_factor * sample_factor

        # Event indicator (1 if actionable variant detected, 0 if censored)
        # Use actual actionability status with some random censoring
        random_censoring = np.random.binomial(1, 0.1, len(patient_data))  # 10% random censoring
        patient_data['event_observed'] = np.where(
            random_censoring == 1, 0,  # Randomly censored
            patient_data['isOncoKbLevel1']  # Otherwise use actual actionability
        )

        # Adjust time for non-events (censoring time)
        max_followup = 24  # 24 months max follow-up
        patient_data.loc[patient_data['event_observed'] == 0, 'time_to_actionable'] = np.minimum(
            patient_data.loc[patient_data['event_observed'] == 0, 'time_to_actionable'],
            max_followup
        )

        logger.info(f"Survival analysis data prepared:")
        logger.info(f"Events observed: {patient_data['event_observed'].sum()}")
        logger.info(f"Median time to event: {patient_data['time_to_actionable'].median():.1f} months")

        return patient_data

    def clinical_utility_modeling(self, patient_data, prediction_target, cancer_types=None):
        """
        Build clinical utility-focused prediction models

        Args:
            patient_data: DataFrame with patient data
            prediction_target: Target variable for prediction
            cancer_types: List of cancer types to include (None for all)
        """

        if cancer_types is None:
            # Use cancer types with sufficient sample size
            type_counts = patient_data['ONCOTREE_CODE'].value_counts()
            cancer_types = type_counts[type_counts >= 100].index.tolist()

        results = {}

        for cancer_type in cancer_types:
            logger.info(f"Modeling {prediction_target} for {cancer_type}")

            # Subset data
            cancer_data = patient_data[patient_data['ONCOTREE_CODE'] == cancer_type].copy()

            if len(cancer_data) < 50:
                logger.warning(f"Insufficient data for {cancer_type}: {len(cancer_data)} patients")
                continue

            # Prepare features
            feature_cols = [col for col in self.demographic_features if col in cancer_data.columns]
            X = cancer_data[feature_cols].copy()

            if prediction_target not in cancer_data.columns:
                logger.warning(f"Target {prediction_target} not available for {cancer_type}")
                continue

            y = cancer_data[prediction_target]

            # Handle missing values
            # Numeric imputation
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                numeric_imputer = IterativeImputer(random_state=42)
                X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])

            # Categorical imputation
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                categorical_imputer = SimpleImputer(strategy='most_frequent')
                X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])

            # Skip if no variation in target
            if len(y.unique()) < 2:
                logger.warning(f"No variation in target for {cancer_type}")
                continue

            # Train-test split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
            except ValueError:
                # If stratification fails due to small classes
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )

            # Build preprocessing pipeline
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns

            preprocessing = ColumnTransformer([
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
            ])

            # Build model pipeline
            if prediction_target in ['testing_strategy']:
                # Multi-class classification
                from sklearn.multiclass import OneVsRestClassifier
                model = Pipeline([
                    ('preprocessor', preprocessing),
                    ('classifier', OneVsRestClassifier(RandomForestClassifier(
                        n_estimators=100, random_state=42, class_weight='balanced'
                    )))
                ])
            else:
                # Binary classification
                model = Pipeline([
                    ('preprocessor', preprocessing),
                    ('classifier', RandomForestClassifier(
                        n_estimators=100, random_state=42, class_weight='balanced'
                    ))
                ])

            # Train model
            model.fit(X_train, y_train)

            # Evaluate model
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            # Calculate metrics
            if len(np.unique(y)) == 2:  # Binary classification
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:  # Multi-class
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')

            # Clinical utility metrics
            model_metrics = {
                'cancer_type': cancer_type,
                'target': prediction_target,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'auc': auc,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }

            # Calculate calibration metrics
            if len(np.unique(y)) == 2:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_test, y_pred_proba[:, 1], n_bins=5
                )
                brier_score = brier_score_loss(y_test, y_pred_proba[:, 1])
                model_metrics.update({
                    'calibration_fraction_positive': fraction_of_positives.tolist(),
                    'calibration_mean_predicted': mean_predicted_value.tolist(),
                    'brier_score': brier_score
                })

            results[cancer_type] = {
                'model': model,
                'metrics': model_metrics,
                'predictions': {
                    'y_true': y_test.values,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
            }

        return results

    def survival_analysis_modeling(self, patient_data, cancer_types=None):
        """
        Perform survival analysis for time-to-actionable-event
        """

        if not SURVIVAL_AVAILABLE:
            logger.warning("Survival analysis not available - lifelines package required")
            return None

        if cancer_types is None:
            type_counts = patient_data['ONCOTREE_CODE'].value_counts()
            cancer_types = type_counts[type_counts >= 50].index.tolist()

        survival_results = {}

        for cancer_type in cancer_types:
            logger.info(f"Survival analysis for {cancer_type}")

            cancer_data = patient_data[patient_data['ONCOTREE_CODE'] == cancer_type].copy()

            if len(cancer_data) < 30:
                continue

            # Kaplan-Meier estimation
            kmf = KaplanMeierFitter()
            kmf.fit(cancer_data['time_to_actionable'],
                   event_observed=cancer_data['event_observed'])

            # Cox proportional hazards model
            # Prepare covariates
            cox_data = cancer_data.copy()

            # Encode categorical variables
            categorical_vars = ['SEX', 'RACE', 'STAGE', 'SAMPLE_TYPE']
            for var in categorical_vars:
                if var in cox_data.columns:
                    le = LabelEncoder()
                    cox_data[f'{var}_encoded'] = le.fit_transform(cox_data[var].fillna('Unknown'))

            # Select covariates for Cox model
            cox_covariates = ['AGE_AT_SEQ_REPORT']
            cox_covariates.extend([f'{var}_encoded' for var in categorical_vars if var in cox_data.columns])

            # Remove rows with missing essential data
            cox_data = cox_data.dropna(subset=['time_to_actionable', 'event_observed'] + cox_covariates)

            if len(cox_data) < 20:
                logger.warning(f"Insufficient data for Cox model in {cancer_type}")
                continue

            try:
                cph = CoxPHFitter()
                cph.fit(cox_data[['time_to_actionable', 'event_observed'] + cox_covariates],
                       duration_col='time_to_actionable',
                       event_col='event_observed')

                survival_results[cancer_type] = {
                    'kaplan_meier': kmf,
                    'cox_model': cph,
                    'median_survival': kmf.median_survival_time_,
                    'data': cox_data
                }

            except Exception as e:
                logger.warning(f"Cox model failed for {cancer_type}: {e}")
                survival_results[cancer_type] = {
                    'kaplan_meier': kmf,
                    'cox_model': None,
                    'median_survival': kmf.median_survival_time_,
                    'data': cox_data
                }

        return survival_results

    def run_comprehensive_clinical_utility_analysis(self):
        """
        Run comprehensive analysis across all clinical utility frameworks
        """

        logger.info("Starting comprehensive clinical utility analysis")

        # Load data
        patient_data, genomic_data, oncokb_data = self.load_comprehensive_data()

        # Create clinical utility targets
        logger.info("Creating optimal testing strategy targets...")
        patient_data = self.create_optimal_testing_strategy_targets(patient_data, genomic_data)

        logger.info("Creating actionability risk stratification...")
        patient_data = self.create_actionability_risk_stratification(patient_data)

        logger.info("Preparing survival analysis data...")
        patient_data = self.prepare_survival_analysis_data(patient_data)

        # Get eligible cancer types
        type_counts = patient_data['ONCOTREE_CODE'].value_counts()
        eligible_types = type_counts[type_counts >= 100].index.tolist()[:10]  # Top 10 for demo

        results = {}

        # 1. Optimal Testing Strategy Models
        logger.info("Building optimal testing strategy models...")
        testing_results = self.clinical_utility_modeling(
            patient_data, 'needs_expanded_testing', eligible_types
        )
        results['optimal_testing'] = testing_results

        # 2. Actionability Risk Stratification
        logger.info("Building actionability risk models...")
        risk_results = self.clinical_utility_modeling(
            patient_data, 'high_actionability_risk', eligible_types
        )
        results['actionability_risk'] = risk_results

        # 3. Survival Analysis
        if SURVIVAL_AVAILABLE:
            logger.info("Performing survival analysis...")
            survival_results = self.survival_analysis_modeling(patient_data, eligible_types)
            results['survival_analysis'] = survival_results

        # Compile summary results
        summary_metrics = []

        for framework, framework_results in results.items():
            if framework == 'survival_analysis':
                continue  # Handle separately

            for cancer_type, result in framework_results.items():
                metrics = result['metrics'].copy()
                metrics['framework'] = framework
                summary_metrics.append(metrics)

        summary_df = pd.DataFrame(summary_metrics)

        # Save results
        output_file = os.path.join(self.output_dir, 'clinical_utility_frameworks_results.csv')
        summary_df.to_csv(output_file, index=False)

        # Create summary visualizations
        self.create_clinical_utility_visualizations(results, summary_df)

        logger.info(f"Clinical utility analysis completed. Results saved to {self.output_dir}")

        return results, summary_df

    def create_clinical_utility_visualizations(self, results, summary_df):
        """Create visualizations for clinical utility analysis"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Performance comparison across frameworks
        ax1 = axes[0, 0]
        framework_performance = summary_df.groupby('framework')['auc'].agg(['mean', 'std']).reset_index()
        ax1.bar(framework_performance['framework'], framework_performance['mean'],
                yerr=framework_performance['std'], capsize=5)
        ax1.set_title('Model Performance by Clinical Framework')
        ax1.set_ylabel('AUC (Mean ± SD)')
        ax1.tick_params(axis='x', rotation=45)

        # 2. Sample size vs performance
        ax2 = axes[0, 1]
        for framework in summary_df['framework'].unique():
            framework_data = summary_df[summary_df['framework'] == framework]
            ax2.scatter(framework_data['n_test'], framework_data['auc'],
                       label=framework, alpha=0.7)
        ax2.set_xlabel('Test Set Size')
        ax2.set_ylabel('AUC')
        ax2.set_title('Sample Size vs Performance')
        ax2.legend()

        # 3. Performance distribution
        ax3 = axes[1, 0]
        auc_data = [summary_df[summary_df['framework'] == fw]['auc'] for fw in summary_df['framework'].unique()]
        ax3.boxplot(auc_data, labels=summary_df['framework'].unique())
        ax3.set_title('AUC Distribution by Framework')
        ax3.set_ylabel('AUC')
        ax3.tick_params(axis='x', rotation=45)

        # 4. Clinical utility heatmap
        ax4 = axes[1, 1]
        pivot_data = summary_df.pivot_table(values='auc', index='cancer_type', columns='framework')
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='viridis', ax=ax4)
        ax4.set_title('Clinical Utility Performance Heatmap')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'clinical_utility_analysis.pdf'),
                   dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main execution function"""

    # Configuration
    base_dir = '/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20250522'
    output_dir = os.path.join(base_dir, 'clinical_utility_frameworks')
    db_path = os.path.join(base_dir, 'balder-harmonized-biomarker-data-v20250522.sqlite')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize clinical utility frameworks
    frameworks = ClinicalUtilityFrameworks(db_path, output_dir)

    # Run comprehensive analysis
    try:
        results, summary_df = frameworks.run_comprehensive_clinical_utility_analysis()

        logger.info("=== CLINICAL UTILITY ANALYSIS SUMMARY ===")
        logger.info(f"Frameworks analyzed: {summary_df['framework'].unique()}")
        logger.info(f"Cancer types covered: {summary_df['cancer_type'].nunique()}")
        logger.info(f"Mean AUC across frameworks: {summary_df['auc'].mean():.3f}")

        # Best performing framework per cancer type
        best_frameworks = summary_df.loc[summary_df.groupby('cancer_type')['auc'].idxmax()]
        logger.info("\nBest performing framework per cancer type:")
        for _, row in best_frameworks.iterrows():
            logger.info(f"{row['cancer_type']}: {row['framework']} (AUC: {row['auc']:.3f})")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == '__main__':
    main()