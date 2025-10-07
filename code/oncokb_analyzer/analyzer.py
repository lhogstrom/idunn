"""
Simple OncoKB Analyzer - Main Orchestrator

Minimal implementation for end-to-end analysis with basic modular structure.
No advanced features - just basic functionality to get working.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from .config import Config
from .data_loader import DataLoader
from .trainer import ModelTrainer


class OncoKBAnalyzer:
    """
    Simple main orchestrator for OncoKB actionability analysis.

    Coordinates basic components: config, data loading, and model training.
    """

    def __init__(self, config: Config):
        """
        Initialize analyzer with configuration.

        Args:
            config: Configuration object containing analysis parameters
        """
        self.config = config

        # Validate configuration
        self.config.validate()

        # Setup basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.data_loader = DataLoader(config.db_path)
        self.trainer = ModelTrainer(
            algorithm=config.algorithm,
            cv_folds=config.cv_folds,
            random_state=config.random_state
        )

        self.logger.info(f"Initialized OncoKBAnalyzer with {config.algorithm} algorithm")

    def run_analysis(self) -> Dict[str, Any]:
        """
        Run basic end-to-end analysis for configured cancer types.

        Returns:
            Dictionary containing analysis results for each cancer type

        Raises:
            Exception: If analysis fails
        """
        start_time = datetime.now()
        self.logger.info("Starting OncoKB actionability analysis...")

        try:
            # Load data
            self.logger.info("Loading patient data...")
            data = self.data_loader.load_patient_data()

            # Get data summary
            data_summary = self.data_loader.get_data_summary(data)
            self.logger.info(f"Loaded data summary: {data_summary['total_patients']} patients, "
                            f"{data_summary['total_cancer_types']} cancer types, "
                            f"{data_summary['actionable_rate']:.1%} actionable rate")

            results = {
                'analysis_metadata': {
                    'start_time': start_time.isoformat(),
                    'config': self.config.to_dict(),
                    'data_summary': data_summary
                },
                'cancer_type_results': {}
            }

            # Analyze each configured cancer type
            for cancer_type in self.config.cancer_types:
                self.logger.info(f"Analyzing cancer type: {cancer_type}")

                try:
                    cancer_result = self._analyze_cancer_type(data, cancer_type)
                    results['cancer_type_results'][cancer_type] = cancer_result

                    self.logger.info(
                        f"  {cancer_type} completed - AUC: {cancer_result['cv_auc_mean']:.3f} ± "
                        f"{cancer_result['cv_auc_std']:.3f}"
                    )

                except Exception as e:
                    self.logger.error(f"Failed to analyze {cancer_type}: {str(e)}")
                    results['cancer_type_results'][cancer_type] = {
                        'error': str(e),
                        'status': 'failed'
                    }

            # Finalize results
            end_time = datetime.now()
            results['analysis_metadata']['end_time'] = end_time.isoformat()
            results['analysis_metadata']['duration_seconds'] = (end_time - start_time).total_seconds()

            # Basic summary
            successful_analyses = [r for r in results['cancer_type_results'].values()
                                 if r.get('status') != 'failed']

            self.logger.info(f"Analysis completed - {len(successful_analyses)}/{len(self.config.cancer_types)} "
                           f"cancer types successful")

            return results

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise

    def _analyze_cancer_type(self, data, cancer_type: str) -> Dict[str, Any]:
        """
        Analyze a single cancer type.

        Args:
            data: Full patient dataset
            cancer_type: Cancer type to analyze

        Returns:
            Dictionary with analysis results

        Raises:
            Exception: If analysis fails for this cancer type
        """
        # Filter data for this cancer type
        cancer_data = self.data_loader.filter_for_cancer_type(data, cancer_type)

        # Check minimum actionable events threshold
        n_actionable = cancer_data[self.config.outcome_column].sum()
        if n_actionable < self.config.min_actionable_events:
            raise ValueError(
                f"Insufficient actionable events for {cancer_type}: {n_actionable} "
                f"(minimum {self.config.min_actionable_events} required)"
            )

        # Prepare features based on configuration
        X, y = self.trainer.prepare_features(cancer_data, self.config.outcome_column, self.config.feature_set)

        # Train and evaluate model
        model_results = self.trainer.train_and_evaluate(X, y)

        # Add cancer type metadata
        model_results.update({
            'cancer_type': cancer_type,
            'n_patients': len(cancer_data),
            'n_actionable': int(n_actionable),
            'actionable_rate': float(n_actionable / len(cancer_data)),
            'n_features': X.shape[1],
            'status': 'completed'
        })

        return model_results

    def get_summary(self, results: Dict[str, Any]) -> str:
        """
        Get formatted summary of analysis results.

        Args:
            results: Analysis results from run_analysis()

        Returns:
            Formatted summary string
        """
        if not results:
            return "No analysis results available."

        metadata = results.get('analysis_metadata', {})
        cancer_results = results.get('cancer_type_results', {})

        # Basic statistics
        successful_results = {k: v for k, v in cancer_results.items()
                            if v.get('status') == 'completed'}

        if successful_results:
            aucs = [r['cv_auc_mean'] for r in successful_results.values()]
            mean_auc = sum(aucs) / len(aucs)
            min_auc = min(aucs)
            max_auc = max(aucs)
        else:
            mean_auc = min_auc = max_auc = 0.0

        summary = f"""
OncoKB Actionability Analysis Summary
====================================

Configuration:
- Algorithm: {metadata.get('config', {}).get('algorithm', 'unknown')}
- Cancer types: {', '.join(self.config.cancer_types)}
- Min actionable events: {self.config.min_actionable_events}

Results:
- Successful analyses: {len(successful_results)}/{len(cancer_results)}
- Mean AUC: {mean_auc:.3f}
- AUC range: {min_auc:.3f} - {max_auc:.3f}

Duration: {metadata.get('duration_seconds', 0):.1f} seconds
        """.strip()

        # Add per-cancer-type results
        if successful_results:
            summary += "\n\nPer Cancer Type Results:"
            for cancer_type, result in successful_results.items():
                summary += f"\n- {cancer_type}: AUC {result['cv_auc_mean']:.3f} ± {result['cv_auc_std']:.3f} " \
                          f"({result['n_patients']} patients, {result['n_actionable']} actionable)"

        return summary