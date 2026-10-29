#!/usr/bin/env python3
"""TIER 5: final model training and held-out test-set evaluation.

Trains one model per eligible cancer type (random forest or logistic regression) on the
training split only, then evaluates once on the held-out test split. Hyperparameters come
from the TIER 3 search result or the module defaults.

Outputs: models (joblib), metrics, per-sample test predictions, ROC / calibration /
confusion plots, AUC summary plots, and a summary report.

Usage:
    python run_final_model_training.py --algorithm logistic --quick-test
"""

import sys
import os
from pathlib import Path
import logging
import json
from datetime import datetime
import argparse
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    brier_score_loss, classification_report
)
from sklearn.calibration import calibration_curve

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))

from oncokb_analyzer.data_loader import DataLoader
from oncokb_analyzer.preprocessing import create_preprocessor, get_feature_list
from db_config import get_default_db_path
from feature_config import load_feature_config, copy_config_to_run_dir, get_config_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Eligible cancer types (>=250 actionable samples)
TIER1_CANCER_TYPES = [
    'LUAD', 'IDC', 'BRCA', 'MEL', 'GIST', 'COAD', 'THPA', 'ILC', 'PRAD', 'AML',
    'BLCA', 'SKCM', 'NSCLC', 'ODG', 'HGSOC', 'ASTR', 'MDLC', 'BRCNOS', 'BRCANOS',
    'COADREAD', 'IHCH'
]

# Default (hand-tuned) random forest hyperparameters; used only when no search result is supplied
DEFAULT_RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'max_features': 0.3,
    'min_samples_leaf': 10,
    'min_samples_split': 20,
    'class_weight': 'balanced_subsample',
    'random_state': 42,
    'n_jobs': -1,
    'oob_score': True,
    'bootstrap': True
}

# Default logistic regression hyperparameters
DEFAULT_LR_PARAMS = {
    'C': 1.0,
    'penalty': 'l2',
    'solver': 'lbfgs',
    'class_weight': 'balanced',
    'max_iter': 5000,
    'random_state': 42
}

ALGORITHM_DISPLAY_NAMES = {
    'random_forest': 'Random Forest',
    'logistic': 'Logistic Regression',
}


def resolve_hyperparams(algorithm, hyperparams_file=None, hyperparams_dir=None):
    """Resolve TIER 5 hyperparameters, preferring a TIER 3 search result.

    Returns (params, provenance). Searched keys are merged onto the defaults so that
    infrastructure settings (random_state, n_jobs, ...) survive. An explicitly requested
    source that cannot be resolved is a hard error rather than a silent fallback.
    """
    base = (DEFAULT_RF_PARAMS if algorithm == 'random_forest' else DEFAULT_LR_PARAMS).copy()

    src = None
    if hyperparams_file:
        src = Path(hyperparams_file)
        if not src.exists():
            raise SystemExit(
                f"FATAL: --hyperparams-file does not exist: {src}\n"
                "Refusing to silently fall back to the hand-tuned module constants.")
    elif hyperparams_dir:
        d = Path(hyperparams_dir)
        cands = sorted(d.glob(f'run_*/{algorithm}_universal_params.json'))
        if not cands:
            # Also accept the params file sitting directly in the given directory
            direct = d / f'{algorithm}_universal_params.json'
            cands = [direct] if direct.exists() else []
        if not cands:
            raise SystemExit(
                f"FATAL: no {algorithm}_universal_params.json under {d}\n"
                "TIER 3 did not run, did not complete, or wrote elsewhere. Re-run TIER 3, "
                "or pass --hyperparams-source defaults to use the hand-tuned constants "
                "deliberately. Refusing to fall back silently.")
        src = cands[-1]

    if src is None:
        return base, {'source': 'module defaults (hand-tuned)',
                      'searched_keys': [], 'default_keys': sorted(base)}

    with open(src) as f:
        data = json.load(f)
    searched = data.get('best_params_consensus', data)
    if not isinstance(searched, dict) or not searched:
        raise SystemExit(f"FATAL: no usable parameters in {src}")

    merged = base.copy()
    merged.update(searched)

    # max_iter is a solver convergence budget, not a capacity hyperparameter: a lower value can only
    # add a numerical defect. Take the max of the searched and default budgets.
    iter_floored = None
    if 'max_iter' in searched and 'max_iter' in base:
        if int(searched['max_iter']) < int(base['max_iter']):
            iter_floored = {'searched': searched['max_iter'], 'applied': base['max_iter'],
                            'reason': 'convergence budget floored at the module default'}
            merged['max_iter'] = base['max_iter']

    return merged, {
        'max_iter_floored': iter_floored,
        'source': str(src),
        'searched_keys': sorted(searched),
        'default_keys': sorted(k for k in merged if k not in searched),
        'overridden': {k: {'default': base.get(k), 'searched': searched[k]}
                       for k in sorted(searched) if k in base and base[k] != searched[k]},
        'search_mean_cv_auc': data.get('mean_score'),
        'search_param_stability': data.get('param_stability'),
        'search_n_samples': data.get('n_samples'),
    }


class FinalModelTrainer:
    """TIER 5: train final models on the training split and evaluate on the held-out test split."""

    def __init__(
        self,
        db_path: str,
        train_test_split_file: str,
        output_dir: str = '../../ml_experiments/final_models',
        hyperparams: dict = None,
        hyperparams_provenance: dict = None,
        feature_set: str = 'both',
        cancer_types: list = None,
        institution_weight_alpha: float = 0.0,
        institution_column: str = 'SUBJECT_CENTER',
        algorithm: str = 'random_forest'
    ):
        """Configure paths, split file, hyperparameters, feature set and institution weighting."""
        if algorithm not in ('random_forest', 'logistic'):
            raise ValueError(f"Algorithm must be 'random_forest' or 'logistic', got: {algorithm}")

        self.algorithm = algorithm
        self.algorithm_display = ALGORITHM_DISPLAY_NAMES[algorithm]
        self.db_path = db_path
        self.train_test_split_file = train_test_split_file
        self.feature_set = feature_set
        self.cancer_types = cancer_types or TIER1_CANCER_TYPES
        self.institution_weight_alpha = institution_weight_alpha
        self.institution_column = institution_column

        if hyperparams is not None:
            self.hyperparams = hyperparams
        elif algorithm == 'random_forest':
            self.hyperparams = DEFAULT_RF_PARAMS.copy()
        else:
            self.hyperparams = DEFAULT_LR_PARAMS.copy()
        # Record where the hyperparameters came from (searched vs hand-tuned)
        self.hyperparams_provenance = hyperparams_provenance or {
            'source': 'module defaults (hand-tuned)', 'searched_keys': [],
            'default_keys': sorted(self.hyperparams)}

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(output_dir) / f'run_{timestamp}'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir = self.output_dir / 'models'
        self.plots_dir = self.output_dir / 'plots'
        self.models_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)

        self._load_train_test_split()

        logger.info(f"FinalModelTrainer initialized")
        logger.info(f"Algorithm: {self.algorithm_display}")
        logger.info(f"Database: {db_path}")
        logger.info(f"Train/test split file: {train_test_split_file}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Feature set: {feature_set}")
        logger.info(f"Cancer types: {len(self.cancer_types)}")
        logger.info(f"Training patients: {len(self.train_patient_ids):,}")
        logger.info(f"Test patients: {len(self.test_patient_ids):,}")
        if self.institution_weight_alpha > 0:
            logger.info(f"Institution weighting: alpha={institution_weight_alpha} on '{institution_column}'")

    def _load_train_test_split(self):
        """Load train/test split from file."""
        split_path = Path(self.train_test_split_file)
        if not split_path.exists():
            raise FileNotFoundError(f"Train/test split file not found: {split_path}")

        split_df = pd.read_csv(split_path)

        self.train_patient_ids = split_df[split_df['split'] == 'train']['PATIENT_ID'].tolist()
        self.test_patient_ids = split_df[split_df['split'] == 'test']['PATIENT_ID'].tolist()
        self.split_df = split_df

        logger.info(f"Loaded train/test split: {len(self.train_patient_ids):,} train, {len(self.test_patient_ids):,} test")

    def _compute_institution_weights(self, train_data: pd.DataFrame) -> np.ndarray:
        """Institution sample weights (1 / count) ** alpha, normalized to mean 1."""
        if self.institution_weight_alpha <= 0:
            return None

        if self.institution_column not in train_data.columns:
            logger.warning(f"Institution column '{self.institution_column}' not found. Skipping weighting.")
            return None

        institution_counts = train_data[self.institution_column].value_counts()

        # weight = (1 / count) ^ alpha
        raw_weights = (1.0 / train_data[self.institution_column].map(institution_counts)) ** self.institution_weight_alpha

        # Normalize so weights sum to N (mean weight = 1.0)
        normalized_weights = raw_weights / raw_weights.sum() * len(raw_weights)

        weight_by_inst = train_data.groupby(self.institution_column).apply(
            lambda x: normalized_weights.loc[x.index].mean()
        ).sort_values(ascending=False)

        logger.info(f"  Institution weights (alpha={self.institution_weight_alpha}):")
        logger.info(f"    Weight range: [{normalized_weights.min():.3f}, {normalized_weights.max():.3f}]")
        logger.info(f"    Top 3 weighted: {dict(weight_by_inst.head(3).round(3))}")
        logger.info(f"    Bottom 3 weighted: {dict(weight_by_inst.tail(3).round(3))}")

        return normalized_weights.values

    def train_all_cancer_types(self) -> dict:
        """Train and evaluate a model for each cancer type; write outputs and the summary."""
        logger.info(f"\n{'#'*90}")
        logger.info(f"TIER 5: FINAL MODEL TRAINING & TEST SET EVALUATION")
        logger.info(f"Training {len(self.cancer_types)} cancer types with {self.algorithm_display}")
        logger.info(f"{'#'*90}\n")

        loader = DataLoader(self.db_path)

        all_results = {}
        summary_rows = []

        for i, cancer_type in enumerate(self.cancer_types, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"[{i}/{len(self.cancer_types)}] Training {cancer_type}")
            logger.info(f"{'='*70}")

            try:
                result = self._train_single_cancer_type(
                    cancer_type, loader
                )
                all_results[cancer_type] = result

                summary_rows.append({
                    'cancer_type': cancer_type,
                    'n_train': result['n_train'],
                    'n_test': result['n_test'],
                    'n_actionable_train': result['n_actionable_train'],
                    'n_actionable_test': result['n_actionable_test'],
                    'train_auc': result['train_auc'],
                    'test_auc': result['test_auc'],
                    'test_accuracy': result['test_accuracy'],
                    'test_precision': result['test_precision'],
                    'test_recall': result['test_recall'],
                    'test_f1': result['test_f1'],
                    'brier_score': result['brier_score'],
                    'status': 'completed'
                })

                logger.info(f"  {cancer_type} completed: Test AUC = {result['test_auc']:.4f}")

            except Exception as e:
                logger.error(f"  {cancer_type} FAILED: {str(e)}")
                all_results[cancer_type] = {'status': 'failed', 'error': str(e)}
                summary_rows.append({
                    'cancer_type': cancer_type,
                    'status': 'failed',
                    'error': str(e)
                })

        summary_df = pd.DataFrame(summary_rows)
        summary_file = self.output_dir / 'performance_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"\nSaved performance summary: {summary_file}")

        # Per-sample test predictions (used for DeLong CIs downstream)
        pred_rows = []
        for cancer_type, result in all_results.items():
            if result.get('status') != 'completed' or 'patient_ids' not in result:
                continue
            for pid, yt, yp in zip(result['patient_ids'],
                                   result['y_test'],
                                   result['y_test_proba']):
                pred_rows.append({
                    'cancer_type': cancer_type,
                    'patient_id': pid,
                    'y_true': int(yt),
                    'y_pred_proba': float(yp),
                })
        if pred_rows:
            preds_df = pd.DataFrame(pred_rows)
            preds_file = self.output_dir / 'test_predictions.csv'
            preds_df.to_csv(preds_file, index=False)
            logger.info(f"Saved test predictions: {preds_file} ({len(preds_df):,} rows)")

        self._plot_auc_summary(summary_df)

        self._generate_report(all_results, summary_df)

        hyperparams_file = self.output_dir / 'hyperparameters_used.json'
        with open(hyperparams_file, 'w') as f:
            json.dump(self.hyperparams, f, indent=2)

        # Provenance sidecar, kept separate so hyperparameters_used.json stays a plain estimator-ready dict
        with open(self.output_dir / 'hyperparameters_provenance.json', 'w') as f:
            json.dump(self.hyperparams_provenance, f, indent=2)

        config = {
            'algorithm': self.algorithm,
            'db_path': self.db_path,
            'train_test_split_file': self.train_test_split_file,
            'feature_set': self.feature_set,
            'cancer_types': self.cancer_types,
            'n_train_patients': len(self.train_patient_ids),
            'n_test_patients': len(self.test_patient_ids),
            'institution_weight_alpha': self.institution_weight_alpha,
            'institution_column': self.institution_column,
            'timestamp': datetime.now().isoformat()
        }
        config_file = self.output_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"\n{'#'*90}")
        logger.info(f"TIER 5 COMPLETE")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"{'#'*90}\n")

        return all_results

    def _create_model(self):
        """Create model instance based on configured algorithm and hyperparameters."""
        if self.algorithm == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**self.hyperparams)
        else:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**self.hyperparams)

    def _train_single_cancer_type(self, cancer_type: str, loader: DataLoader) -> dict:
        """Train and evaluate the model for one cancer type."""
        full_data = loader.load_patient_data()
        cancer_data = full_data[full_data['ONCOTREE_CODE'] == cancer_type].copy()

        if len(cancer_data) == 0:
            raise ValueError(f"No data found for cancer type: {cancer_type}")

        train_data = cancer_data[cancer_data['PATIENT_ID'].isin(self.train_patient_ids)].copy()
        test_data = cancer_data[cancer_data['PATIENT_ID'].isin(self.test_patient_ids)].copy()

        logger.info(f"  Data: {len(train_data):,} train, {len(test_data):,} test")

        if len(train_data) < 50:
            raise ValueError(f"Insufficient training data: {len(train_data)} samples")
        if len(test_data) < 10:
            raise ValueError(f"Insufficient test data: {len(test_data)} samples")

        feature_list = get_feature_list(self.feature_set)
        available_cols = [col for col in feature_list if col in train_data.columns]

        if not available_cols:
            raise ValueError(f"No feature columns found in data")

        X_train = train_data[available_cols].copy()
        y_train = train_data['isOncoKbLevel1'].copy()
        X_test = test_data[available_cols].copy()
        y_test = test_data['isOncoKbLevel1'].copy()

        preprocessor = create_preprocessor(available_columns=available_cols)

        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        logger.info(f"  Features: {len(available_cols)} raw, {X_train_processed.shape[1]} after encoding")
        logger.info(f"  Train actionable: {y_train.sum()} ({y_train.mean():.1%})")
        logger.info(f"  Test actionable: {y_test.sum()} ({y_test.mean():.1%})")

        sample_weights = self._compute_institution_weights(train_data)

        model = self._create_model()
        if sample_weights is not None:
            model.fit(X_train_processed, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train_processed, y_train)

        if self.algorithm == 'logistic':
            n_iter = model.n_iter_[0] if hasattr(model, 'n_iter_') else None
            if n_iter is not None:
                max_iter = self.hyperparams.get('max_iter', 1000)
                logger.info(f"  LR converged in {n_iter} iterations (max_iter={max_iter})")
                if n_iter >= max_iter:
                    logger.warning(f"  WARNING: LR may not have converged (n_iter={n_iter} >= max_iter={max_iter})")

        y_train_pred = model.predict(X_train_processed)
        y_train_proba = model.predict_proba(X_train_processed)[:, 1]
        y_test_pred = model.predict(X_test_processed)
        y_test_proba = model.predict_proba(X_test_processed)[:, 1]

        train_auc = roc_auc_score(y_train, y_train_proba)
        test_auc = roc_auc_score(y_test, y_test_proba)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        brier_score = brier_score_loss(y_test, y_test_proba)

        self._plot_roc_curve(y_test, y_test_proba, cancer_type, test_auc)
        self._plot_calibration_curve(y_test, y_test_proba, cancer_type)
        self._plot_confusion_matrix(y_test, y_test_pred, cancer_type)

        model_file = self.models_dir / f'{cancer_type}_{self.algorithm}_model.joblib'
        joblib.dump({
            'model': model,
            'preprocessor': preprocessor,
            'feature_columns': available_cols,
            'hyperparams': self.hyperparams,
            'algorithm': self.algorithm,
            'cancer_type': cancer_type,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'institution_weight_alpha': self.institution_weight_alpha,
            'institution_column': self.institution_column
        }, model_file)
        logger.info(f"  Saved model: {model_file}")

        return {
            'cancer_type': cancer_type,
            'n_train': len(train_data),
            'n_test': len(test_data),
            'n_actionable_train': int(y_train.sum()),
            'n_actionable_test': int(y_test.sum()),
            'train_auc': float(train_auc),
            'test_auc': float(test_auc),
            'test_accuracy': float(test_accuracy),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_f1': float(test_f1),
            'brier_score': float(brier_score),
            'patient_ids': test_data['PATIENT_ID'].tolist(),
            'y_test': y_test.tolist(),
            'y_test_proba': y_test_proba.tolist(),
            'y_test_pred': y_test_pred.tolist(),
            'status': 'completed'
        }

    def _plot_roc_curve(self, y_true, y_proba, cancer_type: str, auc: float):
        """Generate ROC curve plot."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'{self.algorithm_display} (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {cancer_type}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{cancer_type}_roc_curve.pdf', dpi=150)
        plt.close()

    def _plot_calibration_curve(self, y_true, y_proba, cancer_type: str):
        """Generate calibration curve plot."""
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy='uniform')

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(prob_pred, prob_true, 'b-o', linewidth=2, markersize=8, label=self.algorithm_display)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectly calibrated')
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(f'Calibration Curve - {cancer_type}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{cancer_type}_calibration_curve.pdf', dpi=150)
        plt.close()

    def _plot_confusion_matrix(self, y_true, y_pred, cancer_type: str):
        """Generate confusion matrix plot."""
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)

        classes = ['Not Actionable', 'Actionable']
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               ylabel='True Label',
               xlabel='Predicted Label')

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=14, fontweight='bold')

        ax.set_title(f'Confusion Matrix - {cancer_type}', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{cancer_type}_confusion_matrix.pdf', dpi=150)
        plt.close()

    def _plot_auc_summary(self, summary_df: pd.DataFrame):
        """Save the train vs. test AUC summary figure."""
        successful = summary_df[summary_df['status'] == 'completed'].copy()

        if len(successful) == 0:
            logger.warning("No successful cancer types to plot AUC summary")
            return

        successful = successful.sort_values('test_auc', ascending=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(successful) * 0.4)))

        ax1 = axes[0]
        colors = []
        for auc in successful['test_auc']:
            if auc >= 0.7:
                colors.append('#2ecc71')
            elif auc >= 0.6:
                colors.append('#3498db')
            elif auc >= 0.55:
                colors.append('#f39c12')
            else:
                colors.append('#e74c3c')

        y_pos = np.arange(len(successful))
        bars = ax1.barh(y_pos, successful['test_auc'], color=colors, edgecolor='black', linewidth=0.5)

        for i, (auc, bar) in enumerate(zip(successful['test_auc'], bars)):
            ax1.text(auc + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{auc:.3f}', va='center', ha='left', fontsize=9, fontweight='bold')

        ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, label='Random (0.5)')

        mean_auc = successful['test_auc'].mean()
        ax1.axvline(x=mean_auc, color='darkblue', linestyle='-', linewidth=2,
                   label=f'Mean ({mean_auc:.3f})')

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(successful['cancer_type'], fontsize=10)
        ax1.set_xlabel('Test AUC (ROC)', fontsize=12)
        ax1.set_title('Test Set Performance by Cancer Type', fontsize=14, fontweight='bold')
        ax1.set_xlim([0.4, max(successful['test_auc'].max() + 0.1, 0.85)])
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(axis='x', alpha=0.3)

        ax2 = axes[1]

        width = 0.35
        x_pos = np.arange(len(successful))

        bars_train = ax2.barh(x_pos - width/2, successful['train_auc'], width,
                              label='Train AUC', color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5)
        bars_test = ax2.barh(x_pos + width/2, successful['test_auc'], width,
                             label='Test AUC', color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=0.5)

        ax2.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        ax2.set_yticks(x_pos)
        ax2.set_yticklabels(successful['cancer_type'], fontsize=10)
        ax2.set_xlabel('AUC (ROC)', fontsize=12)
        ax2.set_title('Train vs Test AUC (Overfitting Check)', fontsize=14, fontweight='bold')
        ax2.set_xlim([0.4, max(successful['train_auc'].max() + 0.1, 0.95)])
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(axis='x', alpha=0.3)

        mean_diff = (successful['train_auc'] - successful['test_auc']).mean()
        overfitting_text = f"Mean (Train - Test): {mean_diff:.3f}"
        if mean_diff > 0.05:
            overfitting_text += " (Warning: >0.05)"
            text_color = 'red'
        else:
            overfitting_text += " (OK)"
            text_color = 'green'
        ax2.text(0.98, 0.02, overfitting_text, transform=ax2.transAxes,
                fontsize=10, ha='right', va='bottom', color=text_color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        summary_plot_file = self.output_dir / 'auc_summary_across_cancer_types.pdf'
        plt.savefig(summary_plot_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved AUC summary plot: {summary_plot_file}")

        self._plot_auc_summary_simple(successful)

    def _plot_auc_summary_simple(self, successful: pd.DataFrame):
        """Generate a simpler single-panel AUC summary bar chart."""
        fig, ax = plt.subplots(figsize=(10, max(6, len(successful) * 0.35)))

        successful = successful.sort_values('test_auc', ascending=True)

        colors = []
        for auc in successful['test_auc']:
            if auc >= 0.7:
                colors.append('#27ae60')
            elif auc >= 0.6:
                colors.append('#3498db')
            elif auc >= 0.55:
                colors.append('#f39c12')
            else:
                colors.append('#c0392b')

        y_pos = np.arange(len(successful))
        bars = ax.barh(y_pos, successful['test_auc'], color=colors, edgecolor='black', linewidth=0.5, height=0.7)

        for i, (auc, bar) in enumerate(zip(successful['test_auc'], bars)):
            ax.text(auc + 0.008, bar.get_y() + bar.get_height()/2,
                   f'{auc:.3f}', va='center', ha='left', fontsize=9)

        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Random baseline (0.5)')
        mean_auc = successful['test_auc'].mean()
        ax.axvline(x=mean_auc, color='navy', linestyle='-', linewidth=2, label=f'Mean AUC ({mean_auc:.3f})')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(successful['cancer_type'], fontsize=11)
        ax.set_xlabel('Test Set AUC (ROC)', fontsize=12)
        ax.set_title('TIER 5: Final Model Test AUC by Cancer Type', fontsize=14, fontweight='bold')
        ax.set_xlim([0.45, min(successful['test_auc'].max() + 0.08, 1.0)])
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)

        stats_text = (
            f"N = {len(successful)} cancer types\n"
            f"Mean AUC: {mean_auc:.3f}\n"
            f"Median: {successful['test_auc'].median():.3f}\n"
            f"Range: [{successful['test_auc'].min():.3f}, {successful['test_auc'].max():.3f}]"
        )
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

        plt.tight_layout()

        simple_plot_file = self.output_dir / 'auc_summary_barplot.pdf'
        plt.savefig(simple_plot_file, dpi=150, bbox_inches='tight')

        png_file = self.output_dir / 'auc_summary_barplot.png'
        plt.savefig(png_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved simple AUC summary: {simple_plot_file}")

    def _generate_report(self, all_results: dict, summary_df: pd.DataFrame):
        """Generate comprehensive text report."""
        successful = summary_df[summary_df['status'] == 'completed']
        failed = summary_df[summary_df['status'] == 'failed']

        report_lines = []
        report_lines.append("=" * 90)
        report_lines.append("TIER 5: FINAL MODEL TRAINING & TEST SET EVALUATION REPORT")
        report_lines.append("=" * 90)
        report_lines.append("")
        report_lines.append(f"Timestamp: {datetime.now().isoformat()}")
        report_lines.append(f"Database: {self.db_path}")
        report_lines.append(f"Feature set: {self.feature_set}")
        report_lines.append(f"Algorithm: {self.algorithm_display}")
        report_lines.append("")

        report_lines.append("-" * 90)
        report_lines.append("DATA SPLIT SUMMARY")
        report_lines.append("-" * 90)
        report_lines.append(f"Total training patients: {len(self.train_patient_ids):,} (80%)")
        report_lines.append(f"Total test patients: {len(self.test_patient_ids):,} (20%)")
        report_lines.append(f"Cancer types processed: {len(self.cancer_types)}")
        report_lines.append(f"Successful: {len(successful)}")
        report_lines.append(f"Failed: {len(failed)}")
        report_lines.append("")

        if len(successful) > 0:
            report_lines.append("-" * 90)
            report_lines.append("PERFORMANCE SUMMARY (Test Set - HELD OUT)")
            report_lines.append("-" * 90)
            report_lines.append("")
            report_lines.append(f"{'Cancer Type':<12} {'N_Train':>8} {'N_Test':>8} {'Train AUC':>10} {'Test AUC':>10} {'Accuracy':>10} {'F1':>8}")
            report_lines.append("-" * 90)

            for _, row in successful.iterrows():
                report_lines.append(
                    f"{row['cancer_type']:<12} {row['n_train']:>8,} {row['n_test']:>8,} "
                    f"{row['train_auc']:>10.4f} {row['test_auc']:>10.4f} "
                    f"{row['test_accuracy']:>10.4f} {row['test_f1']:>8.4f}"
                )

            report_lines.append("-" * 90)
            report_lines.append("")

            report_lines.append("AGGREGATE STATISTICS:")
            report_lines.append(f"  Mean Test AUC: {successful['test_auc'].mean():.4f} +/- {successful['test_auc'].std():.4f}")
            report_lines.append(f"  Median Test AUC: {successful['test_auc'].median():.4f}")
            report_lines.append(f"  Min Test AUC: {successful['test_auc'].min():.4f} ({successful.loc[successful['test_auc'].idxmin(), 'cancer_type']})")
            report_lines.append(f"  Max Test AUC: {successful['test_auc'].max():.4f} ({successful.loc[successful['test_auc'].idxmax(), 'cancer_type']})")
            report_lines.append("")
            report_lines.append(f"  Mean Brier Score: {successful['brier_score'].mean():.4f}")
            report_lines.append(f"  Mean F1 Score: {successful['test_f1'].mean():.4f}")
            report_lines.append("")

            train_test_diff = (successful['train_auc'] - successful['test_auc']).mean()
            report_lines.append("OVERFITTING CHECK:")
            report_lines.append(f"  Mean (Train AUC - Test AUC): {train_test_diff:.4f}")
            if train_test_diff > 0.05:
                report_lines.append(f"  WARNING: Possible overfitting (train-test gap > 0.05)")
            else:
                report_lines.append(f"  OK: No significant overfitting detected")
            report_lines.append("")

        if len(failed) > 0:
            report_lines.append("-" * 90)
            report_lines.append("FAILED CANCER TYPES")
            report_lines.append("-" * 90)
            for _, row in failed.iterrows():
                report_lines.append(f"  {row['cancer_type']}: {row.get('error', 'Unknown error')}")
            report_lines.append("")

        report_lines.append("-" * 90)
        report_lines.append(f"HYPERPARAMETERS USED ({self.algorithm_display})")
        report_lines.append("-" * 90)
        report_lines.append(f"  Source: {self.hyperparams_provenance.get('source')}")
        if self.hyperparams_provenance.get('searched_keys'):
            report_lines.append(
                f"  Searched (TIER 3): {', '.join(self.hyperparams_provenance['searched_keys'])}")
        for param, value in self.hyperparams.items():
            report_lines.append(f"  {param}: {value}")
        report_lines.append("")

        report_lines.append("-" * 90)
        report_lines.append("INSTITUTION SAMPLE WEIGHTING")
        report_lines.append("-" * 90)
        if self.institution_weight_alpha > 0:
            report_lines.append(f"  Weighting enabled: Yes")
            report_lines.append(f"  Alpha parameter: {self.institution_weight_alpha}")
            report_lines.append(f"  Institution column: {self.institution_column}")
            report_lines.append(f"  Formula: weight = (1 / institution_count) ^ alpha")
            report_lines.append(f"  Effect: Reduces influence of dominant institutions (MSK, DFCI)")
        else:
            report_lines.append(f"  Weighting enabled: No (alpha=0.0)")
            report_lines.append(f"  All samples weighted equally")
        report_lines.append("")

        report_lines.append("-" * 90)
        report_lines.append("OUTPUT FILES")
        report_lines.append("-" * 90)
        report_lines.append(f"  Models: {self.models_dir}/")
        report_lines.append(f"  Plots: {self.plots_dir}/")
        report_lines.append(f"  Summary: {self.output_dir}/performance_summary.csv")
        report_lines.append(f"  AUC Summary Plot: {self.output_dir}/auc_summary_barplot.pdf")
        report_lines.append(f"  AUC Comparison Plot: {self.output_dir}/auc_summary_across_cancer_types.pdf")
        report_lines.append(f"  Config: {self.output_dir}/config.json")
        report_lines.append("")

        report_lines.append("=" * 90)
        report_lines.append("NOTE: Test set was used for the FIRST TIME in this evaluation.")
        report_lines.append("All prior model development (hyperparameter tuning, feature selection)")
        report_lines.append("used only the training data to prevent data leakage.")
        report_lines.append("=" * 90)

        report_text = "\n".join(report_lines)

        report_file = self.output_dir / 'final_evaluation_report.txt'
        with open(report_file, 'w') as f:
            f.write(report_text)

        print("\n" + report_text)

        logger.info(f"Saved report: {report_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='TIER 5: Final model training and test set evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard run
    python run_final_model_training.py

    # Quick test with one cancer type
    python run_final_model_training.py --quick-test

    # Custom split file
    python run_final_model_training.py \\
        --train-test-split-file path/to/split.csv
        """
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default=get_default_db_path(),
        help='Path to SQLite database'
    )
    parser.add_argument(
        '--train-test-split-file',
        type=str,
        default='../../analysis_output/train_test_split/train_test_split.csv',
        help='Path to train/test split CSV file'
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['random_forest', 'logistic'],
        default='random_forest',
        help='Algorithm to use: random_forest or logistic (default: random_forest)'
    )
    parser.add_argument(
        '--hyperparams-file',
        type=str,
        default=None,
        help='Path to JSON file with model hyperparameters (uses algorithm-specific defaults if not provided)'
    )
    parser.add_argument(
        '--hyperparams-dir',
        type=str,
        default=None,
        help=(
            'TIER 3 output directory (tier3_hyperparameter_optimization/). The newest '
            'run_*/<algorithm>_universal_params.json beneath it is used. Resolved at TIER 5 '
            'runtime, so the orchestrator can pass it before TIER 3 has produced anything. '
            'Fails closed if nothing is found.'
        )
    )
    parser.add_argument(
        '--hyperparams-source',
        type=str,
        choices=['search', 'defaults'],
        default='search',
        help=(
            "Where final-model hyperparameters come from. 'search' (default) uses the TIER 3 "
            "result given by --hyperparams-file/--hyperparams-dir. 'defaults' uses the hand-tuned "
            "module constants -- which are NOT search-derived and were selected with reference to "
            "held-out performance; choosing it is "
            "a disclosable decision, which is why it must be explicit."
        )
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../../ml_experiments/final_models',
        help='Output directory for results'
    )
    parser.add_argument(
        '--feature-set',
        type=str,
        choices=['demographic', 'technical', 'both'],
        default='both',
        help='Feature set to use (default: both)'
    )
    parser.add_argument(
        '--cancer-types',
        type=str,
        nargs='+',
        default=None,
        help='Specific cancer types to process (default: all 21 tier 1 types)'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test with single cancer type (LUAD)'
    )
    parser.add_argument(
        '--feature-config',
        type=str,
        default=None,
        help=(
            'Path to custom feature configuration YAML file. '
            'Default: code/config/features.yaml. '
            'Can also be set via BALDER_FEATURES_CONFIG environment variable.'
        )
    )
    parser.add_argument(
        '--institution-weight-alpha',
        type=float,
        default=0.0,
        help=(
            'Power parameter for institutional sample weighting (0.0 to 1.0). '
            '0.0 = no weighting (default), 0.5 = moderate correction, 1.0 = full inverse frequency. '
            'Higher values reduce the influence of dominant institutions (MSK, DFCI) during training.'
        )
    )
    parser.add_argument(
        '--institution-column',
        type=str,
        default='SUBJECT_CENTER',
        help='Column name for institution grouping (default: SUBJECT_CENTER)'
    )

    args = parser.parse_args()

    if args.feature_config:
        logger.info(f"Loading custom feature config: {args.feature_config}")
        load_feature_config(args.feature_config)
    else:
        load_feature_config()
    logger.info(f"Feature config loaded from: {get_config_path()}")

    if args.quick_test:
        args.cancer_types = ['LUAD']
        logger.info("Quick test mode: processing only LUAD")

    # Resolve hyperparameters: TIER 3 search result if available, else the hand-tuned
    # module constants, but only as a deliberate choice
    if args.hyperparams_source == 'defaults':
        hyperparams, hp_provenance = resolve_hyperparams(args.algorithm)
        logger.warning("Using hand-tuned module constants by explicit request "
                       "(--hyperparams-source defaults). These are NOT the TIER 3 search result.")
    else:
        hyperparams, hp_provenance = resolve_hyperparams(
            args.algorithm, args.hyperparams_file, args.hyperparams_dir)
        if hp_provenance['source'].startswith('module defaults'):
            logger.warning(
                "No --hyperparams-file/--hyperparams-dir given; falling back to the hand-tuned "
                "module constants. Pass a TIER 3 search result to use searched hyperparameters.")
        else:
            logger.info(f"Loaded TIER 3 hyperparameters from: {hp_provenance['source']}")
            logger.info(f"  searched keys : {hp_provenance['searched_keys']}")
            if hp_provenance.get('overridden'):
                logger.info(f"  overrides defaults: {hp_provenance['overridden']}")

    if not Path(args.db_path).exists():
        logger.error(f"Database not found: {args.db_path}")
        sys.exit(1)

    if not Path(args.train_test_split_file).exists():
        logger.error(f"Train/test split file not found: {args.train_test_split_file}")
        logger.error("Run define_train_test_split.py first to create the split file")
        sys.exit(1)

    trainer = FinalModelTrainer(
        db_path=args.db_path,
        train_test_split_file=args.train_test_split_file,
        output_dir=args.output_dir,
        hyperparams=hyperparams,
        hyperparams_provenance=hp_provenance,
        feature_set=args.feature_set,
        cancer_types=args.cancer_types,
        institution_weight_alpha=args.institution_weight_alpha,
        institution_column=args.institution_column,
        algorithm=args.algorithm
    )

    copied_config = copy_config_to_run_dir(str(trainer.output_dir), args.feature_config)
    logger.info(f"Feature config copied to: {copied_config}")

    results = trainer.train_all_cancer_types()

    successful = sum(1 for r in results.values() if r.get('status') == 'completed')
    logger.info(f"\nTIER 5 Complete: {successful}/{len(results)} cancer types processed successfully")
    logger.info(f"Results saved to: {trainer.output_dir}")


if __name__ == '__main__':
    main()
