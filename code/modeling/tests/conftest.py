"""
Shared pytest fixtures for oncokb_analyzer tests.

Provides common fixtures for:
- Synthetic toy data for unit tests (no database required)
- Path setup for imports
- Memory cleanup between tests
"""

import gc
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================================
# Memory Management
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Force garbage collection after each test to prevent memory buildup."""
    yield
    gc.collect()


# ============================================================================
# Synthetic Data Fixtures (No Database Required)
# ============================================================================

@pytest.fixture
def toy_data_binary():
    """Generate toy binary classification data for testing."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=42,
        flip_y=0.05
    )
    return X, y


@pytest.fixture
def toy_dataframe():
    """Generate toy DataFrame with realistic column names."""
    np.random.seed(42)
    n_samples = 500

    df = pd.DataFrame({
        # Demographic features
        'SEX': np.random.choice(['Male', 'Female'], n_samples),
        'RACE': np.random.choice(['White', 'Black', 'Asian', 'Other'], n_samples),
        'ETHNICITY': np.random.choice(['Hispanic', 'Non-Hispanic', 'Unknown'], n_samples),
        'AGE_AT_SEQ_REPORT': np.random.randint(20, 90, n_samples),
        'SAMPLE_TYPE': np.random.choice(['Primary', 'Metastasis', 'Other'], n_samples),
        'SAMPLE_TYPE_DETAILED': np.random.choice(['Primary Tumor', 'Metastatic', 'Local Recurrence'], n_samples),

        # Technical features
        'SUBJECT_CENTER': np.random.choice(['MSK', 'DFCI', 'VICC', 'UHN'], n_samples),
        'number_of_genes': np.random.randint(100, 600, n_samples),
        'preservation_technique': np.random.choice(['FFPE', 'Fresh Frozen'], n_samples),
        'calling_strategy': np.random.choice(['tumor_only', 'tumor_normal'], n_samples),

        # Outcome
        'isOncoKbLevel1': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),

        # Patient ID (for grouping)
        'PATIENT_ID': [f'P{i:04d}' for i in range(n_samples)]
    })

    return df


@pytest.fixture
def toy_multi_cancer_data():
    """Generate synthetic multi-cancer dataset for universal vs specific testing."""
    from sklearn.datasets import make_classification

    cancer_data = {}

    for i, cancer in enumerate(['CANCER_A', 'CANCER_B', 'CANCER_C']):
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            weights=[0.6 + i*0.1, 0.4 - i*0.1],
            random_state=i * 100,
            flip_y=0.05
        )
        cancer_data[cancer] = (X, y, None)

    return cancer_data


@pytest.fixture
def small_dataframe():
    """Small DataFrame for quick tests."""
    np.random.seed(123)
    n_samples = 50

    return pd.DataFrame({
        'SEX': np.random.choice(['Male', 'Female'], n_samples),
        'AGE_AT_SEQ_REPORT': np.random.randint(30, 80, n_samples),
        'SUBJECT_CENTER': np.random.choice(['MSK', 'DFCI'], n_samples),
        'isOncoKbLevel1': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    })
