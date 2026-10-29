"""Tests for StratifiedGroupKFold fold-size constraints.

Covers the default warning behavior and enforce_constraints=True, which raises ValueError.
"""

import numpy as np
import pytest
from oncokb_analyzer.stratified_group_kfold import StratifiedGroupKFold


def _make_data(institution_sizes):
    """Helper to create synthetic data with given institution sizes."""
    groups_list = []
    for name, count in institution_sizes.items():
        groups_list.extend([name] * count)
    n = len(groups_list)
    X = np.random.randn(n, 5)
    y = np.random.randint(0, 2, n)
    groups = np.array(groups_list)
    return X, y, groups


class TestBasicSplitting:
    """Verify core splitting behavior is unchanged."""

    def test_all_samples_used(self):
        X, y, groups = _make_data({'A': 500, 'B': 400, 'C': 300, 'D': 200, 'E': 100})
        cv = StratifiedGroupKFold(n_splits=5, random_state=42)
        all_test = []
        for train_idx, test_idx in cv.split(X, y, groups):
            all_test.extend(test_idx)
        assert sorted(all_test) == list(range(len(X)))

    def test_groups_not_split(self):
        X, y, groups = _make_data({'A': 500, 'B': 400, 'C': 300, 'D': 200, 'E': 100})
        cv = StratifiedGroupKFold(n_splits=5, random_state=42)
        for train_idx, test_idx in cv.split(X, y, groups):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            assert train_groups.isdisjoint(test_groups), "Groups should not appear in both train and test"

    def test_n_splits_folds(self):
        X, y, groups = _make_data({'A': 500, 'B': 400, 'C': 300, 'D': 200, 'E': 100})
        cv = StratifiedGroupKFold(n_splits=3, random_state=42)
        folds = list(cv.split(X, y, groups))
        assert len(folds) == 3


class TestDefaultWarningBehavior:
    """Default mode: violations only warn."""

    def test_small_folds_warn_only(self):
        """With enforce_constraints=False (default), small folds only warn."""
        X, y, groups = _make_data({'A': 100, 'B': 80, 'C': 60, 'D': 40, 'E': 30})
        cv = StratifiedGroupKFold(n_splits=5, min_fold_size=500, random_state=42)
        # Should NOT raise — just warns
        folds = list(cv.split(X, y, groups))
        assert len(folds) == 5

    def test_high_ratio_warn_only(self):
        """Large ratio with default mode should warn, not raise."""
        X, y, groups = _make_data({'A': 5000, 'B': 50, 'C': 50, 'D': 50, 'E': 50})
        cv = StratifiedGroupKFold(n_splits=5, max_fold_ratio=1.5, random_state=42)
        folds = list(cv.split(X, y, groups))
        assert len(folds) == 5


class TestEnforceConstraints:
    """enforce_constraints=True: violations raise."""

    def test_min_fold_size_enforced(self):
        """Folds below min_fold_size should raise ValueError."""
        # 5 groups with small sizes — each fold will be ~62 samples
        X, y, groups = _make_data({'A': 100, 'B': 80, 'C': 60, 'D': 40, 'E': 30})
        cv = StratifiedGroupKFold(
            n_splits=5, min_fold_size=500,
            enforce_constraints=True, random_state=42
        )
        with pytest.raises(ValueError, match="below minimum threshold"):
            list(cv.split(X, y, groups))

    def test_max_fold_ratio_enforced(self):
        """Folds exceeding max ratio should raise ValueError."""
        # One huge group forces imbalanced folds
        X, y, groups = _make_data({'A': 5000, 'B': 50, 'C': 50, 'D': 50, 'E': 50})
        cv = StratifiedGroupKFold(
            n_splits=5, max_fold_ratio=1.5,
            enforce_constraints=True, random_state=42
        )
        with pytest.raises(ValueError, match="exceeds maximum threshold"):
            list(cv.split(X, y, groups))

    def test_min_fold_fraction_enforced(self):
        """Folds below fraction of total should raise ValueError."""
        # With 5 folds, each should be ~20%. Set fraction to 15% and make one fold tiny.
        X, y, groups = _make_data({'A': 5000, 'B': 50, 'C': 50, 'D': 50, 'E': 50})
        cv = StratifiedGroupKFold(
            n_splits=5, min_fold_fraction=0.15,
            min_fold_size=0,  # disable absolute check
            max_fold_ratio=100.0,  # disable ratio check
            enforce_constraints=True, random_state=42
        )
        with pytest.raises(ValueError, match="below 15%"):
            list(cv.split(X, y, groups))

    def test_no_violation_passes(self):
        """Well-balanced data should pass even with enforce_constraints=True."""
        X, y, groups = _make_data({
            'A': 200, 'B': 200, 'C': 200, 'D': 200, 'E': 200,
            'F': 200, 'G': 200, 'H': 200, 'I': 200, 'J': 200
        })
        cv = StratifiedGroupKFold(
            n_splits=5, min_fold_size=300,
            max_fold_ratio=2.0, min_fold_fraction=0.10,
            enforce_constraints=True, random_state=42
        )
        folds = list(cv.split(X, y, groups))
        assert len(folds) == 5

    def test_multiple_violations_reported(self):
        """Multiple violations should all appear in the error message."""
        X, y, groups = _make_data({'A': 5000, 'B': 10, 'C': 10, 'D': 10, 'E': 10})
        cv = StratifiedGroupKFold(
            n_splits=5, min_fold_size=500, max_fold_ratio=1.5,
            min_fold_fraction=0.15,
            enforce_constraints=True, random_state=42
        )
        with pytest.raises(ValueError) as exc_info:
            list(cv.split(X, y, groups))
        msg = str(exc_info.value)
        assert "below minimum threshold" in msg
        assert "exceeds maximum threshold" in msg


class TestNewParameterDefaults:
    """Constraint parameters do not change default behavior."""

    def test_default_enforce_is_false(self):
        cv = StratifiedGroupKFold()
        assert cv.enforce_constraints is False

    def test_default_min_fold_fraction_is_zero(self):
        cv = StratifiedGroupKFold()
        assert cv.min_fold_fraction == 0.0

    def test_backward_compatible_init(self):
        """Construction without constraint parameters still works."""
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42,
                                   min_fold_size=500, max_fold_ratio=2.5)
        assert cv.enforce_constraints is False
        assert cv.min_fold_fraction == 0.0
