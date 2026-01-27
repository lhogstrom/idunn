"""
Stratified GroupKFold Implementation

Implements a cross-validation strategy that:
1. Keeps all samples from the same group (institution) together in the same fold
2. Balances fold sizes to minimize variance in patient counts across folds

This solves the critical fold imbalance problem where standard GroupKFold creates
folds with 2-11× differences in size due to institution size heterogeneity.

"""

import numpy as np
import pandas as pd
from typing import Generator, Tuple
from collections import defaultdict
import logging


class StratifiedGroupKFold:
    """
    Stratified GroupKFold cross-validator.

    Ensures that:
    1. All samples from the same group stay together (no group splitting)
    2. Folds have similar sizes (minimizes patient count variance)

    Algorithm:
    ---------
    1. Sort groups by size (descending)
    2. Assign largest groups round-robin to folds
    3. Iteratively assign remaining groups to the fold with fewest samples
    4. This greedy approach produces near-optimal fold balance

    Parameters:
    ----------
    n_splits : int, default=5
        Number of folds

    shuffle : bool, default=True
        Whether to shuffle groups before splitting (preserves group integrity)

    random_state : int, default=None
        Random seed for shuffling

    min_fold_size : int, default=500
        Minimum acceptable fold size (raise warning if violated)

    max_fold_ratio : float, default=2.5
        Maximum acceptable ratio between largest and smallest fold (raise warning if violated)

    Examples:
    --------
    >>> from stratified_group_kfold import StratifiedGroupKFold
    >>> import numpy as np
    >>>
    >>> X = np.random.randn(1000, 10)
    >>> y = np.random.randint(0, 2, 1000)
    >>> groups = np.random.choice(['MSK', 'DFCI', 'JHU', 'VICC', 'UHN'], 1000)
    >>>
    >>> cv = StratifiedGroupKFold(n_splits=5, random_state=42)
    >>> for train_idx, test_idx in cv.split(X, y, groups):
    >>>     print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    """

    def __init__(self, n_splits=5, shuffle=True, random_state=None,
                 min_fold_size=500, max_fold_ratio=2.5):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.min_fold_size = min_fold_size
        self.max_fold_ratio = max_fold_ratio
        self.logger = logging.getLogger(self.__class__.__name__)

        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations."""
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test set.

        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data

        y : array-like, shape (n_samples,), optional
            Target variable (not used for grouping, but required for sklearn compatibility)

        groups : array-like, shape (n_samples,)
            Group labels for the samples (e.g., institution IDs)

        Yields:
        ------
        train : ndarray
            The training set indices for that split

        test : ndarray
            The testing set indices for that split
        """
        if groups is None:
            raise ValueError("groups parameter is required for StratifiedGroupKFold")

        groups = np.asarray(groups)
        n_samples = len(groups)

        if len(groups) != len(X):
            raise ValueError(f"groups length ({len(groups)}) must match X length ({len(X)})")

        # Get unique groups and their sample counts
        unique_groups, group_counts = np.unique(groups, return_counts=True)
        n_groups = len(unique_groups)

        if n_groups < self.n_splits:
            raise ValueError(
                f"Cannot split {n_groups} groups into {self.n_splits} folds. "
                f"Reduce n_splits or increase number of groups."
            )

        self.logger.info(f"Splitting {n_samples} samples from {n_groups} groups into {self.n_splits} folds")

        # Create group-to-indices mapping
        group_indices = defaultdict(list)
        for idx, group in enumerate(groups):
            group_indices[group].append(idx)

        # Sort groups by size (descending) for better balance
        sorted_groups = sorted(
            unique_groups,
            key=lambda g: len(group_indices[g]),
            reverse=True
        )

        # Shuffle groups while maintaining size-based clustering
        # (shuffling within size buckets maintains balance while adding randomness)
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            # Shuffle within quartiles to maintain approximate size ordering
            quartile_size = len(sorted_groups) // 4 + 1
            for i in range(0, len(sorted_groups), quartile_size):
                quartile = sorted_groups[i:i+quartile_size]
                rng.shuffle(quartile)
                sorted_groups[i:i+quartile_size] = quartile

        # Assign groups to folds using greedy bin packing
        fold_assignments = self._assign_groups_to_folds(sorted_groups, group_indices)

        # Calculate and report fold statistics
        fold_sizes = [len(fold_assignments[i]) for i in range(self.n_splits)]
        fold_group_counts = [len(set(groups[fold_assignments[i]])) for i in range(self.n_splits)]

        self._report_fold_statistics(fold_sizes, fold_group_counts)

        # Generate train/test splits
        for fold_idx in range(self.n_splits):
            test_indices = np.array(fold_assignments[fold_idx])
            train_indices = np.concatenate([
                fold_assignments[i] for i in range(self.n_splits) if i != fold_idx
            ])

            yield train_indices, test_indices

    def _assign_groups_to_folds(self, sorted_groups, group_indices):
        """
        Assign groups to folds using greedy bin packing algorithm.

        This ensures:
        1. Each group goes entirely into one fold (no splitting)
        2. Folds have similar total sample sizes
        """
        # Initialize empty folds
        fold_assignments = [[] for _ in range(self.n_splits)]
        fold_sizes = np.zeros(self.n_splits, dtype=int)

        # Assign each group to the fold with fewest samples (greedy)
        for group in sorted_groups:
            indices = group_indices[group]
            group_size = len(indices)

            # Find fold with minimum size
            min_fold_idx = np.argmin(fold_sizes)

            # Assign this group to that fold
            fold_assignments[min_fold_idx].extend(indices)
            fold_sizes[min_fold_idx] += group_size

        return fold_assignments

    def _report_fold_statistics(self, fold_sizes, fold_group_counts):
        """Calculate and report fold balance statistics."""
        fold_sizes_arr = np.array(fold_sizes)
        mean_size = fold_sizes_arr.mean()
        std_size = fold_sizes_arr.std()
        cv_size = std_size / mean_size if mean_size > 0 else 0

        min_size = fold_sizes_arr.min()
        max_size = fold_sizes_arr.max()
        size_ratio = max_size / min_size if min_size > 0 else np.inf

        self.logger.info(f"Fold sizes: {fold_sizes}")
        self.logger.info(f"Fold group counts: {fold_group_counts}")
        self.logger.info(f"Fold size statistics: mean={mean_size:.0f}, std={std_size:.1f}, CV={cv_size:.3f}")
        self.logger.info(f"Size range: {min_size} to {max_size} (ratio: {size_ratio:.2f}×)")

        # Check for warnings
        if min_size < self.min_fold_size:
            self.logger.warning(
                f"WARNING: Smallest fold ({min_size}) is below minimum threshold ({self.min_fold_size})"
            )

        if size_ratio > self.max_fold_ratio:
            self.logger.warning(
                f"WARNING: Fold size ratio ({size_ratio:.2f}×) exceeds maximum threshold ({self.max_fold_ratio}×)"
            )

        return {
            'fold_sizes': fold_sizes,
            'fold_group_counts': fold_group_counts,
            'mean_size': mean_size,
            'std_size': std_size,
            'cv_size': cv_size,
            'min_size': min_size,
            'max_size': max_size,
            'size_ratio': size_ratio
        }


def calculate_fold_imbalance_metrics(X, y, groups, cv_splitter):
    """
    Calculate comprehensive fold imbalance metrics for a given CV splitter.

    Useful for comparing standard GroupKFold vs StratifiedGroupKFold.

    Parameters:
    ----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    groups : array-like
        Group labels
    cv_splitter : cross-validator
        Either GroupKFold or StratifiedGroupKFold instance

    Returns:
    -------
    dict : Fold balance metrics
        - fold_sizes: List of fold sizes
        - fold_group_counts: List of groups per fold
        - patients_cv: Coefficient of variation for patient counts
        - groups_cv: Coefficient of variation for group counts
        - size_ratio: Max fold size / min fold size
        - actionable_rates: Actionable rate per fold (if y is binary)
    """
    fold_sizes = []
    fold_group_counts = []
    fold_actionable_rates = []

    for train_idx, test_idx in cv_splitter.split(X, y, groups):
        fold_sizes.append(len(test_idx))
        fold_groups = np.unique(groups[test_idx])
        fold_group_counts.append(len(fold_groups))

        if y is not None:
            fold_y = y[test_idx] if isinstance(y, np.ndarray) else y.iloc[test_idx]
            actionable_rate = fold_y.mean() if hasattr(fold_y, 'mean') else np.mean(fold_y)
            fold_actionable_rates.append(actionable_rate)

    fold_sizes_arr = np.array(fold_sizes)
    fold_groups_arr = np.array(fold_group_counts)

    patients_cv = fold_sizes_arr.std() / fold_sizes_arr.mean() if fold_sizes_arr.mean() > 0 else 0
    groups_cv = fold_groups_arr.std() / fold_groups_arr.mean() if fold_groups_arr.mean() > 0 else 0
    size_ratio = fold_sizes_arr.max() / fold_sizes_arr.min() if fold_sizes_arr.min() > 0 else np.inf

    return {
        'fold_sizes': fold_sizes,
        'fold_group_counts': fold_group_counts,
        'patients_cv': patients_cv,
        'groups_cv': groups_cv,
        'size_ratio': size_ratio,
        'actionable_rates': fold_actionable_rates,
        'mean_fold_size': fold_sizes_arr.mean(),
        'std_fold_size': fold_sizes_arr.std(),
        'mean_groups_per_fold': fold_groups_arr.mean(),
        'std_groups_per_fold': fold_groups_arr.std()
    }


if __name__ == '__main__':
    """
    Demonstration and validation of StratifiedGroupKFold.
    """
    import logging
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

    print("="*80)
    print("STRATIFIED GROUPKFOLD DEMONSTRATION")
    print("="*80)

    # Simulate realistic data with heterogeneous institution sizes
    np.random.seed(42)

    # Create institutions with varying sizes (mimicking real cancer center data)
    institutions = {
        'MSK': 3000,      # Large academic center
        'DFCI': 2500,     # Large academic center
        'JHU': 1800,      # Large academic center
        'VICC': 1200,     # Medium center
        'UHN': 1000,      # Medium center
        'YALE': 800,      # Medium center
        'PROV': 600,      # Small center
        'WAKE': 400,      # Small center
        'CHOP': 300,      # Small center
        'GRCC': 200,      # Small center
        'NKI': 150,       # Small center
        'PHS': 100,       # Very small center
    }

    # Create dataset
    groups_list = []
    for inst, count in institutions.items():
        groups_list.extend([inst] * count)

    n_samples = len(groups_list)
    X = np.random.randn(n_samples, 10)
    y = np.random.randint(0, 2, n_samples)
    groups = np.array(groups_list)

    print(f"\nDataset: {n_samples} samples from {len(institutions)} institutions")
    print(f"Institution sizes: {list(institutions.values())}")

    # Compare standard GroupKFold vs Stratified GroupKFold
    from sklearn.model_selection import GroupKFold

    print("\n" + "="*80)
    print("STANDARD GROUPKFOLD")
    print("="*80)
    standard_gkf = GroupKFold(n_splits=5)
    standard_metrics = calculate_fold_imbalance_metrics(X, y, groups, standard_gkf)

    print(f"Fold sizes: {standard_metrics['fold_sizes']}")
    print(f"Patients CV: {standard_metrics['patients_cv']:.3f}")
    print(f"Size ratio: {standard_metrics['size_ratio']:.2f}×")
    print(f"Groups per fold: {standard_metrics['fold_group_counts']}")

    print("\n" + "="*80)
    print("STRATIFIED GROUPKFOLD")
    print("="*80)
    stratified_gkf = StratifiedGroupKFold(n_splits=5, random_state=42)
    stratified_metrics = calculate_fold_imbalance_metrics(X, y, groups, stratified_gkf)

    print(f"\nFold sizes: {stratified_metrics['fold_sizes']}")
    print(f"Patients CV: {stratified_metrics['patients_cv']:.3f}")
    print(f"Size ratio: {stratified_metrics['size_ratio']:.2f}×")
    print(f"Groups per fold: {stratified_metrics['fold_group_counts']}")

    print("\n" + "="*80)
    print("IMPROVEMENT")
    print("="*80)
    cv_improvement = (standard_metrics['patients_cv'] - stratified_metrics['patients_cv']) / standard_metrics['patients_cv'] * 100
    ratio_improvement = (standard_metrics['size_ratio'] - stratified_metrics['size_ratio']) / standard_metrics['size_ratio'] * 100

    print(f"Patients CV: {standard_metrics['patients_cv']:.3f} → {stratified_metrics['patients_cv']:.3f} ({cv_improvement:+.1f}%)")
    print(f"Size ratio: {standard_metrics['size_ratio']:.2f}× → {stratified_metrics['size_ratio']:.2f}× ({ratio_improvement:+.1f}%)")

    if stratified_metrics['patients_cv'] < 0.3:
        print("\n✅ SUCCESS: Stratified GroupKFold achieves well-balanced folds (CV < 0.3)")
    else:
        print(f"\n⚠️  WARNING: Fold imbalance still present (CV = {stratified_metrics['patients_cv']:.3f})")
