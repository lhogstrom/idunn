"""Split an "Unknown" sentinel into a binary indicator plus a mode-imputed value.

For each configured column (e.g. ANCESTRY, ETHNICITY) adds `<col>_missing` and
replaces "Unknown" with the training-set mode. The mode is fit on training data only.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Catch-all categories; warn if mode-imputation resolves to one, since that folds
# missing rows into a heterogeneous group.
DEFAULT_CATCHALL_VALUES: tuple[str, ...] = ("Other",)

# GENIE encodes unrecorded categorical values as the literal string "Unknown" (not NULL),
# so .notna() does not filter it. Shared with the post-hoc tests so both layers exclude the same sentinel.
DEFAULT_MISSING_SENTINEL: str = "Unknown"


class MissingnessIndicatorTransformer(BaseEstimator, TransformerMixin):
    """Add binary missingness indicators and mode-impute the "Unknown" sentinel."""

    def __init__(
        self,
        columns: Iterable[str],
        missing_value: str = DEFAULT_MISSING_SENTINEL,
        indicator_suffix: str = "_missing",
        catchall_values: Iterable[str] = DEFAULT_CATCHALL_VALUES,
    ):
        # sklearn's BaseEstimator requires __init__ to store parameters unchanged.
        self.columns = columns
        self.missing_value = missing_value
        self.indicator_suffix = indicator_suffix
        self.catchall_values = catchall_values

    def _columns_list(self):
        return list(self.columns)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"MissingnessIndicatorTransformer expects a pandas DataFrame; "
                f"got {type(X).__name__}."
            )

        cols = self._columns_list()
        missing_cols = [c for c in cols if c not in X.columns]
        if missing_cols:
            raise ValueError(
                f"Columns not in input DataFrame: {missing_cols}. "
                f"Available: {list(X.columns)}"
            )

        catchall_values = tuple(self.catchall_values)
        modes_: dict[str, str] = {}
        for col in cols:
            series = X[col]
            # Treat NaN as missing too, so a DB NULL cannot bypass the indicator.
            missing_mask = series.isna() | (series == self.missing_value)
            non_missing = series[~missing_mask]
            if len(non_missing) == 0:
                warnings.warn(
                    f"Column '{col}' is 100% missing in fit data "
                    f"(every value is NaN or '{self.missing_value}'); "
                    f"mode falls back to '{self.missing_value}'.",
                    RuntimeWarning,
                )
                modes_[col] = self.missing_value
            else:
                modes_[col] = non_missing.mode().iloc[0]
                # Warn when the imputed value is itself a catch-all bucket.
                if modes_[col] in catchall_values:
                    warnings.warn(
                        f"Column '{col}' mode-imputation value is "
                        f"'{modes_[col]}', a catch-all bucket. Rows previously "
                        f"marked '{self.missing_value}' will be folded into "
                        f"this heterogeneous group — interpret downstream "
                        f"'{modes_[col]}' one-hot effects with caution.",
                        RuntimeWarning,
                    )

        self.modes_ = modes_
        # Record fit-time columns so get_feature_names_out(None) can return them.
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = self.feature_names_in_.shape[0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "modes_"):
            raise RuntimeError(
                "MissingnessIndicatorTransformer must be fit before transform."
            )
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"MissingnessIndicatorTransformer expects a pandas DataFrame; "
                f"got {type(X).__name__}."
            )

        out = X.copy()
        for col in self._columns_list():
            if col not in out.columns:
                raise ValueError(
                    f"Column '{col}' missing in transform input. "
                    f"Available: {list(out.columns)}"
                )
            # NaN and the sentinel both count as missing.
            missing_mask = out[col].isna() | (out[col] == self.missing_value)
            out[f"{col}{self.indicator_suffix}"] = missing_mask.astype(np.int8)
            out[col] = out[col].where(~missing_mask, self.modes_[col])
        return out

    def get_feature_names_out(self, input_features: Optional[Iterable[str]] = None):
        cols = self._columns_list()
        if input_features is None:
            # Pipeline chaining passes None; return the full fit-time column set.
            if hasattr(self, "feature_names_in_"):
                input_features = list(self.feature_names_in_)
            else:
                input_features = cols
        else:
            input_features = list(input_features)
        indicators = [f"{c}{self.indicator_suffix}" for c in cols]
        return np.array(input_features + indicators, dtype=object)
