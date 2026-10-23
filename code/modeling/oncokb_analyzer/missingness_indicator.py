"""
MissingnessIndicatorTransformer
================================

Transforms categorical columns that use the string value "Unknown" to encode
missingness (e.g., ANCESTRY, ETHNICITY) into:

  1. A binary indicator column `<col>_missing` ∈ {0, 1} added to the DataFrame
     (1 where the original value was "Unknown").
  2. The original column with "Unknown" replaced by the training-set mode of
     the non-Unknown values.

The fit step records the mode per column on training data only; the transform
step applies the recorded mode to any data (train, validation, test). This
keeps the train/test boundary clean — the test set never contributes to mode
selection.

Why this exists
---------------
NKI and CRUK are 100% Unknown for both RACE/ANCESTRY and ETHNICITY in the
GENIE cohort. When Unknown is treated as a one-hot category by the downstream
ColumnTransformer, the resulting `ANCESTRY_Unknown` dummy is a near-perfect
identifier for those institutions — turning the demographic-only model into a
covert institutional predictor. By splitting the signal into a binary indicator
(`ANCESTRY_missing`) and a mode-imputed value, we:

  - Preserve the institutional missingness signal in a named, interpretable
    feature that downstream analysis can inspect and remove from production
    interpretation.
  - Remove the institutional fingerprint from the ANCESTRY one-hot dummies so
    they reflect actual ancestry differences when present.

See `code_review_and_feedback/feature_review_report.md` Priority 2 and
BACKLOG.md "Feature Review Fixes" item (b).
"""

from __future__ import annotations

import warnings
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Categorical values that are themselves catch-all buckets. When mode-imputation
# resolves to one of these, the imputed rows get folded into an already-
# heterogeneous group, which partially defeats the missingness-isolation
# purpose of the transformer (pre-lock review #10, 2026-05-22). We warn rather
# than reject so the analyst can decide.
DEFAULT_CATCHALL_VALUES: tuple[str, ...] = ("Other",)

# The string GENIE uses to encode an unrecorded categorical value. It is a
# literal string in the DB, not a SQL NULL, so `.notna()` does NOT filter it —
# any consumer that wants "recorded values only" must exclude it explicitly.
# Exported so the TIER 7 post-hoc tests exclude exactly the same sentinel this
# transformer isolates; the two layers disagreeing is BACKLOG.md item E2k.
DEFAULT_MISSING_SENTINEL: str = "Unknown"


class MissingnessIndicatorTransformer(BaseEstimator, TransformerMixin):
    """
    Add binary missingness indicators and mode-impute the "Unknown" sentinel.

    Parameters
    ----------
    columns : Iterable[str]
        Names of columns where "Unknown" should be treated as missingness.
        Each named column must exist in the input DataFrame at fit and
        transform time.
    missing_value : str, default="Unknown"
        The string value that indicates missingness in the input columns.
    indicator_suffix : str, default="_missing"
        Suffix appended to each column name to form the indicator column.

    Attributes
    ----------
    modes_ : dict[str, str]
        Per-column mode, computed on non-`missing_value` rows during fit.

    Notes
    -----
    - The transformer returns a pandas DataFrame, not a numpy array. The
      surrounding ColumnTransformer must accept DataFrame output (which is the
      default since sklearn 1.2 with `set_output(transform='pandas')`, or
      when consumers select columns by index/name).
    - If a fit-time column is 100% missing (every value equals
      `missing_value`), the mode falls back to `missing_value` itself and a
      RuntimeWarning is emitted — but in practice we don't expect this for
      ANCESTRY / ETHNICITY at the cohort level (only certain institutions
      have 100% Unknown rates).
    """

    def __init__(
        self,
        columns: Iterable[str],
        missing_value: str = DEFAULT_MISSING_SENTINEL,
        indicator_suffix: str = "_missing",
        catchall_values: Iterable[str] = DEFAULT_CATCHALL_VALUES,
    ):
        # sklearn's BaseEstimator requires __init__ to store parameters
        # unchanged. Convert to a list only at use sites (fit / transform).
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
            # Treat both np.nan and the configured sentinel as missingness
            # (pre-lock review #11, 2026-05-22). Previously only the string
            # sentinel counted, so a future R-pipeline NULL→DataFrame NaN
            # would silently flow through with indicator=0.
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
                # Pre-lock review #10 (2026-05-22): warn when the imputed value
                # is itself a catch-all bucket. Folding NKI/CRUK Unknowns into
                # 'Other' re-merges institutional missingness into a
                # heterogeneous group, partly defeating Fix b's purpose.
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
        # Record the full fit-time column set so get_feature_names_out(None)
        # can return all passthrough columns plus the indicators it adds.
        # Without this, sklearn Pipeline.get_feature_names_out() (which passes
        # input_features=None to the first step) silently drops every column
        # not in self.columns.
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
            # Both np.nan and the configured sentinel count as missing
            # (pre-lock review #11, 2026-05-22).
            missing_mask = out[col].isna() | (out[col] == self.missing_value)
            out[f"{col}{self.indicator_suffix}"] = missing_mask.astype(np.int8)
            out[col] = out[col].where(~missing_mask, self.modes_[col])
        return out

    def get_feature_names_out(self, input_features: Optional[Iterable[str]] = None):
        cols = self._columns_list()
        if input_features is None:
            # sklearn Pipeline chaining passes None — return the full fit-time
            # column set so downstream steps (e.g., ColumnTransformer) see every
            # column they were configured for, plus the indicators we add.
            if hasattr(self, "feature_names_in_"):
                input_features = list(self.feature_names_in_)
            else:
                input_features = cols
        else:
            input_features = list(input_features)
        indicators = [f"{c}{self.indicator_suffix}" for c in cols]
        return np.array(input_features + indicators, dtype=object)
