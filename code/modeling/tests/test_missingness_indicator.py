"""Tests for MissingnessIndicatorTransformer."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from oncokb_analyzer.missingness_indicator import MissingnessIndicatorTransformer


@pytest.fixture
def train_df():
    return pd.DataFrame({
        'ANCESTRY': ['European'] * 6 + ['African'] * 2 + ['Unknown'] * 2,
        'ETHNICITY': ['Non-Hispanic'] * 7 + ['Hispanic'] * 1 + ['Unknown'] * 2,
        'SEX': ['Male', 'Female'] * 5,
    })


@pytest.fixture
def test_df():
    return pd.DataFrame({
        'ANCESTRY': ['Unknown', 'European', 'African', 'Unknown'],
        'ETHNICITY': ['Hispanic', 'Unknown', 'Non-Hispanic', 'Unknown'],
        'SEX': ['Male', 'Female', 'Male', 'Female'],
    })


def test_indicator_columns_added(train_df):
    """Indicator columns are 1 where original was 'Unknown', else 0."""
    t = MissingnessIndicatorTransformer(columns=['ANCESTRY', 'ETHNICITY'])
    t.fit(train_df)
    out = t.transform(train_df)

    assert 'ANCESTRY_missing' in out.columns
    assert 'ETHNICITY_missing' in out.columns

    expected_ancestry_missing = (train_df['ANCESTRY'] == 'Unknown').astype(np.int8)
    pd.testing.assert_series_equal(
        out['ANCESTRY_missing'].astype(np.int8),
        expected_ancestry_missing.rename('ANCESTRY_missing'),
        check_names=False,
    )


def test_mode_imputation_replaces_unknown(train_df):
    """After transform, no 'Unknown' values remain in the source columns."""
    t = MissingnessIndicatorTransformer(columns=['ANCESTRY', 'ETHNICITY'])
    t.fit(train_df)
    out = t.transform(train_df)

    assert (out['ANCESTRY'] == 'Unknown').sum() == 0
    assert (out['ETHNICITY'] == 'Unknown').sum() == 0
    # Mode of train (excluding Unknown) is European / Non-Hispanic
    assert (out.loc[train_df['ANCESTRY'] == 'Unknown', 'ANCESTRY'] == 'European').all()
    assert (out.loc[train_df['ETHNICITY'] == 'Unknown', 'ETHNICITY'] == 'Non-Hispanic').all()


def test_train_test_boundary_no_leakage(train_df, test_df):
    """Test set's Unknowns are replaced by the *training-set* mode, not the test mode."""
    t = MissingnessIndicatorTransformer(columns=['ANCESTRY'])
    t.fit(train_df)
    test_out = t.transform(test_df)

    # Recorded mode is from train only (European).
    assert t.modes_['ANCESTRY'] == 'European'
    # Test set's Unknowns become European, not whatever the test-set mode would be.
    test_unknown_rows = test_df['ANCESTRY'] == 'Unknown'
    assert (test_out.loc[test_unknown_rows, 'ANCESTRY'] == 'European').all()


def test_indicator_correct_on_test_set(train_df, test_df):
    """Indicator column on the test set marks the *test-set* Unknowns."""
    t = MissingnessIndicatorTransformer(columns=['ANCESTRY'])
    t.fit(train_df)
    test_out = t.transform(test_df)

    expected = (test_df['ANCESTRY'] == 'Unknown').astype(np.int8)
    pd.testing.assert_series_equal(
        test_out['ANCESTRY_missing'].astype(np.int8),
        expected.rename('ANCESTRY_missing'),
        check_names=False,
    )


def test_no_unknown_in_train_still_works():
    """If the training data has zero Unknowns, transformer still fits and the indicator is zero."""
    df = pd.DataFrame({'ANCESTRY': ['European'] * 6 + ['African'] * 4})
    t = MissingnessIndicatorTransformer(columns=['ANCESTRY'])
    t.fit(df)
    out = t.transform(df)
    assert (out['ANCESTRY_missing'] == 0).all()
    # Recorded mode is the most common non-Unknown value (European; majority 6).
    assert t.modes_['ANCESTRY'] == 'European'


def test_all_unknown_in_train_warns():
    """100% Unknown in training emits a warning and mode falls back to 'Unknown'."""
    df = pd.DataFrame({'ANCESTRY': ['Unknown'] * 10})
    t = MissingnessIndicatorTransformer(columns=['ANCESTRY'])
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        t.fit(df)
        assert any(issubclass(w.category, RuntimeWarning) for w in ws)
    assert t.modes_['ANCESTRY'] == 'Unknown'


def test_fit_rejects_missing_column():
    """If a configured column is missing from the input, fit raises."""
    df = pd.DataFrame({'SEX': ['M', 'F']})
    t = MissingnessIndicatorTransformer(columns=['ANCESTRY'])
    with pytest.raises(ValueError, match='ANCESTRY'):
        t.fit(df)


def test_transform_before_fit_raises(train_df):
    t = MissingnessIndicatorTransformer(columns=['ANCESTRY'])
    with pytest.raises(RuntimeError, match='must be fit'):
        t.transform(train_df)


def test_get_feature_names_out_includes_indicators(train_df):
    t = MissingnessIndicatorTransformer(columns=['ANCESTRY', 'ETHNICITY'])
    t.fit(train_df)
    names = t.get_feature_names_out(['ANCESTRY', 'ETHNICITY', 'SEX'])
    assert 'ANCESTRY_missing' in names
    assert 'ETHNICITY_missing' in names
    assert 'SEX' in names


def test_mode_other_warns():
    """Warn when the imputed mode is a catch-all category such as 'Other'."""
    df = pd.DataFrame({
        # Non-Unknown mode is 'Other' (4 vs European=3, African=1).
        'ANCESTRY': ['Other'] * 4 + ['European'] * 3 + ['African'] * 1 + ['Unknown'] * 2,
    })
    t = MissingnessIndicatorTransformer(columns=['ANCESTRY'])
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        t.fit(df)
    assert t.modes_['ANCESTRY'] == 'Other'
    assert any(
        issubclass(w.category, RuntimeWarning) and 'catch-all' in str(w.message)
        for w in ws
    )


def test_mode_non_other_does_not_warn(train_df):
    """Catch-all warning should NOT fire when the mode is a real category."""
    t = MissingnessIndicatorTransformer(columns=['ANCESTRY', 'ETHNICITY'])
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        t.fit(train_df)
    assert not any('catch-all' in str(w.message) for w in ws)


def test_nan_treated_as_missing():
    """NaN is treated as missing, like the 'Unknown' sentinel."""
    train_df = pd.DataFrame({
        'ANCESTRY': ['European'] * 6 + ['African'] * 2 + ['Unknown'] * 2,
    })
    test_df = pd.DataFrame({
        'ANCESTRY': [np.nan, 'European', 'Unknown', 'African'],
    })
    t = MissingnessIndicatorTransformer(columns=['ANCESTRY'])
    t.fit(train_df)
    out = t.transform(test_df)

    # NaN row gets indicator=1 just like the 'Unknown' row.
    assert out['ANCESTRY_missing'].tolist() == [1, 0, 1, 0]
    # NaN row gets mode-imputed too (no NaN remaining in ANCESTRY).
    assert out['ANCESTRY'].isna().sum() == 0
    assert out.loc[0, 'ANCESTRY'] == 'European'  # NaN → mode
    assert out.loc[2, 'ANCESTRY'] == 'European'  # Unknown → mode


def test_nan_in_fit_counted_as_missing():
    """NaN in the fit data should be excluded from mode computation."""
    df = pd.DataFrame({
        # If NaN were counted, mode might tie or shift.
        'ANCESTRY': ['European'] * 3 + ['African'] * 1 + [np.nan] * 5 + ['Unknown'] * 2,
    })
    t = MissingnessIndicatorTransformer(columns=['ANCESTRY'])
    t.fit(df)
    assert t.modes_['ANCESTRY'] == 'European'


def test_get_feature_names_out_none_returns_fit_columns_plus_indicators(train_df):
    """get_feature_names_out(None) returns the fit-time columns plus the indicators."""
    t = MissingnessIndicatorTransformer(columns=['ANCESTRY', 'ETHNICITY'])
    t.fit(train_df)
    names = t.get_feature_names_out()  # explicit no-arg
    # Every column the transformer saw at fit time must round-trip.
    for col in train_df.columns:
        assert col in names, f"{col} should round-trip through get_feature_names_out(None)"
    # Plus the indicators it adds.
    assert 'ANCESTRY_missing' in names
    assert 'ETHNICITY_missing' in names
