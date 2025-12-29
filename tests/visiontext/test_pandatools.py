import pandas as pd
import pytest

from visiontext.pandatools import (
    check_pandas_str_column_is_empty,
    check_pandas_str_field_is_empty,
)


def test_check_pandas_str_field_is_empty():
    # Test NaN values
    assert check_pandas_str_field_is_empty(None) is True
    assert check_pandas_str_field_is_empty(pd.NA) is True
    assert check_pandas_str_field_is_empty(float("nan")) is True

    # Test empty strings
    assert check_pandas_str_field_is_empty("") is True
    assert check_pandas_str_field_is_empty("   ") is True
    assert check_pandas_str_field_is_empty("\t\n") is True

    # Test short strings with min_len
    assert check_pandas_str_field_is_empty("a", min_len=1) is False
    assert check_pandas_str_field_is_empty("a", min_len=2) is True
    assert check_pandas_str_field_is_empty("ab", min_len=2) is False
    assert check_pandas_str_field_is_empty("ab", min_len=3) is True

    # Test valid non-empty strings
    assert check_pandas_str_field_is_empty("hello") is False
    assert check_pandas_str_field_is_empty("  hello  ") is False
    assert check_pandas_str_field_is_empty("x") is False

    # Test nan_strs parameter (default includes "nan")
    assert check_pandas_str_field_is_empty("nan") is True
    assert check_pandas_str_field_is_empty("NaN") is True
    assert check_pandas_str_field_is_empty("  NAN  ") is True
    assert check_pandas_str_field_is_empty("nan", nan_strs=None) is False

    # Test custom nan_strs
    assert check_pandas_str_field_is_empty("N/A", nan_strs=["n/a", "na"]) is True
    assert check_pandas_str_field_is_empty("na", nan_strs=["n/a", "na"]) is True
    assert check_pandas_str_field_is_empty("  NA  ", nan_strs=["n/a", "na"]) is True
    assert check_pandas_str_field_is_empty("unknown", nan_strs=["unknown", "n/a"]) is True
    assert check_pandas_str_field_is_empty("valid", nan_strs=["n/a", "na"]) is False


def test_check_pandas_str_column_is_empty():
    # Create test series
    series = pd.Series(["hello", "", "   ", None, pd.NA, "a", "ab", "  world  ", "\t\n"])

    # Test with default min_len=1
    result = check_pandas_str_column_is_empty(series)
    expected = pd.Series([False, True, True, True, True, False, False, False, True])
    pd.testing.assert_series_equal(result, expected)

    # Test with min_len=2
    result = check_pandas_str_column_is_empty(series, min_len=2)
    expected = pd.Series([False, True, True, True, True, True, False, False, True])
    pd.testing.assert_series_equal(result, expected)

    # Test with min_len=3
    result = check_pandas_str_column_is_empty(series, min_len=3)
    expected = pd.Series([False, True, True, True, True, True, True, False, True])
    pd.testing.assert_series_equal(result, expected)

    # Test nan_strs parameter (default includes "nan")
    series_with_nan_str = pd.Series(["hello", "nan", "NaN", "  NAN  ", "valid", None])
    result = check_pandas_str_column_is_empty(series_with_nan_str)
    expected = pd.Series([False, True, True, True, False, True])
    pd.testing.assert_series_equal(result, expected)

    # Test with nan_strs=None (should not consider "nan" as empty)
    result = check_pandas_str_column_is_empty(series_with_nan_str, nan_strs=None)
    expected = pd.Series([False, False, False, False, False, True])
    pd.testing.assert_series_equal(result, expected)

    # Test custom nan_strs
    series_custom = pd.Series(["N/A", "na", "  Unknown  ", "valid", "n/a", None])
    result = check_pandas_str_column_is_empty(series_custom, nan_strs=["n/a", "na", "unknown"])
    expected = pd.Series([True, True, True, False, True, True])
    pd.testing.assert_series_equal(result, expected)
