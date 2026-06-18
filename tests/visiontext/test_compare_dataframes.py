import numpy as np
import pandas as pd
import pytest

from visiontext.pandatools import compare_dataframes


@pytest.fixture
def dfs():
    # Base
    df_base = pd.DataFrame(
        {"a": [1, 2, 3], "b": [10.0, 20.0, 30.0], "c": ["x", "y", "z"]},
        index=[0, 1, 2],
    )

    # Identical (copy)
    df_same = df_base.copy(deep=True)

    # Totally different values, same shape/labels
    df_totally_different = pd.DataFrame(
        {"a": [100, 200, 300], "b": [-1.0, -2.0, -3.0], "c": ["u", "v", "w"]},
        index=[0, 1, 2],
    )

    # Slightly different numeric (within default rtol/atol vs outside)
    df_slightly_diff_within_tol = df_base.copy(deep=True)
    df_slightly_diff_within_tol.loc[1, "b"] = 20.0 + 1e-12  # should be equal by default

    df_slightly_diff_outside_tol = df_base.copy(deep=True)
    df_slightly_diff_outside_tol.loc[1, "b"] = 20.0 + 1e-2  # should differ by default

    # Empty dataframes
    df_empty = pd.DataFrame()
    df_empty2 = pd.DataFrame()

    # Different columns
    df_extra_col = df_base.copy(deep=True)
    df_extra_col["new_col"] = [1, 1, 1]

    df_missing_col = df_base.drop(columns=["c"])

    # Different index
    df_diff_index = df_base.copy(deep=True)
    df_diff_index.index = [10, 11, 12]

    # Dtype mismatch but same "values"
    df_dtype_mismatch = df_base.copy(deep=True)
    df_dtype_mismatch["a"] = df_dtype_mismatch["a"].astype("int64")
    df_base_a_as_float = df_base.copy(deep=True)
    df_base_a_as_float["a"] = df_base_a_as_float["a"].astype("float64")

    # NaN vs NaN and NaN vs value
    df_with_nan = df_base.copy(deep=True)
    df_with_nan.loc[1, "b"] = np.nan

    df_with_nan_same = df_base.copy(deep=True)
    df_with_nan_same.loc[1, "b"] = np.nan

    df_with_nan_other = df_base.copy(deep=True)
    df_with_nan_other.loc[1, "b"] = 999.0

    # Coerce-numeric scenario: numeric stored as strings
    df_num_as_str = pd.DataFrame(
        {"a": ["1", "2", "3"], "b": ["10.0", "20.0", "30.0"]}, index=[0, 1, 2]
    )
    df_num_as_num = pd.DataFrame({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]}, index=[0, 1, 2])

    return {
        "base": df_base,
        "same": df_same,
        "totally_different": df_totally_different,
        "slightly_within_tol": df_slightly_diff_within_tol,
        "slightly_outside_tol": df_slightly_diff_outside_tol,
        "empty": df_empty,
        "empty2": df_empty2,
        "extra_col": df_extra_col,
        "missing_col": df_missing_col,
        "diff_index": df_diff_index,
        "dtype_mismatch_int": df_dtype_mismatch,
        "dtype_mismatch_float": df_base_a_as_float,
        "with_nan": df_with_nan,
        "with_nan_same": df_with_nan_same,
        "with_nan_other": df_with_nan_other,
        "num_as_str": df_num_as_str,
        "num_as_num": df_num_as_num,
    }


# -----------------------
# Default kwargs behavior
# -----------------------


def test_default_same(dfs):
    assert compare_dataframes(dfs["base"], dfs["same"]) == []


def test_default_totally_different(dfs):
    diffs = compare_dataframes(dfs["base"], dfs["totally_different"])
    assert diffs  # non-empty
    # At least one cell difference should be reported
    assert any("'a'" in d for d in diffs)


def test_default_empty_equals_empty(dfs):
    assert compare_dataframes(dfs["empty"], dfs["empty2"]) == []


def test_default_slightly_within_tol_equal(dfs):
    assert compare_dataframes(dfs["base"], dfs["slightly_within_tol"]) == []


def test_default_slightly_outside_tol_differs(dfs):
    diffs = compare_dataframes(dfs["base"], dfs["slightly_outside_tol"])
    assert diffs
    assert any("'b'" in d for d in diffs)


def test_default_shape_or_labels_mismatch(dfs):
    diffs = compare_dataframes(dfs["base"], dfs["extra_col"])
    assert diffs
    # Reports column mismatch (now as count)
    assert any("columns missing" in d for d in diffs)


# ----------------------------------------
# One test per kwarg that changes behavior
# ----------------------------------------


def test_kw_na_equal_false_treats_nan_as_not_equal(dfs):
    # Default is na_equal=True, so NaN == NaN
    diffs_default = compare_dataframes(dfs["with_nan"], dfs["with_nan_same"])
    assert diffs_default == []  # should be equal by default

    # With na_equal=False, NaNs should not be equal
    diffs = compare_dataframes(dfs["with_nan"], dfs["with_nan_same"], na_equal=False)
    assert diffs  # should report differences


def test_kw_rtol_atol_can_make_slight_diff_equal(dfs):
    diffs_default = compare_dataframes(dfs["base"], dfs["slightly_outside_tol"])
    assert diffs_default

    diffs = compare_dataframes(dfs["base"], dfs["slightly_outside_tol"], rtol=1e-2, atol=1e-2)
    assert diffs == []


def test_kw_coerce_numeric_true_compares_numeric_strings_as_numbers(dfs):
    diffs_default = compare_dataframes(dfs["num_as_str"], dfs["num_as_num"])
    assert diffs_default  # dtype and values mismatch

    # With coerce_numeric, values are equal but dtypes still differ
    diffs = compare_dataframes(dfs["num_as_str"], dfs["num_as_num"], coerce_numeric=True)
    # Dtype mismatch will still be reported since we always check dtypes
    assert any("dtype mismatch" in d for d in diffs)
