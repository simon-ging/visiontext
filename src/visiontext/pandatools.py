from __future__ import annotations
from typing import List

import numpy as np
import pandas as pd

import webbrowser
from contextlib import contextmanager
from pathlib import Path
from timeit import default_timer
from typing import Optional

import pandas as pd
from loguru import logger
from pandas.io.formats.style import Styler

from packg.iotools.jsonext import load_json
from visiontext.colormaps import redgreen_bright


#################### I/O utils ####################


def load_parquet(file, dtype_backend="pyarrow", verbose: bool = False, **kwargs):
    file = Path(file)
    if verbose:
        start_timer = default_timer()
        file_len = f"{file.stat().st_size / 1024 ** 2:.1f} MB"
        print(f"Load json file {file} with size {file_len}.")
    data = pd.read_parquet(file, dtype_backend=dtype_backend, **kwargs)
    if verbose:
        print(f"Loaded json file {file} in {default_timer() - start_timer:.3f} seconds")
    return data


def dump_parquet(df, file, engine="pyarrow", compression="snappy", **kwargs):
    df.to_parquet(file, engine=engine, compression=compression, **kwargs)


def load_json_to_df(file):
    """Load JSON file into a pandas DataFrame.

    Args:
        file: Path to JSON file. The file must be a dict like {index1: {column1_key: column1_value, ...}, ...}

    Returns:
        pd.DataFrame: DataFrame created from JSON data
    """
    dct = load_json(file)
    df = pd.DataFrame.from_dict(dct, orient="index")
    return df


#################### Output / display ####################


def create_styler(
    formats=None,
    vmin=0,
    vmax=1,
    cmap=redgreen_bright,
    hide_index=True,
    nan_color="white",
):
    """

    Args:
        formats: dictionary {column name: format string or function}
        vmin: min value for background gradient
        vmax: max value for background gradient
        cmap: colormap for background gradient
        hide_index: hide index column
        nan_color: color for NaN values

    Returns:
        function to use as:
            styled_df = df.style.pipe(styler_fn)
    """
    formats = {} if formats is None else formats

    def styler_fn(
        styler: Styler,
    ):
        styler.format(formats)
        styler.background_gradient(cmap=cmap, axis=1, vmin=vmin, vmax=vmax)
        if hide_index:
            styler.hide(axis="index")
        styler.highlight_null(color=nan_color)

        return styler

    return styler_fn


def save_df_to_html(df, title, html_file, open_browser=True):
    html_file = Path(html_file)
    html_file.parent.mkdir(parents=True, exist_ok=True)
    with html_file.open("w", encoding="utf-8") as f:
        f.write(f"<html><head><title>{title}</title></head><body>")
        f.write(f"<h1>{title}</h1>")
        df.to_html(f)
        f.write("</body></html>")

    if open_browser:
        webbrowser.open(html_file.as_posix())
    logger.info(f"Saved HTML output to {html_file}")


@contextmanager
def full_pandas_display(
    max_rows=None, max_columns=None, max_colwidth=None, width=None, expand_frame_repr=False
):
    """
    A context manager for setting various pandas display options to their
    maximum values so that the output is not truncated when printed.

    Available options: https://pandas.pydata.org/docs/reference/api/pandas.get_option.html

    """
    with pd.option_context(
        "display.max_rows",
        max_rows,
        "display.max_columns",
        max_columns,
        "display.max_colwidth",
        max_colwidth,
        "display.width",
        width,
        "display.expand_frame_repr",
        expand_frame_repr,
    ):
        yield


def pd_print_series(series):
    keys = series.index.tolist()
    values = series.values.tolist()
    keys_str = [str(k) for k in keys]
    values_str = [str(v) for v in values]
    max_key_str_len = max([len(k) for k in keys_str])
    for key, value in zip(keys_str, values_str):
        print(f"{key:{max_key_str_len}}: {value}")


def display_df(
    df, max_rows=None, max_columns=None, max_colwidth=None, width=None, expand_frame_repr=False
):
    with full_pandas_display(max_rows, max_columns, max_colwidth, width, expand_frame_repr):
        print(df)


@contextmanager
def pandas_float_format(fmt="{:.2f}"):
    with pd.option_context("display.float_format", fmt.format):
        yield


def print_stats(input_data, title: Optional[str] = None):
    if title is not None:
        print(title)
    print(pd.Series(input_data).describe())


#################### String field validation ####################


def check_pandas_str_field_is_empty(
    value, min_len: int = 1, nan_strs: set[str] | list[str] | None = {"nan"}
) -> bool:
    """Check if a single pandas string field is empty.

    A field is considered empty if:
    - It is NaN/None (pd.isna() returns True)
    - After stripping whitespace, its length is less than min_len
    - Its lowercase stripped value is in nan_strs
    """
    if pd.isna(value):
        return True
    stripped = str(value).strip()
    if len(stripped) < min_len:
        return True
    if nan_strs is None:
        return False
    nan_strs = set(s.lower() for s in nan_strs)
    if stripped.lower() in nan_strs:
        return True
    return False


def check_pandas_str_column_is_empty(
    series: pd.Series, min_len: int = 1, nan_strs: set[str] | list[str] | None = {"nan"}
) -> pd.Series:
    """Check which values in a pandas string column are empty (vectorized)."""
    is_na = pd.isna(series)
    # note that when doing .str... operations on NaNs they will just stay NaN
    stripped = series.astype(str).str.strip()
    is_too_short = stripped.str.len() < min_len
    if nan_strs is None:
        return is_na | is_too_short
    nan_strs = set(s.lower() for s in nan_strs)
    is_in_nan_strs = stripped.str.lower().isin(nan_strs)
    return is_na | is_too_short | is_in_nan_strs


#################### Compare dataframes ####################


def compare_dataframes(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    na_equal: bool = True,
    coerce_numeric: bool = False,
) -> List[str]:
    """
    Compare two pandas DataFrames and return a list of human-readable differences.
    Empty list means equality under the given options.
    
    By default, NaN values are treated as equal (na_equal=True).
    """
    diffs: List[str] = []

    if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
        return [f"Type mismatch: {type(df1)} != {type(df2)}"]

    # ---- index / columns checks ----
    if not df1.index.equals(df2.index):
        missing_in_2 = df1.index.difference(df2.index)
        missing_in_1 = df2.index.difference(df1.index)
        if len(missing_in_2):
            diffs.append(f"{len(missing_in_2)} index rows missing in second df")
        if len(missing_in_1):
            diffs.append(f"{len(missing_in_1)} index rows missing in first df")

    if not df1.columns.equals(df2.columns):
        missing_in_2 = df1.columns.difference(df2.columns)
        missing_in_1 = df2.columns.difference(df1.columns)
        if len(missing_in_2):
            diffs.append(f"{len(missing_in_2)} columns missing in second df")
        if len(missing_in_1):
            diffs.append(f"{len(missing_in_1)} columns missing in first df")

    # ---- alignment ----
    # Always align on union of indices/columns to continue comparison
    union_index = df1.index.union(df2.index)
    union_cols = df1.columns.union(df2.columns)
    a = df1.reindex(index=union_index, columns=union_cols)
    b = df2.reindex(index=union_index, columns=union_cols)

    # ---- dtype checks ----
    # Only check columns that exist in both original dataframes
    cols_to_check = df1.columns.intersection(df2.columns)
    for c in cols_to_check:
        if a[c].dtype != b[c].dtype:
            diffs.append(f"Column dtype mismatch '{c}': {a[c].dtype} != {b[c].dtype}")

    # ---- value comparison ----
    # Only compare columns/rows that exist in both original dataframes
    common_cols = df1.columns.intersection(df2.columns)
    common_idx = df1.index.intersection(df2.index)

    aa = a.loc[common_idx, common_cols].copy()
    bb = b.loc[common_idx, common_cols].copy()

    if coerce_numeric:
        for c in common_cols:
            aa[c] = pd.to_numeric(aa[c], errors="coerce")
            bb[c] = pd.to_numeric(bb[c], errors="coerce")

    for c in common_cols:
        s1 = aa[c]
        s2 = bb[c]

        na1 = s1.isna()
        na2 = s2.isna()

        if na_equal:
            na_mismatch = na1 ^ na2
            for idx in na_mismatch[na_mismatch].index:
                diffs.append(f"[{idx!r}, {c!r}] NA mismatch: {s1.loc[idx]!r} != {s2.loc[idx]!r}")
            mask = ~(na1 | na2)
            s1c = s1[mask]
            s2c = s2[mask]
        else:
            s1c = s1
            s2c = s2

        if s1c.empty:
            continue

        if pd.api.types.is_numeric_dtype(s1c.dtype) and pd.api.types.is_numeric_dtype(s2c.dtype):
            close = np.isclose(
                s1c.to_numpy(dtype=float, copy=False),
                s2c.to_numpy(dtype=float, copy=False),
                rtol=rtol,
                atol=atol,
                equal_nan=na_equal,
            )
            for pos in np.where(~close)[0]:
                idx = s1c.index[pos]
                diffs.append(f"[{idx!r}, {c!r}] {s1.loc[idx]!r} != {s2.loc[idx]!r}")
            continue

        if pd.api.types.is_datetime64_any_dtype(s1c.dtype) or pd.api.types.is_timedelta64_dtype(
            s1c.dtype
        ):
            neq = s1c.ne(s2c)
        else:
            neq = s1c.ne(s2c)

        for idx in neq[neq].index:
            diffs.append(f"[{idx!r}, {c!r}] {s1.loc[idx]!r} != {s2.loc[idx]!r}")

    return diffs
