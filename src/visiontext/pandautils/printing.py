import pandas as pd
from contextlib import contextmanager


@contextmanager
def full_pandas_display():
    """
    A context manager for setting various pandas display options to their
    maximum values so that the output is not truncated when printed.
    """
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.max_colwidth",
        None,
        "display.width",
        None,
        "display.expand_frame_repr",
        False,
    ):
        yield


@contextmanager
def pandas_float_format(fmt="{:.2f}"):
    with pd.option_context("display.float_format", fmt.format):
        yield
