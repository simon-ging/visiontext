import pandas as pd
import webbrowser
from contextlib import contextmanager
from loguru import logger
from pandas.io.formats.style import Styler
from pathlib import Path
from typing import Optional

from visiontext.images import PILImageScaler

from visiontext.colormaps import redgreen_bright


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
