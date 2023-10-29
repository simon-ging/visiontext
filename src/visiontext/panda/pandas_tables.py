from pandas.io.formats.style import Styler

from visiontext.visualizer.colormaps import redgreen_bright


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
