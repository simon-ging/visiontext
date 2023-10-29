from dataclasses import dataclass
from typing import Union, Tuple

from matplotlib import colors as mpl_colors, pyplot as plt, cm as mpl_cm

# transitions from red to green for bad to good values

redgreen_bright = mpl_colors.LinearSegmentedColormap.from_list(
    "rwg", [[1.0, 0.7, 0.7], [1.0, 1.0, 1.0], [0.7, 1.0, 0.7]]
)

redgreen_dark = mpl_colors.LinearSegmentedColormap.from_list(
    "rg", [[0.5, 0, 0], [0, 0, 0], [0, 0.3, 0]]
)


rbg_bright = mpl_colors.LinearSegmentedColormap.from_list(
    "rbg",
    [
        [1.0, 0.7, 0.7],
        [1.0, 1.0, 0.7],
        [1.0, 1.0, 1.0],
        [0.7, 1.0, 1.0],
        [0.7, 1.0, 0.7],
    ],
)


def get_redgreen_for_theme(is_dark_theme: bool):
    return redgreen_dark if is_dark_theme else redgreen_bright


@dataclass
class ColorGetterForColorMap:
    input_cmap: Union[mpl_colors.Colormap, str]
    start_val: Union[int, float]
    stop_val: Union[int, float]
    return_alpha: bool = False
    return_float: bool = False

    def __post_init__(self):
        self.cmap = self.input_cmap
        if isinstance(self.input_cmap, str):
            self.cmap = plt.get_cmap(self.input_cmap)
        self.norm = mpl_colors.Normalize(vmin=self.start_val, vmax=self.stop_val)
        self.scalar_map = mpl_cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_color(self, val) -> Union[Tuple[int, int, int, int], Tuple[int, int, int]]:
        rgba = self.scalar_map.to_rgba(val)
        if not self.return_alpha:
            rgba = rgba[:3]
        if not self.return_float:
            rgba = [int(round(a * 255)) for a in rgba]
        return rgba


def create_colorgetter_from_colormap(
    input_cmap: Union[mpl_colors.Colormap, str],
    return_alpha: bool = False,
    return_float: bool = False,
):
    # getter is now function that takes a value in (0, 1) and returns a color
    color_helper = ColorGetterForColorMap(
        input_cmap, 0, 1, return_alpha=return_alpha, return_float=return_float
    )
    return color_helper.get_color


def create_colormap_for_dark_background(
    n: int = 256, inverted=False, scale=1.0, alpha=1.0, add_brightness: float = 0.0
) -> mpl_colors.Colormap:
    """
    Colormap with only light colors that works well for text on dark background.

    Args:
        n: number of discrete colors to generate in the colormap
        inverted: if True, invert the colormap (gives dark colors for bright background)
        scale: downscale the colormap (make it darker, 1.0 = no scaling)
        alpha: transparency value for all colors, 1.0 = opaque, 0.0 = transparent / invisible
        add_brightness: add this value to all RGB values (0.0 = no change, 1.0 = colors are white)

    Returns:
        colormap
    """
    ccmap_colors = [
        [128, 128, 255],
        [0, 255, 255],
        [0, 255, 0],
        [255, 255, 0],
        [255, 0, 0],
        [255, 0, 255],
        [255, 255, 255],
    ]

    def _convert(c):
        c = c / 255
        if inverted:
            c = 1 - c
        c_scaled = max(min(c * scale, 1.0), 0.0)
        return c_scaled * (1.0 - add_brightness) + add_brightness

    ccmap_colors_01 = [[_convert(a) for a in rgb] + [alpha] for rgb in ccmap_colors]
    ccmap = mpl_colors.LinearSegmentedColormap.from_list("better_nipy", ccmap_colors_01, N=n)
    return ccmap
