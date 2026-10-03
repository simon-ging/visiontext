"""
Matplotlib utilities.
"""

import datetime
import io
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ctime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def mpl_setup(fontsize=16, color="white"):
    print(f"MPL Setup at {ctime}")
    mpl.rc("axes", unicode_minus=False)  # dont remember what this does
    mpl.rc("font", family="Dejavu Sans", serif="cmr10")
    mpl.rcParams.update({"font.size": fontsize})

    def set_font(fig=None):
        if fig is None:
            fig = plt.gcf()
        set_fontsize(fig=fig, fontsize=fontsize, color=color)

    return set_font


def set_fontsize(fig=None, fontsize=16, color=None):
    """
    For each text object of a figure fig, set the font size to fontsize

    See https://matplotlib.org/stable/api/text_api.html
    """
    if fig is None:
        fig = plt.gcf()

    def match(artist):
        return artist.__module__ == "matplotlib.text"

    for textobj in fig.findobj(match=match):
        textobj.set_fontsize(fontsize)
        textobj.set(color=color)
        # textobj.set_fontfamily(fontname)


@dataclass
class FigureSaver:
    """
    Usage:
        fs = FigureSaver(h, w)
        fig, ax = fs.create_fig_and_ax()
        fs.imshow(arr)  # or any other plotting on fig/ax
        data = fs.get_img_from_fig(fig)
        fs.close()
    """

    h: int
    w: int
    dpi: int = 96

    def create_fig_and_ax(self):
        show_dpi = self.dpi
        save_dpi = show_dpi
        fig = plt.figure(
            figsize=(self.w / save_dpi, self.h / save_dpi), dpi=show_dpi, frameon=False
        )
        # fig.set_size_inches()
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        self.fig, self.ax = fig, ax
        return fig, ax

    def imshow(self, img):
        self.ax.imshow(img, aspect="equal")

    def get_img_from_fig(self, as_fp32=False, with_alpha=False, as_pillow=False):
        fig = self.fig
        save_dpi = self.dpi
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=save_dpi)
        buf.seek(0)
        im = Image.open(buf)
        if as_pillow:
            im = im.copy()
        else:
            im = np.array(im)
            if as_fp32:
                im = im.astype(np.float32) / 255.0
            if not with_alpha:
                im = im[:, :, :3]
        buf.close()
        return im

    def close(self):
        self.fig.clear(True)
        plt.close()
