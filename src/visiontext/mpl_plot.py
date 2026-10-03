"""
Matplotlib utilities.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from PIL import Image

from visiontext.colormaps import get_color_from_default_color_cycle

_ = mpl.colors.LinearSegmentedColormap


def plot_image(arr):
    h, w, _ = arr.shape
    # depend figsize on ratio
    fs_base = 12
    ratio = w / h
    fig, ax = plt.subplots(figsize=(fs_base * ratio, fs_base / ratio))
    # arr_crop = scale_crop_image(arr)
    # plt.imshow(arr_crop)
    plt.imshow(arr)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def plot_image_from_file(image_file):
    image = Image.open(image_file)
    arr = np.array(image)
    plot_image(arr)


def plot_int_histogram_and_cumsum(
    histdata, title="", do_log=False, figsize=(12, 12), set_xs=True, xs_res=1
):
    histdata = np.array(histdata)
    hmin, hmax = np.min(histdata), np.max(histdata)
    n_bins = np.round(hmax - hmin + 1).astype(int)
    histed, bins = np.histogram(histdata, bins=n_bins, range=(hmin - 0.5, hmax + 0.5))
    xs = np.linspace(hmin, hmax + 1, n_bins, endpoint=False)

    fig, axes = plt.subplots(2, 1, figsize=figsize)
    ax = axes[0]
    plt.suptitle(title)
    ax.grid()
    ax.bar(xs, histed, width=0.5)
    if set_xs:
        ax.set_xticks(xs[::xs_res])
    if do_log:
        ax.semilogy()

    # also plot the percentages
    ax = axes[1]
    ax.grid()
    ax.plot(xs, np.cumsum(histed) / np.sum(histed), "X-")
    if set_xs:
        ax.set_xticks(xs[::xs_res])

    plt.tight_layout()


def plot_2d_gaussian(mean, cov, color, ax, samples=None, n_ellipses=2, label=None):
    """
    Plots a 2D gaussian distribution

    Usage:
        >>> fig, axis = plt.subplots(figsize=(8, 6))
        >>> plot_2d_gaussian((0, 0), 1., 0, axis, label="unit gaussian")
        >>> plt.show()

    Args:
        mean: array shape (2,) for x, y coordinates of the mean
        cov: float (spherical gaussian) or array shape (2,) (diagonal gaussian)
            or array shape (2, 2) (full gaussian)
        color: matplotlib color or int for one of the default colors
        ax: axis to plot on
        samples: optional samples drawn from the gaussian to plot
        n_ellipses: number of standard deviation ellipses to plot
        label: label for the legend

    Returns:

    """
    if isinstance(color, int):
        color = get_color_from_default_color_cycle(color)

    # calculate the eigen vectors from the covariance matrix
    use_cov = np.copy(cov)
    if use_cov.ndim == 0:
        # spherical gaussian
        use_cov = np.eye(len(mean)) * use_cov
    if use_cov.ndim == 1:
        # diagonal cov gaussian
        use_cov = np.eye(len(mean)) * use_cov
    eig_w, eig_v = np.linalg.eig(use_cov)

    # eigenvectors are orthogonal so we only need to check the first one
    eig_v_0 = eig_v[0]
    # arccos of first eigenvectors x coordinate will give us the angle
    angle = np.arccos(eig_v_0[0])
    # however, this only works for angles in [0, pi]
    # for angles in [pi, 2*pi] we need to consider the
    # y coordinate, too
    if eig_v_0[1] > 0:
        angle = 2 * np.pi - angle
    angle = np.rad2deg(angle)

    # draws different covariances representing 1, 2, 3, ... standard deviations
    # width and height are times by 2 since sqrt of the eigenval only measures half the distance
    for i in range(n_ellipses):
        ell = Ellipse(
            xy=(mean[0], mean[1]),
            width=np.sqrt(eig_w[0]) * 2 * (i + 1),
            height=np.sqrt(eig_w[1]) * 2 * (i + 1),
            angle=angle,
            edgecolor=color,
            lw=2,
            facecolor="none",
        )
        ax.add_artist(ell)

    if samples is not None:
        ax.scatter(samples[:, 0], samples[:, 1], marker="x", color=color, alpha=0.5)
    ax.scatter(mean[0], mean[1], color=color, marker="X", label=label, s=250)


#     plt.grid()
#     plt.show()
