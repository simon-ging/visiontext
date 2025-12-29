"""Utilities for visualising image aspect ratios."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.ticker import FuncFormatter


def plot_ratio_histogram(
    ratios: Sequence[float],
    *,
    bins: int = 50,
    ax: Optional[Axes] = None,
    figsize: tuple[int, int] = (8, 4),
    show: bool = False,
) -> tuple[Figure | SubFigure, Axes]:
    """Plot a histogram of aspect ratios centred on 1.

    This uses a mirrored axis so distances from the minimum ratio to ``1`` and
    from ``1`` to the maximum ratio appear identical, with ratios equal to ``1``
    occupying their own centred bin.
    """
    ratios_array = np.asarray(list(ratios), dtype=float)
    if ratios_array.size == 0:
        raise ValueError("No ratios provided to plot_ratio_histogram.")
    if not np.all(np.isfinite(ratios_array)):
        raise ValueError("Ratios must be finite numbers.")
    if np.any(ratios_array <= 0):
        raise ValueError("Ratios must be strictly positive.")

    max_ratio = float(np.max(ratios_array))
    max_inverse = float(np.max(1.0 / ratios_array))
    bound = max(max_ratio, max_inverse)
    if np.isclose(bound, 1.0):
        bound = 1.0 + 1e-3
    min_bound = 1.0 / bound

    def transform(values: Sequence[float]) -> np.ndarray:
        vals = np.asarray(values, dtype=float)
        out = np.empty_like(vals)
        lt_mask = vals < 1.0
        gt_mask = vals > 1.0
        eq_mask = ~(lt_mask | gt_mask)
        out[lt_mask] = -(1.0 - vals[lt_mask]) / (1.0 - min_bound)
        out[gt_mask] = (vals[gt_mask] - 1.0) / (bound - 1.0)
        out[eq_mask] = 0.0
        return out

    def inverse_transform(coord: float) -> float:
        if coord < 0:
            return 1.0 - (1.0 - min_bound) * (-coord)
        if coord > 0:
            return 1.0 + (bound - 1.0) * coord
        return 1.0

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    side_bins = max(int(bins) // 2, 1)
    left_edges = np.linspace(min_bound, 1.0, side_bins + 1)
    right_edges = np.linspace(1.0, bound, side_bins + 1)

    left_mask = ratios_array < 1.0
    right_mask = ratios_array > 1.0
    center_mask = np.isclose(ratios_array, 1.0, rtol=1e-5, atol=1e-8)

    left_hist, _ = np.histogram(ratios_array[left_mask], bins=left_edges)
    right_hist, _ = np.histogram(ratios_array[right_mask], bins=right_edges)
    center_count = int(center_mask.sum())

    transformed_left_edges = transform(left_edges)
    transformed_right_edges = transform(right_edges)

    left_positions = 0.5 * (transformed_left_edges[:-1] + transformed_left_edges[1:])
    left_widths = np.diff(transformed_left_edges)
    right_positions = 0.5 * (transformed_right_edges[:-1] + transformed_right_edges[1:])
    right_widths = np.diff(transformed_right_edges)

    bar_kwargs = dict(alpha=0.85, edgecolor="white", color="#4C72B0")
    if left_hist.size:
        ax.bar(
            left_positions,
            left_hist,
            width=left_widths,
            align="center",
            label="ratio != 1",
            **bar_kwargs,
        )
    if right_hist.size:
        ax.bar(
            right_positions,
            right_hist,
            width=right_widths,
            align="center",
            label="ratio != 1" if left_hist.sum() == 0 else None,
            **bar_kwargs,
        )

    if center_count > 0:
        central_width_candidates = []
        if left_widths.size:
            central_width_candidates.append(left_widths[-1])
        if right_widths.size:
            central_width_candidates.append(right_widths[0])
        central_width = min(central_width_candidates) if central_width_candidates else 0.1
        if central_width <= 0:
            central_width = 0.1
        ax.bar(
            0.0,
            center_count,
            width=central_width,
            align="center",
            color="#55A868",
            edgecolor="white",
            label="ratio = 1",
        )

    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Aspect ratio (width / height)")
    ax.set_ylabel("Count")
    ax.set_title("Image aspect ratio distribution")
    ax.set_xticks(np.linspace(-1.0, 1.0, num=5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{inverse_transform(val):.2f}"))

    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="best")

    if show:
        plt.show()

    return fig, ax
