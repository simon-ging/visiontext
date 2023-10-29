from __future__ import annotations

import numpy as np


def feat_to_uint8(
    feat: np.ndarray, axis: int = -1, norm: str = "linear"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compress features into uint8 format for storage.

    Args:
        feat: Input float features.
        axis: Which axis to reduce over.
        norm: Norm to use for scaling: "linear" or "exp".
            Depends on the data, but mostly linear is better.

    Returns:
        feat_uint8: Compressed features, dtype uint8, same shape as input.
        mins: Min values used for scaling, same float dtype as input. same shape as input,
            except the reduction axis has length 1.
        maxs: See mins
    """
    if norm == "linear":
        # linear scaling
        mins = np.min(feat, axis=axis, keepdims=True)
        maxs = np.max(feat, axis=axis, keepdims=True)
        # safeguard against div by zero
        problem_idx = np.where((maxs - mins) ** 2 < 1e-16)
        maxs[problem_idx] = mins[problem_idx] + 1e-8
        feat_uint8 = np.round((feat - mins) * 255 / (maxs - mins)).astype(np.uint8)
        return feat_uint8, mins, maxs
    if norm == "exp":
        # exponential scaling
        mins = np.min(feat, axis=axis, keepdims=True)
        term = np.log2(feat - mins + 1)
        maxs = np.max(term)
        feat_uint8 = np.round((term / maxs) * 255).astype(np.uint8)
        return feat_uint8, mins, maxs
    raise ValueError(f"Unknown norm {norm}")


def uint8_to_feat(
    feat_uint8: np.ndarray, mins: np.ndarray, maxs: np.ndarray, norm: str = "linear"
) -> np.ndarray:
    """Undo the compression from feat_to_uint8."""
    if norm == "linear":
        return feat_uint8.astype(np.float32) / 255.0 * (maxs - mins) + mins
    if norm == "exp":
        # exponential
        return (mins - 1 + 2 ** (feat_uint8 * maxs / 255)).astype(np.float32)
    raise ValueError(f"Unknown norm {norm}")
