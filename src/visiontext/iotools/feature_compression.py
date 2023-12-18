from __future__ import annotations

import numpy as np
import torch

from packg import Const


class NormsC(Const):
    LINEAR = "linear"


def compress_fp32_to_uint8_numpy(
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
    if norm == NormsC.LINEAR:
        mins = np.min(feat, axis=axis, keepdims=True)
        maxs = np.max(feat, axis=axis, keepdims=True)
        # safeguard against div by zero
        problem_idx = np.where((maxs - mins) ** 2 < 1e-16)
        maxs[problem_idx] = mins[problem_idx] + 1e-8
        feat_uint8 = np.round((feat - mins) * 255 / (maxs - mins)).astype(np.uint8)
        return feat_uint8, mins, maxs
    raise ValueError(f"Unknown norm '{norm}', possible values {list(NormsC.values())}")


def decompress_uint8_to_fp32_numpy(
    feat_uint8: np.ndarray, mins: np.ndarray, maxs: np.ndarray, norm: str = "linear"
) -> np.ndarray:
    """
    Undo the compression from compress_fp32_to_uint8_numpy
    """
    if norm == NormsC.LINEAR:
        return feat_uint8.astype(np.float32) / 255.0 * (maxs - mins) + mins
    raise ValueError(f"Unknown norm '{norm}', possible values {list(NormsC.values())}")


def compress_fp32_to_uint8_torch(
    feat: torch.Tensor, axis: int = -1, norm: str = "linear"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    See compress_fp32_to_uint8_numpy
    """
    if norm == NormsC.LINEAR:
        mins, _mins_idx = torch.min(feat, dim=axis, keepdim=True)
        maxs, _maxs_idx = torch.max(feat, dim=axis, keepdim=True)
        # safeguard against div by zero
        problem_idx = torch.where((maxs - mins) ** 2 < 1e-16)
        maxs[problem_idx] = mins[problem_idx] + 1e-8
        feat_uint8 = torch.round((feat - mins) * 255 / (maxs - mins)).to(torch.uint8)
        return feat_uint8, mins, maxs
    raise ValueError(f"Unknown norm '{norm}', possible values {list(NormsC.values())}")


def decompress_uint8_to_fp32_torch(
    feat_uint8: torch.Tensor, mins: torch.Tensor, maxs: torch.Tensor, norm: str = "linear"
) -> torch.Tensor:
    """
    Undo the compression from compress_fp32_to_uint8_torch
    """
    if norm == NormsC.LINEAR:
        return feat_uint8.float() / 255.0 * (maxs - mins) + mins
    raise ValueError(f"Unknown norm '{norm}', possible values {list(NormsC.values())}")
