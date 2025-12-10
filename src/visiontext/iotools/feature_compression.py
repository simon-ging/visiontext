from __future__ import annotations

from pathlib import Path

import numpy as np
import safetensors.torch
import torch

from packg import Const
from packg.iotools.compress import CompressorC, compress_data_to_file, decompress_file_to_bytes


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


def convert_to_fp16_torch(feat: torch.Tensor) -> torch.Tensor:
    """
    Convert a float32 tensor to float16, raising an error if overflow would occur.
    """
    FP16_MAX = 65500.0
    abs_max = torch.max(torch.abs(feat)).item()
    if abs_max > FP16_MAX:
        raise ValueError(
            f"Tensor contains values that would overflow fp16. "
            f"Max absolute value: {abs_max}, fp16 max: {FP16_MAX}"
        )
    return feat.half()


def dump_safetensors_zst(
    tensors: dict[str, torch.Tensor],
    save_path: str | Path,
    create_parent: bool = False,
    verbose: bool = False,
    level: int = 3,
) -> None:
    """
    Save a dictionary of tensors to a zstd-compressed safetensors file.

    Args:
        tensors: Dictionary mapping tensor names to torch tensors.
        save_path: Path where to save the compressed file.
        create_parent: If True, create parent directories if they don't exist.
        verbose: If True, print progress information.
    """
    save_path = Path(save_path)
    if create_parent:
        save_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Saving tensors and compressing to {save_path}")
    safetensors_bytes = safetensors.torch.save(tensors)
    compress_data_to_file(
        safetensors_bytes,
        save_path,
        CompressorC.ZSTD,
        create_parent=create_parent,
        level=level,
    )

    if verbose:
        print(f"Successfully saved compressed safetensors to {save_path}")


def dump_single_safetensor_zst(tensor, *args, **kwargs):
    dump_safetensors_zst({"tensor": tensor}, *args, **kwargs)


def load_safetensors_zst(load_path: str | Path, verbose: bool = False) -> dict[str, torch.Tensor]:
    """
    Load a dictionary of tensors from a zstd-compressed safetensors file.

    Args:
        load_path: Path to the compressed safetensors file.
        verbose: If True, print progress information.

    Returns:
        Dictionary mapping tensor names to torch tensors.
    """
    load_path = Path(load_path)

    if not load_path.exists():
        raise FileNotFoundError(f"File not found: {load_path}")

    if verbose:
        print(f"Decompressing and loading {load_path}")
    safetensors_bytes = decompress_file_to_bytes(load_path, CompressorC.ZSTD)

    tensors = safetensors.torch.load(safetensors_bytes)

    if verbose:
        print(f"Successfully loaded {len(tensors)} tensor(s) from {load_path}")

    return tensors


def load_single_safetensor_zst(load_path: str | Path, verbose: bool = False) -> torch.Tensor:
    tensors = load_safetensors_zst(load_path, verbose=verbose)
    if tensors.keys() != {"tensor"}:
        raise KeyError(f"this function expects to find a single 'tensor' but got {tensors.keys()}")
    return tensors["tensor"]
