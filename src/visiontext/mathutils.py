import math
from typing import Optional, Union, Tuple, List

import numpy as np
import torch

from packg.maths import np_round_half_down


def torch_stable_softmax(inp, temp=1.0, dim=-1):
    x: torch.Tensor = inp / temp
    max_values, max_indices = x.max(dim=dim, keepdim=True)
    z = x - max_values
    numerator = torch.exp(z)
    denominator = torch.sum(numerator, dim=dim, keepdim=True)
    softmax = numerator / denominator
    return softmax


def compute_indices(
    num_frames_orig: int, num_frames_target: int, is_train: bool = False
) -> np.ndarray:
    """
    Given two sequence lengths n_orig and n_target, sample n_target indices from the range [0, n_orig-1].

    Random sample approximately from intervals during training:
    with factor f = n_orig / n_target, sample in the range [i*f, (i+1)*f].
    Center sample in the same range during validation.

    Args:
        num_frames_orig: Original sequence length n_orig.
        num_frames_target: Target sequence length n_target.
        is_train: enable random sampling of indices. if False, center sample.

    Returns:
        Indices with shape (n_target)
    """
    # random sampling during training
    if is_train:
        # create rounded start points
        start_points = np.linspace(0, num_frames_orig, num_frames_target, endpoint=False)
        start_points = np_round_half_down(start_points).astype(int)

        # compute random offsets s.t. the sum of offsets equals num_frames_orig
        offsets = start_points[1:] - start_points[:-1]
        np.random.shuffle(offsets)
        last_offset = num_frames_orig - np.sum(offsets)
        offsets = np.concatenate([offsets, np.array([last_offset])])

        # compute new start points as cumulative sum of offsets
        new_start_points = np.cumsum(offsets) - offsets[0]

        # move offsets to the left so they fit the new start points
        offsets = np.roll(offsets, -1)

        # now randomly sample in the uniform intervals given by the offsets
        random_offsets = offsets * np.random.rand(num_frames_target)

        # calculate indices and floor them to get ints
        indices = new_start_points + random_offsets
        indices = np.floor(indices).astype(int)
        return indices
    # center sampling during validation
    # compute the linspace and offset it so its centered
    start_points = np.linspace(0, num_frames_orig, num_frames_target, endpoint=False)
    offset = num_frames_orig / num_frames_target / 2
    indices = start_points + offset
    # floor the result to get ints
    indices = np.floor(indices).astype(int)
    return indices


def expand_video_segment(
    num_frames_video: int, min_frames_seg: int, start_frame_seg: int, stop_frame_seg: int
):
    """
    Expand a given video segment defined by start and stop frame to have at least a minimum number of frames.

    Args:
        num_frames_video: Total number of frames in the video.
        min_frames_seg: Target minimum number of frames in the segment.
        start_frame_seg: Current start frame of the segment.
        stop_frame_seg: Current stop frame of the segment.

    Returns:
        Tuple of start frame, stop frame, flag whether the segment was changed.
    """
    num_frames_seg = stop_frame_seg - start_frame_seg
    changes = False
    if min_frames_seg > num_frames_video:
        min_frames_seg = num_frames_video
    if num_frames_seg < min_frames_seg:
        while True:
            if start_frame_seg > 0:
                start_frame_seg -= 1
                num_frames_seg += 1
                changes = True
            if num_frames_seg == min_frames_seg:
                break
            if stop_frame_seg < num_frames_video:
                stop_frame_seg += 1
                num_frames_seg += 1
                changes = True
            if num_frames_seg == min_frames_seg:
                break
    return start_frame_seg, stop_frame_seg, changes


def chunked_mean(arr, max_n=10000):
    ct = 0
    total = 0.0
    for n in range(0, len(arr), max_n):
        n_stop = n + max_n
        arr_sub = arr[n:n_stop]
        ct += len(arr_sub)
        this_sum = np.sum(arr_sub)
        # print(this_sum)
        total += this_sum
    return total / ct


def np_softmax(x: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = -1) -> np.ndarray:
    x_max = np.max(x, axis, keepdims=True)
    unnormalized = np.exp(x - x_max)
    return unnormalized / np.sum(unnormalized, axis, keepdims=True)


def distribute_evenly(num: int, num_chunks: int) -> List[int]:
    """Distribute `num` evenly into `num_chunks` parts.

    Args:
        num: Number to distribute.
        num_chunks: Number of parts to distribute into.

    Returns:
        Array of length `num_chunks` with the distribution.
    """
    step_size = math.floor(num / num_chunks)
    per_shard = [step_size] * num_chunks
    missing = num - sum(per_shard)
    for i in range(missing):
        per_shard[i] += 1
    return per_shard
