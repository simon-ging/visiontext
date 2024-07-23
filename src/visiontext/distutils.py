"""
Rank: Global rank of the process in the distributed setting.
Local rank: Rank on this machine

Example:
            |    Node1  |   Node2    |
____________| p1 |  p2  |  p3  |  p4 |
local_rank  | 0  |   1  |  0   |   1 |
rank        | 0  |   1  |  2   |   4 |

"""

from __future__ import annotations

import multiprocessing
import os
from multiprocessing.process import BaseProcess

from packg.log import logger


def get_process_info() -> str:
    process: BaseProcess = multiprocessing.current_process()
    return f"Process {process.pid} {repr(process)}"


def get_world_info() -> tuple[int, int]:
    """
    Return global rank and world size from environment variables. If not set, return 0, 1.

    """
    return get_rank(), get_world_size()


def set_expected_world_size(expected_world_size: int, verbose: bool = True):
    """
    In certain scenarios like lightning training you may want to init your dataset or loss
    already before trainer.fit is called. At that point, the global world_size is still 1
    because ligthning hasn't initialized the distributed backend yet.
    However if you know what the world_size will be later, you can set it here and
    initialize your dataset with the correct world_size.
    """
    old_world_size = get_world_size(use_expected_world_size=False)
    if 1 < old_world_size != expected_world_size:
        raise ValueError(
            f"Setting EXPECTED_WORLD_SIZE to {expected_world_size} but WORLD_SIZE is already "
            f"set to {old_world_size}."
        )
    if not isinstance(expected_world_size, int) or expected_world_size < 1:
        raise ValueError(f"Invalid world size: {expected_world_size}")
    if expected_world_size == old_world_size:
        if verbose:
            print_with_rank(
                f"EXPECTED_WORLD_SIZE==WORLD_SIZE=={expected_world_size}, nothing to set."
            )
        return
    if verbose:
        print_with_rank(f"Setting EXPECTED_WORLD_SIZE={expected_world_size}")
    os.environ["EXPECTED_WORLD_SIZE"] = str(expected_world_size)


def get_world_size(use_expected_world_size: bool = False) -> int:
    """
    Args:
        use_expected_world_size: whether to override with EXPECTED_WORLD_SIZE env variable if set.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if use_expected_world_size:
        expected_world_size = os.environ.get("EXPECTED_WORLD_SIZE", None)
        if expected_world_size is not None:
            expected_world_size = int(expected_world_size)
            if 1 < world_size != expected_world_size:
                raise ValueError(
                    f"Got mismatch betwen WORLD_SIZE={world_size} and "
                    f"EXPECTED_WORLD_SIZE={expected_world_size}"
                )
    return world_size


def get_rank() -> int:
    """
    In some cases LOCAL_RANK is set, but RANK is unset. Use LOCAL_RANK in that case.
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank


def print_main(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def print_with_rank(*args, **kwargs):
    rank = get_rank()
    world_size = get_world_size(use_expected_world_size=False)
    print(f"Rank {rank:>2d}/{world_size}:", *args, **kwargs)


def is_main_process():
    rank, _world_size = get_world_info()
    return rank == 0


def barrier_safe():
    """Barrier only if in a distributed torch run. Does not fail if torch package is missing."""
    if is_distributed():
        from torch import distributed as dist

        dist.barrier()


def is_distributed():
    return "WORLD_SIZE" in os.environ and get_world_size() > 1
