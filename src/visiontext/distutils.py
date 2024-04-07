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


def get_process_info() -> str:
    process: BaseProcess = multiprocessing.current_process()
    return f"Process {process.pid} {repr(process)}"


def get_world_info() -> tuple[int, int]:
    """
    Return global rank and world size from environment variables. If not set, return 0, 1.

    """
    return get_rank(), get_world_size()


def get_world_size():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
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
    world_size = get_world_size()
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
