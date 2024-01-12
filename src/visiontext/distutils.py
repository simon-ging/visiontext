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
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size


def get_local_rank():
    """
    Get local rank from environment variables. If not set, return 0.
    """
    local_rank = 0
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    return local_rank


def print_main(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def is_main_process():
    rank, _world_size = get_world_info()
    return rank == 0


def barrier_safe():
    """Barrier only if in a distributed torch run. Does not fail if torch package is missing."""
    if is_distributed():
        from torch import distributed as dist

        dist.barrier()


def is_distributed():
    return "WORLD_SIZE" in os.environ
