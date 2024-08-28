"""
Note that for pytorch lightning you should pass the lightning trainer into WorldInfo


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
from deprecated import deprecated
from multiprocessing.process import BaseProcess


class WorldInfo:
    def __init__(self, trainer=None):
        self.trainer = trainer

    @property
    def global_rank(self):
        if self.trainer is not None:
            return self.trainer.global_rank
        return get_global_rank()

    @property
    def world_size(self):
        if self.trainer is not None:
            return self.trainer.world_size
        return get_world_size()

    @property
    def is_global_zero(self):
        return self.global_rank == 0

    def print_with_rank(self, *args, **kwargs):
        rank = self.global_rank
        world_size = self.world_size
        print(f"Rank {rank:>2d}/{world_size}:", *args, **kwargs)

    def barrier_safe(self):
        if self.trainer is not None:
            return self.trainer.strategy.barrier()
        return barrier_safe()


def get_process_info() -> str:
    process: BaseProcess = multiprocessing.current_process()
    return f"Process {process.pid} {repr(process)}"


def get_world_info() -> tuple[int, int]:
    """
    Return global rank and world size from environment variables. If not set, return 0, 1.
    """
    return get_global_rank(), get_world_size()


def get_world_size() -> int:
    if is_slurm_sbatch():
        return int(os.environ["SLURM_NTASKS"])
    return int(os.environ.get("WORLD_SIZE", 1))


def get_global_rank() -> int:
    """
    In some cases LOCAL_RANK is set, but RANK is unset. Use LOCAL_RANK in that case.
    RANK: global rank of the process in the distributed setting, across all nodes.
    LOCAL_RANK: rank on this machine / node.

    in slurm sbatch scripts, we need to use the slurm env variables instead.
    """
    if is_slurm_sbatch():
        return int(os.environ["SLURM_PROCID"])

    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank


@deprecated(reason="Use get_global_rank instead")
def get_rank():
    return get_global_rank()


def is_slurm_sbatch():
    slurm_job_name = os.environ.get("SLURM_JOB_NAME")
    if slurm_job_name is None:
        # not in slurm job
        return False
    if slurm_job_name == "bash":
        # in foreground slurm job
        return False
    # in background slurm job
    return True


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
