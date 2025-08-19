"""
Utilities to build a webdataset pipeline.

Code adapted from openclip/training/data.py and the webdataset package
"""

import logging
import random
import re
from multiprocessing import Value

import webdataset as wds
from loguru import logger
from torch.utils.data import get_worker_info
from webdataset.filters import _shuffle, default_collation_fn  # noqa

from typedparser.objects import repr_value
from visiontext.distutils import get_global_rank, get_torch_worker_id, print_with_rank

SHARD_SHUFFLE_SIZE = 2000
SHARD_SHUFFLE_INITIAL = 500
SAMPLE_SHUFFLE_SIZE = 5000
SAMPLE_SHUFFLE_INITIAL = 1000


def dict_collation_fn(
    samples,
    keys=None,
    collation_fn=default_collation_fn,
):
    """Collate samples to a batch tuple, then convert the tuple to dict using the keys"""
    batch = collation_fn(samples)
    assert len(batch) == len(keys), f"{len(batch)=} != {len(keys)=} keys: {keys}"
    dict_batch = {k: v for k, v in zip(keys, batch)}
    return dict_batch


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


class wds_filter_unpack_json(wds.PipelineStage):
    def __init__(self, json_key: str, content_keys: tuple[str]):
        self.json_key = json_key
        self.content_keys = content_keys

    def run(self, src):
        for data in src:
            # at this point, wds.decode has decoded the json bytestring and it is already a dict
            json_data = data.pop(self.json_key)
            for content_key in self.content_keys:
                data[content_key] = json_data[content_key]
            yield data


class detshuffle2(wds.PipelineStage):
    def __init__(
        self,
        bufsize=1000,
        initial=100,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


class wds_detshuffle_datapoints(wds.PipelineStage):
    def __init__(
        self,
        bufsize=5000,
        initial=1000,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            raise ValueError("Webdataset missing SharedEpoch")
        rng = random.Random()
        if self.seed < 0:
            raise NotImplementedError("Untested")
            # # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            # seed = pytorch_worker_seed(epoch)
        worker_id = get_torch_worker_id()
        if worker_id < 0:
            logger.warning(
                f"Worker id is {worker_id}. If you have 0 workers (foreground) this is ok. "
                f"Otherwise, the worker seeds will be off!"
            )

        # This seed is deterministic but different for each rank (node and gpu), worker, epoch
        seed = self.seed + epoch + worker_id * 1000 + get_global_rank() * 1000000
        # print_with_rank(f"Worker {get_worker_id()} decided for seed {seed}")
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class wds_print_content(wds.PipelineStage):
    def __init__(self):
        pass

    def run(self, src):
        for i, data in enumerate(src):
            if i == 0 and get_torch_worker_id() <= 0 and get_global_rank() == 0:
                print_with_rank(
                    f"worker: {get_torch_worker_id()} keys: {data.keys()} content:\n{repr_value(data)}"
                )
            yield data
            break
        yield from src


class wds_print_tar_indices(wds.PipelineStage):
    """Useful to figure out whether the tars are properly sorted / shuffled"""

    def __init__(self, extra_text: str | None = None):
        self.extra_text = extra_text

    def run(self, src):
        samples = list(src)
        urls = [sample["url"] for sample in samples]
        re_tar_num = re.compile(r".*?([0-9]+)\.tar")
        nums = [int(re_tar_num.match(url).group(1)) for url in urls]
        text_out = f"worker {get_torch_worker_id()} tar file shuffle: {nums}"
        if self.extra_text is not None:
            text_out = f"{self.extra_text} {text_out}"
        print_with_rank(text_out)
        for sample in samples:
            yield sample


class wds_print_sample_keys(wds.PipelineStage):
    """Useful to figure out whether the datapoints are properly sorted / shuffled"""

    def __init__(self, extra_text: str | None = None, n_show: int = 10):
        self.extra_text = extra_text
        self.n_show = n_show

    def run(self, src):
        examples = []
        for sample in src:
            if len(examples) >= self.n_show:
                break
            examples.append(sample)
        first_keys = [sample["__key__"] for sample in examples]
        text_out = f"Worker {get_torch_worker_id()} first keys: {first_keys}"
        if self.extra_text is not None:
            text_out = f"{self.extra_text} {text_out}"
        print_with_rank(text_out)
        for i, sample in enumerate(examples):
            print_with_rank(f"Worker {get_torch_worker_id()} key sample {i}")
            yield sample
        for i, sample in enumerate(src):
            yield sample
