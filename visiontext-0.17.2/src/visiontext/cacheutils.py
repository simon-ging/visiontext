import lmdb
import msgpack
import os
from pathlib import Path
from typing import Optional

from packg.paths import get_cache_dir
from packg.strings import hash_object
from packg.typext import PathType


class CachePlugin:
    """
    Helper to create a cache for a class.

    cache_name and cache_kwargs define a unique filename.
    cache_kwargs must be jsonable
    """

    def __init__(
        self,
        cache_name: str,
        cache_kwargs: Optional[dict] = None,
        cache_dir: Optional[PathType] = None,
        map_size=10 * 1024**3,  # 10 GB max size of database, increase if necessary
        verbose: bool = False,
    ):
        if cache_dir is None:
            cache_dir = get_cache_dir()
        if cache_kwargs is None:
            kwargs_str = "none"
        else:
            kwargs_str = hash_object(cache_kwargs)

        lmdb_file = Path(cache_dir) / "cache_plugin" / f"{cache_name}~{kwargs_str}.lmdb"
        os.makedirs(lmdb_file.parent, exist_ok=True)
        self.db: lmdb.Environment = lmdb.open(lmdb_file.as_posix(), map_size=map_size)
        self.lmdb_file = lmdb_file
        if verbose:
            print(f"Using cache file: {self.lmdb_file}")

    def get_values(self, list_of_keys):
        outputs = []
        with self.db.begin(write=False) as txn:
            for key in list_of_keys:
                bkey = msgpack.packb(key)
                bvalue = txn.get(bkey)
                if bvalue is None:
                    outputs.append(None)
                else:
                    value = msgpack.unpackb(bvalue)
                    outputs.append(value)
        return outputs

    def put_values(self, list_of_keys, list_of_values):
        assert len(list_of_keys) == len(
            list_of_values
        ), f"Got {len(list_of_keys)} keys and {len(list_of_values)} values."
        if len(list_of_keys) == 0:
            return
        with self.db.begin(write=True) as txn:
            for key, value in zip(list_of_keys, list_of_values):
                bkey = msgpack.packb(key)
                bvalue = msgpack.packb(value)
                txn.put(bkey, bvalue)
