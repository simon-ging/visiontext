import lmdb
import msgpack
import os
from pathlib import Path
from typing import Optional

from packg.paths import get_cache_dir
from packg.strings import hash_object
from packg.typext import PathType

class LmdbAutoGrowEnvironment(lmdb.Environment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        try:
            super().__setitem__(key, value)
        except lmdb.MapFullError:
            self.set_mapsize(self.info()["map_size"] * 2)
            super().__setitem__(key, value)