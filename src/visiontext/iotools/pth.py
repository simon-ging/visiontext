from typing import Any

import os
import torch

from packg import format_exception
from packg.typext import PathType


def torch_save_safely(obj: Any, f: PathType, **kwargs):
    try:
        torch.save(obj, f, **kwargs)
    except KeyboardInterrupt as e:
        print(f"Deleting {f} due to KeyboardInterrupt")
        try:
            os.remove(f)
        except Exception as e2:
            print(f"Exception while deleting {f}: {format_exception(e2)}")
            pass
        raise e
