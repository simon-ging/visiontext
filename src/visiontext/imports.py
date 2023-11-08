"""
Collection of useful imports e.g. for jupyter notebooks. Usage:

>>> from visiontext.imports import *

"""

import base64
import hashlib
import io
import json
import os
import random
import shutil
import sys
import time
import re
from copy import deepcopy
from enum import Enum
from pathlib import Path
from pprint import pprint, pformat
from timeit import default_timer as timer

from typing import (
    Dict,
    List,
    Optional,
    Any,
    Iterable,
    Mapping,
    Tuple,
    Union,
    Callable,
    BinaryIO,
    Sequence,
    Collection,
)

import numpy as np
import torch
from IPython.display import Image, display, HTML
from PIL import Image
from loguru import logger
from collections import defaultdict, Counter
from matplotlib import pyplot as plt

from packg.iotools.jsonext import load_json, dump_json, loads_json, dumps_json
from packg.paths import get_anno_dir, get_cache_dir, get_result_dir, get_data_dir, get_code_dir
from packg.strings import b64_encode_from_bytes
from packg import format_exception
from packg.log import configure_logger
from packg.magic import reload_recursive
from torch import nn
from tqdm import tqdm
from visiontext.notebookutils import NotebookHTMLPrinter, display_html_table
from visiontext.images import PILImageScaler
from sentence_transformers.util import cos_sim

# # the __all__ list below is used to stop pycharm or other tools from removing unused imports
# # to update it after changing the imports above, uncomment the code below and copypaste the output
# imported_modules = [m for m in globals().keys() if not m.startswith("_")]
#
# print(f"__all__ = {repr(sorted(imported_modules))}")

__all__ = [
    "Any",
    "BinaryIO",
    "Callable",
    "Collection",
    "Counter",
    "Dict",
    "Enum",
    "HTML",
    "Image",
    "Iterable",
    "List",
    "Mapping",
    "NotebookHTMLPrinter",
    "Optional",
    "PILImageScaler",
    "Path",
    "Sequence",
    "Tuple",
    "Union",
    "b64_encode_from_bytes",
    "base64",
    "configure_logger",
    "cos_sim",
    "deepcopy",
    "defaultdict",
    "display",
    "display_html_table",
    "dump_json",
    "dumps_json",
    "format_exception",
    "get_anno_dir",
    "get_cache_dir",
    "get_code_dir",
    "get_data_dir",
    "get_result_dir",
    "hashlib",
    "io",
    "json",
    "load_json",
    "loads_json",
    "logger",
    "nn",
    "np",
    "os",
    "pformat",
    "plt",
    "pprint",
    "random",
    "re",
    "reload_recursive",
    "shutil",
    "sys",
    "time",
    "timer",
    "torch",
    "tqdm",
]
