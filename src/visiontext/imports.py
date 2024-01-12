"""
Collection of useful imports e.g. for jupyter notebooks. Usage:

>>> from visiontext.imports import *

"""

from collections import defaultdict, Counter

import base64
import hashlib
import io
import json
import numpy as np
import os
import pandas as pd
import pickle
import random
import re
import shutil
import sys
import time
from IPython.display import (
    Image,
    display,
    HTML,
    JSON,
)
from PIL import Image
from copy import deepcopy
from enum import Enum
from loguru import logger
from matplotlib import pyplot as plt
from pathlib import Path
from pprint import pprint, pformat
from timeit import default_timer as timer
from tqdm import tqdm
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
from natsort import natsorted
from packg import format_exception
from packg.iotools.jsonext import load_json, dump_json, loads_json, dumps_json
from packg.log import configure_logger
from packg.magic import reload_recursive
from packg.paths import get_cache_dir, get_data_dir, get_code_dir, setup_environ
from packg.strings import b64_encode_from_bytes
from packg.tqdmext import tqdm_max_ncols
from typedparser.objects import invert_dict_of_dict, flatten_dict
from visiontext.htmltools import NotebookHTMLPrinter, display_html_table
from visiontext.images import PILImageScaler, open_image_scaled
from visiontext.pandatools import full_pandas_display
from visiontext.nlp import preprocess_text_simple

# the __all__ list below is used to stop pycharm or other tools from removing unused imports
# to update it after changing the imports above, uncomment the code below and copypaste the output
# python -c "from visiontext.imports import *; im = [m for m in globals().keys() if not m.startswith(\"_\")]; print(f\"__all__ = {repr(sorted(im))}\")"
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
    "deepcopy",
    "defaultdict",
    "display",
    "display_html_table",
    "dump_json",
    "dumps_json",
    "flatten_dict",
    "format_exception",
    "full_pandas_display",
    "get_cache_dir",
    "get_code_dir",
    "get_data_dir",
    "hashlib",
    "invert_dict_of_dict",
    "io",
    "json",
    "load_json",
    "loads_json",
    "logger",
    "np",
    "open_image_scaled",
    "os",
    "pd",
    "pformat",
    "pickle",
    "plt",
    "pprint",
    "random",
    "re",
    "reload_recursive",
    "setup_environ",
    "shutil",
    "sys",
    "time",
    "timer",
    "tqdm",
    "tqdm_max_ncols",
    "JSON",
    "preprocess_text_simple",
    "natsorted",
]
