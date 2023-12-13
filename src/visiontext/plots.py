from collections import defaultdict, Counter

import base64
import hashlib
import io
import json
import numpy as np
import pandas as pd
import os
import random
import re
import shutil
import sys
import time
from IPython.display import Image, display, HTML
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

from packg import format_exception
from packg.iotools.jsonext import load_json, dump_json, loads_json, dumps_json
from packg.log import configure_logger
from packg.magic import reload_recursive
from packg.paths import get_cache_dir, get_data_dir, get_code_dir
from packg.strings import b64_encode_from_bytes
from packg.tqdmext import tqdm_max_ncols
from visiontext.htmltools import NotebookHTMLPrinter, display_html_table
from visiontext.images import PILImageScaler, open_image_scaled
from visiontext.pandatools import full_pandas_display
from typedparser.objects import invert_dict_of_dict, flatten_dict


import os
import shutil
import sys
import json
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
from pathlib import Path

import numpy as np
import torch as th
from torch import nn
from pprint import pprint
import time
from timeit import default_timer as timer
from enum import Enum
from tqdm import tqdm

from copy import deepcopy
import seaborn as sns


def plot_hist(data, bins=None, density=False, title="untitled", figsize=None, show=False):
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.hist(data, density=density, bins=bins)
    sns.kdeplot(data, bw_adjust=1)
    plt.grid()
    if show:
        plt.show()
