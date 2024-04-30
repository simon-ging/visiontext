from tempfile import NamedTemporaryFile
from typing import Tuple

import numpy as np
from loguru import logger
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.signal import spectrogram

from packg import Const
from packg.typext import PathType


class AudioConvertC(Const):
    NONE = "none"
    MONO = "mono"
    STEREO = "stereo"


def load_audio_file(
    input_file: PathType, verbose: bool = False, convert=AudioConvertC.STEREO
) -> Tuple[int, np.ndarray]:
    """
    Load audio file and return frequency and data.

    Args:
        input_file:
        verbose: print debug info
        convert: convert to either stereo (shape (N, 2)) or mono (shape (N,)) or leave as is.

    Returns:
        tuple of frequency int and data numpy array of shape (N, 2) (stereo) or (N,) (mono)

    """
    file_format = input_file.name.split(".")[-1]
    if verbose:
        logger.info(f"Reading file {input_file} detected format {file_format}")
    audio = AudioSegment.from_file(input_file, format=file_format)

    # convert to wav
    with NamedTemporaryFile(suffix=".wav") as tmpfile:
        audio.export(tmpfile, format="wav")
        frequency, data = wavfile.read(tmpfile)

    # convert to stereo or mono as needed
    if convert == AudioConvertC.STEREO and len(data.shape) == 1:
        data = np.stack([data, data], axis=-1)  # mono to stereo
    elif convert == AudioConvertC.MONO and len(data.shape) == 2:
        data = data.mean(-1)  # stereo to mono

    if verbose:
        logger.info(f"Got frequency {frequency} shape {data.shape} after conversion '{convert}'")

    return frequency, data


def check_audio_quality(frequency, data, eps=1e-6) -> bool:
    """
    logic: a good quality music file will have audio at 20khz. a bad one will not.
    """
    # dmin, dmax = data.min(), data.max()
    # dnorm = (data - dmin) / (dmax - dmin)
    # print_ndarray_lovely(data)  # raw mono audio
    # example data shape 11,796,960 freq 48000 => 246 seconds

    freqs, time, spec = spectrogram(data, fs=frequency, nperseg=256, noverlap=128)

    # print_ndarray_lovely(freqs)  # frequency bins
    # select all bins within 19 and 21 khz
    freqs_to_use = np.less_equal(19000, freqs) & np.less_equal(freqs, 21000)
    # example: 129 equally spaced values from 0 ... 24000

    # print_ndarray_lovely(time)  # time bins
    # print_ndarray_lovely(spec)  # spec shape (n_freq, n_time)
    smin, smax = spec.min(), spec.max()
    snorm = (spec - smin) / (smax - smin)

    srel = snorm[freqs_to_use]
    return srel.max() > eps
    # now we want to normalize with something between mean (1-norm) and max (inf-norm)
    # srel_norm_freq = np.linalg.norm(srel, ord=np.inf, axis=0)
    # srel_norm_all = np.linalg.norm(srel_norm_freq, ord=np.inf, axis=0)
    # return srel_norm_all
