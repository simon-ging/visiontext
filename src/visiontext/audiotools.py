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


def measure_audio_quality(
    frequency,
    data,
    high_band=(18000, 20000),
    ref_band=(1000, 5000),
) -> float:
    """
    Measure how much high-frequency content a music file has, in dB relative to a mid-band
    reference. A lossless / high-bitrate file carries real signal up to ~20 kHz, while a
    low-bitrate mp3 is hard-cut around 16 kHz and the high band drops to the noise floor.

    Args:
        frequency: sample rate in Hz
        data: audio samples, shape (N,) mono or (N, channels)
        high_band: (lo, hi) Hz band whose energy indicates quality
        ref_band: (lo, hi) Hz mid band whose peak level is the 0 dB reference

    Returns:
        energy of high_band relative to ref_band in dB. Higher means more high-frequency
        content, i.e. better quality. Returns -inf if the sample rate is too low to contain
        the high band at all.
    """
    if data.ndim == 2:
        data = data.mean(-1)  # stereo to mono
    data = data.astype(np.float64)

    # long-term average power spectrum; large window for fine frequency resolution
    nperseg = min(4096, len(data))
    freqs, _time, spec = spectrogram(data, fs=frequency, nperseg=nperseg, noverlap=nperseg // 2)
    mean_pow = spec.mean(axis=1)  # average power per frequency bin over the whole track

    ref_sel = (freqs >= ref_band[0]) & (freqs <= ref_band[1])
    high_sel = (freqs >= high_band[0]) & (freqs <= high_band[1])
    if not high_sel.any() or not ref_sel.any():
        # sample rate does not even reach the high band -> cannot be high quality
        return -np.inf

    ref_level = 10 * np.log10(mean_pow[ref_sel].max() + 1e-20)
    high_level = 10 * np.log10(mean_pow[high_sel].mean() + 1e-20)
    return high_level - ref_level


def check_audio_quality(frequency, data, threshold_db=-60.0) -> bool:
    """
    Classify a music file as good quality if it has meaningful signal in the high band.

    Measured on labelled examples: good files land around -50 dB, low-bitrate mp3/m4a around
    -70 dB or lower, so a -60 dB threshold separates them with ~10 dB margin.
    """
    return measure_audio_quality(frequency, data) > threshold_db
