"""
Compute spectrogram of an audio file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from attrs import define
from loguru import logger

from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from typedparser import VerboseQuietArgs, add_argument, TypedParser
from visiontext.audiotools import load_audio_file, AudioConvertC, check_audio_quality


@define
class Args(VerboseQuietArgs):
    input_file: Path = add_argument("input_file", type=str, help="Audio file to check")


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")

    frequency, data = load_audio_file(args.input_file, verbose=True, convert=AudioConvertC.MONO)

    print(f"quality: {check_audio_quality(frequency, data)}")

    # plot the spectogram
    # plt.specgram(data, Fs=frequency, NFFT=128, noverlap=0)  # plot
    # plt.show()


if __name__ == "__main__":
    main()
