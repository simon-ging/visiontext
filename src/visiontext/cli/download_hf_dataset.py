from pathlib import Path
from typing import (
    Optional,
)

from attrs import define
from datasets import load_dataset
from loguru import logger

from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from packg.paths import get_data_dir
from typedparser import VerboseQuietArgs, add_argument, TypedParser


@define
class Args(VerboseQuietArgs):
    base_dir: Optional[Path] = add_argument(
        shortcut="-d", type=str, help="Base dir", default=get_data_dir()
    )
    dataset_name: str = add_argument(positional=True, type=str, help="Dataset name")
    options: list[str] | None = add_argument(shortcut="-o", action="append", help="dataset kwargs")


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")

    dict_dotlist = load_dotlist(args.options)  # TODO ADD TO VISIONTEXT
    # takes ~26 min
    dataset = load_dataset(
        args.dataset_name,
        use_auth_token=True,  # required
        streaming=False,  # this downloads to disk
        split="train",
        cache_dir=(args.base_dir / "huggingface-en-2301").as_posix(),
    )


if __name__ == "__main__":
    main()
