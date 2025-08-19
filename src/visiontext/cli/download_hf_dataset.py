"""
download a dataset from huggingface.

potentially useful args:
    -o name=DatasetConfigName
    -o use_auth_token=True
    -o split="train"
"""

from pathlib import Path

from attrs import define
from loguru import logger

from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from packg.paths import get_packg_data_dir
from typedparser import TypedParser, VerboseQuietArgs, add_argument
from visiontext.configutils import load_dotlist


@define
class Args(VerboseQuietArgs):
    base_dir: Path | None = add_argument(
        shortcut="-d", type=str, help="Base dir", default=get_packg_data_dir()
    )
    dataset_name: str = add_argument(positional=True, type=str, help="Dataset name")
    options: list[str] | None = add_argument(shortcut="-o", action="append", help="dataset kwargs")
    subdir: str | None = add_argument(
        shortcut="-s", type=str, help="Subdir (none=auto)", default=None
    )


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")

    dict_dotlist = load_dotlist(args.options)
    subdir = args.subdir
    if subdir is None:
        subdir = args.dataset_name.replace("/", "__")

    logger.info(f"Downloading dataset {args.dataset_name} to {args.base_dir / subdir}")
    try:
        from datasets import load_dataset
    except ImportError as e:
        logger.error(f"datasets is not installed. Please install it with `pip install datasets`.")
        raise e
    dataset = load_dataset(
        args.dataset_name,
        streaming=False,  # this downloads to disk
        cache_dir=(args.base_dir / subdir).as_posix(),
        **dict_dotlist,
    )


if __name__ == "__main__":
    main()
