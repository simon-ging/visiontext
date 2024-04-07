from pathlib import Path
from typing import Optional

from attrs import define
from loguru import logger

from packg.iotools import dump_json
from packg.iotools.pathspec_matcher import PathSpecArgs
from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from typedparser import VerboseQuietArgs, add_argument, TypedParser
from visiontext.iotools.tar_lookup import TarLookup


@define
class Args(PathSpecArgs, VerboseQuietArgs):
    base_dir: Optional[Path] = add_argument(
        shortcut="-b", type=str, help="Source base dir", default=None
    )
    tar_files_glob: str = add_argument(
        shortcut="-t", type=str, help="Glob pattern for tar files", default="**/*.tar"
    )
    index_file: Optional[Path] = add_argument(type=str, help="Index file", default=None)


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")
    tar_files = list(args.base_dir.glob(args.tar_files_glob))
    logger.info(f"Found {len(tar_files)} tar files in {args.base_dir}.")
    if args.index_file is None:
        args.index_file = args.base_dir / "index.sqlite"
    logger.info(f"Creating tarlookup and indexing to {args.index_file}")
    tar_lookup = TarLookup(
        args.base_dir,
        tar_files,
        args.index_file,
        verbose=True,
        sort_tar_files=True,
        worker_id=0,
        delete_index=False,
    )
    logger.info(f"Got {len(tar_lookup)} files inside the tars.")
    plaintext_filenames_file = Path(f"{args.index_file.as_posix()}.json")
    dump_json(
        {"filenames": tar_lookup.file_names},
        plaintext_filenames_file,
        indent=2,
        custom_format=False,
    )


if __name__ == "__main__":
    main()
