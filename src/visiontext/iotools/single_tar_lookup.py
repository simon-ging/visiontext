"""
simple tar index, with one json index for each tar file, and only one tar file read at a time.

compared to the sqlite based tar indexer this is simpler, but probably less efficient
for large datasets because you always need to read the full index before searching,
instead of relying on dataset selects.
"""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Optional

from natsort import natsorted

from packg import format_exception
from packg.iotools import dump_json, load_json
from packg.log import logger
from packg.typext import PathType


class SingleTarLookup:
    def __init__(
        self,
        tar_file: PathType,
        index_file: Optional[PathType] = None,
        force_rebuild_index: bool = False,
        worker_id: int = 0,
        check_stat: bool = True,
    ):
        self.tar_file = Path(tar_file)
        self.index = get_tar_index(tar_file, index_file, force_rebuild_index, check_stat=check_stat)
        self.filenames = list(self.index["files"].keys())
        if len(self.filenames) == 0:
            raise ValueError(f"No files in tar: {tar_file}")
        self.filepointers = {}
        self.worker_id = worker_id

    def __repr__(self):
        type_name = type(self).__name__
        return (
            f"{type_name}(tar_file={self.tar_file}, "
            f"worker_id={self.worker_id}) with {len(self.filenames)} files, example first "
            f"file: {self.filenames[0]}"
        )

    def get_filenames(self):
        return self.filenames

    def get_file_info(self, filename_in: str) -> tuple[int, int, float]:
        if filename_in not in self.index["files"]:
            raise FileNotFoundError(f"{filename_in} not found in {self.tar_file}")
        return self.index["files"][filename_in]

    def get_file_content(self, filename_in: str) -> bytes:
        offset, size, mtime = self.get_file_info(filename_in)
        if self.worker_id not in self.filepointers:
            self.filepointers[self.worker_id] = self.tar_file.open("rb")
        tarf = self.filepointers[self.worker_id]
        tarf.seek(offset)
        content = tarf.read(size)
        return content

    def has_file(self, filename_in: str) -> bool:
        return filename_in in self.index["files"]

    def close(self):
        if self.worker_id in self.filepointers:
            self.filepointers[self.worker_id].close()
            del self.filepointers[self.worker_id]


def get_index_file_for_tar_file(tar_file: PathType) -> Path:
    tar_file = Path(tar_file)
    return Path(f"{tar_file.as_posix()}.json")


def get_tar_index(
    tar_file: PathType,
    index_file: Optional[PathType] = None,
    force: bool = False,
    check_stat: bool = True,
) -> dict[str, Any]:
    """

    Args:
        tar_file: tar file to index
        index_file: where to save the index, default f"{tar_file}.json"
        force: always overwrite the index
        check_stat: verify tar file size and mtime in the index (default True is slower)

    Returns:
        dictionary of
            size: tar file size
            mtime: tar file modification time
            files: dictionary of file name to (offset, size) in the tar

    """
    tar_file = Path(tar_file)
    tar_size, tar_mtime = None, None
    if check_stat:
        assert tar_file.is_file(), f"File not found: {tar_file}"
        tar_file_stat = tar_file.stat()
        tar_size, tar_mtime = tar_file_stat.st_size, tar_file_stat.st_mtime

    if index_file is None:
        index_file = get_index_file_for_tar_file(tar_file)
    else:
        index_file = Path(index_file)

    if not force and index_file.is_file():
        existing_index = load_json(index_file)
        size, mtime = existing_index["size"], existing_index["mtime"]
        if not check_stat or (size == tar_size and mtime == tar_mtime):
            # index already exists and is up to date
            return existing_index

    # index must be rebuilt
    try:
        fileindex = build_tar_fileindex(tar_file)
    except tarfile.ReadError as e:
        logger.error(f"Failed to read tar file {tar_file}: {format_exception(e)}")
        raise
    index = {"size": tar_size, "mtime": tar_mtime, "files": fileindex}
    dump_json(index, index_file, verbose=False)
    return index


def build_tar_fileindex(tar_file: PathType) -> dict[str, tuple[int, int, float]]:
    tar = tarfile.open(tar_file)
    fileindex = {}
    for tarinfo in tar:
        if not tarinfo.isfile():
            continue
        fileindex[tarinfo.name] = [tarinfo.offset_data, tarinfo.size, tarinfo.mtime]
    return fileindex


def search_and_compress_files_to_tar(
    tar_file: PathType,
    base_dir: PathType,
    patterns: tuple[str] = ("**/*",),
    verify: bool = True,
    delete_files: bool = False,
    sort_fn: Optional[callable] = natsorted,
):
    rel_files = []
    for pattern in patterns:
        rel_files.extend([file.relative_to(base_dir).as_posix() for file in base_dir.glob(pattern)])
    if len(rel_files) == 0:
        raise FileNotFoundError(f"No files found to compress for\n{base_dir=}\n{patterns=}")
    if sort_fn is not None:
        rel_files = sort_fn(rel_files)
    logger.info(f"Found {len(rel_files)} files in {base_dir}, compressing to {tar_file}")
    compress_files_to_tar(tar_file, base_dir, rel_files, verify=verify, delete_files=delete_files)


def compress_files_to_tar(
    tar_file: PathType,
    base_dir: PathType,
    rel_files: list[PathType],
    delete_files: bool = False,
    verify: bool = True,
):
    if tar_file.is_file():
        if verify:
            assert verify_tar(tar_file, base_dir, rel_files), f"Verification failed for {tar_file}"
            return
        return
    # tar does not exist, create the tar
    if len(rel_files) == 0:
        raise FileNotFoundError(f"No files given to compress.")
    with tarfile.open(tar_file.as_posix(), "w") as tar:
        for file_rel in rel_files:
            file_full = base_dir / file_rel
            info = tar.gettarinfo(file_full.as_posix())
            # write dummy user info
            info.uid = 10999
            info.gid = 10999
            info.uname = "bigdata"
            info.gname = "bigdata"
            # info.mtime = file_fullstat().mtime
            # info.size = file_full.stat().st_size
            info.mode = 0o666
            # save relative paths
            info.name = file_rel
            tar.addfile(info, file_full.open("rb"))
    if verify:
        assert verify_tar(tar_file, base_dir, rel_files), f"Verification failed for {tar_file}"
    if delete_files:
        logger.warning(f"Deleting {len(rel_files)} files in {base_dir} after compression.")
        for file_rel in rel_files:
            (base_dir / file_rel).unlink()


def verify_tar(tar_file: PathType, base_dir: PathType, rel_files: list[PathType]) -> bool:
    # check tar contains all files from above, and only those
    file_missing = {f: True for f in rel_files}
    try:
        with tarfile.open(tar_file.as_posix(), "r") as tar:
            for tarinfo in tar:
                assert tarinfo.isfile()
                assert file_missing[tarinfo.name], f"Duplicate file: {tarinfo.name}"
                file_missing[tarinfo.name] = False
    except tarfile.ReadError as e:
        logger.error(f"Failed to read tar file {tar_file}: {format_exception(e)}")
        return False

    missing_files = [f for f, missing in file_missing.items() if missing]
    assert len(missing_files) == 0, f"Missing files: {missing_files}"

    # check all files have correct content
    try:
        lookup = SingleTarLookup(tar_file)
        assert natsorted(lookup.get_filenames()) == rel_files
        for file_rel in rel_files:
            content = lookup.get_file_content(file_rel)
            gt_content = (base_dir / file_rel).read_bytes()
            assert content == gt_content
            offset, size, mtime = lookup.get_file_info(file_rel)
            gt_stat = (base_dir / file_rel).stat()
            assert size == gt_stat.st_size
            assert mtime == gt_stat.st_mtime
    except AssertionError as e:
        logger.error(f"Verification failed for tar file {tar_file}: {format_exception(e)}")
        return False
    return True
