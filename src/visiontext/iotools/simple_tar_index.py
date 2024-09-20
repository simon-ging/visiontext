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

from packg.iotools import load_json, dump_json
from packg.typext import PathType


class SingleTarLookup:
    def __init__(
        self,
        tar_file: PathType,
        index_file: Optional[PathType] = None,
        force_rebuild_index: bool = False,
        worker_id: int = 0,
    ):
        self.tar_file = Path(tar_file)
        self.index = get_tar_index(tar_file, index_file, force_rebuild_index)
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

    def close(self):
        if self.worker_id in self.filepointers:
            self.filepointers[self.worker_id].close()
            del self.filepointers[self.worker_id]


def get_index_file_for_tar_file(tar_file: PathType) -> Path:
    tar_file = Path(tar_file)
    return Path(f"{tar_file.as_posix()}.json")


def get_tar_index(
    tar_file: PathType, index_file: Optional[PathType] = None, force: bool = False
) -> dict[str, any]:
    """

    Args:
        tar_file: tar file to index
        index_file: where to save the index, default f"{tar_file}.json"
        force: always overwrite the index

    Returns:
        dictionary of
            size: tar file size
            mtime: tar file modification time
            files: dictionary of file name to (offset, size) in the tar

    """
    tar_file = Path(tar_file)
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
        if size == tar_size and mtime == tar_mtime:
            # index already exists and is up to date
            return existing_index

    # index must be rebuilt
    fileindex = build_tar_fileindex(tar_file)
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
