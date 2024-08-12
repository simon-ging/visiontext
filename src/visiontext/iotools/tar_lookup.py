import json
import os
import sqlite3
import time
from pathlib import Path
from timeit import default_timer
from typing import List, Tuple, Union

from visiontext.iotools.tar_indexer import index_tar
from visiontext.distutils import is_main_process, barrier_safe


class TarLookup:
    def __init__(
        self,
        base_path: Union[str, Path],
        tar_files_rel: List[Union[str, Path]],
        index_file: Union[str, Path],
        verbose: bool = True,
        sort_tar_files: bool = True,
        worker_id: int = 0,
        delete_index: bool = False,
    ):
        """
        Allow to lookup files inside tar archives via random access.

        Note: This class does not have update functionality. So if the tars change
        the index must be deleted.

        Args:
            base_path: Path where tar files are stored. The index will reference files relative
                to this path.
            tar_files_rel: List of tar files to index, relative to base_path OR absolute but inside
                base_path.
            index_file: Absolute path to sqlite index file.
            verbose: Print verbose output.
            sort_tar_files: Sort tar file input list before indexing.
            worker_id: Worker id to use for file pointers. When using multiprocessing dataloaders,
                each worker that uses this lookup needs its own file pointer to the tars.
                In torch use following code to find it:
                    from torch.utils.data import get_worker_info
                    worker = get_worker_info()
                    worker = worker.id if worker else None
            delete_index: Delete index file(s) and rebuild.

        """
        base_path = Path(base_path)
        index_file = Path(index_file)

        if verbose:
            print(
                f"Creating TarLookup with index {index_file} and base path {base_path} "
                f"with {len(tar_files_rel)} tar files."
            )
        if len(tar_files_rel) == 0:
            raise ValueError(f"No tar files to index for base path {base_path}.")

        filename_cache_file = Path(f"{index_file.as_posix()}.filenames.json")
        if delete_index:
            if verbose:
                print(f"Deleting {index_file} and {filename_cache_file} if they exist.")
            if index_file.is_file():
                index_file.unlink()
            if filename_cache_file.is_file():
                filename_cache_file.unlink()

        if not index_file.is_file():
            # index the tars
            if is_main_process():
                os.makedirs(base_path, exist_ok=True)
                os.makedirs(index_file.parent, exist_ok=True)
                if sort_tar_files:
                    tar_files_rel = sorted(tar_files_rel)
                for i, tar_file_rel in enumerate(tar_files_rel):
                    tar_file_rel = Path(tar_file_rel)
                    if tar_file_rel.is_absolute():
                        try:
                            tar_file_rel = tar_file_rel.relative_to(base_path)
                        except ValueError as e:
                            raise ValueError(
                                f"Tar file {tar_file_rel} must be relative to base_path {base_path}."
                            ) from e
                    tar_file_abs = base_path / tar_file_rel
                    if not tar_file_abs.is_file():
                        raise FileNotFoundError(f"Tar file {tar_file_abs} not found.")
                    if verbose:
                        print(
                            f"Indexing file {i}/{len(tar_files_rel)} dir {base_path} "
                            f"filename {tar_file_rel} to {index_file}"
                        )
                    # create sqlite index
                    index_tar(base_path, tar_file_rel, index_file, verbose=verbose)
            barrier_safe()

        # connect to database and load content
        try:
            connection = sqlite3.connect(str(index_file), check_same_thread=False)
        except sqlite3.OperationalError as e:
            print(f"Could not load {index_file}.")
            raise e
        cursor = connection.cursor()

        t1 = default_timer()

        # how many tarfiles are stored in the index?
        # fields in file_data are (file_id, file_name, mtime)
        # here file_id and file_name are referring to the tar files themselves
        # not worth caching since there will never be so many tar files
        cursor.execute("SELECT COUNT(file_id) FROM file_data")
        self.n_tar_files = int(cursor.fetchone()[0])
        if verbose:
            print(f"{default_timer() -t1:7.1f}s: Counted {self.n_tar_files} tar files")

        # select the filenames to build the index for this dataset
        # fields in offset_data are (file_id, file_name, offset, file_size)
        # here, file_id and file_name are referring to content files inside the tar.
        # with a 2GB index, 1200 tar, 30M files inside those, this select takes ? alot of time
        # therefore it is worth caching
        if not filename_cache_file.is_file():
            sql = "SELECT file_name FROM offset_data ORDER BY file_name"
            file_names = [data[0] for data in cursor.execute(sql).fetchall()]
            json.dump(file_names, filename_cache_file.open("w", encoding="utf-8"))
            if verbose:
                print(f"{default_timer() -t1:7.1f}s: Built index of {len(file_names)} filenames.")
        else:
            file_names = json.load(filename_cache_file.open(encoding="utf-8"))
            if verbose:
                print(
                    f"{default_timer() -t1:7.1f}s: Loaded index of {len(file_names)} filenames "
                    f"from {filename_cache_file}."
                )

        self.n_content_files = len(file_names)
        self.filepointers = {}

        self.file_names = file_names
        self.verbose = verbose
        self.base_path = base_path
        self.worker_id = worker_id
        self.index_file = index_file
        self.cursor = cursor
        self.connection = connection

    def get_fileinfo_from_filename(
        self, filename_in: str, n_tries: int = 3
    ) -> Tuple[str, str, int, int]:
        """
        Yield file information matching file_name from TAR archives.

        Args:
            filename_in: filename to search
            n_tries: in case of sqlite bugging out, try again a few times

        Returns:
            filename, tarfilename, offset (position inside the tar), file_size

        """
        fails = 0
        while True:
            try:
                self.cursor.execute(
                    """SELECT o.file_name, f.file_name, offset, file_size
                        FROM offset_data as o JOIN file_data as f ON o.file_id=f.file_id
                        WHERE o.file_name=?""",
                    (filename_in,),
                )
                break
            except Exception as e:
                print(
                    f"WARNING: {e} for file {self.index_file} "
                    f"looking for {filename_in} in index."
                )
                fails += 1
                time.sleep(1)
                if fails >= n_tries:
                    raise e
        # get information needed to load the file
        fetch = self.cursor.fetchone()
        if fetch is None:
            raise KeyError(
                f"File {filename_in} not found in index {self.index_file}. "
                f"Try deleting the index to rebuild it automatically."
            )
        filename, tarfilename, offset, file_size = fetch
        return filename, tarfilename, offset, file_size

    def get_content_from_filename(
        self, filename_in: str, n_tries: int = 3
    ) -> Tuple[str, str, bytes]:
        """
        Yield files matching file_name from TAR archives.

        Args:
            filename_in: filename to search
            n_tries: in case of sqlite bugging out, try again a few times

        Returns:
            filename, tarfilename, content (bytes)
        """
        filename, tarfilename, offset, file_size = self.get_fileinfo_from_filename(
            filename_in, n_tries
        )

        tarfn_full = self.base_path / tarfilename
        # create one filepointer for each combination of worker and tarfile
        worker = self.worker_id
        key = (worker, tarfilename)
        if key not in self.filepointers:
            # open tarfile
            self.filepointers[key] = open(tarfn_full, "rb")

        tarf = self.filepointers[key]
        # seek
        tarf.seek(offset)
        # return content
        content = tarf.read(file_size)
        return filename, tarfilename, content

    def get_content_from_index(self, idx: int) -> Tuple[str, str, bytes]:
        """
        Load content from index (number of a single file).

        Args:
            idx:

        Returns:
            filename, tarfilename, content (bytes)

        """
        file_name = self.file_names[idx]
        return self.get_content_from_filename(file_name)

    def __len__(self):
        return self.n_content_files
