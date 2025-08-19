"""
TarLookup uses sqlite3 to store an index and allows indexing of multiple tar archives.

Note, only raw (uncompressed) tar files are accepted as native tar.gz cannot be random accessed.
But you can compress each file using zlib before adding it to tar.
The tool requires sqlite3 - install on the system or with conda

index_tar, get_cursor, prepare_db are modified version of https://github.com/lpryszcz/tar_indexer
license: GPL-3.0 https://www.gnu.org/licenses/gpl-3.0.en.html
in accordance with the license, these functions are still licensed as GPL-3.0.
author: l.p.pryszcz@gmail.com - Mizerow, 23/04/2014
changes: adapt for python3, remove script usage, small tweaks

"""

from __future__ import annotations

import json
import os
import sqlite3
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from timeit import default_timer
from typing import List, Optional, Tuple, Union

from visiontext.distutils import barrier_safe, is_main_process


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
                connection, cursor = get_cursor(index_file)
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
                    index_tar(base_path, tar_file_rel, cursor, verbose=verbose)
                connection.close()
            barrier_safe()

        try:
            connection = sqlite3.connect(str(index_file), check_same_thread=False)
        except sqlite3.OperationalError as e:
            raise RuntimeError(f"Could not load {index_file}.") from e
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

    def get_files_per_shard(
        self, endswith: Optional[str] = None, sorter: callable = sorted
    ) -> dict[str, List[str]]:
        """
        Group files by shard name.

        Args:
            endswith: to count only e.g. .jpg files
            sorter: the default sorted is fine for sorting tars named 00000.tar, 00001.tar, ...
                however if tar filenames do not have leading zeros, set it to natsort.natsorted

        Returns:
            dict of shardname to list of filenames
        """
        self.cursor.execute(
            "SELECT o.file_name, f.file_name, offset, file_size FROM offset_data as o JOIN file_data as f ON o.file_id=f.file_id"
        )
        fetches = self.cursor.fetchall()
        shard2fi = defaultdict(list)
        for filename, shardname, _, _ in fetches:
            if endswith is not None:
                if not filename.endswith(endswith):
                    continue
            shard2fi[shardname].append(filename)

        return dict(sorter(shard2fi.items(), key=lambda x: x[0]))

    def __len__(self):
        return self.n_content_files


def index_tar(base_path, tarfn, cursor, verbose=True, sleep_duration=0.1):
    """Index tar file and create sqlite index."""
    # get archive size
    tarfn_str = str(tarfn)
    tarfn_full = Path(base_path) / tarfn
    if not tarfn_full.is_file():
        raise FileNotFoundError(f"File not found: {tarfn_full}")
    tarsize = os.path.getsize(tarfn_full)
    # prepare db
    file_id = prepare_db(base_path, cursor, tarfn_str, verbose)
    if not file_id:
        return
    tar = tarfile.open(tarfn_full)
    data = []
    i = 1
    n_files = 0
    for tarinfo in tar:
        if not tarinfo.isfile():
            continue
        data.append((file_id, tarinfo.name, tarinfo.offset_data, tarinfo.size))
        n_files += 1
        # upload
        if i % 100 == 0:
            cursor.executemany("INSERT INTO offset_data VALUES (?, ?, ?, ?)", data)
            if verbose:
                print(f" {i} [{tarinfo.offset_data / tarsize:.2%}]      ", end="\r")
            # free ram...
            data = []
            tar.members = []
        i += 1
    # upload last batch
    if len(data) > 0:
        cursor.executemany("INSERT INTO offset_data VALUES (?, ?, ?, ?)", data)
    # finally commit changes
    cursor.connection.commit()
    print(f"Wrote {n_files} files. Sleeping {sleep_duration}s to let the DB work...")
    time.sleep(sleep_duration)


def get_cursor(indexfn):
    """Return cur object"""
    # create/connect to sqlite3 database
    cnx = sqlite3.connect(indexfn)
    cur = cnx.cursor()
    # asyn execute >50x faster ##http://www.sqlite.org/pragma.html#pragma_synchronous
    # cur.execute("PRAGMA synchronous=OFF")
    # prepare db schema #type='table' and
    cur.execute("SELECT * FROM sqlite_master WHERE name='file_data'")
    result = cur.fetchone()
    if not result:
        # create file data table
        cur.execute(
            """CREATE TABLE file_data (file_id INTEGER PRIMARY KEY,
                    file_name TEXT, mtime FLOAT)"""
        )
        cur.execute("CREATE INDEX file_data_file_name ON file_data (file_name)")
        # create offset_data table
        cur.execute(
            """CREATE TABLE offset_data (file_id INTEGER,
                    file_name TEXT, offset INTEGER, file_size INTEGER)"""
        )  # ,PRIMARY KEY (file_id, file_name))
        cur.execute("CREATE INDEX offset_data_file_id ON offset_data (file_id)")
        cur.execute("CREATE INDEX offset_data_file_name ON offset_data (file_name)")
    return cnx, cur


def prepare_db(base_path, cur, tarfn_str, verbose):
    """Prepare database and add file"""
    tarfn_full = Path(base_path) / tarfn_str
    mtime = os.path.getmtime(tarfn_full)
    # check if file already indexed
    cur.execute("SELECT file_id, mtime FROM file_data WHERE file_name = ?", (tarfn_str,))
    result = cur.fetchone()  # ; print result
    if result:
        file_id, pmtime = result  # ; print file_id, pmtime
        # skip if index with newer mtime than archives exists
        if mtime <= pmtime:
            if verbose:
                print(" Archive already indexed.")
            return None
        # else update mtime and remove previous index
        cur.execute("UPDATE file_data SET mtime = ? WHERE file_name = ?", (mtime, tarfn_str))
        cur.execute("DELETE FROM offset_data WHERE file_id = ?", (file_id,))
    else:
        cur.execute("SELECT MAX(file_id) FROM file_data")
        (max_file_id,) = cur.fetchone()  # ; print max_file_id
        if max_file_id:
            file_id = max_file_id + 1
        else:
            file_id = 1
        # add file info
        cur.execute("INSERT INTO file_data VALUES (?, ?, ?)", (file_id, tarfn_str, mtime))
    return file_id
