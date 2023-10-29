"""
Original source: https://github.com/lpryszcz/tar_indexer
LICENSE is GNUGPLV3.0

Author:
    l.p.pryszcz@gmail.com
    Mizerow, 23/04/2014
Changes:
    - 2to3
    - remove script usage
    - small tweaks

TAR indexer uses sqlite3 for index storing and allows indexing of multiple tar archives.
Note, only raw (uncompressed) tar files are accepted as native tar.gz cannot be random accessed.
But you can compress each file using zlib before adding it to tar.

Prerequisites
    Python 3.7+
    sqlite3 - install with sudo apt-get install / with conda

"""

import os
import sqlite3
import tarfile
import time
from pathlib import Path


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
    return cur


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


def index_tar(base_path, tarfn, indexfn, verbose=True, sleep_duration=5):
    """Index tar file and create sqlite index."""
    if verbose:
        print(f"Indexing tarfile {base_path} / {tarfn} to {indexfn}")
    if not indexfn:
        raise ValueError(f"Index missing {indexfn}")
    # get archive size
    tarfn_str = str(tarfn)
    tarfn_full = Path(base_path) / tarfn
    if not tarfn_full.is_file():
        raise FileNotFoundError(f"File not found: {tarfn_full}")
    tarsize = os.path.getsize(tarfn_full)
    # prepare db
    cur = get_cursor(indexfn)
    file_id = prepare_db(base_path, cur, tarfn_str, verbose)
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
            cur.executemany("INSERT INTO offset_data VALUES (?, ?, ?, ?)", data)
            if verbose:
                print(f" {i} [{tarinfo.offset_data / tarsize:.2%}]      ", end="\r")
            # free ram...
            data = []
            tar.members = []
        i += 1
    # upload last batch
    if len(data) > 0:
        cur.executemany("INSERT INTO offset_data VALUES (?, ?, ?, ?)", data)
    # finally commit changes
    cur.connection.commit()
    print(f"Wrote {n_files} files to {indexfn}. Sleep {sleep_duration}s to let the DB work...")
    time.sleep(sleep_duration)
