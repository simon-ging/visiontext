import os
import tarfile
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from packg.iotools import dump_json, load_json
from visiontext.iotools.single_tar_lookup import (
    build_tar_fileindex,
    get_tar_index,
    get_index_file_for_tar_file,
    SingleTarLookup,
)


@pytest.fixture(scope="session")
def temp_data_for_tar_test(tmp_path_factory):
    temp_dir = Path(tmp_path_factory.mktemp("data"))

    # create random image
    width, height = 100, 100  # Size of the image
    random_image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(random_image_array)
    image_path = temp_dir / "random_image.jpg"
    image.save(image_path.as_posix())

    # create random json file
    random_json_data = {
        "name": "Random Name",
        "value": np.random.randint(1, 100),
        "valid": bool(np.random.randint(0, 2)),
    }
    json_path = temp_dir / "random_data.json"
    dump_json(random_json_data, json_path, verbose=False)

    # tar the data
    tar_path = temp_dir / "files_archive.tar"
    with tarfile.open(tar_path.as_posix(), "w") as tar:
        for file_path in [image_path, json_path]:
            info = tar.gettarinfo(file_path.as_posix())
            # delete actual user info
            info.uid = 10999
            info.gid = 10999
            info.uname = "bigdata"
            info.gname = "bigdata"
            info.mtime = file_path.stat().st_mtime
            info.size = file_path.stat().st_size
            info.mode = 0o444
            # save relative paths
            info.name = file_path.relative_to(tar_path.parent).as_posix()
            tar.addfile(info, open(file_path, "rb"))
    return image_path, json_path, tar_path


def test_random_image_and_json(temp_data_for_tar_test):
    """
    Test if the fixture above works
    """
    image_path, json_path, tar_path = temp_data_for_tar_test
    assert os.path.exists(image_path)
    assert os.path.exists(json_path)
    assert os.path.exists(tar_path)
    image = Image.open(image_path.as_posix())
    assert image.size == (100, 100)
    json_data = load_json(json_path)
    assert "name" in json_data
    assert "value" in json_data
    assert "valid" in json_data


def _test_fileindex(image_path, json_path, fileindex):
    for file_path in [image_path, json_path]:
        assert file_path.name in fileindex
        offset, size, mtime = fileindex[file_path.name]
        assert offset >= 0
        assert size > 0
        assert size == file_path.stat().st_size
        assert mtime == file_path.stat().st_mtime


def test_build_tar_fileindex(temp_data_for_tar_test):
    image_path, json_path, tar_path = temp_data_for_tar_test
    fileindex = build_tar_fileindex(tar_path)
    _test_fileindex(image_path, json_path, fileindex)


def test_get_tar_index(temp_data_for_tar_test):
    image_path, json_path, tar_path = temp_data_for_tar_test
    index = get_tar_index(tar_path)
    tar_start = tar_path.stat()
    size, mtime = tar_start.st_size, tar_start.st_mtime
    assert index["size"] == size
    assert index["mtime"] == mtime
    fileindex = index["files"]
    _test_fileindex(image_path, json_path, fileindex)

    # now the index file should already exist and not be rewritten
    index_file = get_index_file_for_tar_file(tar_path)
    assert index_file.is_file()
    index_file_mtime = index_file.stat().st_mtime
    index_again = get_tar_index(tar_path)
    assert index_file_mtime == index_file.stat().st_mtime
    assert index == index_again, f"{index} != {index_again}"

    # unless force is True
    time.sleep(0.01)
    index_again = get_tar_index(tar_path, force=True)
    assert index_file_mtime != index_file.stat().st_mtime, f"{index_file}"
    assert index == index_again

    # finally test setting custom index
    custom_index_file = tar_path.with_suffix(".custom_index.json")
    index_again = get_tar_index(tar_path, index_file=custom_index_file)
    assert custom_index_file.is_file()
    assert index == index_again
    index_file_mtime = custom_index_file.stat().st_mtime
    index_again = get_tar_index(tar_path, index_file=custom_index_file)
    assert index_file_mtime == custom_index_file.stat().st_mtime
    assert index == index_again
    time.sleep(0.01)
    index_again = get_tar_index(tar_path, index_file=custom_index_file, force=True)
    assert index_file_mtime != custom_index_file.stat().st_mtime
    assert index == index_again


def test_lookup(temp_data_for_tar_test):
    image_path, json_path, tar_path = temp_data_for_tar_test
    gt_filepaths = [image_path, json_path]
    gt_filenames = [file_path.name for file_path in gt_filepaths]
    for index_file in [None, tar_path.with_suffix(".custom_index.json")]:
        for force_rebuild_index in [False, True]:
            for worked_id in [0, 1]:
                lookup = SingleTarLookup(
                    tar_path,
                    index_file=index_file,
                    force_rebuild_index=force_rebuild_index,
                    worker_id=worked_id,
                )
                filenames = lookup.get_filenames()
                for i, filename in enumerate(filenames):
                    gt_filename = gt_filenames[i]
                    assert filename == gt_filename
                    content = lookup.get_file_content(filename)
                    gt_filepath = gt_filepaths[i]
                    gt_content = gt_filepath.read_bytes()
                    assert content == gt_content
