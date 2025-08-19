"""
Webdataset extension where jpeg images are in a tar file,
and metadata information is in a separate json file with the same name.

This makes it easier to update or read the metadata later, since it is not sitting in a tar file.

pipeline is:
- SimpleShardList
- shuffle shards
- split by node
- split by worker
- NEW: wds_pipeline_expand_tarfile_and_json_to_samples
    - calls wds_tarfile_expander_with_metadata which loads tar + json into samples.
        this also loads the jpeg bytes.
    - calls wds_convert_jpg_files_and_metadata_to_samples to further process the samples to dicts
- shuffle samples
- NEW: wds_decode_with_metadata decodes the jpeg and keeps the metadata as dict.

the filtering function will get as input dCCJy5Fr-JGv2LciCB0CtYCJm6bq6v4NM04DJg/4.jpg
and returns False to skip the file.

"""

import re
import tarfile
from typing import Any, Callable, Dict, Iterable, Iterator, Optional

import webdataset as wds
from webdataset import autodecode, gzfilter, pipelinefilter, reraise_exception
from webdataset.autodecode import Continue, basichandlers
from webdataset.tariterators import base_plus_ext, tar_file_iterator, url_opener

from packg.iotools import load_json


class wds_pipeline_expand_tarfile_and_json_to_samples(wds.PipelineStage):
    """Given a stream of tar files, yield samples.

    based on fucnctions tar_file_expander and tarfile_to_samples

    Args:
        handler: exception handler
        select_files: function that selects files to be included
        rename_files: function that renames files in the tars (i.e. change their ending)
        tar_file_iter: function that iterates over the tar file

    Returns:
        stream of samples
    """

    def __init__(
        self,
        handler=reraise_exception,
        select_files: Optional[Callable[[str], bool]] = None,
        rename_files: Optional[Callable[[str], str]] = None,
        tar_file_iter=tar_file_iterator,
        add_metadata_to_iter=False,
        metadata_filename: str | None = None,
    ):
        self.handler = handler
        self.select_files = select_files
        self.rename_files = rename_files
        self.tar_file_iter = tar_file_iter
        self.add_metadata_to_iter = add_metadata_to_iter
        self.metadata_filename = metadata_filename

    def run(self, src):
        """Given a stream of tar files, yield samples."""
        streams = url_opener(src, handler=self.handler)
        files = wds_tarfile_expander_with_metadata(
            streams,
            self.handler,
            self.select_files,
            self.rename_files,
            tar_file_iter=self.tar_file_iter,
            add_metadata_to_iter=self.add_metadata_to_iter,
            metadata_filename=self.metadata_filename,
        )
        samples = wds_convert_jpg_files_and_metadata_to_samples(files, handler=self.handler)
        return samples


def wds_tarfile_expander_with_metadata(
    src,
    handler,
    select_files,
    rename_files,
    tar_file_iter=tar_file_iterator,
    add_metadata_to_iter=False,
    metadata_filename: str | None = None,
):
    for source in src:
        url = source["url"]
        # here we load the json metadata that belongs to the tar (same name)
        # assuming url is a string (other format not supported) e.g.
        # /.../dataset/v10/lowres_train/shard_00098.tar
        assert isinstance(url, str) and url.endswith(".tar")
        if metadata_filename is None:
            json_url = f"{url[:-4]}.json"
        else:
            url_folder, url_filename = url.rsplit("/", 1)
            shard_num = int(url_filename.split("_")[1].split(".")[0])
            json_url = url_folder + "/" + metadata_filename.format(shard_num)
        metadata = load_json(json_url)

        try:
            assert isinstance(source, dict)
            assert "stream" in source
            # tar file iterator is where the actual data is read.
            # select_files is the last chance to skip before the bytes are read
            # source is just the shard info and the opened stream on the tar file
            iter_kwargs = dict(
                handler=handler,
                select_files=select_files,
                rename_files=rename_files,
            )
            if add_metadata_to_iter:
                iter_kwargs["metadata"] = metadata
            for sample in tar_file_iter(source["stream"], **iter_kwargs):
                assert isinstance(sample, dict) and "data" in sample and "fname" in sample
                # we will get the key from fname: {query_hash}/{imgnum}.jpg
                sample["__url__"] = url
                image_key = sample["fname"][:-4]
                image_metadata = metadata[image_key]
                # # serialize the metadata back to json string
                # sample["metadata"] = json.dumps(image_metadata)
                sample["metadata"] = image_metadata
                # print(sample["fname"], sample["metadata"]["image_file"])
                yield sample
        except Exception as exn:
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                break


meta_prefix = "__"
meta_suffix = "__"


def wds_tar_file_iterator_with_metadata(
    fileobj: tarfile.TarFile,
    skip_meta: Optional[str] = r"__[^/]*__($|/)",
    handler: Callable[[Exception], bool] = reraise_exception,
    select_files: Optional[Callable[[str, dict], bool]] = None,
    rename_files: Optional[Callable[[str], str]] = None,
    metadata: dict | None = None,
) -> Iterator[Dict[str, Any]]:
    """for v4 we need to add the metadata to the skip function already"""

    stream = tarfile.open(fileobj=fileobj, mode="r|*")
    for tarinfo in stream:
        fname = tarinfo.name
        try:
            if not tarinfo.isreg():
                continue
            if fname is None:
                continue
            if "/" not in fname and fname.startswith(meta_prefix) and fname.endswith(meta_suffix):
                # skipping metadata for now
                continue
            if skip_meta is not None and re.match(skip_meta, fname):
                continue
            if rename_files:
                fname = rename_files(fname)
            assert fname.endswith(".jpg"), f"{fname=}"
            fmetadata = metadata[fname[:-4]]
            if select_files is not None and not select_files(fname, fmetadata):
                continue
            data = stream.extractfile(tarinfo).read()
            result = dict(fname=fname, data=data)
            yield result
            stream.members = []
        except Exception as exn:
            if hasattr(exn, "args") and len(exn.args) > 0:
                exn.args = (str(exn.args[0]) + " @ " + str(fileobj),) + exn.args[1:]
            if handler(exn):
                continue
            else:
                break
    del stream


def wds_convert_jpg_files_and_metadata_to_samples(
    data: Iterable[dict[str, Any]],
    keys: Callable[[str], tuple[str, str]] = base_plus_ext,
    lcase: bool = True,
    suffixes: Optional[set[str]] = None,
    handler: Callable[[Exception], bool] = reraise_exception,
) -> Iterator[dict[str, Any]]:
    """Group tarfile contents by keys and yield samples.
    The original function could group multiple files into one key.
    This function instead assumes 1 file per key (the jpeg) and the metadata comes separately
    in the field "metadata".

    Args:
        data: iterator over tarfile contents
        keys: function that takes a file name and returns a key and a suffix.
        lcase: whether to lowercase the suffix.
        suffixes: list of suffixes to keep.
        handler: exception handler.

    Raises:
        ValueError: raised if there are duplicate file names in the tar file.

    Yields:
        iterator over samples.
    """
    for i, filesample in enumerate(data):
        try:
            assert isinstance(filesample, dict)
            fname = filesample["fname"]
            value = filesample["data"]
            metadata = filesample["metadata"]
            url = filesample["__url__"]
            # print(f"{i=} {fname=} {len(value)=} {url=} {metadata=}")
            prefix, suffix = keys(fname)
            if prefix is None:
                raise ValueError(f"{fname=} has empty prefix {prefix=}")
            if suffix != "jpg":
                raise ValueError(f"{fname=} has invalid suffix {suffix=}")
            if lcase:
                suffix = suffix.lower()
            current_sample = dict(__key__=prefix, __url__=url, metadata=metadata)
            current_sample[suffix] = value  # suffix will be jpg here
            yield current_sample
        except Exception as exn:
            # exn.args = exn.args + (filesample.get("stream"), filesample.get("url"))
            if handler(exn):
                continue
            else:
                break


default_pre_handlers = [gzfilter]
default_post_handlers = [basichandlers]


class DecoderWithMetadata:
    """Decode samples using a list of handlers.

    For each key/data item, this iterates through the list of
    handlers until some handler returns something other than None.
    """

    def __init__(self, handlers, pre=None, post=None, only=None, partial=False):
        """Create a Decoder.

        :param handlers: main list of handlers
        :param pre: handlers called before the main list (.gz handler by default)
        :param post: handlers called after the main list (default handlers by default)
        :param only: a list of extensions; when give, only ignores files with those extensions
        :param partial: allow partial decoding (i.e., don't decode fields that aren't of type bytes)
        """
        if isinstance(only, str):
            only = only.split()
        self.only = only if only is None else set(only)
        if pre is None:
            pre = default_pre_handlers
        if post is None:
            post = default_post_handlers
        assert all(callable(h) for h in handlers), f"one of {handlers} not callable"
        assert all(callable(h) for h in pre), f"one of {pre} not callable"
        assert all(callable(h) for h in post), f"one of {post} not callable"
        self.handlers = pre + handlers + post
        self.partial = partial

    def decode1(self, key, data):
        """Decode a single field of a sample.

        :param key: file name extension
        :param data: binary data
        """
        key = "." + key
        for f in self.handlers:
            result = f(key, data)
            if isinstance(result, Continue):
                key, data = result.key, result.data
                continue
            if result is not None:
                return result
        return data

    def decode(self, sample):
        """Decode an entire sample.

        :param sample: the sample, a dictionary of key value pairs
        """
        result = {}
        assert isinstance(sample, dict), sample
        for k, v in list(sample.items()):
            if k[:2] == "__":
                if isinstance(v, bytes):
                    try:
                        v = v.decode("utf-8")
                    except Exception:
                        print(f"Can't decode v of k = {k} as utf-8: v = {v}")
                result[k] = v
                continue
            if self.only is not None and k not in self.only:
                result[k] = v
                continue
            assert v is not None
            if self.partial:
                if isinstance(v, bytes):
                    result[k] = self.decode1(k, v)
                else:
                    result[k] = v
            elif isinstance(v, dict) and k == "metadata":
                # keep dict metadata as is
                result[k] = v
            else:
                assert isinstance(v, bytes), f"k,v = {k}, {v}"
                result[k] = self.decode1(k, v)
        return result

    def __call__(self, sample):
        """Decode an entire sample.

        :param sample: the sample
        """
        assert isinstance(sample, dict), (len(sample), sample)
        return self.decode(sample)


def _decode_with_metadata(data, *args, handler=reraise_exception, **kw):
    """Decode data based on the decoding functions given as arguments."""

    decoder = lambda x: autodecode.imagehandler(x) if isinstance(x, str) else x
    handlers = [decoder(x) for x in args]
    f = DecoderWithMetadata(handlers, **kw)

    for sample in data:
        assert isinstance(sample, dict), sample
        try:
            decoded = f(sample)
        except Exception as exn:  # skipcq: PYL-W0703
            if handler(exn):
                continue
            else:
                break
        yield decoded


wds_decode_with_metadata = pipelinefilter(_decode_with_metadata)
