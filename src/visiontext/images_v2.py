"""
Load, scale and save images with pillow without breaking metadata.



Metadata that the output format cannot store is dropped, e.g. PNG text chunks when writing jpg.
Everything the format supports is carried over.

Examples:
    >>> image, metadata = load_image("photo.jpg")
    >>> image = scale_to_bigger_side(image, 500)
    >>> save_image(image, f"photo_small{resolve_extension(image, '.jpg')}", metadata)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image, ImageChops, ImageOps, ImageStat, PngImagePlugin
from PIL.Image import Resampling

from packg import format_exception
from packg.typext import PathType

AUTO = "auto"
DEFAULT_UPSAMPLING_METHOD = "bicubic"
DEFAULT_DOWNSAMPLING_METHOD = "area"
DEFAULT_QUALITY = 95

SAMPLING_MAP = {
    "nearest": Resampling.NEAREST,
    "bilinear": Resampling.BILINEAR,
    "bicubic": Resampling.BICUBIC,
    "lanczos": Resampling.LANCZOS,
    "area": Resampling.BOX,
    "box": Resampling.BOX,
    "hamming": Resampling.HAMMING,
}

EXTENSION_TO_FORMAT = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".jpe": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".bmp": "BMP",
}
FORMATS_WITH_ALPHA = {"PNG", "WEBP", "TIFF"}
ALPHA_FALLBACK_EXTENSION = ".png"

# which metadata pillow writes for a given format, see the _save functions of the plugins.
# PNG has no xmp kwarg, it goes into the text chunks instead.
FORMAT_METADATA_KEYS = {
    "JPEG": ("exif", "icc_profile", "xmp", "dpi"),
    "PNG": ("exif", "icc_profile", "dpi"),
    "WEBP": ("exif", "icc_profile", "xmp"),
    "TIFF": ("exif", "icc_profile", "dpi"),
    "BMP": ("dpi",),
}
FORMAT_QUALITY_KEYS = {"JPEG", "WEBP"}
# modes a format can write, everything else is converted to RGB or RGBA
FORMAT_MODES = {
    "JPEG": ("L", "RGB", "CMYK"),
    "PNG": ("1", "L", "LA", "P", "I;16", "RGB", "RGBA"),
    "WEBP": ("RGB", "RGBA"),
    "TIFF": ("1", "L", "LA", "P", "I;16", "RGB", "RGBA", "CMYK"),
    "BMP": ("1", "L", "P", "RGB"),
}
XMP_TEXT_KEY = "XML:com.adobe.xmp"

# an extra frame whose rms difference to the first is below this holds the same picture again,
# usually a smaller copy, and can be dropped. above it the frame shows something else
MULTI_FRAME_RMS_TOLERANCE = 0.10
# rms alone cannot separate the two cases: copies measured up to 0.087 and the most similar
# hdr gain map 0.086. a gain map is grayscale though, so a frame that lost the colour of the
# first one is auxiliary data whatever its rms difference says
MULTI_FRAME_SATURATION_TOLERANCE = 0.02

# exif sub-directories, rebuilt from their content instead of copying the offset
EXIF_IFD_POINTERS = (ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo, ExifTags.IFD.Interop)
# tags recording the pixel size. a decoder takes the size from the jpeg or png header, so these
# are redundant when correct and misleading once the pixels change
EXIF_SIZE_TAGS = (
    ExifTags.Base.ImageWidth,
    ExifTags.Base.ImageLength,
    ExifTags.Base.ExifImageWidth,
    ExifTags.Base.ExifImageHeight,
)
# everything clean_exif removes, apart from the thumbnail directory
EXIF_DROP_TAGS = (ExifTags.Base.Orientation,) + EXIF_SIZE_TAGS
# TIFF pixel storage does not describe a re-encoded image. XMP and ICC travel separately.
TIFF_STORAGE_TAGS = frozenset(
    {
        ExifTags.Base.NewSubfileType,
        ExifTags.Base.SubfileType,
        ExifTags.Base.BitsPerSample,
        ExifTags.Base.Compression,
        ExifTags.Base.PhotometricInterpretation,
        ExifTags.Base.Thresholding,
        ExifTags.Base.CellWidth,
        ExifTags.Base.CellLength,
        ExifTags.Base.FillOrder,
        ExifTags.Base.StripOffsets,
        ExifTags.Base.SamplesPerPixel,
        ExifTags.Base.RowsPerStrip,
        ExifTags.Base.StripByteCounts,
        ExifTags.Base.MinSampleValue,
        ExifTags.Base.MaxSampleValue,
        ExifTags.Base.PlanarConfiguration,
        ExifTags.Base.FreeOffsets,
        ExifTags.Base.FreeByteCounts,
        ExifTags.Base.GrayResponseUnit,
        ExifTags.Base.GrayResponseCurve,
        ExifTags.Base.T4Options,
        ExifTags.Base.T6Options,
        ExifTags.Base.Predictor,
        ExifTags.Base.ColorMap,
        ExifTags.Base.TileWidth,
        ExifTags.Base.TileLength,
        ExifTags.Base.TileOffsets,
        ExifTags.Base.TileByteCounts,
        ExifTags.Base.SubIFDs,
        ExifTags.Base.ExtraSamples,
        ExifTags.Base.SampleFormat,
        ExifTags.Base.SMinSampleValue,
        ExifTags.Base.SMaxSampleValue,
        ExifTags.Base.JPEGTables,
        ExifTags.Base.JPEGProc,
        ExifTags.Base.JpegIFOffset,
        ExifTags.Base.JpegIFByteCount,
        ExifTags.Base.JpegRestartInterval,
        ExifTags.Base.JpegLosslessPredictors,
        ExifTags.Base.JpegPointTransforms,
        ExifTags.Base.JpegQTables,
        ExifTags.Base.JpegDCTables,
        ExifTags.Base.JpegACTables,
        ExifTags.Base.YCbCrCoefficients,
        ExifTags.Base.YCbCrSubSampling,
        ExifTags.Base.YCbCrPositioning,
        ExifTags.Base.ReferenceBlackWhite,
        ExifTags.Base.XMLPacket,
        ExifTags.Base.InterColorProfile,
    }
)
# the directories that hold content, in the order they are reported
EXIF_CONTENT_IFDS = (
    ("IFD0", None),
    ("Exif", ExifTags.IFD.Exif),
    ("GPS", ExifTags.IFD.GPSInfo),
    ("Interop", ExifTags.IFD.Interop),
)


@dataclass
class ImageMetadata:
    """Metadata read from an image, to be passed back into Image.save()."""

    exif: bytes | None = None
    icc_profile: bytes | None = None
    xmp: bytes | None = None
    dpi: tuple[float, float] | None = None
    png_text: dict[str, str] = field(default_factory=dict)
    # format the image was read from as pillow names it, e.g. JPEG. not written back.
    format: str | None = None

    def get_save_kwargs(self, image_format: str) -> dict[str, Any]:
        kwargs = {}
        for key in FORMAT_METADATA_KEYS.get(image_format, ()):
            value = getattr(self, key)
            if value:
                kwargs[key] = value
        if image_format == "PNG":
            png_info = self.get_png_info()
            if png_info is not None:
                kwargs["pnginfo"] = png_info
        return kwargs

    def get_png_info(self) -> PngImagePlugin.PngInfo | None:
        if not self.png_text and not self.xmp:
            return None
        png_info = PngImagePlugin.PngInfo()
        for key, value in self.png_text.items():
            if key == XMP_TEXT_KEY:
                continue  # written below from self.xmp
            png_info.add_text(key, value)
        if self.xmp:
            png_info.add_itxt(XMP_TEXT_KEY, self.xmp.decode("utf-8", errors="replace"))
        return png_info

    def with_clean_exif(self) -> ImageMetadata:
        """Return a copy whose exif no longer describes the old pixel layout, see clean_exif()."""
        if self.exif is None:
            return replace(self)
        return replace(self, exif=clean_exif(self.exif))


def describe_exif_change(old_exif: bytes, new_exif: bytes | None) -> str:
    """
    Summarize what changed between two exif blocks, to log what a clean_exif call did.

    Args:
        old_exif: exif block before
        new_exif: exif block after, None when nothing was left to write

    Returns:
        a compact description of the dropped, changed and added tags
    """
    old, new = Image.Exif(), Image.Exif()
    old.load(old_exif)
    if new_exif is not None:
        new.load(new_exif)
    dropped, changed, added = [], [], []
    old_thumbnail = int(old.get_ifd(ExifTags.IFD.IFD1).get(ExifTags.Base.JpegIFByteCount, 0))
    new_thumbnail = int(new.get_ifd(ExifTags.IFD.IFD1).get(ExifTags.Base.JpegIFByteCount, 0))
    if old_thumbnail > new_thumbnail:
        dropped.append(f"thumbnail {(old_thumbnail - new_thumbnail) / 1024:.0f}KB")
    # the interoperability directory hangs off a pointer, which the loop below skips
    if ExifTags.IFD.Interop in dict(old.get_ifd(ExifTags.IFD.Exif)) and (
        ExifTags.IFD.Interop not in dict(new.get_ifd(ExifTags.IFD.Exif))
    ):
        dropped.append("Interop directory")
    for ifd in (None, ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo):
        names = ExifTags.GPSTAGS if ifd == ExifTags.IFD.GPSInfo else ExifTags.TAGS
        old_tags = _exif_content_tags(old, ifd)
        new_tags = _exif_content_tags(new, ifd)
        for tag, old_value in old_tags.items():
            name = names.get(tag, f"0x{tag:04x}")
            if tag not in new_tags:
                dropped.append(name)
            elif _exif_values_differ(old_value, new_tags[tag]):
                changed.append(
                    f"{name} {_short_exif_value(old_value)}->" f"{_short_exif_value(new_tags[tag])}"
                )
        added.extend(names.get(tag, f"0x{tag:04x}") for tag in new_tags if tag not in old_tags)
    parts = [
        f"{label} {', '.join(items)}"
        for label, items in (("dropped", dropped), ("set", changed), ("added", added))
        if items
    ]
    if parts:
        return "; ".join(parts)
    return "empty block dropped" if new_exif is None else "nothing changed"


def _exif_values_differ(old_value: Any, new_value: Any) -> bool:
    """
    Compare two tag values as content rather than as python objects.

    Pillow reads some tags of undefined type as text and writes them back as bytes, e.g.
    ExifVersion, and a rational tag can hold a division by zero which reads back as nan.
    Neither is a change worth reporting.
    """
    if isinstance(old_value, str) and isinstance(new_value, bytes):
        return old_value.encode("utf-8", errors="replace") != new_value
    if isinstance(old_value, bytes) and isinstance(new_value, str):
        return old_value != new_value.encode("utf-8", errors="replace")
    old_text, new_text = str(old_value), str(new_value)
    if old_text == new_text:
        return False
    return old_value != new_value


def _exif_content_tags(exif: Image.Exif, ifd: int | None) -> dict[int, Any]:
    """Tags of one exif directory without the pointers to the other directories."""
    tags = dict(exif) if ifd is None else dict(exif.get_ifd(ifd))
    return {tag: value for tag, value in tags.items() if tag not in EXIF_IFD_POINTERS}


def _short_exif_value(value: Any) -> str:
    if isinstance(value, bytes):
        return f"{len(value)}B"
    text = str(value)
    return text if len(text) <= 24 else f"{text[:21]}..."


def clean_exif(exif: bytes) -> bytes | None:
    """
    Rebuild an exif block without the parts that describe the old pixel layout.

    Three things are dropped. The embedded thumbnail, which stops matching once the pixels
    change and is most of the bytes of a phone photo. The orientation tag, so the caller has to
    bake the rotation into the pixels. The pixel size tags, because a decoder reads the size
    from the jpeg or png header and a stale value there is worse than none.

    Everything else is kept, including the interoperability directory, the gps position and the
    tags outside the exif standard that hold vendor specific data.

    Args:
        exif: exif block as read from an image

    Returns:
        the new exif block, or None when no tag is left to write
    """
    old = Image.Exif()
    old.load(exif)
    return _rebuild_exif(old, EXIF_DROP_TAGS)


def _rebuild_exif(
    old: Image.Exif,
    drop_tags: tuple[int, ...] = (),
    *,
    ifd0_drop_tags: frozenset[int] = frozenset(),
) -> bytes | None:
    """Rebuild content directories without thumbnails or the requested tags."""
    new = Image.Exif()
    new.endian = old.endian
    for tag, value in old.items():
        if tag in EXIF_IFD_POINTERS or tag in drop_tags or tag in ifd0_drop_tags:
            continue
        new[tag] = value
    old_exif_ifd = dict(old.get_ifd(ExifTags.IFD.Exif))
    exif_ifd = {
        tag: value
        for tag, value in old_exif_ifd.items()
        if tag not in EXIF_IFD_POINTERS and tag not in drop_tags
    }
    # the interoperability directory hangs off the exif directory, pillow writes a nested dict
    if ExifTags.IFD.Interop in old_exif_ifd:
        interop_ifd = dict(old.get_ifd(ExifTags.IFD.Interop))
        if interop_ifd:
            exif_ifd[ExifTags.IFD.Interop] = interop_ifd
    gps_ifd = dict(old.get_ifd(ExifTags.IFD.GPSInfo))
    if len(new) == 0 and len(exif_ifd) == 0 and len(gps_ifd) == 0:
        return None
    if exif_ifd:
        new[ExifTags.IFD.Exif] = exif_ifd
    if gps_ifd:
        new[ExifTags.IFD.GPSInfo] = gps_ifd
    return new.tobytes()


def check_exif_survived(image_file: PathType, expected_exif: bytes | None) -> list[str]:
    """
    Read a written image back and check its exif against what was meant to be written.

    Call this on the temporary file before renaming it into place, so a format or encoder that
    cannot carry a tag is caught instead of losing metadata silently.

    Args:
        image_file: the file that was just written
        expected_exif: the exif block handed to save_image, None when none was written

    Returns:
        descriptions of tags whose value changed, which the caller should report. Pointer tags
        are skipped, their values are offsets that have to change when a block is rebuilt.

    Raises:
        ValueError: when the file lost a tag or the whole block
    """
    with Image.open(image_file) as opened:
        written_exif = opened.info.get("exif")
    if not expected_exif:
        if written_exif:
            raise ValueError(f"{image_file} has an exif block although none was written")
        return []
    if not written_exif:
        raise ValueError(
            f"{image_file} has no exif block, {len(expected_exif)} bytes were written to it"
        )
    expected, written = Image.Exif(), Image.Exif()
    expected.load(expected_exif)
    written.load(written_exif)
    missing, drifted = [], []
    for label, ifd in EXIF_CONTENT_IFDS:
        expected_tags = _exif_content_tags(expected, ifd)
        written_tags = _exif_content_tags(written, ifd)
        names = ExifTags.GPSTAGS if label == "GPS" else ExifTags.TAGS
        for tag, value in expected_tags.items():
            name = f"{label}/{names.get(tag, f'0x{tag:04x}')}"
            if tag not in written_tags:
                missing.append(name)
            elif _exif_values_differ(value, written_tags[tag]):
                drifted.append(
                    f"{name} {_short_exif_value(value)}->" f"{_short_exif_value(written_tags[tag])}"
                )
    if missing:
        raise ValueError(
            f"{image_file} is missing {len(missing)} exif tags that were written to it: "
            f"{', '.join(missing[:10])}"
        )
    return drifted


def describe_exif_change(old_exif: bytes, new_exif: bytes | None) -> str:
    """
    Summarize what changed between two exif blocks, to log what a clean_exif call did.

    Args:
        old_exif: exif block before
        new_exif: exif block after, None when nothing was left to write

    Returns:
        a compact description of the dropped, changed and added tags
    """
    old, new = Image.Exif(), Image.Exif()
    old.load(old_exif)
    if new_exif is not None:
        new.load(new_exif)
    dropped, changed, added = [], [], []
    old_thumbnail = int(old.get_ifd(ExifTags.IFD.IFD1).get(ExifTags.Base.JpegIFByteCount, 0))
    new_thumbnail = int(new.get_ifd(ExifTags.IFD.IFD1).get(ExifTags.Base.JpegIFByteCount, 0))
    if old_thumbnail > new_thumbnail:
        dropped.append(f"thumbnail {(old_thumbnail - new_thumbnail) / 1024:.0f}KB")
    for label, ifd in EXIF_CONTENT_IFDS:
        names = ExifTags.GPSTAGS if label == "GPS" else ExifTags.TAGS
        old_tags = _exif_content_tags(old, ifd)
        new_tags = _exif_content_tags(new, ifd)
        for tag, old_value in old_tags.items():
            name = names.get(tag, f"0x{tag:04x}")
            if tag not in new_tags:
                dropped.append(name)
            elif _exif_values_differ(old_value, new_tags[tag]):
                changed.append(
                    f"{name} {_short_exif_value(old_value)}->" f"{_short_exif_value(new_tags[tag])}"
                )
        added.extend(names.get(tag, f"0x{tag:04x}") for tag in new_tags if tag not in old_tags)
    parts = [
        f"{label} {', '.join(items)}"
        for label, items in (("dropped", dropped), ("set", changed), ("added", added))
        if items
    ]
    if parts:
        return "; ".join(parts)
    return "empty block dropped" if new_exif is None else "nothing changed"


def _exif_content_tags(exif: Image.Exif, ifd: int | None) -> dict[int, Any]:
    """Tags of one exif directory without the pointers to the other directories."""
    if ifd is None:
        tags = dict(exif)
    else:
        # pillow raises when a sub-directory pointer is absent
        try:
            tags = dict(exif.get_ifd(ifd))
        except KeyError:
            return {}
    return {tag: value for tag, value in tags.items() if tag not in EXIF_IFD_POINTERS}


def _exif_values_differ(old_value: Any, new_value: Any) -> bool:
    """
    Compare two tag values as content rather than as python objects.

    Pillow reads some tags of undefined type as text and writes them back as bytes, e.g.
    ExifVersion, and a rational tag can hold a division by zero which reads back as nan.
    Neither is a change worth reporting.
    """
    if isinstance(old_value, str) and isinstance(new_value, bytes):
        return old_value.encode("utf-8", errors="replace") != new_value
    if isinstance(old_value, bytes) and isinstance(new_value, str):
        return old_value != new_value.encode("utf-8", errors="replace")
    old_text, new_text = str(old_value), str(new_value)
    if old_text == new_text:
        return False
    return old_value != new_value


def _short_exif_value(value: Any) -> str:
    if isinstance(value, bytes):
        return f"{len(value)}B"
    text = str(value)
    return text if len(text) <= 24 else f"{text[:21]}..."


# new orientation tag after rotating the displayed image clockwise, keyed by the old tag and
# the angle. derived from the 8 exif transforms rather than copied, see the unit test that
# re-derives it. tags 2, 4, 5 and 7 mirror as well as rotate, so the mirroring has to follow
ROTATED_ORIENTATION = {
    (1, 90): 6,
    (1, 180): 3,
    (1, 270): 8,
    (2, 90): 7,
    (2, 180): 4,
    (2, 270): 5,
    (3, 90): 8,
    (3, 180): 1,
    (3, 270): 6,
    (4, 90): 5,
    (4, 180): 2,
    (4, 270): 7,
    (5, 90): 2,
    (5, 180): 7,
    (5, 270): 4,
    (6, 90): 3,
    (6, 180): 8,
    (6, 270): 1,
    (7, 90): 4,
    (7, 180): 5,
    (7, 270): 2,
    (8, 90): 1,
    (8, 180): 6,
    (8, 270): 3,
}
# an exif block sits in an APP1 segment that starts with this
EXIF_APP1_PREFIX = b"Exif\x00\x00"


def find_exif_segment(data: bytes) -> tuple[int, int, int] | None:
    """
    Locate the exif APP1 segment of a jpeg.

    Args:
        data: the whole file

    Returns:
        start of the marker, end of the segment and start of the payload, or None when the
        file has no exif segment
    """
    position = 2
    while position < len(data) - 1 and data[position] == 0xFF:
        marker = data[position + 1]
        if marker == 0xD8:  # another start of image, no length field
            position += 2
            continue
        length = int.from_bytes(data[position + 2 : position + 4], "big")
        payload = position + 4
        if marker == 0xE1 and data[payload : payload + len(EXIF_APP1_PREFIX)] == EXIF_APP1_PREFIX:
            return position, position + 2 + length, payload
        if marker == 0xDA:  # start of scan, the compressed data follows
            break
        position += 2 + length
    return None


def rotate_by_orientation_tag(image_file: PathType, out_file: PathType, rotation_cw: int) -> int:
    """
    Turn a jpeg upright by rewriting its exif orientation tag, leaving the pixels alone.

    Only the exif segment is replaced, every other part of the file is copied byte for byte:
    both frames of a multi frame jpeg, the xmp, the gain map marker of an hdr photo, the icc
    profile and any vendor segment. That makes this the only way to rotate a photo whose extra
    data would not survive a rewrite. The embedded exif thumbnail is dropped, as everywhere
    else, because the exif block is rebuilt.

    A viewer that ignores the orientation tag still shows the old rotation, the same trade as
    rotating a video by its container rotation.

    Args:
        image_file: input jpeg
        out_file: where to write, may be the same path
        rotation_cw: 90, 180 or 270 degrees clockwise

    Returns:
        the orientation tag that was written

    Raises:
        ValueError: for another angle, or for a file without an exif segment to rewrite
    """
    image_file, out_file = Path(image_file), Path(out_file)
    if rotation_cw not in (90, 180, 270):
        raise ValueError(f"Cannot rotate by {rotation_cw} degrees, only 90, 180 and 270")
    data = image_file.read_bytes()
    found = find_exif_segment(data)
    if found is None:
        raise ValueError(
            f"{image_file} has no exif segment, so there is no orientation tag to write. "
            f"Rotate the pixels instead."
        )
    seg_start, seg_end, payload = found
    payload_length = int.from_bytes(data[seg_start + 2 : seg_start + 4], "big") - 2
    exif = Image.Exif()
    exif.load(data[payload : payload + payload_length])
    old = exif.get(ExifTags.Base.Orientation, 1)
    if old not in range(1, 9):
        old = 1  # some cameras write 0, which no viewer applies
    new = ROTATED_ORIENTATION[(old, rotation_cw)]
    exif[ExifTags.Base.Orientation] = new
    new_payload = exif.tobytes()
    segment = b"\xff\xe1" + (len(new_payload) + 2).to_bytes(2, "big") + new_payload
    out_file.write_bytes(data[:seg_start] + segment + data[seg_end:])
    return new


def rms_difference(image_a: Image.Image, image_b: Image.Image) -> float:
    """
    Root mean square difference of two images, 0 for identical and 1 for the maximum.

    The bigger image is scaled down to the size of the smaller one first, so a picture and a
    smaller copy of it compare as nearly equal. The worst channel counts.
    """
    size = (min(image_a.width, image_b.width), min(image_a.height, image_b.height))
    first = image_a if image_a.size == size else image_a.resize(size)
    second = image_b if image_b.size == size else image_b.resize(size)
    difference = ImageChops.difference(first.convert("RGB"), second.convert("RGB"))
    return max(ImageStat.Stat(difference).rms) / 255


def colour_saturation(image: Image.Image) -> float:
    """Mean spread between the colour channels, 0 for a grayscale image."""
    red, green, blue = image.convert("RGB").split()
    spread = ImageStat.Stat(ImageChops.difference(red, green)).mean
    spread += ImageStat.Stat(ImageChops.difference(green, blue)).mean
    return sum(spread) / len(spread) / 2 / 255


def extra_frames_are_redundant(image_file: PathType) -> tuple[bool, str]:
    """
    Whether the frames after the first only hold the same picture again.

    A jpeg can carry more than one image, which pillow reports as the MPO format. Cameras use
    the extra frame for a smaller copy of the same shot, which is redundant. Phones use it for
    auxiliary data, an iphone stores a grayscale hdr gain map there, and dropping that loses
    the hdr rendering.

    Args:
        image_file: input file

    Returns:
        whether every extra frame is a copy that can be dropped, and a description of what was
        measured for the log. True with an empty description for a single frame image.
    """
    with Image.open(image_file) as opened:
        n_frames = getattr(opened, "n_frames", 1)
        if n_frames < 2:
            return True, ""
        opened.seek(0)
        first = opened.convert("RGB")
        first_saturation = colour_saturation(first)
        reports = []
        redundant = True
        for index in range(1, n_frames):
            try:
                opened.seek(index)
                extra = opened.convert("RGB")
            except (OSError, SyntaxError, ValueError) as e:
                reports.append(f"frame {index} undecodable ({format_exception(e)})")
                redundant = False
                continue
            difference = rms_difference(first, extra)
            lost_colour = first_saturation - colour_saturation(extra)
            reports.append(
                f"frame {index} {extra.width}x{extra.height} rms={difference:.3f} "
                f"desaturated={lost_colour:.3f}"
            )
            if difference >= MULTI_FRAME_RMS_TOLERANCE:
                redundant = False
            if lost_colour > MULTI_FRAME_SATURATION_TOLERANCE:
                redundant = False
    return redundant, ", ".join(reports)


def extra_frame_differences(image_file: PathType) -> list[float]:
    """
    Compare every frame after the first of a multi frame image against the first frame.

    A jpeg can hold more than one picture, which pillow reports as the MPO format. Cameras use
    the extra frame for a smaller copy of the same shot, phones for auxiliary data such as an
    hdr gain map. Only the first kind can be dropped, see MULTI_FRAME_RMS_TOLERANCE.

    Args:
        image_file: input file

    Returns:
        the rms difference per extra frame, empty for a single frame image. A frame that cannot
        be decoded counts as infinitely different, so a caller that drops frames leaves the
        file alone instead of throwing away something it could not check.
    """
    with Image.open(image_file) as opened:
        n_frames = getattr(opened, "n_frames", 1)
        if n_frames < 2:
            return []
        opened.seek(0)
        first = opened.convert("RGB")
        differences = []
        for index in range(1, n_frames):
            try:
                opened.seek(index)
                differences.append(rms_difference(first, opened.convert("RGB")))
            except (OSError, SyntaxError, ValueError):
                differences.append(math.inf)
    return differences


def load_image(
    image_file: PathType, fix_rotation: bool = True
) -> tuple[Image.Image, ImageMetadata]:
    """
    Open an image, apply the EXIF orientation and read its metadata.

    Palette images with transparency are converted to RGBA so that the transparency survives
    independently of the palette. TIFF metadata is rebuilt without pixel-storage fields.

    Args:
        image_file: input file
        fix_rotation: rotate the pixels as described by the EXIF orientation tag and remove the tag

    Returns:
        tuple of image and its metadata
    """
    with Image.open(image_file) as opened_image:
        source_format = opened_image.format
        if source_format == "TIFF":
            # TIFF stores EXIF in file-backed directories rather than info["exif"]. Read
            # nested directories before load() closes the file, including nested Interop.
            tiff_exif = opened_image.getexif()
            exif_ifd = tiff_exif.get_ifd(ExifTags.IFD.Exif)
            tiff_exif.get_ifd(ExifTags.IFD.GPSInfo)
            if ExifTags.IFD.Interop in exif_ifd:
                tiff_exif.get_ifd(ExifTags.IFD.Interop)
        opened_image.load()
        if source_format == "TIFF":
            # Pillow's TIFF loader already applies orientation during load(). Use the
            # updated EXIF so exif_transpose below cannot rotate the pixels a second time.
            tiff_bytes = _rebuild_exif(tiff_exif, ifd0_drop_tags=TIFF_STORAGE_TAGS)
            if tiff_bytes is not None:
                opened_image.info["exif"] = tiff_bytes
        png_text = dict(getattr(opened_image, "text", {}))
        if fix_rotation:
            # exif_transpose rotates the pixels and removes the orientation tag from the exif of
            # the returned image, so the exif has to be read after this call, not before.
            image = ImageOps.exif_transpose(opened_image)
        else:
            image = opened_image.copy()
    if image.mode == "P" and image.info.get("transparency") is not None:
        image = image.convert("RGBA")
    metadata = ImageMetadata(
        exif=image.info.get("exif"),
        icc_profile=image.info.get("icc_profile"),
        xmp=image.info.get("xmp"),
        dpi=image.info.get("dpi"),
        png_text=png_text,
        format=source_format,
    )
    return image, metadata


def save_image(
    image: Image.Image,
    image_file: PathType,
    metadata: ImageMetadata | None = None,
    quality: int | str = DEFAULT_QUALITY,
    **save_kwargs: Any,
) -> None:
    """
    Save an image, writing back all metadata the output format supports.

    The format is taken from the file extension. The mode is converted if the format cannot
    store it, e.g. RGBA to RGB for jpg. Use resolve_extension() to avoid losing an alpha channel.
    """
    image_file = Path(image_file)
    image_format = get_format(image_file)
    image = convert_mode_for_format(image, image_format)
    kwargs = {} if metadata is None else metadata.get_save_kwargs(image_format)
    if image_format in FORMAT_QUALITY_KEYS:
        kwargs["quality"] = quality
    kwargs.update(save_kwargs)
    image_file.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_file.as_posix(), format=image_format, **kwargs)


def resolve_extension(image: Image.Image, extension: str) -> str:
    """
    Return the given extension, or .png if the image has transparency the format cannot store.
    """
    extension = extension if extension.startswith(".") else f".{extension}"
    if not has_transparency(image):
        return extension
    if EXTENSION_TO_FORMAT[extension.lower()] in FORMATS_WITH_ALPHA:
        return extension
    return ALPHA_FALLBACK_EXTENSION


def has_transparency(image: Image.Image) -> bool:
    """Check for an alpha channel that is not fully opaque everywhere."""
    if image.mode == "P":
        return image.info.get("transparency") is not None
    if "A" not in image.getbands():
        return False
    return image.getchannel("A").getextrema()[0] < 255


def scale_to_smaller_side(
    image: Image.Image, smaller_side: int, method: str = AUTO, **method_kwargs: str
) -> Image.Image:
    width, height = image.size
    if height < width:
        target_height = smaller_side
        target_width = round(width * smaller_side / height)
    else:
        target_width = smaller_side
        target_height = round(height * smaller_side / width)
    return scale_image(image, target_width, target_height, method, **method_kwargs)


def scale_to_bigger_side(
    image: Image.Image, bigger_side: int, method: str = AUTO, **method_kwargs: str
) -> Image.Image:
    width, height = image.size
    if height > width:
        target_height = bigger_side
        target_width = round(width * bigger_side / height)
    else:
        target_width = bigger_side
        target_height = round(height * bigger_side / width)
    return scale_image(image, target_width, target_height, method, **method_kwargs)


def scale_image(
    image: Image.Image,
    target_width: int,
    target_height: int,
    method: str = AUTO,
    upsampling_method: str = DEFAULT_UPSAMPLING_METHOD,
    downsampling_method: str = DEFAULT_DOWNSAMPLING_METHOD,
) -> Image.Image:
    """
    Resize an image. With method=auto, downsampling uses area and upsampling bicubic.
    """
    width, height = image.size
    if (target_width, target_height) == (width, height):
        return image
    if method == AUTO:
        is_downsampling = target_width <= width and target_height <= height
        method = downsampling_method if is_downsampling else upsampling_method
    if method not in SAMPLING_MAP:
        raise KeyError(f"Unknown sampling method {method}, available: {list(SAMPLING_MAP.keys())}")
    return image.resize((target_width, target_height), SAMPLING_MAP[method])


def get_format(image_file: PathType) -> str:
    extension = Path(image_file).suffix.lower()
    if extension not in EXTENSION_TO_FORMAT:
        raise ValueError(
            f"Unsupported extension {extension} for {image_file}, "
            f"available: {sorted(EXTENSION_TO_FORMAT.keys())}"
        )
    return EXTENSION_TO_FORMAT[extension]


def convert_mode_for_format(image: Image.Image, image_format: str) -> Image.Image:
    """Convert the image mode if the output format cannot store it."""
    if image.mode in FORMAT_MODES[image_format]:
        return image
    if has_transparency(image) and image_format in FORMATS_WITH_ALPHA:
        return image.convert("RGBA")
    return image.convert("RGB")
