import struct
from io import BytesIO

import pytest
from PIL import ExifTags, Image

from visiontext.images_v2 import (
    ImageMetadata,
    check_exif_survived,
    clean_exif,
    describe_exif_change,
    load_image,
    save_image,
)

VENDOR_TAG = 0x9999  # not part of the exif standard, phones use it for their own data
SIZE = (40, 30)


def _make_exif() -> bytes:
    """Exif of a 30x40 photo with a real tag, a thumbnail, a maker note and a vendor tag."""
    exif = Image.Exif()
    exif[ExifTags.Base.Make] = "TestCam"
    exif[ExifTags.Base.Model] = "TestModel"
    exif[ExifTags.Base.Orientation] = 6
    exif[ExifTags.Base.ImageWidth] = 30
    exif[ExifTags.Base.ImageLength] = 40
    exif[VENDOR_TAG] = '{"scene":"night"}'
    exif[ExifTags.IFD.Exif] = {
        ExifTags.Base.DateTimeOriginal: "2026:08:10 15:42:19",
        ExifTags.Base.ISOSpeedRatings: 80,
        ExifTags.Base.ExifImageWidth: 30,
        ExifTags.Base.ExifImageHeight: 40,
        ExifTags.Base.MakerNote: b"\x00" * 512,
        VENDOR_TAG: '{"scene":"night"}',
    }
    exif[ExifTags.IFD.GPSInfo] = {ExifTags.GPS.GPSAltitude: 100}
    return exif.tobytes()


def test_clean_exif_keeps_the_real_tags():
    cleaned = Image.Exif()
    cleaned.load(clean_exif(_make_exif()))
    assert cleaned[ExifTags.Base.Make] == "TestCam"
    assert cleaned[ExifTags.Base.Model] == "TestModel"
    exif_ifd = cleaned.get_ifd(ExifTags.IFD.Exif)
    assert exif_ifd[ExifTags.Base.DateTimeOriginal] == "2026:08:10 15:42:19"
    assert exif_ifd[ExifTags.Base.ISOSpeedRatings] == 80
    assert cleaned.get_ifd(ExifTags.IFD.GPSInfo)[ExifTags.GPS.GPSAltitude] == 100


def test_clean_exif_drops_the_thumbnail_and_the_orientation():
    original = _make_exif()
    cleaned_bytes = clean_exif(original)
    assert len(cleaned_bytes) < len(original)
    cleaned = Image.Exif()
    cleaned.load(cleaned_bytes)
    assert cleaned.get_ifd(ExifTags.IFD.IFD1) == {}
    # the rotation is expected to be baked into the pixels by the caller
    assert ExifTags.Base.Orientation not in cleaned


def test_clean_exif_keeps_the_vendor_tags():
    """Tags outside the exif standard are kept, only a tool that reads them knows their use."""
    cleaned = Image.Exif()
    cleaned.load(clean_exif(_make_exif()))
    exif_ifd = cleaned.get_ifd(ExifTags.IFD.Exif)
    assert cleaned[VENDOR_TAG] == '{"scene":"night"}'
    assert exif_ifd[VENDOR_TAG] == '{"scene":"night"}'
    assert exif_ifd[ExifTags.Base.MakerNote] == b"\x00" * 512


def test_clean_exif_drops_the_size_tags():
    """The size a decoder uses comes from the jpeg or png header, so these are redundant."""
    cleaned = Image.Exif()
    cleaned.load(clean_exif(_make_exif()))
    assert ExifTags.Base.ImageWidth not in cleaned
    assert ExifTags.Base.ImageLength not in cleaned
    exif_ifd = cleaned.get_ifd(ExifTags.IFD.Exif)
    assert ExifTags.Base.ExifImageWidth not in exif_ifd
    assert ExifTags.Base.ExifImageHeight not in exif_ifd


def test_clean_exif_without_any_tag_returns_nothing():
    """An exif block with nothing worth keeping is dropped instead of written back empty."""
    assert clean_exif(Image.Exif().tobytes()) is None


def test_clean_exif_does_not_add_size_tags():
    """Most phone photos carry no size tags, inventing them would add metadata that was absent."""
    exif = Image.Exif()
    exif[ExifTags.Base.Make] = "TestCam"
    cleaned = Image.Exif()
    cleaned.load(clean_exif(exif.tobytes()))
    assert cleaned[ExifTags.Base.Make] == "TestCam"
    assert ExifTags.Base.ImageWidth not in cleaned
    assert ExifTags.Base.ImageLength not in cleaned


def test_with_clean_exif_keeps_the_other_metadata():
    metadata = ImageMetadata(exif=_make_exif(), icc_profile=b"icc", xmp=b"<xmp/>", dpi=(72, 72))
    cleaned = metadata.with_clean_exif()
    assert cleaned.icc_profile == metadata.icc_profile
    assert cleaned.xmp == metadata.xmp
    assert cleaned.dpi == metadata.dpi
    assert len(cleaned.exif) < len(metadata.exif)
    # the input is not modified
    assert metadata.exif == _make_exif()


def test_with_clean_exif_without_exif():
    metadata = ImageMetadata(icc_profile=b"icc")
    cleaned = metadata.with_clean_exif()
    assert cleaned.exif is None
    assert cleaned.icc_profile == b"icc"


@pytest.mark.parametrize("extension", [".jpg", ".png", ".webp"])
def test_clean_exif_survives_a_save_and_load(tmp_path, extension):
    image = Image.new("RGB", SIZE, "red")
    metadata = ImageMetadata(exif=_make_exif()).with_clean_exif()
    image_file = tmp_path / f"photo{extension}"
    save_image(image, image_file, metadata)
    loaded, loaded_metadata = load_image(image_file)
    assert loaded.size == SIZE
    reread = Image.Exif()
    reread.load(loaded_metadata.exif)
    assert reread[ExifTags.Base.Make] == "TestCam"
    assert reread.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.DateTimeOriginal] == (
        "2026:08:10 15:42:19"
    )


def test_describe_exif_change_lists_what_was_dropped_and_set():
    original = _make_exif()
    description = describe_exif_change(original, clean_exif(original))
    assert "Orientation" in description
    assert "ImageWidth" in description
    assert "ImageLength" in description
    # the fixture has no thumbnail, so nothing about one should be claimed
    assert "thumbnail" not in description
    # the vendor tags are kept, so they must not show up as dropped
    assert "0x9999" not in description
    assert "MakerNote" not in description


def test_describe_exif_change_of_a_dropped_block():
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = 6
    original = exif.tobytes()
    assert clean_exif(original) is None
    assert describe_exif_change(original, None) == "dropped Orientation"


def test_describe_exif_change_of_an_unchanged_block():
    original = _make_exif()
    assert describe_exif_change(original, original) == "nothing changed"


def test_describe_exif_change_ignores_a_bytes_or_text_round_trip():
    """Pillow reads ExifVersion as text and writes it back as bytes, which is not a change."""
    exif = Image.Exif()
    exif[ExifTags.Base.ImageWidth] = SIZE[0]
    exif[ExifTags.Base.ImageLength] = SIZE[1]
    exif[ExifTags.IFD.Exif] = {ExifTags.Base.ExifVersion: b"0220"}
    original = exif.tobytes()
    description = describe_exif_change(original, clean_exif(original))
    # the size tags go, but ExifVersion must not be reported as changed
    assert "ExifVersion" not in description
    assert description == "dropped ImageWidth, ImageLength"


def test_clean_exif_keeps_the_interop_directory():
    exif = Image.Exif()
    exif[ExifTags.Base.Make] = "TestCam"
    exif[ExifTags.IFD.Exif] = {
        ExifTags.Base.DateTimeOriginal: "2026:08:10 15:42:19",
        ExifTags.IFD.Interop: {1: "R98", 2: b"0100"},
    }
    cleaned = Image.Exif()
    cleaned.load(clean_exif(exif.tobytes()))
    assert dict(cleaned.get_ifd(ExifTags.IFD.Interop)) == {1: "R98", 2: b"0100"}


def test_check_exif_survived_accepts_a_faithful_write(tmp_path):
    image = Image.new("RGB", SIZE, "red")
    metadata = ImageMetadata(exif=_make_exif()).with_clean_exif()
    image_file = tmp_path / "photo.jpg"
    save_image(image, image_file, metadata)
    # the vendor tags and the maker note all have to come back out of the file
    assert check_exif_survived(image_file, metadata.exif) == []


def test_check_exif_survived_raises_when_the_block_is_gone(tmp_path):
    image = Image.new("RGB", SIZE, "red")
    image_file = tmp_path / "photo.jpg"
    save_image(image, image_file, None)
    with pytest.raises(ValueError, match="no exif block"):
        check_exif_survived(image_file, _make_exif())


def test_check_exif_survived_raises_when_a_tag_is_missing(tmp_path):
    """A bmp cannot carry exif at all, which must be caught rather than ignored."""
    image = Image.new("RGB", SIZE, "red")
    image_file = tmp_path / "photo.bmp"
    save_image(image, image_file, ImageMetadata(exif=_make_exif()))
    with pytest.raises(ValueError, match="no exif block"):
        check_exif_survived(image_file, _make_exif())


def test_check_exif_survived_without_any_exif(tmp_path):
    image = Image.new("RGB", SIZE, "red")
    image_file = tmp_path / "photo.jpg"
    save_image(image, image_file, None)
    assert check_exif_survived(image_file, None) == []


@pytest.mark.parametrize("extension", [".jpg", ".png"])
@pytest.mark.parametrize("clean", [False, True])
def test_upright_orientation_is_removed_only_when_cleaning(tmp_path, extension, clean):
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = 1
    source = tmp_path / f"source{extension}"
    Image.new("RGB", SIZE, "red").save(source, exif=exif)
    image, metadata = load_image(source)
    if clean:
        metadata = metadata.with_clean_exif()
        assert metadata.exif is None
    target = tmp_path / f"target{extension}"
    save_image(image, target, metadata)
    assert check_exif_survived(target, metadata.exif) == []
    with Image.open(target) as written:
        assert written.getexif().get(ExifTags.Base.Orientation) == (None if clean else 1)


@pytest.mark.parametrize("orientation", [1, 6])
def test_default_rotation_can_drop_the_thumbnail(tmp_path, orientation):
    thumbnail = BytesIO()
    Image.new("RGB", (4, 3), "red").save(thumbnail, format="JPEG")
    payload = thumbnail.getvalue()
    # Little-endian TIFF: IFD0 at 8, IFD1 at 38, JPEG thumbnail at 68.
    exif = b"Exif\0\0" + struct.pack("<2sHI", b"II", 42, 8)
    exif += struct.pack("<H", 2)
    exif += struct.pack("<HHI4s", 274, 3, 1, struct.pack("<H", orientation) + b"\0\0")
    exif += struct.pack("<HHI4s", 271, 2, 4, b"Cam\0") + struct.pack("<I", 38)
    exif += struct.pack("<H", 2)
    exif += struct.pack("<HHII", 513, 4, 1, 68)
    exif += struct.pack("<HHII", 514, 4, 1, len(payload)) + struct.pack("<I", 0) + payload
    source = tmp_path / "source.jpg"
    Image.new("RGB", SIZE, "red").save(source, exif=exif)
    with Image.open(source) as original:
        assert original.getexif().get_ifd(ExifTags.IFD.IFD1)[514] == len(payload)
    image, metadata = load_image(source)
    target = tmp_path / "target.jpg"
    save_image(image, target, metadata)
    with Image.open(target) as written:
        assert written.getexif()[ExifTags.Base.Make] == "Cam"
        if orientation == 1:
            assert written.info["exif"] == exif
        else:
            assert written.size == SIZE[::-1]
            assert written.getexif().get_ifd(ExifTags.IFD.IFD1) == {}
            assert payload not in written.info["exif"]


@pytest.mark.parametrize("extension", [".jpg", ".png"])
@pytest.mark.parametrize(
    "orientation_field", ['tiff:Orientation="6"', "<tiff:Orientation>6</tiff:Orientation>"]
)
def test_transposing_removes_only_orientation_from_xmp(tmp_path, extension, orientation_field):
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = 6
    xmp = f'<rdf:Description {orientation_field} test="preserve"/>'.encode()
    if orientation_field.startswith("<"):
        xmp = f'<rdf:Description test="preserve">{orientation_field}</rdf:Description>'.encode()
    source = tmp_path / f"source{extension}"
    save_image(Image.new("RGB", SIZE), source, ImageMetadata(exif=exif.tobytes(), xmp=xmp))
    image, metadata = load_image(source)
    target = tmp_path / f"target{extension}"
    save_image(image, target, metadata.with_clean_exif())
    _, written_metadata = load_image(target)
    assert written_metadata.xmp == xmp.replace(orientation_field.encode(), b"")


@pytest.mark.parametrize("orientation", [1, 6, 8])
@pytest.mark.parametrize("clean", [False, True])
def test_tiff_to_jpeg_preserves_descriptive_metadata(tmp_path, orientation, clean):
    exif = Image.Exif()
    exif.load(_make_exif())
    exif[ExifTags.Base.Orientation] = orientation
    exif[ExifTags.Base.ImageWidth], exif[ExifTags.Base.ImageLength] = SIZE
    exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.IFD.Interop] = {1: "R98", 2: b"0100"}
    xmp = f'<rdf:Description tiff:Orientation="{orientation}" test="preserve"/>'.encode()
    exif[ExifTags.Base.XMLPacket] = xmp
    source = tmp_path / "source.tiff"
    pixels = Image.new("RGB", SIZE, "red")
    pixels.paste("blue", (0, 0, 10, 10))
    pixels.save(source, exif=exif, icc_profile=b"test-icc")

    image, metadata = load_image(source)
    expected_pixels = pixels
    if orientation != 1:
        method = Image.Transpose.ROTATE_270 if orientation == 6 else Image.Transpose.ROTATE_90
        expected_pixels = pixels.transpose(method)
    assert image.size == expected_pixels.size
    assert image.tobytes() == expected_pixels.tobytes()
    assert metadata.exif is not None
    assert metadata.exif[6:8] == source.read_bytes()[:2]
    if clean:
        metadata = metadata.with_clean_exif()
    target = tmp_path / "target.jpg"
    save_image(image, target, metadata)
    assert check_exif_survived(target, metadata.exif) == []
    with Image.open(target) as written:
        # getexif() can synthesize Orientation from the preserved XMP on upright images.
        tags = Image.Exif()
        tags.load(written.info["exif"])
        assert tags[ExifTags.Base.Make] == "TestCam"
        assert tags[ExifTags.Base.Model] == "TestModel"
        assert tags[VENDOR_TAG] == '{"scene":"night"}'
        exif_ifd = tags.get_ifd(ExifTags.IFD.Exif)
        assert exif_ifd[ExifTags.Base.DateTimeOriginal] == "2026:08:10 15:42:19"
        assert exif_ifd[ExifTags.Base.MakerNote] == b"\x00" * 512
        assert exif_ifd[VENDOR_TAG] == '{"scene":"night"}'
        assert tags.get_ifd(ExifTags.IFD.GPSInfo)[ExifTags.GPS.GPSAltitude] == 100
        assert tags.get_ifd(ExifTags.IFD.Interop) == {1: "R98", 2: b"0100"}
        assert tags.get(ExifTags.Base.Orientation) == (
            1 if orientation == 1 and not clean else None
        )
        assert ExifTags.Base.StripOffsets not in tags
        assert ExifTags.Base.StripByteCounts not in tags
        assert ExifTags.Base.Compression not in tags
        assert ExifTags.Base.PhotometricInterpretation not in tags
        assert ExifTags.Base.XMLPacket not in tags
        assert ExifTags.Base.InterColorProfile not in tags
        assert written.info["icc_profile"] == b"test-icc"
        expected_xmp = (
            xmp
            if orientation == 1
            else xmp.replace(f'tiff:Orientation="{orientation}"'.encode(), b"")
        )
        assert written.info["xmp"] == expected_xmp


@pytest.mark.parametrize("orientation", [1, 6])
def test_compressed_tiff_to_jpeg_preserves_metadata(tmp_path, orientation):
    # Pillow's libtiff writer cannot write the nested dictionaries in the richer fixture.
    exif = Image.Exif()
    exif[ExifTags.Base.Make] = "TestCam"
    exif[ExifTags.Base.Orientation] = orientation
    source = tmp_path / "source.tiff"
    Image.new("RGB", SIZE, "red").save(source, exif=exif, compression="tiff_lzw", dpi=(96, 96))
    image, metadata = load_image(source)
    target = tmp_path / "target.jpg"
    save_image(image, target, metadata)
    with Image.open(target) as written:
        assert written.size == (SIZE if orientation == 1 else SIZE[::-1])
        assert written.getexif()[ExifTags.Base.Make] == "TestCam"
        assert ExifTags.Base.StripOffsets not in written.getexif()
        assert written.info["dpi"] == (96, 96)


@pytest.mark.parametrize("metadata_flags", [[], ["-E"], ["--destroy_metadata"]])
def test_downscale_cli_converts_tiff_with_metadata(tmp_path, monkeypatch, metadata_flags):
    # the cli is not part of the published package, there this test is skipped
    pytest.importorskip("gutil")
    from gutil.cli.files.downscale_images import main

    exif = Image.Exif()
    exif[ExifTags.Base.Make] = "TestCam"
    exif[ExifTags.Base.Orientation] = 6
    source = tmp_path / "photo.tiff"
    Image.new("RGB", SIZE, "red").save(source, exif=exif)
    source_mtime = source.stat().st_mtime
    monkeypatch.setattr(
        "sys.argv", ["downscale_images", str(tmp_path), "-O", "-b", "20", "-w", *metadata_flags]
    )
    main()
    target = tmp_path / "photo.jpg"
    assert not source.exists()
    assert target.stat().st_mtime == pytest.approx(source_mtime, rel=0, abs=1e-6)
    with Image.open(target) as written:
        assert written.size == (15, 20)
        tags = written.getexif()
        if metadata_flags == ["--destroy_metadata"]:
            assert "exif" not in written.info
        else:
            assert tags[ExifTags.Base.Make] == "TestCam"
            assert ExifTags.Base.Orientation not in tags
            assert ExifTags.Base.StripOffsets not in tags
            assert (ExifTags.Base.ImageWidth in tags) == (metadata_flags != ["-E"])
