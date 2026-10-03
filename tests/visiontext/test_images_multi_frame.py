from __future__ import annotations  # py 3.9 support

import pytest
from PIL import Image

from visiontext import images_v2
from visiontext.images_v2 import (
    MULTI_FRAME_RMS_TOLERANCE,
    MULTI_FRAME_SATURATION_TOLERANCE,
    colour_saturation,
    extra_frames_are_redundant,
    rms_difference,
)

SIZE = (64, 48)


def _photo(size=SIZE, shift=0):
    """A colourful gradient, so both saturation and rms are meaningful."""
    image = Image.new("RGB", size)
    for x in range(size[0]):
        for y in range(size[1]):
            image.putpixel((x, y), ((x * 4 + shift) % 256, (y * 5) % 256, 128))
    return image


class FakeMultiFrame:
    """
    Stands in for a multi frame jpeg, which pillow can read but not write.

    A frame given as None raises on decode, like the truncated extra frames that do occur.
    """

    def __init__(self, frames: list[Image.Image | None]):
        self.frames = frames
        self.index = 0

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def seek(self, index: int) -> None:
        if index >= len(self.frames):
            raise EOFError("no more images in file")
        self.index = index

    def convert(self, mode: str) -> Image.Image:
        frame = self.frames[self.index]
        if frame is None:
            raise OSError("broken data stream when reading image file")
        return frame.convert(mode)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def fake_frames(monkeypatch):
    """Make extra_frames_are_redundant read the frames handed to the returned setter."""

    def use(frames):
        monkeypatch.setattr(images_v2.Image, "open", lambda _: FakeMultiFrame(frames))

    return use


def test_rms_difference_of_identical_images_is_zero():
    assert rms_difference(_photo(), _photo()) == 0.0


def test_rms_difference_ignores_a_size_change():
    """A smaller copy of the same picture must compare as nearly equal."""
    assert rms_difference(_photo(), _photo().resize((32, 24))) < 0.1


def test_rms_difference_of_unrelated_images_is_large():
    assert rms_difference(Image.new("RGB", SIZE, "black"), Image.new("RGB", SIZE, "white")) == 1.0


def test_colour_saturation():
    assert colour_saturation(Image.new("RGB", SIZE, (80, 80, 80))) == 0.0
    assert colour_saturation(_photo()) > 0.05


def test_a_single_frame_image_is_always_fine(tmp_path):
    image_file = tmp_path / "one.jpg"
    _photo().save(image_file, format="JPEG")
    assert extra_frames_are_redundant(image_file) == (True, "")


def test_a_smaller_copy_as_second_frame_is_redundant(fake_frames, tmp_path):
    main = _photo()
    fake_frames([main, main.resize((32, 24))])
    redundant, report = extra_frames_are_redundant(tmp_path / "copy.jpg")
    assert redundant
    assert "frame 1 32x24" in report


def test_a_grayscale_second_frame_is_kept(fake_frames, tmp_path):
    """An hdr gain map is the same picture without colour, and dropping it loses the hdr."""
    main = _photo()
    fake_frames([main, main.convert("L").convert("RGB")])
    redundant, report = extra_frames_are_redundant(tmp_path / "gainmap.jpg")
    assert not redundant
    assert "desaturated" in report


def test_a_different_picture_as_second_frame_is_kept(fake_frames, tmp_path):
    fake_frames([_photo(), Image.new("RGB", SIZE, (200, 30, 30))])
    redundant, _ = extra_frames_are_redundant(tmp_path / "other.jpg")
    assert not redundant


def test_an_undecodable_frame_keeps_the_file(fake_frames, tmp_path):
    """What cannot be checked is not thrown away."""
    fake_frames([_photo(), None])
    redundant, report = extra_frames_are_redundant(tmp_path / "broken.jpg")
    assert not redundant
    assert "undecodable" in report


def test_one_bad_frame_among_several_keeps_the_file(fake_frames, tmp_path):
    main = _photo()
    fake_frames([main, main.resize((32, 24)), Image.new("RGB", SIZE, "white")])
    redundant, report = extra_frames_are_redundant(tmp_path / "mixed.jpg")
    assert not redundant
    # both extra frames are reported, so the log says which one objected
    assert "frame 1" in report and "frame 2" in report


def test_the_tolerances_are_ordered_sensibly():
    assert 0 < MULTI_FRAME_SATURATION_TOLERANCE < MULTI_FRAME_RMS_TOLERANCE < 1


def test_rotated_orientation_table_matches_the_exif_transforms():
    """Re-derive the table from the 8 transforms, so a typo in it cannot survive."""
    import numpy as np

    from visiontext.images_v2 import ROTATED_ORIENTATION

    stored = Image.fromarray(np.arange(6 * 4 * 3, dtype=np.uint8).reshape(4, 6, 3))
    methods = {
        1: None,
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }
    clockwise = {
        90: Image.Transpose.ROTATE_270,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_90,
    }

    def displayed(tag):
        method = methods[tag]
        return stored if method is None else stored.transpose(method)

    assert len(ROTATED_ORIENTATION) == 24
    for tag in range(1, 9):
        for degrees, turn in clockwise.items():
            want = np.asarray(displayed(tag).transpose(turn))
            expected = ROTATED_ORIENTATION[(tag, degrees)]
            assert np.array_equal(np.asarray(displayed(expected)), want), (tag, degrees)


@pytest.mark.parametrize("rotation_cw", [90, 180, 270])
def test_rotate_by_orientation_tag_keeps_the_pixels(tmp_path, rotation_cw):
    import numpy as np

    from visiontext.images_v2 import rotate_by_orientation_tag

    source = tmp_path / "photo.jpg"
    exif = Image.Exif()
    exif[271] = "TestCam"
    _photo().save(source, format="JPEG", quality=95, exif=exif.tobytes())
    before = source.read_bytes()

    out = tmp_path / "rotated.jpg"
    tag = rotate_by_orientation_tag(source, out, rotation_cw)
    assert tag == {90: 6, 180: 3, 270: 8}[rotation_cw]

    # the compressed image data is untouched, only the exif segment differs
    opened = Image.open(out)
    assert opened.getexif()[274] == tag
    assert np.array_equal(
        np.asarray(opened.convert("RGB")), np.asarray(Image.open(source).convert("RGB"))
    )
    # the tag that was already there is kept alongside the new one
    assert opened.getexif()[271] == "TestCam"
    assert source.read_bytes() == before  # the source was not modified


def test_rotate_by_orientation_tag_composes_with_an_existing_tag(tmp_path):
    from visiontext.images_v2 import rotate_by_orientation_tag

    source = tmp_path / "photo.jpg"
    exif = Image.Exif()
    exif[274] = 6  # already turned 90 clockwise on display
    _photo().save(source, format="JPEG", exif=exif.tobytes())
    # another 90 clockwise makes 180 in total
    assert rotate_by_orientation_tag(source, tmp_path / "a.jpg", 90) == 3
    # and 270 more brings it back to upright
    assert rotate_by_orientation_tag(source, tmp_path / "b.jpg", 270) == 1


def test_rotate_by_orientation_tag_refuses_a_file_without_exif(tmp_path):
    from visiontext.images_v2 import rotate_by_orientation_tag

    source = tmp_path / "bare.jpg"
    _photo().save(source, format="JPEG")
    with pytest.raises(ValueError, match="no exif segment"):
        rotate_by_orientation_tag(source, tmp_path / "out.jpg", 90)


def test_rotate_by_orientation_tag_refuses_another_angle(tmp_path):
    from visiontext.images_v2 import rotate_by_orientation_tag

    source = tmp_path / "photo.jpg"
    exif = Image.Exif()
    exif[274] = 1
    _photo().save(source, format="JPEG", exif=exif.tobytes())
    for rotation_cw in (0, 45, 360):
        with pytest.raises(ValueError, match="Cannot rotate"):
            rotate_by_orientation_tag(source, tmp_path / "out.jpg", rotation_cw)
