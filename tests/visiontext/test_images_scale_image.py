# todo tests supports both string and the default pil stuff

import pytest
from PIL import Image

from visiontext.images import PILImageScaler, open_image_scaled, SamplingMapPIL


# Create a function to generate a small sample image and save it to a temporary path
@pytest.fixture
def sample_image(tmp_path):
    # Create an image with size 200x100 and fill it with red
    img = Image.new("RGB", (200, 100), color="red")

    # Save the image to the temporary path
    img_path = tmp_path / "sample_image.jpg"
    img.save(img_path)

    return img_path


@pytest.fixture
def image_scaler():
    return PILImageScaler(return_pillow=True)


def test_resampling_methods_pillow_enum_value(sample_image, image_scaler):
    # here method is passed as int
    print(f"Use enum value:")
    img = Image.open(sample_image)
    for method_enum in Image.Resampling:
        method = method_enum.value
        print(f"    Method {method} type {type(method)}")
        result = image_scaler.scale_image(img, 50, 100, method=method)
        assert result.size == (100, 50)


def test_resampling_methods_pillow_enum_name(sample_image, image_scaler):
    print(f"Use enum name:")
    img = Image.open(sample_image)
    for method_enum in Image.Resampling:
        method = method_enum.name
        print(f"    Method {method} type {type(method)}")
        result = image_scaler.scale_image(img, 50, 100, method=method)
        assert result.size == (100, 50)


def test_resampling_methods_pillow_enum(sample_image, image_scaler):
    print(f"Use enum directly:")
    img = Image.open(sample_image)
    for method_enum in Image.Resampling:
        method = method_enum
        print(f"    Method {method} type {type(method)}")
        result = image_scaler.scale_image(img, 50, 100, method=method)
        assert result.size == (100, 50)
    result = image_scaler.scale_image(img, 50, 100, method=Image.Resampling.BICUBIC)
    assert result.size == (100, 50)


def test_resampling_methods_dict(sample_image, image_scaler):
    img = Image.open(sample_image)
    for method in SamplingMapPIL.keys():
        result = image_scaler.scale_image(img, 50, 100, method=method)
        assert result.size == (100, 50)


def test_scale_image_smaller_side(sample_image, image_scaler):
    img = Image.open(sample_image)
    result = image_scaler.scale_image_smaller_side(img, 50)
    assert result.size == (100, 50)


def test_scale_image_bigger_side(sample_image, image_scaler):
    img = Image.open(sample_image)
    result = image_scaler.scale_image_bigger_side(img, 250)
    assert result.size == (250, 125)


def test_open_image_scaled_smaller_side(sample_image):
    result = open_image_scaled(sample_image, smaller_side=50)
    assert result.size == (100, 50)


def test_open_image_scaled_bigger_side(sample_image):
    result = open_image_scaled(sample_image, bigger_side=250)
    assert result.size == (250, 125)


def test_open_image_scaled_error_both_sides_given(sample_image):
    with pytest.raises(
        ValueError, match=r".*Only one of smaller_side.*and bigger_side.*can be given"
    ):
        open_image_scaled(sample_image, smaller_side=50, bigger_side=250)


def test_open_image_scaled_error_no_side_given(sample_image):
    with pytest.raises(ValueError, match=r".*One of smaller_side.*and bigger_side.*must be given"):
        open_image_scaled(sample_image)
