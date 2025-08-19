import time

import numpy as np
import pytest
from PIL import Image

from visiontext.images import JPEGDecoderConst, decode_jpeg, encode_jpeg, show_image_pil


def _image_delta(i1, i2):
    if i1.ndim == 2:
        i1 = i1[..., None]
    if i2.ndim == 2:
        i2 = i2[..., None]
    # convert uint8 images ranged [0, 255] to float range [0, 1] and compute Root MSE
    err = np.mean((i1.astype(np.float64) / 255 - i2.astype(np.float64) / 255) ** 2) ** 0.5
    # print(f"Error: {err:.3e}")

    # for debugging
    if err >= 1e-1:
        print(f"Error is {err:.3e} instead of < 1e-1")
        show_image_pil(np.concatenate([i1, i2], axis=0))
    # assert err < 1e-2, f"Error is {err:.3e} instead of < 1e-2"

    return err


def _assert_shape_dtype(image, shape):
    assert image.shape == shape, f"Shape is {image.shape} instead of {shape}"
    assert image.dtype == np.uint8, f"Dtype is {image.dtype} instead of {np.uint8}"


def _assert_bytes(bytes_):
    assert isinstance(bytes_, bytes), f"Encoded image is not bytes, but {type(bytes_)}"
    assert len(bytes_) > 0, f"Encoded image is empty"


class _Timer:
    def __init__(self):
        self.start = time.perf_counter_ns()

    def stop(self, n=1, title=""):
        print(f"{title}T={(time.perf_counter_ns() - self.start) / n / 1000000:.3f}ms", end=" ")


@pytest.mark.full
# noinspection PyUnboundLocalVariable
def test_jpeg_coding():
    h, w = 10, 10

    img_gray_orig = np.round(np.linspace(0, 1, h * w).reshape((h, w)) * 255).astype(np.uint8)
    img_gray_3d = img_gray_orig[..., None]
    img_gray_as_rgb = np.repeat(img_gray_3d, repeats=3, axis=2)
    img_rgb_data = np.linspace(0, 1, h * w * 3)
    img_rgb_backwards = np.round(img_rgb_data.reshape((3, h, w)) * 255).astype(np.uint8)
    img_rgb = np.transpose(img_rgb_backwards, (1, 2, 0))

    runs = 100

    # pillow_image expects / produces PIL.Image.Image and needs to be converted here
    pillow_image_method = JPEGDecoderConst.PILLOW_IMAGE
    img_gray = Image.fromarray(img_gray_orig, mode="L")
    t = _Timer()
    for n in range(runs):
        bytes_gray = encode_jpeg(img_gray, method=pillow_image_method)
    _assert_bytes(bytes_gray)
    t.stop(runs, "Enc given gray PIL Image")

    for i_enc, encoder_method in enumerate(JPEGDecoderConst.values()):
        if encoder_method == JPEGDecoderConst.PILLOW_IMAGE:
            continue

        for i_dec, decoder_method in enumerate(JPEGDecoderConst.values()):
            if decoder_method == JPEGDecoderConst.PILLOW_IMAGE:
                continue

            print(f"---------- encoder {encoder_method} decoder {decoder_method} ----------")

            for img_gray in [img_gray_orig, img_gray_3d, img_gray_as_rgb]:
                print(f"Test gray: {str(img_gray.shape):15s}", end=" ")

                # encode gray image to bytes
                t = _Timer()
                if encoder_method == JPEGDecoderConst.PILLOW_IMAGE:
                    img_gray = Image.fromarray(img_gray)

                for n in range(runs):
                    bytes_gray = encode_jpeg(img_gray, encoder_method)
                t.stop(runs, "Enc")

                _assert_bytes(bytes_gray)
                print(f"length={len(bytes_gray)}", end=" ")

                # decode gray image to grayscale
                t = _Timer()
                for n in range(runs):
                    img_gray_re = decode_jpeg(bytes_gray, decoder_method, is_gray=True)
                t.stop(runs, "DecGray")

                _assert_shape_dtype(img_gray_re, (h, w))
                err = _image_delta(img_gray, img_gray_re)
                print(f"EG={err:.1e}", end=" ")

                # decode gray image to rgb
                t = _Timer()
                for n in range(runs):
                    img_gray_re_as_rgb = decode_jpeg(bytes_gray, decoder_method, is_gray=False)
                t.stop(runs, "DecRGB")
                _assert_shape_dtype(img_gray_re_as_rgb, (h, w, 3))
                err = _image_delta(img_gray, img_gray_re_as_rgb)
                print(f"ER={err:.1e}")

            print(f"Test RGB : {str(img_rgb.shape):15s}", end=" ")

            # encode rgb image to bytes
            t = _Timer()
            for n in range(runs):
                bytes_rgb = encode_jpeg(img_rgb, encoder_method)
            t.stop(runs, "Enc")
            _assert_bytes(bytes_rgb)
            print(f"length={len(bytes_gray)}", end=" ")

            # decode rgb image to rgb
            t = _Timer()
            for n in range(runs):
                img_rgb_re = decode_jpeg(bytes_rgb, decoder_method, is_gray=False)
            t.stop(runs, "Dec")

            # show_image(img_rgb)
            # show_image(img_rgb_re)

            _assert_shape_dtype(img_rgb_re, (h, w, 3))
            err = _image_delta(img_rgb, img_rgb_re)
            print(f"E={err:.1e}")
            # img_rgb_re_as_gray = decode_jpeg(bytes_rgb, is_gray=True)
