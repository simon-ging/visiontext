"""
Image utilities using opencv-python, libturbojpeg, pillow

which sampling method to use:
    - bilinear is a classic
    - area might be best for downsampling
    - nearest for upsampling without blurring
    - cubic/lanczos might look better for upsampling but are slower

https://www.libjpeg-turbo.org/
documentation on tjCompress2 method:
https://rawcdn.githack.com/libjpeg-turbo/libjpeg-turbo/main/doc/html/group___turbo_j_p_e_g.html

conda install -c conda-forge libjpeg-turbo -y
pip install -U pyturbojpeg opencv-python pillow

Todo: Write tests, does it work with all kinds of images (grayscale, RGB, RGBA)

Todo: this CMYK file breaks libjpeg-turbo

/misc/lmbssd/gings/datasets/imagenet1k/val/n13133613/ILSVRC2012_val_00019877.JPEG

Examples:
    >>> from IPython.display import display
    >>> image = open_image_scaled("image.png", bigger_side=500)
    >>> display(image)
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Union, Optional

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage, Resampling
from loguru import logger

from packg import format_exception
from packg.constclass import Const


class SamplingConst:
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"
    AREA = "area"
    AUTO = "auto"


# optional opencv-python
try:
    import cv2

    # ref: https://chadrick-kwag.net/cv2-resize-interpolation-methods/
    SamplingMapCV2 = {
        SamplingConst.NEAREST: cv2.INTER_NEAREST,
        SamplingConst.BILINEAR: cv2.INTER_LINEAR,
        SamplingConst.AREA: cv2.INTER_AREA,
        SamplingConst.BICUBIC: cv2.INTER_CUBIC,
        SamplingConst.LANCZOS: cv2.INTER_LANCZOS4,
    }
except ImportError:
    cv2, SamplingMapCV2 = [None] * 2

# optional libturbojpeg
try:
    import turbojpeg as tjpeg
except ImportError:
    tjpeg = None

SamplingMapPIL = {
    SamplingConst.NEAREST: Resampling.NEAREST,
    SamplingConst.BILINEAR: Resampling.BILINEAR,
    SamplingConst.BICUBIC: Resampling.BICUBIC,
    SamplingConst.LANCZOS: Resampling.LANCZOS,
    SamplingConst.AREA: Resampling.BOX,
    "box": Resampling.BOX,  # backward compatibility
    "hamming": Resampling.HAMMING,
}
DEFAULT_UPSAMPLING_METHOD = SamplingConst.BICUBIC
DEFAULT_DOWNSAMPLING_METHOD = SamplingConst.AREA
JPEG_QUAL_FFMPEG = 2  # 2 best, 31 worst
AUTO = "auto"

ImageType = Union[np.ndarray, Image.Image]
ResamplingMethodType = Union[str, int, Resampling]


class JPEGDecoderConst(Const):
    OPENCV = "opencv"
    LIBTURBOJPEG_DEFAULT = "libturbojpeg_default"
    LIBTURBOJPEG_FASTEST = "libturbojpeg_fastest"
    PILLOW = "pillow"
    PILLOW_IMAGE = "pillow_image"  # returns PIL.Image instead of np.ndarray


class _TjpegGetter:
    def __init__(self):
        self.jpeg = None

    def get(self):
        if self.jpeg is None:
            if tjpeg is None:
                raise ImportError(
                    "turbojpeg not installed. To install, run:\n"
                    "conda install -c conda-forge libjpeg-turbo -y\n"
                    "pip install -U pyturbojpeg"
                )
            self.jpeg = tjpeg.TurboJPEG()
        return self.jpeg


_jpeg_getter = _TjpegGetter()


def get_dummy_image_np() -> np.array:
    """
    Returns:
        numpy array image type uint8, shape (100, 100, 3), y-axis gradient from black to white
    """
    return (
        np.linspace(0, 255, num=100)
        .astype(np.uint8)[:, None, None]
        .repeat(3, axis=2)
        .repeat(100, axis=1)
    )


def check_turbojpeg_available() -> bool:
    if tjpeg is None:
        return False
    try:
        test_method = JPEGDecoderConst.LIBTURBOJPEG_DEFAULT
        image_in = get_dummy_image_np()
        jpeg_bytes = encode_jpeg(image_in, method=test_method)
        image_out = decode_jpeg(jpeg_bytes, method=test_method)
        np.testing.assert_allclose(image_in, image_out, atol=1, rtol=1e-3)
    except Exception as e:
        print(f"Error in turbojpeg test: {format_exception(e)}")
        return False
    return True


def decode_jpeg(
    jpeg_arr: Union[np.ndarray, bytes],
    method: str = JPEGDecoderConst.LIBTURBOJPEG_DEFAULT,
    is_gray: bool = False,
) -> Union[np.ndarray, PILImage]:
    """
    Decode jpeg encoded image.
    Don't care about autodetecting gray jpegs, probably not worth it.

    Args:
        jpeg_arr: input jpeg encoded image shape (num_bytes,)
        method: decoder
        is_gray: if true, decode to grayscale

    Returns:
        decoded image shape (h, w, 3) dtype uint8 in [0, 255] OR pillow image
    """
    decoded_arr = None
    if method == JPEGDecoderConst.OPENCV:
        if isinstance(jpeg_arr, bytes):
            jpeg_arr = np.frombuffer(jpeg_arr, dtype=np.uint8)
        if is_gray:
            decoded_arr = cv2.imdecode(jpeg_arr, cv2.IMREAD_GRAYSCALE)
        else:
            decoded_arr = cv2.imdecode(jpeg_arr, cv2.IMREAD_COLOR)
            decoded_arr = cv2.cvtColor(decoded_arr, cv2.COLOR_BGR2RGB)
    if (
        method == JPEGDecoderConst.LIBTURBOJPEG_DEFAULT
        or method == JPEGDecoderConst.LIBTURBOJPEG_FASTEST
    ):
        flags = 0
        if method == JPEGDecoderConst.LIBTURBOJPEG_FASTEST:
            flags |= tjpeg.TJFLAG_FASTUPSAMPLE | tjpeg.TJFLAG_FASTDCT
        try:
            decoded_arr = _jpeg_getter.get().decode(
                jpeg_arr, pixel_format=tjpeg.TJPF_GRAY if is_gray else tjpeg.TJPF_RGB, flags=flags
            )
        except OSError as e:
            # it seems there exist images that cannot be decoded correctly with libjpeg-turbo
            # but pillow decodes them perfectly fine. therefore catch those errors and call pillow.
            error_str = format_exception(e)
            if (
                error_str == "OSError: Unsupported color conversion request"
                or "Could not determine subsampling type for JPEG image" in error_str
            ):
                logger.error(
                    f"{error_str} for image bytes of length {len(jpeg_arr)}. "
                    f"Decoding with pillow instead of libjpeg-turbo."
                )
                method = JPEGDecoderConst.PILLOW
                decoded_arr = None
            else:
                raise e
    if method == JPEGDecoderConst.PILLOW:
        # noinspection PyTypeChecker
        decoded_arr = np.array(Image.open(io.BytesIO(jpeg_arr)).convert("L" if is_gray else "RGB"))
    if method == JPEGDecoderConst.PILLOW_IMAGE:
        pil_image = Image.open(io.BytesIO(jpeg_arr)).convert("L" if is_gray else "RGB")
        return pil_image
    if decoded_arr is None:
        raise ValueError(f"Unknown JPEG decoding method {method}")
    if is_gray and decoded_arr.ndim == 3:
        decoded_arr = np.squeeze(decoded_arr, axis=-1)
    return decoded_arr


def encode_jpeg(
    np_arr: Union[np.ndarray, PILImage],
    method: str = JPEGDecoderConst.LIBTURBOJPEG_DEFAULT,
    quality=95,
) -> bytes:
    """
    Encode image to jpeg. expects (h, w) or (h, w, 1) for gray-scale and
    (h, w, 3) for RGB images.

    Compared losses and speed for different combinations of en-/decoder and quality
    both with random noise and some real images b/c noise probably cant be encoded well.
    Result: libturbojpeg_fastest has too many artifacts.
    libturbojpeg_default is fast and good quality, use that.
    Others are slower and not better quality-wise.

    quality: 95 best and cv2 default, 75 pillow default, 85 libturbojpeg default, 0-31 worst
    """
    is_gray = None
    if method == JPEGDecoderConst.PILLOW_IMAGE:
        is_gray = np_arr.mode == "L"
    elif np_arr.ndim == 2:
        is_gray = True
    elif np_arr.ndim == 3:
        if np_arr.shape[-1] == 1:
            is_gray = True
            np_arr = np.squeeze(np_arr, axis=2)
        elif np_arr.shape[-1] == 3:
            is_gray = False

    if is_gray is None:
        raise ValueError("Unknown image shape: " + str(np_arr.shape))

    if method == JPEGDecoderConst.OPENCV:
        params = (cv2.IMWRITE_JPEG_QUALITY, quality)
        if is_gray:
            encoded_arr = cv2.imencode(".jpg", np_arr, params=params)[1]
        else:
            np_arr_bgr = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
            encoded_arr = cv2.imencode(".jpg", np_arr_bgr, params=params)[1]
        # creates np.ndarray with shape (num_bytes) dtype uint8
        return encoded_arr.tobytes()
    if method == JPEGDecoderConst.PILLOW:
        mode = "L" if is_gray else "RGB"
        bio = io.BytesIO()
        # noinspection PyTypeChecker
        Image.fromarray(np_arr, mode).save(bio, format="JPEG", quality=quality)
        return bio.getvalue()
    if method == JPEGDecoderConst.PILLOW_IMAGE:
        bio = io.BytesIO()
        # noinspection PyTypeChecker
        np_arr.save(bio, format="JPEG", quality=quality)
        return bio.getvalue()
    if method == JPEGDecoderConst.LIBTURBOJPEG_DEFAULT:
        np_arr_bgr = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
        encoded_arr = _jpeg_getter.get().encode(
            np_arr_bgr, quality=quality, **_get_tjpeg_kwargs(is_gray)
        )
        return encoded_arr
    if method == JPEGDecoderConst.LIBTURBOJPEG_FASTEST:
        np_arr_bgr = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
        encoded_arr = _jpeg_getter.get().encode(
            np_arr_bgr,
            quality=quality,
            flags=tjpeg.TJFLAG_FASTUPSAMPLE | tjpeg.TJFLAG_FASTDCT,
            **_get_tjpeg_kwargs(is_gray),
        )
        return encoded_arr
    raise ValueError(f"Unknown JPEG encoding method {method}")


def _get_tjpeg_kwargs(is_gray: bool) -> dict[str, int]:
    return {
        "pixel_format": tjpeg.TJPF_GRAY if is_gray else tjpeg.TJPF_BGR,
        "jpeg_subsample": tjpeg.TJSAMP_GRAY if is_gray else tjpeg.TJSAMP_422,
    }


@dataclass
class PILImageScaler:
    """
    Scale an image using PIL.

    todo in a new version set the default to return pillow and
        return_type as enum with pillow, uint8, fp32
    """

    upsampling_method: Union[str, int] = DEFAULT_UPSAMPLING_METHOD
    downsampling_method: Union[str, int] = DEFAULT_DOWNSAMPLING_METHOD
    return_pillow: bool = False
    return_fp32: bool = False

    def scale_image_smaller_side(
        self, img: ImageType, smaller_side: int, method: ResamplingMethodType = AUTO
    ):
        img = self._ensure_pil(img)
        w, h = img.size
        if h < w:
            target_h = smaller_side
            target_w = round(w * smaller_side / h)
        else:
            target_w = smaller_side
            target_h = round(h * smaller_side / w)
        method = self._get_method(h, w, target_h, target_w, method)
        img = img.resize((target_w, target_h), method)
        return self._prepare_return(img)

    def scale_image_bigger_side(
        self, img: ImageType, bigger_side: int, method: ResamplingMethodType = AUTO
    ):
        img = self._ensure_pil(img)
        w, h = img.size
        if h > w:
            target_h = bigger_side
            target_w = int(w * bigger_side / h)
        else:
            target_w = bigger_side
            target_h = int(h * bigger_side / w)
        method = self._get_method(h, w, target_h, target_w, method)
        img = img.resize((target_w, target_h), method)
        return self._prepare_return(img)

    def scale_image(
        self, img: ImageType, target_h: int, target_w: int, method: ResamplingMethodType = AUTO
    ):
        img = self._ensure_pil(img)
        w, h = img.size
        method = self._get_method(h, w, target_h, target_w, method)
        img = img.resize((target_w, target_h), method)
        return self._prepare_return(img)

    def crop_square(self, img: ImageType):
        img = self._ensure_pil(img)
        w, h = img.size
        if w == h:
            return img
        if w > h:
            left = (w - h) // 2
            right = left + h
            top = 0
            bottom = h
        else:
            top = (h - w) // 2
            bottom = top + w
            left = 0
            right = w
        img = img.crop((left, top, right, bottom))
        return self._prepare_return(img)

    def _ensure_pil(self, img: ImageType):
        if isinstance(img, Image.Image):
            return img
        return Image.fromarray(img)

    def _get_method(
        self, h: int, w: int, target_h: int, target_w: int, method: ResamplingMethodType = AUTO
    ) -> Resampling:
        if isinstance(method, str):
            method = method.lower()
        if method == AUTO:
            method = (
                self.downsampling_method
                if _is_downsampling(h, w, target_h, target_w)
                else self.upsampling_method
            )
        if isinstance(method, str):
            method = SamplingMapPIL[method]
        return method

    def _prepare_return(self, img: ImageType):
        if self.return_pillow:
            return img
        img = np.array(img)
        if self.return_fp32:
            img = img.astype(np.float32) / 255.0
            img = np.clip(img, 0.0, 1.0)
        return img


pil_editor = PILImageScaler(return_pillow=True)


def _is_downsampling(h, w, target_h, target_w):
    return target_h <= h and target_w <= w


def open_image_scaled(
    image_file,
    smaller_side: Optional[int] = None,
    bigger_side: Optional[int] = None,
    method: str = AUTO,
    upsampling_method: str = DEFAULT_UPSAMPLING_METHOD,
    downsampling_method: str = DEFAULT_DOWNSAMPLING_METHOD,
    convert: Optional[str] = None,  # RGB, L, ...
):
    scaler = PILImageScaler(
        upsampling_method=upsampling_method,
        downsampling_method=downsampling_method,
        return_pillow=True,
    )
    image = Image.open(image_file)
    if convert is not None:
        image = image.convert(convert)
    if smaller_side is not None and bigger_side is not None:
        raise ValueError(
            f"Only one of smaller_side={smaller_side} and bigger_side={bigger_side} can be given"
        )
    elif smaller_side is not None:
        return scaler.scale_image_smaller_side(image, smaller_side, method=method)
    elif bigger_side is not None:
        return scaler.scale_image_bigger_side(image, bigger_side, method=method)
    else:
        raise ValueError(
            f"One of smaller_side={smaller_side} and bigger_side={bigger_side} must be given"
        )


def get_properties(image_file):
    """
    Modes: https://pillow.readthedocs.io/en/stable/handbook/concepts.html

    Args:
        image_file:

    Returns:

    """
    image = Image.open(image_file)
    mode = image.mode  # L (grayscale), RGB, RGBA
    assert mode in ["L", "RGB", "RGBA"], f"Unsupported mode {mode} for {image_file}"
    width, height = image.size
    return mode, width, height


def show_image_pil(image: ImageType):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.show()
