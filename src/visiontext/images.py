"""
Examples:
    >>> from IPython.display import display
    >>> image = open_image_scaled("image.png", bigger_side=500)
    >>> display(image)
"""
import numpy as np
from PIL import Image
from PIL.Image import Resampling
from dataclasses import dataclass
from typing import Union, Optional

AUTO = "auto"
DEFAULT_UPSAMPLING_METHOD = Resampling.BICUBIC
DEFAULT_DOWNSAMPLING_METHOD = Resampling.BOX

SamplingMapPIL = {e.name.lower(): e.value for e in Resampling}
ImageType = Union[np.ndarray, Image.Image]
ResamplingMethodType = Union[str, int, Resampling]


@dataclass
class PILImageScaler:
    """
    Scale an image using PIL.

    todo in a new version set the default to return pillow and
        return_type as enum with pillow, uint8, fp32
    """

    upsampling_method: str = DEFAULT_UPSAMPLING_METHOD
    downsampling_method: str = DEFAULT_DOWNSAMPLING_METHOD
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
