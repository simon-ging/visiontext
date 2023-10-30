from dataclasses import dataclass
from typing import Union, Optional

import numpy as np
from PIL import Image
from PIL.Image import Resampling

from packg import Const


class ResamplingC(Const):
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"
    AREA = "area"
    AUTO = "auto"


DEFAULT_UPSAMPLING_METHOD = ResamplingC.BICUBIC
DEFAULT_DOWNSAMPLING_METHOD = ResamplingC.AREA

SamplingMapPIL = {
    ResamplingC.NEAREST: Resampling.NEAREST,
    ResamplingC.BILINEAR: Resampling.BILINEAR,
    ResamplingC.BICUBIC: Resampling.BICUBIC,
    ResamplingC.LANCZOS: Resampling.LANCZOS,
    ResamplingC.AREA: Resampling.BOX,
}

ImageType = Union[np.ndarray, Image.Image]


@dataclass
class PILImageScaler:
    upsampling_method: str = DEFAULT_UPSAMPLING_METHOD
    downsampling_method: str = DEFAULT_DOWNSAMPLING_METHOD
    return_pillow: bool = False
    return_fp32: bool = False

    def scale_image_smaller_side(
        self, img: ImageType, smaller_side: int, method: str = ResamplingC.AUTO
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
        img = img.resize((target_w, target_h), SamplingMapPIL[method])
        return self._prepare_return(img)

    def scale_image_bigger_side(
        self, img: ImageType, bigger_side: int, method: str = ResamplingC.AUTO
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
        img = img.resize((target_w, target_h), SamplingMapPIL[method])
        return self._prepare_return(img)

    def scale_image(self, img, target_h, target_w, method=ResamplingC.AUTO):
        img = self._ensure_pil(img)
        w, h = img.size
        method = self._get_method(h, w, target_h, target_w, method)
        img = img.resize((target_w, target_h), method)
        return self._prepare_return(img)

    def _ensure_pil(self, img):
        return img if isinstance(img, Image.Image) else Image.fromarray(img)

    def _get_method(self, h, w, target_h, target_w, method):
        if method != ResamplingC.AUTO:
            return method
        return (
            self.downsampling_method
            if _is_downsampling(h, w, target_h, target_w)
            else self.upsampling_method
        )

    def _prepare_return(self, img):
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
    method: str = ResamplingC.AUTO,
    upsampling_method: str = DEFAULT_UPSAMPLING_METHOD,
    downsampling_method: str = DEFAULT_DOWNSAMPLING_METHOD,
):
    scaler = PILImageScaler(
        upsampling_method=upsampling_method,
        downsampling_method=downsampling_method,
        return_pillow=True,
    )
    image = Image.open(image_file).convert("RGB")
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
