# from reportlab.graphics import renderPM
# from svglib.svglib import svg2rlg
# from svglib.svglib import logger as svglib_logger
# svglib_logger.setLevel("ERROR")


import io
import os
import random
import signal
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import PIL
import requests
from PIL import ExifTags, Image, ImageFile
from PIL.Image import Resampling

from packg import format_exception
from packg.log import logger
from packg.system.timeout import run_function_with_timeout
from packg.typext import PathType
from visiontext.images import pil_editor

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from cairosvg import svg2png
except Exception as e:
    svg2png = None
    # print(f"WARNING: cairosvg not available, svg2png will not work: {format_exception(e)} ")

try:
    import pillow_avif  # pip install pillow-avif-plugin

    _ = pillow_avif
except Exception as e:
    warnings.warn(
        f"WARNING: pillow_avif not available: {format_exception(e)} so visiontext.image_downloader"
        f" will not be able to read or write avif images. Install:\npip install pillow-avif-plugin"
    )


class APIDisabledError(Exception):
    pass


def ensure_svg_support():
    if svg2png is None:
        from cairosvg import svg2png as svg2png_again

        print(svg2png_again)


def download_data(
    url, timeout: float = 10, user_agent_token: str | None = None, verbose: bool = False
) -> tuple[requests.Response | None, str | None]:
    """Adapted from img2dataset"""
    user_agent = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
    if user_agent_token:
        user_agent += f" (compatible; {user_agent_token}; +https://github.com/rom1504/img2dataset)"
    try:
        res = requests.get(
            url, params=None, data=None, headers={"User-Agent": user_agent}, timeout=timeout
        )
        if res.status_code != 200:
            err = f"status_code {res.status_code}"
            if verbose:
                logger.warning(f"status code {res.status_code} for {url}")
            return None, err
        return res, None
    except Exception as e:
        err = format_exception(e)
        if verbose:
            logger.warning(f"error {err} for {url}")
        return None, format_exception(e)


def random_sleep(sleep: float, randomness: float = 0.1):
    if randomness < 0 or randomness > 1:
        raise ValueError(f"randomness must be in [0,1] but is {randomness}")
    min_factor = 1 - randomness
    sleep_here = (min_factor + random.random() * (2 * randomness)) * sleep
    time.sleep(sleep_here)


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds: int):
    def signal_handler(signum, frame):
        raise TimeoutException(f"timeout out after {seconds}s")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def download_data_with_retry(
    url,
    timeout: float = 10,
    tries: int = 3,
    sleep: float = 0.2,
    user_agent_token: str | None = None,
    verbose: bool = False,
) -> tuple[requests.Response | None, str | None]:
    assert tries > 0, f"retries must be > 0 but is '{tries}'"
    res, err = None, None
    for i in range(tries):
        if i > 0:
            random_sleep(sleep)

        try:
            with time_limit(max(1, round(timeout))):
                res, err = download_data(url, timeout, user_agent_token, verbose=verbose)
        except TimeoutException as e:
            err = format_exception(e)

        if res is not None:
            break
    if res is None:
        return None, err

    return res, None


def save_image_robustly(
    pillow_img, target_file, quality, delete_xmp_on_fail=True
) -> tuple[Image.Image | None, str | None]:
    try:
        pillow_img.save(target_file.as_posix(), quality=quality)
        return pillow_img, None
    except (OSError, ValueError, TypeError) as e2:
        # sometimes the xmp metadata is corrupt, try deleting it and saving again
        if "xmp" in pillow_img.info:
            del pillow_img.info["xmp"]
            return save_image_robustly(pillow_img, target_file, quality, delete_xmp_on_fail=False)

        err = format_exception(e2)
        logger.error(
            f"Finally could not save image also without exif: "
            f"{format_exception(e2)} for {target_file}"
        )
        return None, err


def save_image_robustly_with_exif(
    pillow_img, target_file, quality, exif
) -> tuple[Image.Image | None, str | None]:
    os.makedirs(target_file.parent, exist_ok=True)
    try:
        pillow_img.save(target_file.as_posix(), quality=quality, exif=exif)
        return pillow_img, None
    except (OSError, ValueError, TypeError) as e:
        broken_exif_file = target_file.with_suffix(".broken-exif-bin")
        logger.warning(
            f"Could not save raw_exif data: {format_exception(e)} for {target_file}\n"
            f"Discarding it and writing to {broken_exif_file} instead."
        )
        broken_exif_file.write_bytes(exif)
    return save_image_robustly(pillow_img, target_file, quality)


def extract_alpha_channel(img: Image.Image) -> tuple[Image.Image, Image.Image | None, bytes]:
    """
    Convert image to RGB and extract alpha channel if available

    Args:
        img: image to convert

    Returns:
        tuple:
            RGB image data if successful (mode="RGB")
            alpha channel if succesful and alpha available (mode="L" i.e. grayscale)
            raw exif data
    """
    raw_exif = img.info.get("exif", b"")
    if img.mode == "P" and img.info.get("transparency") is not None:
        # convert palette image with transparency to RGBA first
        img = img.convert("RGBA")
    # try to find the alpha channel
    try:
        alpha_index = img.getbands().index("A")
    except ValueError:
        alpha_index = None
    if alpha_index is not None:
        # transparency channel detected, convert to RGB with white background
        rgb_img = Image.new("RGB", img.size, (255, 255, 255))  # (0,0,0))
        alpha_img = img.split()[alpha_index]
        rgb_img.paste(img, box=(0, 0), mask=alpha_img)
        return rgb_img, alpha_img, raw_exif
    else:
        # no transparency channel detected, convert to RGB
        img = img.convert("RGB")
        return img, None, raw_exif


def parse_image(
    data: bytes,
    verbose: bool = False,
    reraise_errors: bool = False,
) -> tuple[Image.Image | None, Image.Image | None, bytes, str | None]:
    """
    TODO fix decompression bomb warnings (images exist that are 70MB big)
        catch decompression bomb warning and raise it as an error, class DecompressionBombWarning
        with warnings.catch_warnings():
            warnings.simplefilter("error", DecompressionBombWarning)
            try:
                rgb_img, a_img, raw_exif, error_str = parse_image(file_bytes, reraise_errors=True)
            except DecompressionBombWarning as e:
                raise DecompressionBombWarning(f"Decompression bomb: {file_name}") from e

    TODO ignore alpha masks that are all 255s, e.g.:
        if a_img is not None:
        has_alpha = True
        a_arr = np.array(a_img)
        assert a_arr.dtype == np.uint8, f"{file_name}: Expected uint8, got {a_arr.dtype}"
        assert a_arr.ndim == 2, f"{file_name}: Expected 2D, got {a_arr.ndim}"
        if a_arr.min() == 255:
            has_alpha = False

    Returns:
        tuple:
            RGB image data if successful (mode="RGB")
            alpha channel if succesful and alpha available (mode="L" i.e. grayscale)
            exif_data, empty bytestring if not exists
            error string if any
    """
    errors = []
    try:
        img = Image.open(io.BytesIO(data))
        rgb_img, a_img, raw_exif = extract_alpha_channel(img)
        return rgb_img, a_img, raw_exif, None
    except Exception as e:
        errors.append(f"PIL: {format_exception(e)}")
        if reraise_errors:
            raise e

    try:
        # # svglib + reportlab[renderpm] broke in some cases
        # drawing = svg2rlg(io.BytesIO(data))
        # img = renderPM.drawToPIL(drawing, dpi=96)
        # # svglib + reportlab + cairo backend might also work

        # so far cairosvg seems to be robust
        # it can hang forever though in some cases.
        png_bytes = run_function_with_timeout(60, svg2png, data)
        # png_bytes = svg2png(data)
        img = Image.open(io.BytesIO(png_bytes))
        rgb_img, a_img, raw_exif = extract_alpha_channel(img)
        return rgb_img, a_img, raw_exif, None
    except Exception as e:
        errors.append(f"svg: {format_exception(e)}")
        if reraise_errors:
            raise e

    error_str = "\n".join(errors)
    if verbose:
        logger.warning(
            f"error parsing image of size {len(data)} type {type(data).__name__}:\n{error_str}"
        )
    return None, None, b"", error_str


def download_image_with_retry(
    url: str,
    target_file: PathType,  # should be jpg
    # seems to be a decent tradeoff for visual quality vs file size
    quality="web_very_high",
    timeout: float = 10,
    retries: int = 3,
    sleep: float = 1,
    user_agent_token: str | None = None,
    fix_url: bool = True,
    save_rgb: bool = True,
    save_alpha: bool = True,
) -> tuple[Image.Image | None, Image.Image | None, bytes, str | None]:
    """
    Returns:
        rgb pillow image,
        alpha channel pillow image if available,
        raw exif data,
        error string if any
    """
    target_file = Path(target_file)
    errors = []
    res, err = download_data_with_retry(
        url, timeout, retries, sleep, user_agent_token, verbose=False
    )
    if res is None:
        errors.append(err)
        if not fix_url:
            return None, None, b"", "\n".join(errors)

        # try to fix url
        new_urls = []
        if not any(url.startswith(s) for s in ["http://", "https://"]):
            url_clean = url.lstrip("/:.")
            new_urls.append(f"https://{url_clean}")
            new_urls.append(f"http://{url_clean}")
        for new_url in new_urls:
            res, err = download_data_with_retry(
                new_url, timeout, retries, sleep, user_agent_token, verbose=False
            )
            if res is not None:
                break
            else:
                errors.append(err)

    if res is None:
        # download failed finally
        return None, None, b"", "\n".join(errors)

    # decode image
    data = res.content
    res.close()
    logger.debug(f"parsing image of size {len(data)} from {url}")
    rgb_img, alpha_img, raw_exif, err = parse_image(data, verbose=False)
    if rgb_img is None:
        # parsing image failed
        return None, None, b"", err

    if save_rgb:
        rgb_img, err = save_image_robustly_with_exif(rgb_img, target_file, quality, raw_exif)
        if rgb_img is None:
            return None, None, b"", err
    if save_alpha and alpha_img is not None:
        target_file_alpha = target_file.with_suffix(".alpha.jpg")
        os.makedirs(target_file_alpha.parent, exist_ok=True)
        alpha_img, alpha_img_err = save_image_robustly(alpha_img, target_file_alpha, quality)
        if alpha_img_err is not None:
            logger.warning(f"error saving alpha image: {alpha_img_err}")
    return rgb_img, alpha_img, raw_exif, None


def download_image_with_retry_only_once(
    url,
    target_file,
    # seems to be a decent tradeoff for visual quality vs file size
    quality="web_very_high",
    timeout: float = 10,
    retries: int = 3,
    sleep: float = 1,
    user_agent_token: str | None = None,
    verbose: bool = False,
) -> tuple[Image.Image | None, Image.Image | None, bytes | None, str | None]:
    error_file = target_file.with_suffix(".error")
    if target_file.is_file():
        rgb_img = None
        try:
            rgb_img = Image.open(target_file.as_posix()).convert("RGB")
        except PIL.UnidentifiedImageError as e:
            logger.error(f"error opening {target_file}: {format_exception(e)}")
            os.remove(target_file.as_posix())
        if rgb_img is not None:
            alpha_img = None
            target_file_alpha = target_file.with_suffix(".alpha.jpg")
            if target_file_alpha.is_file():
                try:
                    alpha_img = Image.open(target_file_alpha.as_posix()).convert("L")
                except PIL.UnidentifiedImageError as e:
                    logger.error(f"error opening {target_file_alpha}: {format_exception(e)}")
            return rgb_img, alpha_img, None, None
    if error_file.is_file():
        err = error_file.read_text(encoding="utf-8", errors="replace")
        if verbose:
            logger.warning(
                f"skipping {url} because of previous error as indicated by {error_file}: {err}"
            )
        return None, None, None, err
    rgb_img, alpha_img, exif_data, err = download_image_with_retry(
        url, target_file, quality, timeout, retries, sleep, user_agent_token
    )
    if err is not None:
        os.makedirs(error_file.parent, exist_ok=True)
        error_file.write_text(err, encoding="utf-8", errors="replace")
    return rgb_img, alpha_img, exif_data, err


def get_exif_from_pillow_image(img) -> tuple[dict[str, Any], list[str]]:
    return_dict = {}
    img_exif = img.getexif()
    IFD_CODE_LOOKUP = {i.value: i.name for i in ExifTags.IFD}
    errors = []

    for tag_code, value in img_exif.items():
        # if the tag is an IFD block, nest into it
        if tag_code in IFD_CODE_LOOKUP:
            ifd_tag_name = IFD_CODE_LOOKUP[tag_code]
            # print(f"IFD '{ifd_tag_name}' (code {tag_code}):")
            try:
                ifd_data = img_exif.get_ifd(tag_code)
            except KeyError as e:
                error = format_exception(e)
                errors.append(f"{tag_code=} {ifd_tag_name=} {error=}")
                continue
            if ifd_data is None:
                errors.append(
                    f"{tag_code=} {ifd_tag_name=} img_exif.get_ifd(tag_code) returned None"
                )
                continue

            for nested_key, nested_value in ifd_data.items():
                nested_tag_name = (
                    ExifTags.GPSTAGS.get(nested_key, None)
                    or ExifTags.TAGS.get(nested_key, None)
                    or nested_key
                )
                # print(f"  {nested_tag_name}: {nested_value}")

                full_name = f"{ifd_tag_name}/{nested_tag_name}"
                return_dict[full_name] = nested_value

        else:
            # root-level tag
            tag_name = ExifTags.TAGS.get(tag_code)
            # print(f"{tag_name}: {value}")
            return_dict[tag_name] = value
    return return_dict, errors


def create_lowres_image(
    source_file, target_file_base, target_size: int, force: bool = False
) -> Path:
    source_file = Path(source_file)
    target_file = Path(f"{Path(target_file_base).as_posix()}_{target_size}px.jpg")
    if target_file.is_file() and not force:
        return target_file
    img = Image.open(source_file.as_posix()).convert("RGB")
    img_square = pil_editor.crop_square(img)
    w, h = img_square.size
    assert w == h
    img_lowres: Image.Image = pil_editor.scale_image(
        img_square, target_size, target_size, method=Resampling.BILINEAR
    )
    img_lowres.save(target_file, quality="web_very_high")
    return target_file
