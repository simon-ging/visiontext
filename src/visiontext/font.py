from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile

import requests
from PIL import ImageFont
from platformdirs import user_cache_path


def get_dejavusans_font_path() -> str:
    """
    Find font path of DejaVuSans.ttf or download it to cache folder

    Returns:
        Absolute path to DejaVuSans.ttf
    """
    font_name = "DejaVuSans.ttf"
    font_url = (
        "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/"
        "version_2_37/dejavu-fonts-ttf-2.37.zip"
    )

    try:
        font = ImageFont.truetype(font_name, 14)
        return font.path
        # e.g. /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
    except OSError:
        pass

    try:
        response = requests.get(font_url)
    except requests.RequestException as e:
        return f"Error downloading the font: {e}"

    response.raise_for_status()  # Check if the download was successful

    font_dir = user_cache_path("python_visiontext") / "fonts"
    font_dir_temp = font_dir / "dejavusans"
    font_archive = font_dir_temp / "archive.zip"

    os.makedirs(font_archive.parent, exist_ok=True)
    with font_archive.open("wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(font_archive.as_posix(), "r") as zip_ref:
        zip_ref.extractall(font_dir_temp / "extracted")

    ttf_src = list(font_dir_temp.glob("**/DejaVuSans.ttf"))[0]
    ttf_dst = font_dir / "DejaVuSans.ttf"
    os.makedirs(ttf_dst.parent, exist_ok=True)
    os.rename(ttf_src, ttf_dst)
    shutil.rmtree(font_dir_temp)
    return ttf_dst.as_posix()


def get_dejavusans_font(font_size: int):
    font_path = get_dejavusans_font_path()
    return ImageFont.truetype(font_path, size=font_size)


def get_notosans_font_path() -> str:
    font_url = "https://github.com/googlefonts/noto-fonts/blob/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf?raw=true"
    font_path = user_cache_path("python_visiontext") / "fonts" / "NotoSans-Regular.ttf"
    os.makedirs(os.path.dirname(font_path), exist_ok=True)

    if not os.path.isfile(font_path):
        print(f"Downloading Noto Sans font... If this breaks, make sure that {font_path} exists")
        urllib.request.urlretrieve(font_url, font_path)
    return font_path.as_posix()


def get_notosans_font(font_size: int):
    font_path = get_notosans_font_path()
    return ImageFont.truetype(font_path, size=font_size)


def main():
    print(get_dejavusans_font_path())


if __name__ == "__main__":
    main()
