import os
import requests
import shutil
import zipfile
from PIL import ImageFont

from packg.paths import get_cache_dir


def get_font_path_dejavusans() -> str:
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

    font_dir = get_cache_dir() / "fonts"
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


def main():
    print(get_font_path_dejavusans())


if __name__ == "__main__":
    main()
