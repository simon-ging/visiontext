
import base64
import io
from IPython.display import display, HTML
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt


def convert_image_to_html(pil_image: Image.Image) -> str:
    """
    Usage:
        display(HTML(convert_image_to_html(pil_image)))

    Args:
        pil_image: pillow image object

    Returns:
        Image as embedded html <img> tag string
    """

    bio = io.BytesIO()
    pil_image.save(bio, "png")
    bios = bio.getbuffer()
    biosb64 = str(base64.b64encode(bios), "ascii")
    html_str = f'<img src="data:image/png;base64,{biosb64}"/>'
    return html_str
