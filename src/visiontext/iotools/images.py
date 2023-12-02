from PIL import Image


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
