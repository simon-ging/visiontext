from PIL import Image, ImageDraw, ImageFont
from matplotlib import colors as mpl_colors
from typing import Dict, List, Tuple, Optional, Union

from packg.maths import clip_rectangle_coords
from visiontext.font import get_font_path_dejavusans
from visiontext.colormaps import (
    create_colormap_for_dark_background,
    create_colorgetter_from_colormap,
)


def create_bbox_images(
    bx: int,
    by: int,
    bw: int,
    bh: int,
    w: int,
    h: int,
    box_color=(0, 0, 255, 255),
    bbox_width: int = 5,
    mask_opacity: int = 64,
) -> Dict[str, Image.Image]:
    """
    Create box overlay images, originally used in the OVAD visualizer e.g.:
    https://lmb.informatik.uni-freiburg.de/resources/datasets/ovad/?imgnum=14#top

    This function creates transparent PNGs that can be overlayed on top of an image to show or
    highlight a bounding box.
    """
    # create one image with transparent background and one with half transparent
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    img2 = Image.new("RGBA", (w, h), (0, 0, 0, mask_opacity))
    draw2 = ImageDraw.Draw(img2)

    # boxes should be approximately the same width after scaling the image
    # so we have to scale the box thickness now
    target_w = 800
    rel_w = w / target_w  # if small image, create smaller boxes
    rel_bbox_width = round(bbox_width * rel_w)

    # the inside of the thick drawn box will be the true bbox
    # this means that the box can clip outside the image and become almost invisible
    # to avoid that, we move the edges of the box outwards and then clip them
    x1 = bx - rel_bbox_width
    y1 = by - rel_bbox_width
    x2 = bx + bw + rel_bbox_width
    y2 = by + bh + rel_bbox_width
    (x1, y1, x2, y2) = clip_rectangle_coords((x1, y1, x2, y2), w, h)

    for bbox_step in range(rel_bbox_width):
        rectangle_coords = (
            x1 + bbox_step,
            y1 + bbox_step,
            x2 - bbox_step,
            y2 - bbox_step,
        )
        # a 2nd clip should now not be necessary anymore and we assert instead
        assert (
            rectangle_coords[0] >= 0
            and rectangle_coords[1] >= 0
            and rectangle_coords[2] <= w
            and rectangle_coords[3] <= h
        ), f"{rectangle_coords} outside {w}x{h}"

        draw.rectangle(rectangle_coords, fill=(0, 0, 0, 0), outline=box_color)
        draw2.rectangle(rectangle_coords, fill=(0, 0, 0, 0), outline=box_color)

    return {"box": img, "box-mask": img2}


def convert_bbox_abs_to_rel(bx, by, bw, bh, image_w, image_h):
    """
    Convert bounding box coordinates to relative coordinates.

    Args:
        bx: x coordinate of the top-left corner of the bounding box
        by: y coordinate of the top-left corner of the bounding box
        bw: width of the bounding box
        bh: height of the bounding box
        image_w: width of the image
        image_h: height of the image

    Returns:
        Tuple of relative coordinates (x, y, w, h)

    """
    x = bx / image_w
    y = by / image_h
    w = bw / image_w
    h = bh / image_h
    return x, y, w, h


def convert_bbox_rel_to_abs(x, y, w, h, image_w, image_h):
    """
    Convert bounding box coordinates to absolute coordinates.

    Args:
        x: x coordinate of the top-left corner of the bounding box
        y: y coordinate of the top-left corner of the bounding box
        w: width of the bounding box
        h: height of the bounding box
        image_w: width of the image
        image_h: height of the image

    Returns:
        Tuple of absolute coordinates (bx, by, bw, bh)

    """
    bx = x * image_w
    by = y * image_h
    bw = w * image_w
    bh = h * image_h
    return bx, by, bw, bh


def get_bbox_bounds(
    box_x: float,
    box_y: float,
    box_w: float,
    box_h: float,
    image_w: int,
    image_h: int,
    min_w=32,
    min_h=32,
    create_squares=False,
):
    """
    Get the bounding box boundaries for cropping the image.
    Increase the box size if it is too small.

    Args:
        box_x: absolute box coordinates for the given image
        box_y:
        box_w:
        box_h:
        image_w: image size
        image_h:
        min_w: minimum box size
        min_h:
        create_squares: if True, try to create a square bounding box instead of a rectangle

    Returns:
        Cropping parameters tuple (x1, y1, x2, y2)

    """
    # Calculate the center coordinates of the bounding box
    center_x = box_x + (box_w / 2)
    center_y = box_y + (box_h / 2)

    # Calculate the new width and height of the bounding box
    new_w = max(box_w, min_w)
    new_h = max(box_h, min_h)

    if create_squares:
        new_w = max(new_w, new_h)
        new_h = max(new_w, new_h)
        pass

    # Calculate the new x and y coordinates of the top-left corner of the bounding box
    new_x = center_x - (new_w / 2)
    new_y = center_y - (new_h / 2)

    # Adjust the new x coordinate if the box is too close to the left border
    if new_x < 0:
        new_x = 0
        new_x2 = min(new_x + new_w, image_w)
    else:
        new_x2 = min(new_x + new_w, image_w)
        if new_x2 > image_w:
            new_x2 = image_w
            new_x = max(new_x2 - new_w, 0)

    # Adjust the new y coordinate if the box is too close to the top border
    if new_y < 0:
        new_y = 0
        new_y2 = min(new_y + new_h, image_h)
    else:
        new_y2 = min(new_y + new_h, image_h)
        if new_y2 > image_h:
            new_y2 = image_h
            new_y = max(new_y2 - new_h, 0)

    # Again make sure all bounds are respected
    new_x = max(new_x, 0)
    new_y = max(new_y, 0)
    new_x2 = min(new_x2, image_w)
    new_y2 = min(new_y2, image_h)

    # Return the cropping boundaries as integers
    return int(new_x), int(new_y), int(new_x2), int(new_y2)


def draw_bounding_boxes_pil(
    image: Image.Image,
    boxes: List[Tuple[float, float, float, float]],  # x, y, w, h
    labels: Tuple[List[str], str, None] = "numbers",
    colors: Optional[List[Tuple[int, int, int]]] = None,
    width: int = 3,
    font_size: int = 14,
    cmap: Optional[Union[str, mpl_colors.Colormap]] = None,
):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(get_font_path_dejavusans(), font_size)

    if colors is None:
        if cmap is None:
            cmap = create_colormap_for_dark_background(add_brightness=0.5)
        # automatically generate a bunch of colors from the colormap
        getter = create_colorgetter_from_colormap(cmap)
        colors = [tuple(getter(i / len(boxes))) for i in range(len(boxes))]

    if isinstance(labels, str):
        if labels == "numbers":
            labels = [str(i) for i in range(len(boxes))]
        else:
            raise ValueError(f"Unknown label type: {labels}")

    for i, (box, color) in enumerate(zip(boxes, colors)):
        x, y, w, h = box
        x1, y1, x2, y2 = get_bbox_bounds(x, y, w, h, image.width, image.height, min_w=0, min_h=0)
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=width)

        text_args = dict(font=font, stroke_width=2)
        if labels is not None:
            label = labels[i]

            # # removed in pillow 10...
            # text_width, text_height = draw.textsize(label, **text_args)

            # new version with textbbox
            tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), label, **text_args)
            text_width = tx1 - tx0
            text_height = ty1 - ty0

            text_x = min(x1 + width + 2, image.width - text_width)
            text_y = min(y1 + width + 2, image.height - text_height)
            draw.text(
                (text_x, text_y),
                label,
                fill=color,
                anchor="lt",
                stroke_fill="black",
                **text_args,
            )

    return image
