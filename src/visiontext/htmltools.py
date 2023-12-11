from io import BytesIO

import base64

import io

from IPython.display import display, HTML
from PIL import Image
from matplotlib import pyplot as plt


class NotebookHTMLPrinter:
    """
    Class for printing things as HTML (Useful for jupyter notebooks)
    Uses an internal buffer to aggregate a bunch of HTML and then creates a single HTML output
    This is better than calling display(HTML(...))) separately for each line because it allows
    for things to stay inside one output cell.

    Usage:
        pr = NotebookHTMLPrinter()
        pr.print("<b>Hello World</b>", output=True)  # output=True to instantly display the HTML

        pr.open_table()
        pr.add_table_row("col1", "col2", "col3", is_header=True)
        pr.add_table_row("row1_col1", "row1_col2", "row1_col3")
        pr.close_table()  # here output=True is the default, so this will output the table

    """

    def __init__(self):
        self.reset()
        self.grid_max_width = 300

    def reset(self):
        self.buffer = []

    def print(self, *text, output=False, sep=" ", end="<br/>"):
        text = [str(t) for t in text]
        self.buffer.append(f"{sep.join(text)}{end}")
        if output:
            self.output()

    def get_html(self):
        html_txt = "".join(self.buffer)
        self.buffer = []
        return html_txt

    def output(self):
        display(HTML(self.get_html()))

    def open_table(self, font_size=1.5, other_style=""):
        """Start creating a HTML table e.g. for data output"""
        self.print(f"<table style='font-size:{font_size:.0%};{other_style}'>", end="")

    def add_table_row(self, *args, is_header=False):
        tag = "th" if is_header else "td"
        self.print("<tr>", end="")
        for arg in args:
            self.print(f"<{tag}>{arg}</{tag}>", end="")
        self.print("</tr>", end="")

    def close_table(self, output=True):
        self.print(f"</table>", output=output, end="")

    def open_grid(self, max_width=300, box_css="display: flex;", col_css=""):
        """Start creating a flexbox grid to output several HTML contents side by side"""
        self.grid_max_width = max_width
        self.print(f"<div style='{box_css}'>", end="")
        self._open_grid_column(col_css=col_css)

    def _open_grid_column(self, col_css=""):
        width_css = ""
        if self.grid_max_width is not None:
            width_css = f"max-width:{self.grid_max_width}px;"
        self.print(f"<div style='margin-right:10px;{width_css} {col_css}'>", end="")

    def _close_grid_column(self):
        self.print(f"</div>", end="")

    def next_grid_column(self, col_css=""):
        self._close_grid_column()
        self._open_grid_column(col_css=col_css)

    def close_grid(self, output=True):
        self._close_grid_column()
        self.print(f"</div>", output=output, end="")


def display_html_table(
    lines, header_row_indices=(0,), scroll_height=None, font_size=1.0, table_other_style=""
):
    """

    Args:
        lines: table data in format [[row1_col1, ..., row1_colN], ..., [rowM_col1, ..., rowM_colN]]
        header_row_indices: which rows should be considered headers and printed bold
        scroll_height: maximum height of the table in pixels, if None, infinite
        font_size: multiplier for font size of the table
        table_other_style: other style attributes for the table
    """
    header_row_indices = set() if header_row_indices is None else set(header_row_indices)
    p1 = NotebookHTMLPrinter()
    p1.open_table(font_size=font_size, other_style=table_other_style)
    for i, l in enumerate(lines):
        p1.add_table_row(*l, is_header=i in header_row_indices)
    p1.close_table(output=False)
    html_content = p1.get_html()
    display_scrollable(html_content, scroll_height=scroll_height)


def display_html_table_from_dict(table_dict, **kwargs):
    """
    Args:
        table_dict: dict of format
            {header1: [row1_col1, ..., row1_colN], ..., headerM: [rowM_col1, ..., rowM_colN]}
        **kwargs: passed to display_html_table
    """
    lines = [[k] + v for k, v in table_dict.items()]
    display_html_table(lines, **kwargs)


def display_scrollable(html_content, scroll_height=200):
    if scroll_height is not None and scroll_height > 0:
        # Puts the scrollbar next to the content
        css_style = "height: 200px; overflow: auto; width: fit-content; resize: both;"
        display(HTML(f"<div style='{css_style}'>{html_content}</div>"))
    else:
        display(HTML(html_content))


def get_colored_html_text(color_tuple, *text_args, sep=" ", end="\n"):
    assert len(color_tuple) in [1, 3, 4] and [
        0 <= ca <= 255 for ca in color_tuple
    ], f"Invalid rgb tuple: {color_tuple}"
    if len(color_tuple) == 1:
        color_tuple = color_tuple * 3
    hex_str = "".join([f"{ca:02x}" for ca in color_tuple])
    args_joined = sep.join(text_args)
    text = f"<span style='color:#{hex_str};'>{args_joined}</span>{end}"
    text = text.replace("\n", "<br />")
    return text


def get_colored_html_text_from_lists(c_list, t_list, sep=""):
    assert len(c_list) == len(  #
        t_list
    ), f"got {len(c_list)} colors and {len(t_list)} texts: {c_list}, {t_list}"
    texts = []
    for c, t in zip(c_list, t_list):
        text = get_colored_html_text(c, t, sep="", end="")
        texts.append(text)
    full_text = sep.join(texts)
    return full_text


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


def convert_figure_to_base64(fig=plt):
    # todo test this if fig is actually not plt
    img = BytesIO()
    fig.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    fig.close()  # Close the plot so that it won't be displayed immediately
    return plot_url


def convert_base64_img_to_tag(base64_img):
    return f'<img src="data:image/png;base64,{base64_img}" />'


def convert_figure_to_html_tag(fig=plt):
    return convert_base64_img_to_tag(convert_figure_to_base64(fig=fig))
