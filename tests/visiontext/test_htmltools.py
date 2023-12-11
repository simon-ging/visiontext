import numpy as np
import re
from matplotlib import pyplot as plt

from visiontext.htmltools import convert_figure_to_html_tag


def test_figure_to_html_tag():
    plt.figure(figsize=(8, 6))
    plt.plot(np.sin(np.linspace(0, 20, 100)))
    plt.title("Sine Wave")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plot_tag = convert_figure_to_html_tag()
    # just check if the function runs and produces a sane tag, do not check the content
    check_re = re.compile('<img src="data:image/png;base64,[a-zA-Z0-9+/=]*" />')
    assert check_re.match(plot_tag), f"Tag does not match regex: {plot_tag} vs {check_re}"
