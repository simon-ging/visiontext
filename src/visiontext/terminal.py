"""
Utilities for terminals
"""

import termcolor


class ColorFormatter:
    def __init__(self, theme="dark"):
        if theme is None:
            theme = "none"
        themes = ["dark", "light", "none"]
        assert theme in themes, f"Unknown color theme {theme}, should be any of {themes}"
        self.theme = theme

    def fmt_bold(self, inp):
        if self.theme == "none":
            return inp
        return termcolor.colored(inp, attrs=["bold"])

    def fmt_correct(self, inp):
        if self.theme == "none":
            return inp
        if self.theme == "dark":
            return termcolor.colored(inp, "light_green")
        return termcolor.colored(inp, "green")

    def fmt_wrong(self, inp):
        if self.theme == "none":
            return inp
        if self.theme == "dark":
            return termcolor.colored(inp, "light_red")
        return termcolor.colored(inp, "red")

    def fmt_neutral(self, inp):
        if self.theme == "none":
            return inp
        return inp
        # return termcolor.colored(inp, "white")

    def fmt_underline(self, inp):
        if self.theme == "none":
            return inp
        return termcolor.colored(inp, attrs=["underline"])

    def fmt_grey(self, inp):
        if self.theme == "none":
            return inp
        if self.theme == "dark":
            return termcolor.colored(inp, "light_grey")
        return termcolor.colored(inp, "grey")
