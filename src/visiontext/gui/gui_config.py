from attr import define

PYGAME_UI_PRELOAD_FONTS = []
for point_size in [14, 18]:
    for style in ["", "bold", "italic", "bold_italic"]:
        PYGAME_UI_PRELOAD_FONTS += [
            {"name": "notosans", "point_size": point_size, "style": style},
            {"name": "notosansmono", "point_size": point_size, "style": style},
        ]


PYGAME_UI_THEME = {
    "defaults": {
        "colours": {
            "normal_bg": "#101010",  # button bg
            "hovered_bg": "#303030",  # button hover bg
            "disabled_bg": "#501010",  # probably disabled button bg
            "selected_bg": "#ffffff",  # ? maybe selectbox
            "dark_bg": "#101010",  # textbox bg
            "normal_text": "#a9b7c6",  # all text
            "hovered_text": "#ffffff",  # button hover text
            # "selected_text": "#FFFFFF",
            # "disabled_text": "#6d736f",
            # "link_text": "#0000EE",
            # "link_hover": "#2020FF",
            # "link_selected": "#551A8B",
            # "text_shadow": "#777777",
            # "normal_border": "#DDDDDD",
            # "hovered_border": "#B0B0B0",
            # "disabled_border": "#808080",
            # "selected_border": "#8080B0",
            # "active_border": "#8080B0",
            # "filled_bar": "#f4251b",
            # "unfilled_bar": "#CCCCCC",
        },
    },
    "button": {
        "normal": {"text": "Say Hello"},
        "hover": {"text": "Say Hello"},
        "selected": {"text": "Say Hello"},
        "selected_hover": {"text": "Say Hello"},
    },
    "@message_box": {
        "font": {
            "name": "notosans",
            "size": 14,
        }
    },
    "@fps_label": {
        "font": {
            "name": "notosansmono",
            "size": 14,
            "bold": 1,
            "italic": 0,
        },
        "misc": {
            "text_horiz_alignment": "left",
        },
    },
}


@define
class GuiConfig:
    width: int = 800
    height: int = 600
    window_title: str = "App"
    is_resizable: bool = False
    background_color: str = "#03041e"
    fps_update_every: float = 0.2
