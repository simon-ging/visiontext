from visiontext.visualizer.colormaps import DEFAULT_COLOR_CYCLE, get_color_from_default_color_cycle


def test_color_cycles():
    assert DEFAULT_COLOR_CYCLE == [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    assert get_color_from_default_color_cycle(0) == DEFAULT_COLOR_CYCLE[0]
    assert get_color_from_default_color_cycle(
        1 + 5 * len(DEFAULT_COLOR_CYCLE)
    ) == get_color_from_default_color_cycle(1)
