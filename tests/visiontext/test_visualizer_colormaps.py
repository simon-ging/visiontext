from visiontext.colormaps import DEFAULT_COLOR_CYCLE, get_color_from_default_color_cycle


def test_color_cycles():
    assert len(DEFAULT_COLOR_CYCLE) > 0
    assert get_color_from_default_color_cycle(0) == DEFAULT_COLOR_CYCLE[0]
    n = len(DEFAULT_COLOR_CYCLE)
    assert get_color_from_default_color_cycle(1 + 5 * n) == get_color_from_default_color_cycle(1)
