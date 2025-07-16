import os
import urllib
import urllib.parse
import webbrowser
from pathlib import Path

from loguru import logger
from pyinstrument import Profiler, renderers

from packg.dtime import get_timestamp_for_filename
from packg.misc import format_exception
from packg.typext import PathType

current_profiler: Profiler = None


def start_pyinstrument_profiler():
    global current_profiler
    if current_profiler is not None:
        logger.warning("Profiler already running!")
        return
    current_profiler = Profiler()
    current_profiler.start()


def stop_pyinstrument_profiler(
    open_in_browser=True,
    output_text=True,
    print_fn=print,
    unicode=True,
    color=True,
    output_dir: PathType = Path.home(),  # browser maybe can't access /tmp, so use home as default
) -> str:
    if current_profiler is None:
        logger.error("No profiler running!")
        return ""
    current_profiler.stop()

    # setup output dir
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # profile to text
    output_name = f"py_profile_{get_timestamp_for_filename()}"
    text = current_profiler.output_text(unicode=unicode, color=color)
    output_file_text = output_dir / f"{output_name}.txt"
    output_file_text.write_text(text, encoding="utf-8")
    if output_text:
        print_fn(text)

    # profile to HTML
    session = current_profiler._get_last_session_or_fail()  # noqa
    html_render = renderers.HTMLRenderer(timeline=False)
    html_render_str = html_render.render(session)
    output_file_html = output_dir / f"{output_name}.html"
    Path(output_file_html).write_text(html_render_str, encoding="utf-8")
    print(f"Saved profiler text output to {output_file_text} and HTML output to {output_file_html}")
    if open_in_browser:
        try:
            url = urllib.parse.urlunparse(("file", "", output_file_html, "", "", ""))
            webbrowser.open(url)
        except Exception as e:
            logger.warning(f"Failed to open profiler in browser: {format_exception(e)}")

    return text
