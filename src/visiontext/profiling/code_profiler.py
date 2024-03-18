from pathlib import Path

from loguru import logger
from pyinstrument import Profiler, renderers

from packg.dtime import get_timestamp_for_filename

current_profiler: Profiler = None


def start_pyinstrument_profiler():
    global current_profiler
    if current_profiler is not None:
        logger.warning("Profiler already running!")
        return
    current_profiler = Profiler()
    current_profiler.start()


def stop_pyinstrument_profiler(
    open_in_browser=True, output_text=True, print_fn=print, unicode=True, color=True
) -> str:
    if current_profiler is None:
        logger.error("No profiler running!")
        return ""
    current_profiler.stop()
    text = current_profiler.output_text(unicode=unicode, color=color)
    if output_text:
        print_fn(text)
    if open_in_browser:
        # in ubuntu, firefox can't access /tmp - hack into pyinstrument and save to home
        # current_profiler.open_in_browser()  # <- doesn't work because it saves to /tmp/XXX.html
        tf = (Path.home() / f"py_profile_{get_timestamp_for_filename()}.html").as_posix()
        session = current_profiler._get_last_session_or_fail()  # noqa
        return renderers.HTMLRenderer(timeline=False).open_in_browser(session, output_filename=tf)
    return text
