import psutil
from loguru import logger
from pyinstrument import Profiler

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
        current_profiler.open_in_browser()
    return text

