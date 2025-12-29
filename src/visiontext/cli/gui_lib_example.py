"""
Simple example on how to use visiontext gui library
"""

import os
import re
import threading
import time

import pygame
import pygame_gui
from attrs import define
from loguru import logger
from pygame.event import Event, post

from packg import format_exception
from packg.iotools import loads_json
from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from packg.system import systemcall
from typedparser import TypedParser, VerboseQuietArgs, add_argument
from visiontext.gui.gui_base import GuiBase
from visiontext.gui.gui_config import GuiConfig
from visiontext.gui.gui_modules import GuiModule
from visiontext.profiling import start_pyinstrument_profiler, stop_pyinstrument_profiler

from gutil.web.check_ip import find_own_ip


@define
class Args(VerboseQuietArgs):
    profiling: bool = add_argument(
        shortcut="-p", action="store_true", help="Run with pyinstrument profiler"
    )


def main():
    print(f"GUI {get_context()}")
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")
    if args.profiling:
        start_pyinstrument_profiler()
    config = GuiConfig(window_title="ExampleGui")
    gui = GuiBase(config)
    manager = gui.ui_manager

    vpn_module = ExampleModule(run_every=60.0)

    # example button
    hello_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((350, 475), (100, 50)), text="Say Hello", manager=manager
    )
    vpn_textbox = pygame_gui.elements.UITextBox(
        relative_rect=pygame.Rect((300, 0), (270, 300)),
        html_text="testor",
        manager=manager,
        object_id=pygame_gui.core.ObjectID(
            object_id="#message_box",
            class_id="@message_box",
        ),
    )
    try:
        while gui.is_running:
            gui.step()
            for event in pygame.event.get():
                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == hello_button:
                        gui.start_event("say_hello", say_hello_fn)
                gui.process_events(event)
            vpn_module.maybe_run(gui.current_time, gui, vpn_textbox)
            pygame.display.update()
    except KeyboardInterrupt as e:
        logger.error(f"{format_exception(e)}, exiting")

    if args.profiling:
        stop_pyinstrument_profiler()


def say_hello_fn():
    time.sleep(0.2)
    print("Half done")
    post(Event(pygame.USEREVENT, {"task_name": "say_hello", "message": "Half done"}))
    time.sleep(0.2)
    print("Done")
    post(Event(pygame.USEREVENT, {"task_name": "say_hello", "message": "Done"}))


class ExampleModule(GuiModule):
    def run(self, gui: GuiBase, module_textbox: pygame_gui.elements.UITextBox):
        gui.start_event(type(self).__name__, example_background_fn, on_complete=self.done)
        self.module_textbox = module_textbox

    def done(self, event):
        print(f"foreground done: {get_context()} {event.output}")
        self.module_textbox.html_text = f"<b>{event.output['payload']}</b>"
        console_text = self.module_textbox.html_text.replace("<br>", "\n")
        logger.info(f"Text:\n{console_text}")
        self.module_textbox.rebuild()


def example_background_fn():
    time.sleep(1.0)
    print(f"example_background_fn running in {get_context()}")
    return {"payload": "done"}


def get_context():
    return f"pid={os.getpid()} thread={threading.get_ident()}"


if __name__ == "__main__":
    main()
