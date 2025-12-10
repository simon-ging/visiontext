import io
from threading import Thread
from timeit import default_timer

import pygame
import pygame_gui
from loguru import logger
from pygame import Event
from pygame.event import post
from pygame_gui.core import ObjectID
from pygame_gui.elements import UIButton, UILabel, UITextBox

from packg.iotools import dumps_json

from .gui_config import PYGAME_UI_PRELOAD_FONTS, PYGAME_UI_THEME, GuiConfig
from .gui_modules import FpsCounter
from .message_box import MessageQueue


class GuiBase:
    config: GuiConfig

    def __init__(self, config: GuiConfig):
        pygame.init()
        pygame.display.set_caption(config.window_title)
        flags = pygame.RESIZABLE if config.is_resizable else 0
        window_surface = pygame.display.set_mode((config.width, config.height), flags)
        logger.info(f"Init pygame-gui, {type(window_surface)}")
        manager = pygame_gui.UIManager(
            (config.width, config.height), io.StringIO(dumps_json(PYGAME_UI_THEME))
        )
        # preload fonts to solve "UserWarning: Finding font with id: ... that is not already loaded."
        manager.preload_fonts(PYGAME_UI_PRELOAD_FONTS)

        self.x_button: UIButton = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((config.width - 40, 0), (40, 40)), text="X", manager=manager
        )

        # timer label
        self.fps_label_background = pygame.Surface((270, 30))
        # set background color to theme "normal_bg" color
        self.fps_label_background.fill(manager.ui_theme.get_colour("normal_bg"))
        self.fps_label: UILabel = UILabel(
            relative_rect=pygame.Rect((5, 0), (260, 30)),
            text="Booting...",
            manager=manager,
            object_id=ObjectID(
                object_id="#fps_label",
                class_id="@fps_label",
            ),
        )

        # message box
        self.message_box: UITextBox = UITextBox(
            relative_rect=pygame.Rect((0, 30), (270, 300)),
            html_text="",
            manager=manager,
            object_id=ObjectID(
                object_id="#message_box",
                class_id="@message_box",
            ),
        )
        self.message_q = MessageQueue()
        self.message_box_checkbox: UIButton = UIButton(
            relative_rect=pygame.Rect((0, 330), (270, 30)),
            text="[ ] show all",
            manager=manager,
        )

        self.start_time: float = default_timer()
        self.message_q.info(f"Starting <b>{config.window_title}</b>")

        self.ui_manager: pygame_gui.UIManager = manager
        self.window_surface: pygame.Surface = window_surface

        self.is_running: bool = True
        self.clock: pygame.time.Clock = pygame.time.Clock()

        self.fps_counter = FpsCounter(config.fps_update_every)
        self.running_events = {}
        self.actual_width, self.actual_height = config.width, config.height
        self.num_steps = 0
        self.config = config
        self.reset_background()

    def reset_background(self):
        self.background = pygame.Surface((self.actual_width, self.actual_height))
        self.background.fill(pygame.Color(self.config.background_color))

    def step(self):
        clock = self.clock
        time_delta = clock.tick(60) / 1000.0
        time_used = clock.get_rawtime()
        busy_percent = time_used / time_delta / 1000
        new_fps = clock.get_fps()
        current_time = default_timer() - self.start_time
        if new_fps > 1.0 or current_time > 1.0:
            # ignore the first few frames with 0 FPS for the FPS counter
            self.fps_counter.maybe_run(
                current_time, self.fps_label, new_fps, current_time, busy_percent
            )

        # message box update, TODO move all this to the message box class
        message_str, should_update = self.message_q.get_text()
        message_box = self.message_box
        if should_update:
            height_adjustment = None
            if message_box.scroll_bar is not None:
                # height adjustment is the absolute scroll of the text
                height_adjustment = (
                    message_box.scroll_bar.start_percentage
                    * message_box.text_box_layout.layout_rect.height
                )
                # the easiest implementation (current) is to show the last message at the top,
                # and keep the height_adjustment fixed between updates
            message_box.set_text(message_str)
            if height_adjustment is not None and message_box.scroll_bar is not None:
                new_start_percentage = (
                    height_adjustment / message_box.text_box_layout.layout_rect.height
                )
                message_box.scroll_bar.set_scroll_from_start_percentage(new_start_percentage)

        self.ui_manager.update(time_delta)
        self.window_surface.blit(self.background, (0, 0))
        self.window_surface.blit(self.fps_label_background, (0, 0))
        self.ui_manager.draw_ui(self.window_surface)
        self.num_steps += 1
        self.current_time = current_time

    def process_events(self, event: Event):
        if event.type == pygame.QUIT:
            self.is_running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                self.is_running = False
        if event.type == pygame.WINDOWRESIZED:
            self.actual_width, self.actual_height = event.x, event.y
            self.message_q.info(f"Window resized to {self.actual_width}x{self.actual_height}")
            self.reset_background()
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.x_button:
                self.is_running = False
            if event.ui_element == self.message_box_checkbox:
                self.message_q.show_only_max = not self.message_q.show_only_max
                self.message_q.should_update = True
                self.message_box_checkbox.set_text(
                    "[ ] show all" if self.message_q.show_only_max else "[X] show all"
                )

        if event.type == pygame.USEREVENT and hasattr(event, "task_name"):
            task_name = event.task_name
            if hasattr(event, "event"):
                # async event handling
                if event.event == "done":
                    on_complete = self.running_events[task_name].get("on_complete")
                    if on_complete is not None:
                        on_complete(event)
                    del self.running_events[task_name]
                    logger.debug(f"Task complete: {task_name}")
            if hasattr(event, "message"):
                # message event, display in console and in the message box
                message = event.message
                self.message_q.info(f"{task_name}: {message}")
        self.ui_manager.process_events(event)

    def start_event(self, task_name: str, task_fn: callable, *args, on_complete=None, **kwargs):
        if task_name in self.running_events:
            logger.warning(f"Task {task_name} already running")
            return
        self.running_events[task_name] = {"running": True, "on_complete": on_complete}
        logger.debug(f"Task started: {task_name}")
        Thread(target=run_task, args=(task_name, task_fn, *args), kwargs=kwargs).start()


def run_task(task_name: str, task_fn: callable, *args, **kwargs):
    output = task_fn(*args, **kwargs)
    post(Event(pygame.USEREVENT, {"task_name": task_name, "event": "done", "output": output}))
