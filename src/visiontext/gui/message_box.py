import logging
from timeit import default_timer

from attr import field
from attrs import define
from loguru import logger

from packg.dtime import format_timestamp
from packg.log import get_level_as_str


@define
class MessageQueue:
    max_messages: int = 10
    max_time: float = 0.0
    scrollback_buffer: int = 100000
    show_only_max: bool = True
    last_text: str = field(init=False)
    queue: list[tuple[str, float]] = field(init=False)
    should_update: bool = field(init=False)

    def __attrs_post_init__(self):
        self.reset()

    def reset(self):
        self.queue = []
        self.last_text = None
        self.should_update = True

    def info(self, message: str, *args, **kwargs):
        self.log(logging.INFO, message, *args, **kwargs)

    def log(self, level: int | str, message: str, *args, **kwargs):
        # TODO care about level
        level_str = get_level_as_str(level)
        logger.log(level_str, message, *args, **kwargs)
        emit_time = default_timer()
        dtime = format_timestamp(format_str="%H:%M:%S")
        message_fmt = " ".join((dtime, message.format(*args, **kwargs)))
        self.queue.append((message_fmt, emit_time))
        # start deleting messages once they get in MB range of text
        if len(self.queue) > self.scrollback_buffer > 0:
            self.queue.pop(0)
        self.should_update = True

    def get_text(self) -> str:
        if not self.should_update:
            return self.last_text, False
        now = default_timer()
        messages, to_delete_indices = [], []
        for i, (message, emit_time) in enumerate(self.queue):
            if now - emit_time > self.max_time > 0:
                continue
            messages.append(message)
        if self.show_only_max:
            if len(messages) > self.max_messages > 0:
                messages = messages[-self.max_messages :]
            show_all_text = f"{len(messages)}/"
        else:
            show_all_text = ""
        text = "<br />".join(
            [f"Showing <b>{show_all_text}{len(self.queue)}</b> messages."] + messages[::-1]
        )
        self.last_text = text
        self.should_update = False
        return text, True
