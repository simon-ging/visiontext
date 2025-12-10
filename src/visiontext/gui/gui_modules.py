import math

import pygame_gui


class GuiModule:
    def __init__(self, run_every: float):
        self.run_every = run_every
        self.last_run = -32767.0

    def maybe_run(self, current_time: float, *args, **kwargs):
        if current_time - self.last_run >= self.run_every:
            self.last_run = current_time
            self.run(*args, **kwargs)

    def run(self, *args, **kwargs):
        raise NotImplementedError()


class FpsCounter(GuiModule):
    def __init__(self, run_every: float):
        super().__init__(run_every)
        self.busy_percent_ema = EmaMeter()
        self.new_fps_ema = EmaMeter()

    def run(
        self,
        fps_label: pygame_gui.elements.UILabel,
        new_fps: float,
        current_time: float,
        busy_percent: float,
    ):
        """
        Update the timer label

        Args:
            fps_label: pygame UILabel to write the info text
            new_fps: current fps in 1/sec,  output from clock.get_fps()
            current_time: time since start in sec, output from default_timer() - start_time
            busy_percent: percentage of time not idle
        """
        # tms = math.floor((current_time * 1000) % 1000)
        ts = math.floor(current_time) % 60
        tm = math.floor(current_time / 60) % 60
        th = math.floor(tm / 60) % 24
        td = math.floor(th / 24)
        current_time_str = f"{td:d}-{th:02d}:{tm:02d}:{ts:02d}"  # .{tms:03d}"
        busy_percent = self.busy_percent_ema.update(busy_percent)
        new_fps = self.new_fps_ema.update(new_fps)
        fps_label.set_text(f"{round(new_fps):>3d} {busy_percent:>4.0%} Time: {current_time_str}")


class EmaMeter:
    def __init__(self, ema: float = 0.9):
        self.ema = ema
        self.value = None

    def update(self, value: float):
        if self.value is None:
            self.value = value
        else:
            self.value = self.value * self.ema + value * (1 - self.ema)
        return self.value

    def get(self):
        return self.value
