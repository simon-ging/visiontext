"""
Video helpers for dataloading: extract frames from video bytes with ffmpeg, and write a list of
frames as a video.

requires ffmpeg installation on the system, either via apt-get or conda-forge

pip install memory-tempfile ffmpeg-python numpy opencv-python
"""

import os
import tempfile
from math import floor
from pathlib import Path
from subprocess import PIPE, Popen
from typing import List, Tuple, Union

import ffmpeg
import memory_tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class FrameExtractor:
    def __init__(self):
        self.tempf, in_memory = get_memory_tempfile_factory()
        self.font_file = get_default_font_file(ignore_errors=True)
        if not in_memory:
            print(f"WARN: Temporary mp4s will be written to disk instead of memory (slow)")

    def _open_file(self, video_bytes: bytes):
        # create temporary file in order to make the content seekable for ffmpeg
        fh = self.tempf.NamedTemporaryFile(mode="wb", buffering=0, delete=False)
        filename = fh.name
        assert filename is not None, f"Got no name for {fh}"
        fh.write(video_bytes)
        fh.close()
        return fh, filename

    def _read_ffmpeg_output(self, process, ffmpeg_cmd_display=""):
        out, err = process.communicate()
        if len(out) == 0:
            raise RuntimeError(
                f"Got empty ffmpeg output\n\n{ffmpeg_cmd_display}\n\n{err.decode('utf-8')}"
            )
        return out

    def extract_approx_frames_from_bytes(
        self,
        video_bytes: bytes,
        start_time: float,
        n_frames: int,
        fps: float,
        h: int,
        w: int,
        input_format: str = "mp4",
        plot_frame_time=False,
        verbose=False,
    ):
        """
        The ffmpeg filter will sample the input video at the given fps and create
        n_frames output frames (or stop earlier when the video is done).
        Frames are center sampled. Given start_time and fps, frame i will be sampled as:
            target_time = (start_time + (0.5 + i) / fps )
            frame_source_index = argmin(frame_time - target_time) s.t. frame_time < target_time
        So the frame at or just before that timestamp will be selected.

        For  the ffmpeg drawtext filter see https://ffmpeg.org/ffmpeg-filters.html#drawtext-1
        """
        fh, filename = self._open_file(video_bytes)
        out_video = self.extract_approx_frames_from_file(
            filename, start_time, n_frames, fps, h, w, input_format, plot_frame_time, verbose
        )
        os.unlink(filename)
        return out_video

    def extract_approx_frames_from_file(
        self,
        filename: Union[str, Path],
        start_time: float,
        n_frames: int,
        fps: float,
        h: int,
        w: int,
        input_format: str = "mp4",
        plot_frame_time=False,
        verbose=False,
    ):
        ss = convert_seconds_to_ffmpeg_hms(start_time)
        drawtext_str = ""
        if plot_frame_time:
            start_time_real = start_time + 0.5 / fps
            drawtext_str = (
                f",drawtext=fontfile={self.font_file}:"
                r"text='%{pts\:hms\:"
                f"{start_time_real}"
                "}':"
                "x=(w-tw)/2:y=h-(2*lh):fontcolor=white:box=1:boxcolor=0x00000000@1:"
                "fontsize=30"
            )
        ffmpeg_cmd = (
            f"ffmpeg -f {input_format} -ss {ss} -i {filename} "
            f"-vf fps={fps}{drawtext_str} "
            f"-f rawvideo -pix_fmt rgb24 -vframes {n_frames} -vsync 0 pipe:"
        )
        if verbose:
            print(ffmpeg_cmd)
        with Popen(ffmpeg_cmd.split(), stdout=PIPE, stderr=PIPE) as process:
            out = self._read_ffmpeg_output(process, ffmpeg_cmd)
        out_video = np.frombuffer(out, np.uint8).reshape((-1, h, w, 3))
        return out_video

    def _OLD_extract_indexed_frames_from_bytes(
        self,
        video_bytes: bytes,
        frame_indices: List[int],
        h: int,
        w: int,
        input_format: str = "mp4",
    ):
        # reading input at e.g. 30 FPS and then selecting some frames is extremely slow
        fh, filename = self._open_file(video_bytes)
        stream = ffmpeg.input(filename, format=input_format)
        # select frames given index frames (very slow)
        stream = ffmpeg.filter(stream, "select", "+".join([f"eq(n,{f})" for f in frame_indices]))
        stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt="rgb24", vsync=0)
        out_video = _run_stream(stream, video_bytes, h, w)
        fh.close()
        if out_video.shape[0] < len(frame_indices):
            # ffmpeg automatically removes duplicate frames given the selector here
            # so we have to re-add them to get the expected number of frames
            _, dup_index = np.unique(frame_indices, return_inverse=True)
            out_video = out_video[dup_index]
        return out_video


def convert_seconds_to_ffmpeg_hms(seconds):
    sec_floor, remainder = divmod(seconds, 1)
    sec_floor = round(sec_floor)
    m, s = divmod(sec_floor, 60)
    h, m = divmod(m, 60)
    remainder_ms = floor(remainder * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{remainder_ms}"


def _run_stream(stream, video_bytes, h, w):
    cmd = f"ffmpeg {' '.join(stream.get_args())}"

    stream = ffmpeg.run_async(
        stream, pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True
    )
    out, err = stream.communicate(input=video_bytes)
    if len(out) == 0:
        raise RuntimeError(f"Got empty ffmpeg output\n\n{cmd}\n\n{err.decode('utf-8')}")
    out_video = np.frombuffer(out, np.uint8).reshape((-1, h, w, 3))
    return out_video


def get_default_font_file(ignore_errors=False) -> str:
    """
    font to draw frametime onto the videos, useful for debugging if the video is loaded correctly
    """
    possible_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for file in possible_paths:
        if Path(file).is_file():
            return Path(file).as_posix()
    if ignore_errors:
        return ""

    raise FileNotFoundError(f"Could not find any font file from the list: {possible_paths}")


def get_memory_tempfile_factory() -> Tuple[memory_tempfile.MemoryTempfile, bool]:
    if os.name == "nt":
        # noinspection PyTypeChecker
        return tempfile, False
    tempfile_factory = memory_tempfile.MemoryTempfile()
    return tempfile_factory, tempfile_factory.found_mem_tempdir()


# ---------- video creation ----------


def create_video_from_frames_moviepy(
    vid_file: Union[str, Path], frames: List[np.ndarray], fps: int
):
    """Create an mp4 video from frames using moviepy 2. Needs ffmpeg, which moviepy brings
    along through imageio-ffmpeg."""
    from moviepy import VideoClip

    duration_sec = len(frames) / fps

    def make_frame(t):
        frame_num = int(np.round(t * fps))
        # saveguard against loading 1 frame too much
        if frame_num == len(frames):
            frame_num -= 1
        return frames[frame_num]

    clip = VideoClip(make_frame, duration=duration_sec)
    clip.write_videofile(Path(vid_file).as_posix(), fps=float(fps), logger=None)
    clip.close()


def create_video_from_frames_cv2(vid_file: Union[str, Path], frames: List[np.ndarray], fps: int):
    h, w = frames[0].shape[:2]
    print(f"Write video {vid_file}, {h}x{w} px, {fps} fps, {len(frames)} frames")
    import cv2

    output = cv2.VideoWriter(
        Path(vid_file).as_posix(), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for _i, frame in enumerate(frames):
        # print(i, frame.dtype, frame.shape, frame.min(), frame.max())
        # cv2 expects BGR instead of RGB
        output.write(frame[:, :, ::-1])
    output.release()


def create_test_video(test_vid_file, test_dur=10, test_fps=30, h=240, w=320):
    # pil_font = ImageFont.truetype(get_default_font_file(), 40)
    # only works with integer fps
    pil_font = ImageFont.load_default()
    frames = []
    framenum = 0
    while True:
        total_dur = framenum / test_fps
        if total_dur >= test_dur:
            break
        img = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(img)
        for x in range(0, w, 20):
            for y in range(0, h, 20):
                draw.rectangle((x, y, x + 19, y + 19), fill=None, outline=(255, 0, 0))
        draw.text(
            (20, 20), f"{framenum:>3d}: {total_dur:.3f}s", fill=(255, 255, 255), font=pil_font
        )
        # noinspection PyTypeChecker
        arr = np.array(img)
        frames.append(arr)
        framenum += 1

    create_video_from_frames_cv2(test_vid_file, frames, test_fps)
