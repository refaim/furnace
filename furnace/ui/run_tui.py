from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

import psutil
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import RichLog, Static

from furnace.core.models import (
    AudioAction,
    AudioInstruction,
    DownmixMode,
    Job,
    SubtitleAction,
    SubtitleInstruction,
)
from furnace.core.progress import TrackerSnapshot
from furnace.core.quality import final_output_dimensions
from furnace.core.target_quality import grain_uses_svt
from furnace.ui.fmt import fmt_size

BAR_WIDTH = 40


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _fmt_bitrate(bps: int | None) -> str:
    if bps is None or bps == 0:
        return ""
    kbps = bps // 1000
    if kbps > 0:
        return f"{kbps}kbps"
    return f"{bps}bps"


def _channel_layout_short(layout: str | None) -> str:
    if not layout:
        return ""
    return layout.split("(")[0]


_CH_LAYOUT_MAP = {1: "1.0", 2: "2.0", 6: "5.1", 8: "7.1"}

_AUDIO_STEP_TEMPLATES: dict[AudioAction, str] = {
    AudioAction.COPY: "Copy audio{num} ({codec}{ch})",
    AudioAction.DENORM: "Denorm audio{num} ({codec}{ch})",
    AudioAction.DECODE_ENCODE: "Recode audio{num} ({codec} -> AAC)",
    AudioAction.FFMPEG_ENCODE: "Recode audio{num} ({codec} -> AAC)",
}


_DOWNMIX_TARGET_CHANNELS: dict[DownmixMode, int] = {
    DownmixMode.MONO: 1,
    DownmixMode.STEREO: 2,
    DownmixMode.DOWN6: 6,
}


def _audio_step_label(instr: AudioInstruction, index: int, total: int) -> str:
    codec = instr.codec_name.upper()
    ch = ""
    if instr.channels:
        ch = " " + _CH_LAYOUT_MAP.get(instr.channels, f"{instr.channels}ch")

    num = f" {index + 1}" if total > 1 else ""

    if instr.action in (AudioAction.DECODE_ENCODE, AudioAction.FFMPEG_ENCODE) and instr.downmix is not None:
        tgt_ch = _DOWNMIX_TARGET_CHANNELS[instr.downmix]
        tgt_layout = _CH_LAYOUT_MAP.get(tgt_ch, f"{tgt_ch}ch")
        return f"Recode audio{num} ({codec}{ch} -> AAC {tgt_layout})"

    return _AUDIO_STEP_TEMPLATES[instr.action].format(num=num, codec=codec, ch=ch)


def _sub_step_label(instr: SubtitleInstruction, index: int, total: int) -> str:
    codec = instr.codec_name.upper()
    num = f" {index + 1}" if total > 1 else ""

    if instr.action == SubtitleAction.COPY_RECODE:
        return f"Recode subs{num} ({codec} -> UTF-8)"
    return f"Copy subs{num} ({codec})"


def _build_steps(job: Job) -> list[str]:
    steps: list[str] = []

    for i, audio_instr in enumerate(job.audio):
        steps.append(_audio_step_label(audio_instr, i, len(job.audio)))

    for i, sub_instr in enumerate(job.subtitles):
        steps.append(_sub_step_label(sub_instr, i, len(job.subtitles)))

    vp = job.video_params
    if not vp.passthrough:
        if vp.dv_mode is not None:
            steps.append("Extract DV RPU")
        steps.append("Search quality")
    steps.extend(["Encode video", "Assemble MKV", "Set metadata", "Optimize index"])
    return steps


def _build_source_text(job: Job) -> str:
    lines: list[str] = []

    vp = job.video_params
    codec = vp.source_codec.upper() or "?"
    bitrate_str = f" {_fmt_bitrate(vp.source_bitrate)}" if vp.source_bitrate else ""
    lines.append(f"Video: {codec} {vp.source_width}x{vp.source_height}{bitrate_str}")

    for i, audio_instr in enumerate(job.audio):
        codec = audio_instr.codec_name.upper()
        ch = _channel_layout_short(None)
        if audio_instr.channels:
            ch_map = {1: "1.0", 2: "2.0", 6: "5.1", 8: "7.1"}
            ch = ch_map.get(audio_instr.channels, f"{audio_instr.channels}ch")
        br = _fmt_bitrate(audio_instr.bitrate)
        prefix = "Audio:" if i == 0 else "      "
        parts = [p for p in [codec, ch, br] if p]
        lines.append(f"{prefix} {' '.join(parts)}")

    for i, sub_instr in enumerate(job.subtitles):
        codec = sub_instr.codec_name.upper()
        prefix = "Subs: " if i == 0 else "      "
        lines.append(f"{prefix} {codec} {sub_instr.language}")

    if job.source_size > 0:
        lines.append(f"Size:  {fmt_size(job.source_size)}")

    return "\n".join(lines)


def _target_channels(instr: AudioInstruction) -> int | None:
    if instr.downmix is not None:
        return _DOWNMIX_TARGET_CHANNELS[instr.downmix]
    return instr.channels


def _target_channel_layout(instr: AudioInstruction) -> str:
    ch = _target_channels(instr)
    if ch is None:
        return ""
    return _CH_LAYOUT_MAP.get(ch, f"{ch}ch")


_AUDIO_TARGET_PARTS: dict[AudioAction, Callable[[str], tuple[str, str]]] = {
    AudioAction.COPY: lambda src: (src, "(copy)"),
    AudioAction.DENORM: lambda src: (src, "(denorm)"),
    AudioAction.DECODE_ENCODE: lambda src: ("AAC", f"(from {src})"),
    AudioAction.FFMPEG_ENCODE: lambda src: ("AAC", f"(from {src})"),
}


def _audio_target_label(instr: AudioInstruction) -> str:
    src_codec = instr.codec_name.upper()
    layout = _target_channel_layout(instr)
    head, tag = _AUDIO_TARGET_PARTS[instr.action](src_codec)
    return " ".join(p for p in [head, layout, tag] if p)


def _sub_target_label(instr: SubtitleInstruction) -> str:
    codec = instr.codec_name.upper()
    if instr.action == SubtitleAction.COPY_RECODE:
        return f"{codec} {instr.language} (UTF-8)"
    return f"{codec} {instr.language}"


def _build_target_text(job: Job) -> str:
    lines: list[str] = []

    vp = job.video_params
    if vp.passthrough:
        codec = vp.source_codec.upper() or "?"
        lines.append(f"Video: {codec} {vp.source_width}x{vp.source_height} (copy)")
    else:
        final_w, final_h = final_output_dimensions(vp)
        knob = "CRF" if grain_uses_svt(vp) else "QVBR"
        value = str(job.chosen_cq) if job.chosen_cq is not None else "target"
        lines.append(f"Video: AV1 {final_w}x{final_h} {knob} {value}")

    for i, audio_instr in enumerate(job.audio):
        prefix = "Audio:" if i == 0 else "      "
        lines.append(f"{prefix} {_audio_target_label(audio_instr)}")

    for i, sub_instr in enumerate(job.subtitles):
        prefix = "Subs: " if i == 0 else "      "
        lines.append(f"{prefix} {_sub_target_label(sub_instr)}")

    return "\n".join(lines)


class HeaderWidget(Static):
    DEFAULT_CSS = """
    HeaderWidget {
        height: 1;
        background: $surface;
        padding: 0 1;
        color: $text;
        text-style: bold;
    }
    """


class SourceWidget(Static):
    DEFAULT_CSS = """
    SourceWidget {
        width: 50%;
        border: double $primary;
        padding: 0 1;
        height: auto;
    }
    """


class TargetWidget(Static):
    DEFAULT_CSS = """
    TargetWidget {
        width: 50%;
        border: double $primary;
        padding: 0 1;
        height: auto;
    }
    """


class StepsWidget(Static):
    DEFAULT_CSS = """
    StepsWidget {
        width: 36;
        border: double $primary;
        padding: 0 1;
        height: 1fr;
    }
    """


class OutputLog(RichLog):
    DEFAULT_CSS = """
    OutputLog {
        border: double $primary;
        height: 1fr;
    }
    """


class ProgressWidget(Static):
    DEFAULT_CSS = """
    ProgressWidget {
        height: 2;
        padding: 0 1;
    }
    """


class RunApp(App[None]):
    CSS = """
    #source-target {
        height: auto;
        max-height: 10;
    }
    #middle {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("escape", "quit_app", "Quit", show=True),
        Binding("ctrl+q", "quit_app", "Quit", show=False),
    ]

    def __init__(
        self,
        total_jobs: int,
        shutdown_event: threading.Event,
        executor_fn: Callable[..., None],
    ) -> None:
        super().__init__()
        self._total_jobs = total_jobs
        self._shutdown_event = shutdown_event
        self._executor_fn = executor_fn

        self._job: Job | None = None
        self._job_idx = 0
        self._steps: list[str] = []
        self._current_step_idx: int = -1
        self._snapshot: TrackerSnapshot | None = None
        self._start_time = 0.0
        self._target_base_text = ""
        self._output_size = 0
        self._tool_output_muted = False

    def compose(self) -> ComposeResult:
        yield HeaderWidget("Waiting...", id="header")
        yield Horizontal(
            SourceWidget("", id="source"),
            TargetWidget("", id="target"),
            id="source-target",
        )
        yield Horizontal(
            StepsWidget("", id="steps"),
            OutputLog(id="output", auto_scroll=True, max_lines=5000),
            id="middle",
        )
        yield ProgressWidget("", id="progress")

    def on_mount(self) -> None:
        self.run_worker(self._run_executor, thread=True)

    def _run_executor(self) -> None:
        self._executor_fn(self)

    def action_quit_app(self) -> None:
        self._shutdown_event.set()
        parent = psutil.Process(os.getpid())
        for child in parent.children(recursive=True):
            with contextlib.suppress(psutil.NoSuchProcess):
                child.kill()
        os._exit(0)

    def _safe_call(self, fn: Callable[..., object], *args: object) -> None:
        with contextlib.suppress(Exception):
            self.call_from_thread(fn, *args)

    def start_job(self, job: Job, job_index: int) -> None:
        self._safe_call(self._do_start_job, job, job_index)

    def update_progress(self, snapshot: TrackerSnapshot) -> None:
        self._safe_call(self._do_update_progress, snapshot)

    def update_status(self, message: str) -> None:
        self._safe_call(self._do_update_status, message)

    def add_tool_line(self, line: str) -> None:
        self._safe_call(self._do_add_tool_line, line)

    def tool_output(self, line: str) -> None:
        if self._tool_output_muted:
            return
        self._safe_call(self._do_add_tool_line, line)

    def mute_tool_output(self) -> None:
        self._tool_output_muted = True

    def unmute_tool_output(self) -> None:
        self._tool_output_muted = False

    def finish_job(self, job: Job) -> None:
        self._safe_call(self._do_finish_job, job)

    def update_output_size(self, size_bytes: int) -> None:
        self._safe_call(self._do_update_output_size, size_bytes)

    def set_chosen_quality(self, cq: int) -> None:
        self._safe_call(self._do_set_chosen_quality, cq)

    def stop(self) -> None:
        self._safe_call(self.exit)

    def _do_start_job(self, job: Job, job_index: int) -> None:
        self._job = job
        self._job_idx = job_index
        self._snapshot = None
        self._start_time = time.monotonic()

        filename = Path(job.output_file).name
        header = self.query_one("#header", HeaderWidget)
        header.update(f"[{job_index + 1}/{self._total_jobs}] {filename}")

        source_w = self.query_one("#source", SourceWidget)
        source_w.update(_build_source_text(job))

        self._target_base_text = _build_target_text(job)
        self._output_size = 0
        self._render_target()

        self._steps = _build_steps(job)
        self._current_step_idx = -1
        self._refresh_steps()

        output_log = self.query_one("#output", OutputLog)
        output_log.clear()

        progress_w = self.query_one("#progress", ProgressWidget)
        progress_w.update("")

    def _do_update_progress(self, snapshot: TrackerSnapshot) -> None:
        self._snapshot = snapshot
        self._refresh_progress()

    def _do_update_status(self, message: str) -> None:
        self._snapshot = None
        self._start_time = time.monotonic()

        if self._current_step_idx < len(self._steps) - 1:
            self._current_step_idx += 1
            self._refresh_steps()

        progress_w = self.query_one("#progress", ProgressWidget)
        progress_w.update(message)

    def _do_add_tool_line(self, line: str) -> None:
        output_log = self.query_one("#output", OutputLog)
        output_log.write(line)

    def _do_update_output_size(self, size_bytes: int) -> None:
        self._output_size = size_bytes
        self._render_target()

    def _do_set_chosen_quality(self, cq: int) -> None:
        if self._job is None:
            return
        self._job.chosen_cq = cq
        self._target_base_text = _build_target_text(self._job)
        self._render_target()

    def _render_target(self) -> None:
        size_str = fmt_size(self._output_size) if self._output_size > 0 else "..."
        target_w = self.query_one("#target", TargetWidget)
        target_w.update(f"{self._target_base_text}\nSize:  {size_str}")

    def _do_finish_job(self, job: Job) -> None:
        self._job = job
        self._snapshot = None

        self._current_step_idx = len(self._steps)
        self._refresh_steps()

        progress_w = self.query_one("#progress", ProgressWidget)
        progress_w.update("Done")

    def _refresh_steps(self) -> None:
        lines: list[str] = []
        for i, step in enumerate(self._steps):
            if i < self._current_step_idx:
                lines.append(f"+ {step}")
            elif i == self._current_step_idx:
                lines.append(f"> {step}")
            else:
                lines.append(f"  {step}")

        steps_w = self.query_one("#steps", StepsWidget)
        steps_w.update("\n".join(lines))

    def _refresh_progress(self) -> None:
        if self._snapshot is None:
            return

        frac = self._snapshot.fraction
        filled = int(BAR_WIDTH * frac)
        bar = "\u2588" * filled + "\u2591" * (BAR_WIDTH - filled)

        elapsed = time.monotonic() - self._start_time
        eta_s = self._snapshot.eta_s
        time_part = f"{_fmt_time(elapsed)} / ~{_fmt_time(eta_s)}" if eta_s is not None else _fmt_time(elapsed)

        speed_part = ""
        if self._snapshot.speed is not None:
            speed_part = f" | {self._snapshot.speed:.1f}x"

        text = f"{bar} {frac * 100:5.1f}% | {time_part}{speed_part}"

        progress_w = self.query_one("#progress", ProgressWidget)
        progress_w.update(text)
