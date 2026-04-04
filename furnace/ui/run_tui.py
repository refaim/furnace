"""Textual TUI for the furnace run (encoding) phase.

Replaces Rich Live with a full Textual App that owns the terminal,
eliminating logging conflicts in cmd.exe.  ASCII-only borders
throughout for Windows compatibility.

Layout:
    +-- [1/3] Movie Name (2020) ----------------------------------+
    +-- Source -------------------------+-- Target ----------------+
    | Video: H.264 1920x1080 8.5Mbps   | Video: HEVC 1920x800 CQ25|
    | Audio: DTS 5.1 755kbps           | Audio: DTS 5.1 (denorm)  |
    +-----------------------------------+--------------------------+
    +-- Steps ------+-- Output ------------------------------------+
    |   Extract     |  Running in normal mode ...                   |
    | > Denormalize |  Creating file "audio_2_denorm.dts"...        |
    +---------------+----------------------------------------------+
    +-- ████████░░░░░░ 42.5% | 3:20 / ~4:40 | 1.2x ---------------+
"""
from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import RichLog, Static

from furnace.core.models import (
    AudioAction,
    AudioInstruction,
    Job,
    SubtitleAction,
    SubtitleInstruction,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Progress line patterns from various tools
_EAC3TO_RE = re.compile(r"^process:\s*(\d+)%$")              # eac3to: "process: 42%"
_MKVMERGE_RE = re.compile(r"^Progress:\s*(\d+)%$")            # mkvmerge: "Progress: 42%"
_MKCLEAN_RE = re.compile(r"^Progress\s+(\d)/3:\s*(\d+)%$")    # mkclean: "Progress 1/3:   42%"
_QAAC_RE = re.compile(r"^\[(\d+(?:\.\d+)?)%\]")               # qaac: "[42.5%] 0:30/1:43"

BAR_WIDTH = 40


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _fmt_size(n: int) -> str:
    mb = n / (1024 * 1024)
    return f"{mb:,.0f} MB"


def _fmt_bitrate(bps: int | None) -> str:
    if bps is None or bps == 0:
        return ""
    kbps = bps // 1000
    if kbps > 0:
        return f"{kbps}kbps"
    return f"{bps}bps"


def _channel_layout_short(layout: str | None) -> str:
    """Simplify channel layout: '5.1(side)' -> '5.1'."""
    if not layout:
        return ""
    return layout.split("(")[0]


def _audio_step_label(instr: AudioInstruction, index: int, total: int) -> str:
    """Build a human-readable step label for an audio track."""
    codec = instr.codec_name.upper()
    ch = ""
    if instr.channels:
        ch_map = {1: "1.0", 2: "2.0", 6: "5.1", 8: "7.1"}
        ch = " " + ch_map.get(instr.channels, f"{instr.channels}ch")

    num = f" {index + 1}" if total > 1 else ""

    if instr.action == AudioAction.COPY:
        return f"Copy audio{num} ({codec}{ch})"
    if instr.action == AudioAction.DENORM:
        return f"Denorm audio{num} ({codec}{ch})"
    if instr.action == AudioAction.DECODE_ENCODE:
        return f"Recode audio{num} ({codec} -> AAC)"
    if instr.action == AudioAction.FFMPEG_ENCODE:
        return f"Recode audio{num} ({codec} -> AAC)"
    return f"Audio{num} ({codec})"


def _sub_step_label(instr: SubtitleInstruction, index: int, total: int) -> str:
    """Build a human-readable step label for a subtitle track."""
    codec = instr.codec_name.upper()
    num = f" {index + 1}" if total > 1 else ""

    if instr.action == SubtitleAction.COPY_RECODE:
        return f"Recode subs{num} ({codec} -> UTF-8)"
    return f"Copy subs{num} ({codec})"


def _build_steps(job: Job, *, vmaf_enabled: bool = False) -> list[str]:
    """Build the dynamic step list matching actual pipeline execution order."""
    steps: list[str] = []

    # Audio tracks (each processed fully: extract + denorm/decode/encode)
    for i, audio_instr in enumerate(job.audio):
        steps.append(_audio_step_label(audio_instr, i, len(job.audio)))

    # Subtitle tracks
    for i, sub_instr in enumerate(job.subtitles):
        steps.append(_sub_step_label(sub_instr, i, len(job.subtitles)))

    # Always present
    steps.extend(["Encode video", "Assemble MKV", "Set metadata", "Optimize index"])
    if vmaf_enabled:
        steps.append("VMAF")
    return steps


def _build_source_text(job: Job) -> str:
    """Build source info block from Job data."""
    lines: list[str] = []

    vp = job.video_params
    codec = vp.source_codec.upper() or "?"
    bitrate_str = f" {_fmt_bitrate(vp.source_bitrate)}" if vp.source_bitrate else ""
    lines.append(f"Video: {codec} {vp.source_width}x{vp.source_height}{bitrate_str}")

    for i, audio_instr in enumerate(job.audio):
        codec = audio_instr.codec_name.upper()
        ch = _channel_layout_short(None)
        if audio_instr.channels:
            # Approximate layout from channel count
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
        lines.append(f"Size:  {_fmt_size(job.source_size)}")

    return "\n".join(lines)


def _audio_target_label(instr: AudioInstruction) -> str:
    """Describe what this audio track becomes."""
    codec = instr.codec_name.upper()
    ch = ""
    if instr.channels:
        ch_map = {1: "1.0", 2: "2.0", 6: "5.1", 8: "7.1"}
        ch = ch_map.get(instr.channels, f"{instr.channels}ch")

    if instr.action == AudioAction.COPY:
        return f"{codec} {ch} (copy)".strip()
    if instr.action == AudioAction.DENORM:
        return f"{codec} {ch} (denorm)".strip()
    if instr.action in (AudioAction.DECODE_ENCODE, AudioAction.FFMPEG_ENCODE):
        return f"AAC (from {codec})".strip()
    return codec


def _sub_target_label(instr: SubtitleInstruction) -> str:
    codec = instr.codec_name.upper()
    if instr.action == SubtitleAction.COPY_RECODE:
        return f"{codec} {instr.language} (UTF-8)"
    return f"{codec} {instr.language}"


def _build_target_text(job: Job) -> str:
    """Build target info block from Job data."""
    lines: list[str] = []

    vp = job.video_params
    if vp.crop:
        res = f"{vp.crop.w}x{vp.crop.h}"
    else:
        res = f"{vp.source_width}x{vp.source_height}"
    lines.append(f"Video: HEVC {res} CQ{vp.cq}")

    for i, audio_instr in enumerate(job.audio):
        prefix = "Audio:" if i == 0 else "      "
        lines.append(f"{prefix} {_audio_target_label(audio_instr)}")

    for i, sub_instr in enumerate(job.subtitles):
        prefix = "Subs: " if i == 0 else "      "
        lines.append(f"{prefix} {_sub_target_label(sub_instr)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class HeaderWidget(Static):
    """Top bar: [X/N] filename."""

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
    """Source track info."""

    DEFAULT_CSS = """
    SourceWidget {
        width: 50%;
        border: double $primary;
        padding: 0 1;
        height: auto;
    }
    """


class TargetWidget(Static):
    """Target track info."""

    DEFAULT_CSS = """
    TargetWidget {
        width: 50%;
        border: double $primary;
        padding: 0 1;
        height: auto;
    }
    """


class StepsWidget(Static):
    """Pipeline step list with > and + markers."""

    DEFAULT_CSS = """
    StepsWidget {
        width: 36;
        border: double $primary;
        padding: 0 1;
        height: 1fr;
    }
    """


class OutputLog(RichLog):
    """Scrollable tool output (auto-scroll)."""

    DEFAULT_CSS = """
    OutputLog {
        border: double $primary;
        height: 1fr;
    }
    """


class ProgressWidget(Static):
    """Unicode progress bar at the bottom."""

    DEFAULT_CSS = """
    ProgressWidget {
        height: 2;
        padding: 0 1;
    }
    """


# ---------------------------------------------------------------------------
# RunApp
# ---------------------------------------------------------------------------

class RunApp(App[None]):
    """Textual app for the furnace run (encoding) phase.

    Public API (called from worker thread via call_from_thread):
        start_job(job, job_index)
        update_encode(pct, speed)
        update_status(message)
        add_tool_line(line)
        finish_job(job)
        stop()
    """

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
    ]

    def __init__(
        self,
        total_jobs: int,
        shutdown_event: threading.Event,
        executor_fn: Callable[..., None],
        vmaf_enabled: bool = False,
    ) -> None:
        super().__init__()
        self._total_jobs = total_jobs
        self._shutdown_event = shutdown_event
        self._executor_fn = executor_fn
        self._vmaf_enabled = vmaf_enabled

        # State
        self._job: Job | None = None
        self._job_idx = 0
        self._steps: list[str] = []
        self._current_step_idx: int = -1
        self._pct = 0.0
        self._speed = ""
        self._encoding = False
        self._start_time = 0.0
        self._target_base_text = ""
        self._output_size = 0

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        """Start executor in a worker thread."""
        self.run_worker(self._run_executor, thread=True)

    def _run_executor(self) -> None:
        """Worker thread entry point — calls executor_fn with self as progress."""
        self._executor_fn(self)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_quit_app(self) -> None:
        """ESC pressed: graceful shutdown."""
        self._shutdown_event.set()
        # Kill child process tree
        try:
            import os as _os
            import psutil
            parent = psutil.Process(_os.getpid())
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
        except ImportError:
            pass
        except Exception:  # noqa: S110
            pass
        self.exit()

    # ------------------------------------------------------------------
    # Public API (called from worker thread)
    # ------------------------------------------------------------------

    def _safe_call(self, fn: Callable[..., object], *args: object) -> None:
        """Call from thread, silently ignore if app already exited."""
        try:
            self.call_from_thread(fn, *args)
        except Exception:  # noqa: S110
            pass

    def start_job(self, job: Job, job_index: int) -> None:
        """New job started — update all widgets."""
        self._safe_call(self._do_start_job, job, job_index)

    def update_encode(self, pct: float, speed: str) -> None:
        """Update encoding progress bar."""
        self._safe_call(self._do_update_encode, pct, speed)

    def update_status(self, message: str) -> None:
        """Update current step in the steps list."""
        self._safe_call(self._do_update_status, message)

    def add_tool_line(self, line: str) -> None:
        """Append one line of tool output."""
        self._safe_call(self._do_add_tool_line, line)

    def finish_job(self, job: Job) -> None:
        """Job completed — mark all steps done."""
        self._safe_call(self._do_finish_job, job)

    def update_output_size(self, size_bytes: int) -> None:
        """Update current output size in the Target block."""
        self._safe_call(self._do_update_output_size, size_bytes)

    def stop(self) -> None:
        """All jobs done — exit app."""
        self._safe_call(self.exit)

    # ------------------------------------------------------------------
    # Internal (main thread)
    # ------------------------------------------------------------------

    def _do_start_job(self, job: Job, job_index: int) -> None:
        self._job = job
        self._job_idx = job_index
        self._pct = 0.0
        self._speed = ""
        self._encoding = False
        self._start_time = time.monotonic()

        # Header
        filename = Path(job.output_file).name
        header = self.query_one("#header", HeaderWidget)
        header.update(f"[{job_index + 1}/{self._total_jobs}] {filename}")

        # Source / Target
        source_w = self.query_one("#source", SourceWidget)
        source_w.update(_build_source_text(job))

        self._target_base_text = _build_target_text(job)
        self._output_size = 0
        target_w = self.query_one("#target", TargetWidget)
        target_w.update(f"{self._target_base_text}\nSize:  ?")

        # Steps
        self._steps = _build_steps(job, vmaf_enabled=self._vmaf_enabled)
        self._current_step_idx = -1
        self._refresh_steps()

        # Clear output log
        output_log = self.query_one("#output", OutputLog)
        output_log.clear()

        # Clear progress
        progress_w = self.query_one("#progress", ProgressWidget)
        progress_w.update("")

    def _do_update_encode(self, pct: float, speed: str) -> None:
        self._encoding = True
        self._pct = pct
        self._speed = speed

        # Advance to "Encode video" step
        if "Encode video" in self._steps:
            idx = self._steps.index("Encode video")
            if self._current_step_idx < idx:
                self._current_step_idx = idx
                self._refresh_steps()

        self._refresh_progress()

    def _do_update_status(self, message: str) -> None:
        self._encoding = False
        self._pct = 0.0
        self._speed = ""

        # Advance to next step sequentially
        if self._current_step_idx < len(self._steps) - 1:
            self._current_step_idx += 1
            self._refresh_steps()

        # Show status in progress bar area
        progress_w = self.query_one("#progress", ProgressWidget)
        progress_w.update(message)

    def _set_tool_pct(self, pct: float) -> None:
        """Update progress percentage, resetting timer if a new tool started."""
        if pct < self._pct:
            # Progress went backwards — new tool started
            self._start_time = time.monotonic()
        self._encoding = True
        self._pct = pct
        self._refresh_progress()

    def _do_add_tool_line(self, line: str) -> None:
        stripped = line.strip()

        # eac3to: "process: 42%"
        m = _EAC3TO_RE.match(stripped)
        if m:
            self._set_tool_pct(float(m.group(1)))
            return

        # mkvmerge: "Progress: 42%"
        m = _MKVMERGE_RE.match(stripped)
        if m:
            self._set_tool_pct(float(m.group(1)))
            return

        # mkclean: "Progress 1/3:   42%" → stage 1=0-33%, stage 2=33-66%, stage 3=66-100%
        m = _MKCLEAN_RE.match(stripped)
        if m:
            stage = int(m.group(1))  # 1, 2, or 3
            stage_pct = float(m.group(2))
            self._set_tool_pct((stage - 1) * 33.3 + stage_pct * 0.333)
            return

        # qaac: "[42.5%] 0:30/1:43:01.600 (30.5x), ETA 3:20"
        m = _QAAC_RE.match(stripped)
        if m:
            self._set_tool_pct(float(m.group(1)))
            return

        output_log = self.query_one("#output", OutputLog)
        output_log.write(line)

    def _do_update_output_size(self, size_bytes: int) -> None:
        self._output_size = size_bytes
        target_w = self.query_one("#target", TargetWidget)
        size_str = _fmt_size(size_bytes) if size_bytes > 0 else "..."
        target_w.update(f"{self._target_base_text}\nSize:  {size_str}")

    def _do_finish_job(self, job: Job) -> None:
        self._job = job
        self._encoding = False

        # Mark all steps completed
        self._current_step_idx = len(self._steps)
        self._refresh_steps()

        # Show done in progress
        progress_w = self.query_one("#progress", ProgressWidget)
        progress_w.update("Done")

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _refresh_steps(self) -> None:
        """Re-render the steps widget."""
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
        """Re-render the progress bar."""
        if not self._encoding:
            return

        filled = int(BAR_WIDTH * self._pct / 100)
        bar = "\u2588" * filled + "\u2591" * (BAR_WIDTH - filled)

        elapsed = time.monotonic() - self._start_time

        if self._pct > 0 and elapsed > 0:
            total_est = elapsed / (self._pct / 100)
            remaining = total_est - elapsed
            time_part = f"{_fmt_time(elapsed)} / ~{_fmt_time(remaining)}"
        else:
            time_part = _fmt_time(elapsed)

        speed_part = f" | {self._speed}" if self._speed else ""

        text = f"{bar} {self._pct:5.1f}% | {time_part}{speed_part}"

        progress_w = self.query_one("#progress", ProgressWidget)
        progress_w.update(text)
