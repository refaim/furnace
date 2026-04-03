"""Encoding progress display (Rich Live, crucible-style).

Layout:
    [furnace] File 3 of 12 | movie.mkv
    [furnace] 1920x1080 (2.07M px) -> CQ 25

      frame= 1234 fps=120 q=25.0 ...     <-- last N lines of tool stderr
      frame= 1235 fps=120 q=25.0 ...

    [furnace] ████████░░░░░░  42.5% | 3:20 elapsed | ~4:40 left | 1.2x | 512 MB / 4.0 GB

For non-video steps (audio, mux, mkclean):
    [furnace] File 3 of 12 | movie.mkv
    [furnace] Processing audio 1/2 (AC3 rus)...
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from furnace.core.models import Job, JobStatus, Plan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_size(n: int | None) -> str:
    if n is None or n == 0:
        return "?"
    val = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if val < 1024:
            return f"{val:.0f} {unit}"
        val /= 1024
    return f"{val:.1f} TB"


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


TAG = "[furnace] "
BAR_WIDTH = 30
MAX_OUTPUT_LINES = 20

# Regex for tool progress lines (eac3to "process: 42%")
_PROGRESS_RE = re.compile(r"^process:\s*(\d+)%$")


# ---------------------------------------------------------------------------
# EncodingProgress
# ---------------------------------------------------------------------------

class EncodingProgress:
    """Crucible-style Rich Live progress display.

    Public API (called by Executor):
        start_job(job, job_index)    — new file started
        update_encode(pct, speed, stderr_lines)  — ffmpeg/vmaf progress tick
        update_status(message)       — non-video step status text
        finish_job(job)              — file done
        stop()                       — tear down Live
    """

    def __init__(self, console: Console, total_jobs: int) -> None:
        self._console = console
        self._total = total_jobs
        self._job: Job | None = None
        self._job_idx = 0

        # Encoding state
        self._pct = 0.0
        self._speed = ""
        self._output_lines: list[str] = []
        self._status_text = ""
        self._start_time = 0.0
        self._encoding = False

        self._live = Live(
            self._render(),
            console=console,
            refresh_per_second=4,
        )
        self._live.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_tool_line(self, line: str) -> None:
        """Receive one line of tool output for display.

        Lines matching progress patterns (eac3to 'process: XX%') are
        routed to the progress bar instead of the output frame.
        """
        m = _PROGRESS_RE.match(line.strip())
        if m:
            pct = float(m.group(1))
            self._encoding = True
            self._pct = pct
            self._refresh()
            return

        self._output_lines.append(line)
        if len(self._output_lines) > MAX_OUTPUT_LINES:
            self._output_lines.pop(0)
        self._refresh()

    def start_job(self, job: Job, job_index: int) -> None:
        self._job = job
        self._job_idx = job_index
        self._pct = 0.0
        self._speed = ""
        self._output_lines = []
        self._status_text = ""
        self._encoding = False
        self._start_time = time.monotonic()
        self._refresh()

    def update_encode(self, pct: float, speed: str) -> None:
        """Update encoding progress (pct/speed). Tool output comes via add_tool_line."""
        self._encoding = True
        self._pct = pct
        self._speed = speed
        self._status_text = ""
        self._refresh()

    def update_status(self, message: str) -> None:
        """Update status for non-encoding steps (audio, mux, etc.)."""
        self._encoding = False
        self._pct = 0.0
        self._speed = ""
        self._output_lines = []
        self._status_text = message
        self._refresh()

    def finish_job(self, job: Job) -> None:
        self._job = job
        self._encoding = False
        self._status_text = "Done"
        self._refresh()

    def stop(self) -> None:
        self._live.stop()

    def __enter__(self) -> EncodingProgress:
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> Text:
        text = Text()

        # Header: batch + filename
        if self._job is not None:
            name = Path(self._job.output_file).name
            text.append(TAG, style="cyan")
            text.append(f"File {self._job_idx + 1} of {self._total}", style="bold white")
            text.append(f" | {name}\n", style="white")

            # Video info
            vp = self._job.video_params
            area = vp.source_width * vp.source_height
            if area >= 1_000_000:
                area_label = f"{area / 1_000_000:.2f}M px"
            else:
                area_label = f"{area // 1000}K px"
            text.append(TAG, style="cyan")
            text.append(f"{vp.source_width}x{vp.source_height}", style="bold white")
            text.append(f" ({area_label})", style="dim")
            text.append(" -> CQ ", style="cyan")
            text.append(f"{vp.cq}\n", style="bold yellow")
        else:
            text.append(TAG, style="cyan")
            text.append("Waiting...\n", style="dim")

        # Tool output lines (stderr frame)
        if self._output_lines:
            text.append("\n")
            for line in self._output_lines:
                text.append(f"  {line}\n", style="yellow")
            text.append("\n")

        # Progress bar or status
        if self._encoding and self._job is not None:
            self._render_bar(text)
        elif self._status_text:
            text.append(TAG, style="cyan")
            text.append(f"{self._status_text}\n", style="green")

        return text

    def _render_bar(self, text: Text) -> None:
        """Render crucible-style progress bar."""
        filled = int(BAR_WIDTH * self._pct / 100)
        bar = "\u2588" * filled + "\u2591" * (BAR_WIDTH - filled)

        elapsed = time.monotonic() - self._start_time

        if self._pct > 0 and elapsed > 0:
            total_est = elapsed / (self._pct / 100)
            remaining = total_est - elapsed
            time_part = f"{_fmt_time(elapsed)} elapsed | ~{_fmt_time(remaining)} left"
        else:
            time_part = f"{_fmt_time(elapsed)} elapsed"

        speed_part = f" | {self._speed}" if self._speed else ""

        # Current output size vs source
        size_part = ""
        if self._job is not None and self._job.source_size:
            cur_size = 0
            out_path = Path(self._job.output_file)
            if out_path.exists():
                try:
                    cur_size = out_path.stat().st_size
                except OSError:
                    pass
            if cur_size:
                size_part = f" | {_fmt_size(cur_size)} / {_fmt_size(self._job.source_size)}"

        text.append(TAG, style="cyan")
        text.append(bar, style="bold green")
        text.append(f" {self._pct:5.1f}%", style="bold green")
        text.append(f" | {time_part}{speed_part}{size_part}\n", style="green")

    def _refresh(self) -> None:
        self._live.update(self._render())


# ---------------------------------------------------------------------------
# ReportPrinter
# ---------------------------------------------------------------------------

class ReportPrinter:
    """Print a final summary report after all jobs complete."""

    def print_report(self, plan: Plan, console: Console) -> None:
        done_jobs = [j for j in plan.jobs if j.status == JobStatus.DONE]
        error_jobs = [j for j in plan.jobs if j.status == JobStatus.ERROR]
        pending_jobs = [j for j in plan.jobs if j.status == JobStatus.PENDING]

        total = len(plan.jobs)
        n_done = len(done_jobs)
        n_error = len(error_jobs)
        n_skipped = len(pending_jobs)

        total_source = sum(j.source_size for j in done_jobs if j.source_size)
        total_output = sum(j.output_size for j in done_jobs if j.output_size is not None)

        console.print("-" * 80)

        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold")
        summary.add_column()
        summary.add_row("Files processed:", str(n_done))
        summary.add_row("Files skipped:", str(n_skipped))
        summary.add_row("Files with errors:", str(n_error))
        summary.add_row("Total files:", str(total))
        console.print(summary)
        console.print()

        if done_jobs and total_source > 0:
            size_table = Table.grid(padding=(0, 2))
            size_table.add_column(style="bold")
            size_table.add_column()
            size_table.add_row("Total source size:", _fmt_size(total_source))
            if total_output:
                size_table.add_row("Total output size:", _fmt_size(total_output))
                saved = total_source - total_output
                pct = saved / total_source * 100
                sign = "-" if saved >= 0 else "+"
                savings_str = f"{_fmt_size(abs(saved))} ({sign}{abs(pct):.1f}%)"
                size_table.add_row("Space saved:", savings_str)
            console.print(size_table)
            console.print()

        if plan.vmaf_enabled:
            vmaf_scores = [j.vmaf_score for j in done_jobs if j.vmaf_score is not None]
            if vmaf_scores:
                avg_vmaf = sum(vmaf_scores) / len(vmaf_scores)
                console.print(f"[bold]Average VMAF:[/bold] {avg_vmaf:.2f}  (n={len(vmaf_scores)})")
                console.print()

        if error_jobs:
            console.print("[bold red]Errors:[/bold red]")
            for job in error_jobs:
                name = Path(job.output_file).name
                err = job.error or "unknown error"
                console.print(f"  [red]{name}[/red]: {err}")
            console.print()

        console.print("-" * 80)
