from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from furnace.core.models import Job, JobStatus, Plan


def _fmt_size(n: int | None) -> str:
    """Human-readable file size."""
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


def _savings_pct(before: int, after: int) -> str:
    if before == 0:
        return "N/A"
    pct = (before - after) / before * 100
    sign = "-" if pct >= 0 else "+"
    return f"{sign}{abs(pct):.1f}%"


# ---------------------------------------------------------------------------
# EncodingProgress
# ---------------------------------------------------------------------------

class EncodingProgress:
    """Rich Live display during encoding.

    Usage::

        prog = EncodingProgress(console, total_jobs=5)
        prog.start_job(job, job_index=0)
        prog.update(pct=42.5, speed="1.2x", fps="24.0")
        prog.finish_job(job)
    """

    def __init__(self, console: Console, total_jobs: int) -> None:
        self._console = console
        self._total_jobs = total_jobs
        self._current_job: Job | None = None
        self._job_index: int = 0

        # Progress bar for the current file
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>5.1f}%"),
            TimeRemainingColumn(),
            TextColumn("{task.fields[extra]}"),
            console=console,
            transient=False,
        )
        self._task_id: TaskID | None = None
        self._live = Live(
            self._build_renderable(),
            console=console,
            refresh_per_second=4,
        )
        self._live.start()

        # State updated by update()
        self._pct: float = 0.0
        self._speed: str = ""
        self._fps: str = ""
        self._output_size: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_job(self, job: Job, job_index: int) -> None:
        """Begin displaying a new job."""
        self._current_job = job
        self._job_index = job_index
        self._pct = 0.0
        self._speed = ""
        self._fps = ""
        self._output_size = None

        filename = Path(job.output_file).name
        vp = job.video_params
        resolution = f"{vp.source_width}x{vp.source_height}"
        description = f"{filename}  [{resolution}  CQ {vp.cq}]"

        if self._task_id is not None:
            self._progress.remove_task(self._task_id)

        self._task_id = self._progress.add_task(
            description,
            total=100.0,
            extra="",
        )
        self._refresh()

    def update(self, pct: float, speed: str, fps: str) -> None:
        """Update progress percentage, encoding speed, and fps."""
        self._pct = pct
        self._speed = speed
        self._fps = fps

        if self._task_id is not None:
            extra_parts = []
            if fps:
                extra_parts.append(f"fps={fps}")
            if speed:
                extra_parts.append(f"speed={speed}")
            extra = "  ".join(extra_parts)
            self._progress.update(
                self._task_id,
                completed=pct,
                extra=extra,
            )
        self._refresh()

    def finish_job(self, job: Job) -> None:
        """Mark a job as complete and record its output size."""
        self._output_size = job.output_size
        if self._task_id is not None:
            self._progress.update(self._task_id, completed=100.0, extra="done")
        self._refresh()

    def stop(self) -> None:
        """Stop the Live display."""
        self._live.stop()

    def __enter__(self) -> EncodingProgress:
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_renderable(self) -> Panel:
        grid = Table.grid(padding=(0, 1))
        grid.add_column()

        # Batch progress line
        batch_text = Text(
            f"File {self._job_index + 1} of {self._total_jobs}",
            style="bold cyan",
        )
        grid.add_row(batch_text)

        # Size comparison
        if self._current_job is not None:
            src = self._current_job.source_size
            out = self._output_size
            src_str = _fmt_size(src) if src else "?"
            out_str = _fmt_size(out) if out else "..."
            if src and out:
                savings = _savings_pct(src, out)
                size_line = f"Source: {src_str}  Output: {out_str}  ({savings})"
            else:
                size_line = f"Source: {src_str}  Output: {out_str}"
            grid.add_row(Text(size_line, style="dim"))

        # Progress bar
        grid.add_row(self._progress)

        return Panel(grid, title="[bold]Furnace[/bold]", border_style="blue")

    def _refresh(self) -> None:
        self._live.update(self._build_renderable())


# ---------------------------------------------------------------------------
# ReportPrinter
# ---------------------------------------------------------------------------

class ReportPrinter:
    """Print a final summary report after all jobs complete."""

    def print_report(self, plan: Plan, console: Console) -> None:
        """Print summary to console.

        Counts jobs by status, computes size savings, and shows average VMAF
        score if enabled.
        """
        done_jobs = [j for j in plan.jobs if j.status == JobStatus.DONE]
        error_jobs = [j for j in plan.jobs if j.status == JobStatus.ERROR]
        pending_jobs = [j for j in plan.jobs if j.status == JobStatus.PENDING]

        total = len(plan.jobs)
        n_done = len(done_jobs)
        n_error = len(error_jobs)
        n_skipped = len(pending_jobs)

        total_source = sum(j.source_size for j in done_jobs if j.source_size)
        total_output = sum(j.output_size for j in done_jobs if j.output_size is not None)

        console.rule("[bold blue]Furnace Report[/bold blue]")

        # Job counts table
        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold")
        summary.add_column()
        summary.add_row("Files processed:", str(n_done))
        summary.add_row("Files skipped:", str(n_skipped))
        summary.add_row("Files with errors:", str(n_error))
        summary.add_row("Total files:", str(total))
        console.print(summary)

        console.print()

        # Size summary
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

        # VMAF average
        if plan.vmaf_enabled:
            vmaf_scores = [j.vmaf_score for j in done_jobs if j.vmaf_score is not None]
            if vmaf_scores:
                avg_vmaf = sum(vmaf_scores) / len(vmaf_scores)
                console.print(f"[bold]Average VMAF:[/bold] {avg_vmaf:.2f}  (n={len(vmaf_scores)})")
                console.print()

        # List errored jobs
        if error_jobs:
            console.print("[bold red]Errors:[/bold red]")
            for job in error_jobs:
                name = Path(job.output_file).name
                err = job.error or "unknown error"
                console.print(f"  [red]{name}[/red]: {err}")
            console.print()

        console.rule()
