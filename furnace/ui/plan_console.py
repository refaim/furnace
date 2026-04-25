"""Structured terminal reporter for ``furnace plan``.

Owns stdout for the entire plan command. Renders phase headers, per-row
events, and a single floating progress bar at the bottom of the terminal.
Raw tool output never flows here; the reporter is fed structured events by
services and adapter progress parsers.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.progress import Progress, ProgressColumn, SpinnerColumn, Task, TaskID, TextColumn
from rich.text import Text

from furnace.core.models import DiscType

_DISC_TYPE_NAMES: dict[DiscType, str] = {
    DiscType.BLURAY: "BDMV",
    DiscType.DVD: "DVD",
}

_ASCII_SPINNER = "line"  # Rich built-in ASCII spinner: |/-\

# Visual nesting for demux titles under their disc — disc name remains flush left.
_TITLE_INDENT = "  "


class _ChunkBarColumn(ProgressColumn):
    """Bar built from full-height block chars throughout (no thin incomplete line).

    Width is fixed at 40 cells.
    """

    _WIDTH = 40

    def render(self, task: Task) -> Text:
        pct = task.percentage or 0
        filled = int(self._WIDTH * pct / 100)
        return Text(
            "█" * filled + "░" * (self._WIDTH - filled),
            style="white",
        )


class RichPlanReporter:
    """Implements ``PlanReporter`` against a Rich ``Console``.

    ``ascii_only=True`` forces ASCII bar/spinner glyphs (cmd.exe-safe).
    """

    def __init__(
        self,
        *,
        source: Path,
        output: Path,
        console: Console | None = None,
        ascii_only: bool = True,
    ) -> None:
        self._source = source
        self._output = output
        self._console = console or Console(highlight=False)
        self._ascii_only = ascii_only
        self._detect_started = False
        self._demux_started = False
        self._current_disc_label: str | None = None
        self._current_title_num: int | None = None
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._scan_started = False
        self._analyze_started = False
        self._current_file: str | None = None
        self._plan_started = False
        # Tracks whether any phase header has been emitted yet — controls the
        # "blank line before phase header" rule for non-first phases.
        self._any_phase_started = False

    def _start_progress(self, *, has_progress: bool) -> Progress | None:
        """Start a transient Rich Progress.

        ``has_progress=True`` -> ``[Description, Bar, Percent]`` (bar on the right).
        ``has_progress=False`` -> ``[Description, Spinner]`` (spinner where the bar
        would have been). The two are mutually exclusive — never both at once.

        Returns ``None`` on non-TTY consoles (e.g. when stdout is piped to a
        file): a Live display in that mode would emit stray blank lines, so
        the floating bar is suppressed entirely. Persistent rows still print
        normally via plain ``console.print``.
        """
        assert self._progress is None, "previous progress not stopped"  # noqa: S101
        if not self._console.is_terminal:
            return None
        columns: list[ProgressColumn | SpinnerColumn | TextColumn]
        if has_progress:
            columns = [
                TextColumn("{task.description}"),
                _ChunkBarColumn(),
                TextColumn("{task.percentage:>3.0f}%"),
            ]
        else:
            columns = [
                TextColumn("{task.description}"),
                SpinnerColumn(spinner_name=_ASCII_SPINNER),
            ]
        progress = Progress(
            *columns,
            console=self._console,
            transient=True,
            expand=False,
        )
        progress.start()
        self._progress = progress
        return progress

    def _stop_progress(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._task_id = None
            self._progress = None

    def _emit_phase_header(self, name: str) -> None:
        """Print a bold phase header on its own line, flush left.

        For every phase except the first one, an empty line is emitted first
        to visually separate phase blocks.
        """
        if self._any_phase_started:
            self._console.print()
        self._console.print(f"[bold]{name}[/bold]", highlight=False)
        self._any_phase_started = True

    def start(self) -> None:
        """Print the header. Called once at the beginning of ``plan``."""
        self._console.print(f"Source: {self._source}", highlight=False)
        self._console.print(f"Output: {self._output}", highlight=False)
        self._console.print()

    def stop(self) -> None:
        """Flush. Called once at end of ``plan``."""
        self._stop_progress()

    # -- Detect ---------------------------------------------------------------

    def detect_disc(self, disc_type: DiscType, rel_path: str) -> None:
        if not self._detect_started:
            self._emit_phase_header("Detect")
            self._detect_started = True
        type_name = _DISC_TYPE_NAMES[disc_type]
        self._console.print(f"{type_name:<5} {rel_path}", highlight=False)

    # -- Demux ---------------------------------------------------------------

    def _ensure_demux_header(self) -> None:
        if not self._demux_started:
            self._emit_phase_header("Demux")
            self._demux_started = True

    def demux_disc_cached(self, label: str) -> None:
        self._stop_progress()
        self._ensure_demux_header()
        self._console.print(f"{label} -> from cache", highlight=False)

    def demux_disc_start(self, label: str) -> None:
        self._stop_progress()
        self._ensure_demux_header()
        self._console.print(label, highlight=False)
        self._current_disc_label = label

    def demux_title_start(self, title_num: int) -> None:
        self._stop_progress()
        self._current_title_num = title_num

    def demux_title_substep(self, label: str, *, has_progress: bool) -> None:
        self._stop_progress()
        if self._current_title_num is None:
            return
        progress = self._start_progress(has_progress=has_progress)
        if progress is None:
            return  # non-TTY: skip floating bar entirely
        title_label = f"title {self._current_title_num}"
        desc = f"{_TITLE_INDENT}{title_label} -> {label}"
        self._task_id = progress.add_task(desc, total=100 if has_progress else None)

    def demux_title_progress(self, fraction: float) -> None:
        if self._progress is None or self._task_id is None:
            return
        self._progress.update(self._task_id, completed=fraction * 100)

    def demux_title_done(self) -> None:
        self._stop_progress()
        if self._current_title_num is not None:
            title_label = f"title {self._current_title_num}"
            self._console.print(
                f"{_TITLE_INDENT}{title_label} -> done",
                highlight=False,
            )
        self._current_title_num = None

    def demux_title_failed(self, reason: str) -> None:
        self._stop_progress()
        if self._current_title_num is not None:
            title_label = f"title {self._current_title_num}"
            self._console.print(
                f"{_TITLE_INDENT}{title_label} -> FAILED — {reason}",
                highlight=False,
            )
        self._current_title_num = None

    # -- Scan -----------------------------------------------------------------

    def _ensure_scan_header(self) -> None:
        if not self._scan_started:
            self._emit_phase_header("Scan")
            self._scan_started = True

    def scan_file(self, name: str) -> None:
        self._stop_progress()
        self._ensure_scan_header()
        self._console.print(name, highlight=False)

    def scan_skipped(self, name: str, reason: str) -> None:
        self._stop_progress()
        self._ensure_scan_header()
        self._console.print(f"{name} -> SKIPPED — {reason}", highlight=False)

    # -- Analyze --------------------------------------------------------------

    def _ensure_analyze_header(self) -> None:
        if not self._analyze_started:
            self._emit_phase_header("Analyze")
            self._analyze_started = True

    def analyze_file_start(self, name: str) -> None:
        self._stop_progress()
        self._current_file = name

    def analyze_microop(self, label: str, *, has_progress: bool) -> None:
        self._stop_progress()
        if self._current_file is None:
            return
        progress = self._start_progress(has_progress=has_progress)
        if progress is None:
            return  # non-TTY: skip floating bar entirely
        desc = f"{self._current_file} -> {label}"
        self._task_id = progress.add_task(desc, total=100 if has_progress else None)

    def analyze_progress(self, fraction: float) -> None:
        if self._progress is None or self._task_id is None:
            return
        self._progress.update(self._task_id, completed=fraction * 100)

    def analyze_file_done(self, summary: str) -> None:
        self._stop_progress()
        if self._current_file is not None:
            self._ensure_analyze_header()
            self._console.print(
                f"{self._current_file} -> {summary}",
                highlight=False,
            )
        self._current_file = None

    def analyze_file_failed(self, reason: str) -> None:
        self._stop_progress()
        if self._current_file is not None:
            self._ensure_analyze_header()
            self._console.print(
                f"{self._current_file} -> FAILED — {reason}",
                highlight=False,
            )
        self._current_file = None

    def analyze_file_skipped(self, reason: str) -> None:
        self._stop_progress()
        if self._current_file is not None:
            self._ensure_analyze_header()
            self._console.print(
                f"{self._current_file} -> SKIPPED — {reason}",
                highlight=False,
            )
        self._current_file = None

    # -- Plan -----------------------------------------------------------------

    def _ensure_plan_header(self) -> None:
        if not self._plan_started:
            self._emit_phase_header("Plan")
            self._plan_started = True

    def plan_file_start(self, name: str) -> None:
        self._stop_progress()
        self._current_file = name

    def plan_microop(self, label: str, *, has_progress: bool) -> None:
        self._stop_progress()
        if self._current_file is None:
            return
        progress = self._start_progress(has_progress=has_progress)
        if progress is None:
            return  # non-TTY: skip floating bar entirely
        desc = f"{self._current_file} -> {label}"
        self._task_id = progress.add_task(desc, total=100 if has_progress else None)

    def plan_progress(self, fraction: float) -> None:
        if self._progress is None or self._task_id is None:
            return
        self._progress.update(self._task_id, completed=fraction * 100)

    def plan_file_done(self, summary: str) -> None:
        self._stop_progress()
        if self._current_file is not None:
            self._ensure_plan_header()
            self._console.print(
                f"{self._current_file} -> {summary}",
                highlight=False,
            )
        self._current_file = None

    # -- Final / lifecycle ---------------------------------------------------

    def plan_saved(self, path: Path, n_jobs: int) -> None:
        """No-op for visible output. Kept for Protocol compatibility.

        The user explicitly does not want a final ``-> furnace-plan.json
        (N jobs)`` line. We still stop any lingering progress for safety.
        """
        del path, n_jobs  # unused — method retained for Protocol compatibility
        self._stop_progress()

    def interrupted(self) -> None:
        self._stop_progress()
        self._console.print("interrupted", highlight=False)

    def pause(self) -> None:
        self._stop_progress()

    def resume(self) -> None:
        # Next *_microop call will recreate the Progress; nothing to do here
        return
