from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.progress import Progress, ProgressColumn, SpinnerColumn, Task, TaskID, TextColumn
from rich.text import Text

from furnace.core.models import AnalyzeStatus, DiscType

_DISC_TYPE_NAMES: dict[DiscType, str] = {
    DiscType.BLURAY: "BDMV",
    DiscType.DVD: "DVD",
}

_ASCII_SPINNER = "line"

_TITLE_INDENT = "  "

_BATCH_STATUS_PREFIX: dict[AnalyzeStatus, str] = {
    AnalyzeStatus.DONE: "",
    AnalyzeStatus.SKIPPED: "SKIPPED — ",
    AnalyzeStatus.FAILED: "FAILED — ",
}


class _ChunkBarColumn(ProgressColumn):
    _WIDTH = 40

    def render(self, task: Task) -> Text:
        pct = task.percentage or 0
        filled = int(self._WIDTH * pct / 100)
        return Text(
            "█" * filled + "░" * (self._WIDTH - filled),
            style="white",
        )


class RichPlanReporter:
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
        self._current_file: str | None = None
        self._plan_started = False
        self._any_phase_started = False
        self._current_detect_disc_type: DiscType | None = None
        self._current_detect_rel_path: str | None = None

    def _start_progress(self, *, has_progress: bool) -> Progress | None:
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
        if self._any_phase_started:
            self._console.print()
        self._console.print(f"[bold]{name}[/bold]", highlight=False)
        self._any_phase_started = True

    def start(self) -> None:
        self._console.print(f"Source: {self._source}", highlight=False)
        self._console.print(f"Output: {self._output}", highlight=False)
        self._console.print()

    def stop(self) -> None:
        self._stop_progress()

    def detect_disc(self, disc_type: DiscType, rel_path: str) -> None:
        self._stop_progress()
        if not self._detect_started:
            self._emit_phase_header("Detect")
            self._detect_started = True
        self._current_detect_disc_type = disc_type
        self._current_detect_rel_path = rel_path
        type_name = _DISC_TYPE_NAMES[disc_type]
        progress = self._start_progress(has_progress=False)
        if progress is None:
            return
        desc = f"{type_name:<6}{rel_path} -> scanning"
        self._task_id = progress.add_task(desc, total=None)

    def detect_disc_titles_done(self, n_titles: int) -> None:
        self._stop_progress()
        if self._current_detect_rel_path is None:
            return
        assert self._current_detect_disc_type is not None  # noqa: S101
        type_name = _DISC_TYPE_NAMES[self._current_detect_disc_type]
        word = "title" if n_titles == 1 else "titles"
        self._console.print(
            f"{type_name:<6}{self._current_detect_rel_path} -> {n_titles} {word}",
            highlight=False,
        )
        self._current_detect_rel_path = None
        self._current_detect_disc_type = None

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
            return
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

    def analyze_batch_start(self, total: int) -> None:
        self._stop_progress()
        self._emit_phase_header("Analyze")
        if not self._console.is_terminal:
            self._progress = None
            return
        progress = Progress(
            TextColumn("Analyzing"),
            _ChunkBarColumn(),
            TextColumn("{task.completed:>4.1f}/{task.total:.0f}"),
            console=self._console,
            transient=True,
            expand=False,
        )
        progress.start()
        self._progress = progress
        self._task_id = progress.add_task("", total=total)

    def analyze_batch_progress(self, completed: float) -> None:
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, completed=completed)

    def analyze_batch_item(self, name: str, detail: str, *, status: AnalyzeStatus) -> None:
        line = f"{name} -> {_BATCH_STATUS_PREFIX[status]}{detail}"
        if self._progress is not None and self._task_id is not None:
            self._progress.console.print(line, highlight=False)
        else:
            self._console.print(line, highlight=False)

    def analyze_batch_finish(self) -> None:
        self._stop_progress()

    def _ensure_plan_header(self) -> None:
        if not self._plan_started:
            self._emit_phase_header("Plan")
            self._plan_started = True

    def plan_file_start(self, name: str) -> None:
        self._stop_progress()
        self._current_file = name

    def plan_file_done(self, summary: str) -> None:
        self._stop_progress()
        if self._current_file is not None:
            self._ensure_plan_header()
            self._console.print(
                f"{self._current_file} -> {summary}",
                highlight=False,
            )
        self._current_file = None

    def plan_saved(self, path: Path, n_jobs: int) -> None:
        del path, n_jobs
        self._stop_progress()

    def interrupted(self) -> None:
        self._stop_progress()
        self._console.print("interrupted", highlight=False)

    def pause(self) -> None:
        self._stop_progress()

    def resume(self) -> None:
        return
