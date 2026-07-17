from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from furnace.core.models import AnalyzeStatus, DiscType


@dataclass(frozen=True)
class Event:
    method: str
    args: tuple[object, ...]
    kwargs: tuple[tuple[str, object], ...] = ()


class RecordingPlanReporter:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def _record(self, name: str, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        self.events.append(Event(name, args, tuple(sorted(kwargs.items()))))

    def detect_disc(self, disc_type: DiscType, rel_path: str) -> None:
        self._record("detect_disc", (disc_type, rel_path), {})

    def detect_disc_titles_done(self, n_titles: int) -> None:
        self._record("detect_disc_titles_done", (n_titles,), {})

    def demux_disc_cached(self, label: str) -> None:
        self._record("demux_disc_cached", (label,), {})

    def demux_disc_start(self, label: str) -> None:
        self._record("demux_disc_start", (label,), {})

    def demux_title_start(self, title_num: int) -> None:
        self._record("demux_title_start", (title_num,), {})

    def demux_title_substep(self, label: str, *, has_progress: bool) -> None:
        self._record("demux_title_substep", (label,), {"has_progress": has_progress})

    def demux_title_progress(self, fraction: float) -> None:
        self._record("demux_title_progress", (fraction,), {})

    def demux_title_done(self) -> None:
        self._record("demux_title_done", (), {})

    def demux_title_failed(self, reason: str) -> None:
        self._record("demux_title_failed", (reason,), {})

    def scan_file(self, name: str) -> None:
        self._record("scan_file", (name,), {})

    def scan_skipped(self, name: str, reason: str) -> None:
        self._record("scan_skipped", (name, reason), {})

    def analyze_batch_start(self, total: int) -> None:
        self._record("analyze_batch_start", (total,), {})

    def analyze_batch_progress(self, completed: float) -> None:
        self._record("analyze_batch_progress", (completed,), {})

    def analyze_batch_item(self, name: str, detail: str, *, status: AnalyzeStatus) -> None:
        self._record("analyze_batch_item", (name, detail), {"status": status})

    def analyze_batch_finish(self) -> None:
        self._record("analyze_batch_finish", (), {})

    def plan_file_start(self, name: str) -> None:
        self._record("plan_file_start", (name,), {})

    def plan_file_done(self, summary: str) -> None:
        self._record("plan_file_done", (summary,), {})

    def plan_saved(self, path: Path, n_jobs: int) -> None:
        self._record("plan_saved", (path, n_jobs), {})

    def interrupted(self) -> None:
        self._record("interrupted", (), {})

    def pause(self) -> None:
        self._record("pause", (), {})

    def resume(self) -> None:
        self._record("resume", (), {})
