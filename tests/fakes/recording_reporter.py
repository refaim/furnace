"""Test double for ``furnace.core.ports.PlanReporter``.

Captures every method call as ``Event(method, args, kwargs)`` in a list,
in invocation order. Use to assert event sequences from services.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from furnace.core.models import DiscType


@dataclass(frozen=True)
class Event:
    method: str
    args: tuple[object, ...]
    kwargs: tuple[tuple[str, object], ...] = ()


class RecordingPlanReporter:
    """Records every method call as an ``Event``. Returns ``None`` from all calls.

    Every ``PlanReporter`` method is defined explicitly so that ``mypy
    --strict`` sees the attributes and ``inspect.getattr_static`` (used by
    CPython 3.13's runtime ``isinstance`` against ``@runtime_checkable``
    Protocols) finds them on the class.
    """

    def __init__(self) -> None:
        self.events: list[Event] = []

    def _record(self, name: str, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        self.events.append(Event(name, args, tuple(sorted(kwargs.items()))))

    # Detect
    def detect_disc(self, disc_type: DiscType, rel_path: str) -> None:
        self._record("detect_disc", (disc_type, rel_path), {})

    def detect_disc_titles_done(self, n_titles: int) -> None:
        self._record("detect_disc_titles_done", (n_titles,), {})

    # Demux
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

    # Scan
    def scan_file(self, name: str) -> None:
        self._record("scan_file", (name,), {})

    def scan_skipped(self, name: str, reason: str) -> None:
        self._record("scan_skipped", (name, reason), {})

    # Analyze
    def analyze_file_start(self, name: str) -> None:
        self._record("analyze_file_start", (name,), {})

    def analyze_microop(self, label: str, *, has_progress: bool) -> None:
        self._record("analyze_microop", (label,), {"has_progress": has_progress})

    def analyze_progress(self, fraction: float) -> None:
        self._record("analyze_progress", (fraction,), {})

    def analyze_file_done(self, summary: str) -> None:
        self._record("analyze_file_done", (summary,), {})

    def analyze_file_failed(self, reason: str) -> None:
        self._record("analyze_file_failed", (reason,), {})

    def analyze_file_skipped(self, reason: str) -> None:
        self._record("analyze_file_skipped", (reason,), {})

    # Plan
    def plan_file_start(self, name: str) -> None:
        self._record("plan_file_start", (name,), {})

    def plan_microop(self, label: str, *, has_progress: bool) -> None:
        self._record("plan_microop", (label,), {"has_progress": has_progress})

    def plan_progress(self, fraction: float) -> None:
        self._record("plan_progress", (fraction,), {})

    def plan_file_done(self, summary: str) -> None:
        self._record("plan_file_done", (summary,), {})

    # Final
    def plan_saved(self, path: Path, n_jobs: int) -> None:
        self._record("plan_saved", (path, n_jobs), {})

    def interrupted(self) -> None:
        self._record("interrupted", (), {})

    # Lifecycle
    def pause(self) -> None:
        self._record("pause", (), {})

    def resume(self) -> None:
        self._record("resume", (), {})
