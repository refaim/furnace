# Plan-phase CLI Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the noisy, unstructured terminal output of `furnace plan` with a single, phase-aware reporter that prints clean per-phase rows and floats real progress bars (or spinners where the tool exposes no `%`) at the bottom of the terminal. Raw tool output (`makemkvcon`, `eac3to`, `mkvmerge`) never reaches the terminal — it stays in per-tool log files only.

**Architecture:** A new `PlanReporter` Protocol in `core/ports.py`, satisfied by `RichPlanReporter` (Rich-based, in `furnace/ui/plan_console.py`) for production and by `RecordingPlanReporter` (capture events to a list) for tests. Services (`Scanner`, `Analyzer`, `DiscDemuxer`, `PlannerService`) accept the reporter via DI and call typed methods to report phase/file/sub-step transitions and progress fractions. Existing adapter-level progress parsers (`process: NN%` for eac3to, `Progress: NN%` for mkvmerge) feed into the reporter unchanged; new parsers are added for MakeMKV (`PRGV:cur,tot,max`) and per-sample-point progress for `idet` / `cropdetect` / `audio profile` (which run multiple `ffmpeg` invocations each).

**Tech Stack:** Python 3.12+, Rich (`rich.console.Console`, `rich.progress.Progress`), pytest + coverage (line+branch), Typer, existing `furnace.adapters._subprocess.run_tool`.

**Quality gates per CLAUDE.md:**
- TDD strict — failing test first, every change.
- 100% line+branch coverage on new/touched code (`furnace/`, `tests/`).
- Tests run via `make check` only (NOT `uv run pytest` directly).
- No worktrees — work in main checkout.
- Opus only for any subagent dispatch.
- One commit per task at the end of the task. No intermediate commits inside a task.
- Version bump (`furnace/__init__.py` AND `pyproject.toml`) only on the FINAL task (Task 16) — that's the commit that flips user-facing behavior. Intermediate tasks add infra without changing what `furnace plan` prints.

---

## File Structure

**New files:**
- `furnace/ui/plan_console.py` — `RichPlanReporter` implementation (single cohesive module).
- `tests/fakes/__init__.py` — empty.
- `tests/fakes/recording_reporter.py` — `RecordingPlanReporter` test fake.
- `tests/ui/__init__.py` — empty.
- `tests/ui/test_plan_console.py` — `RichPlanReporter` rendering tests.
- `tests/ui/snapshots/canonical_plan.txt` — golden-file smoke snapshot.
- `tests/adapters/test_makemkv_progress.py` — PRGV parser tests.
- `tests/adapters/test_ffmpeg_idet_progress.py` — `run_idet` per-point progress.
- `tests/adapters/test_ffmpeg_cropdetect_progress.py` — `detect_crop` per-point progress.
- `tests/adapters/test_ffmpeg_audio_profile_progress.py` — `profile_audio_track` per-point progress.
- `tests/services/test_scanner_reports.py` — Scanner event emission.
- `tests/services/test_analyzer_reports.py` — Analyzer event emission.
- `tests/services/test_disc_demuxer_reports.py` — DiscDemuxer event emission.
- `tests/services/test_planner_reports.py` — PlannerService event emission.
- `tests/test_plan_output_integration.py` — end-to-end smoke.

**Modified files:**
- `furnace/core/ports.py` — add `PlanReporter` Protocol; add `on_progress` to `Prober.run_idet`, `Prober.detect_crop`, `Prober.profile_audio_track`.
- `furnace/adapters/_subprocess.py` — add `cancel_event: threading.Event | None` parameter to `run_tool` so plan-phase Ctrl+C can kill child processes.
- `furnace/adapters/makemkv.py` — implement `_parse_makemkv_progress_line` (PRGV/PRGT/PRGC); add `-r` flag to `demux_title`; wire `on_progress` parameter properly.
- `furnace/adapters/ffmpeg.py` — add `on_progress` to `run_idet`, `detect_crop`, `profile_audio_track`; emit `ProgressSample(fraction=points_done/total)` after each sample window.
- `furnace/services/scanner.py` — accept `PlanReporter` (constructor or method arg), emit `scan_file` events.
- `furnace/services/analyzer.py` — accept `PlanReporter`, emit `analyze_file_*` and `analyze_microop` events with progress callbacks for idet / audio profile.
- `furnace/services/disc_demuxer.py` — accept `PlanReporter` in `demux()`, emit `demux_disc_*`, `demux_title_*` and `demux_title_substep` events with progress callbacks for rip / transcode / remux.
- `furnace/services/planner.py` — accept `PlanReporter`, emit `plan_file_*` and `plan_microop("cropdetect", has_progress=True)` with progress callback.
- `furnace/cli.py` — wire `RichPlanReporter` into `plan` command, remove `_console_output`, replace `typer.echo` plan messages with reporter calls, set `console=False` in `_setup_logging` for plan, add Ctrl+C handling with `cancel_event`.
- `furnace/__init__.py` — bump `VERSION = "1.14.0"` (last task only).
- `pyproject.toml` — bump `version = "1.14.0"` (last task only).

---

### Task 1: PlanReporter Protocol + RecordingPlanReporter fake

**Files:**
- Modify: `furnace/core/ports.py` (add Protocol at end of file)
- Create: `tests/fakes/__init__.py` (empty)
- Create: `tests/fakes/recording_reporter.py`
- Create: `tests/core/test_plan_reporter_fake.py`

- [ ] **Step 1: Write the failing test** for `RecordingPlanReporter`.

Create `tests/core/test_plan_reporter_fake.py`:

```python
from pathlib import Path

from furnace.core.models import DiscType
from tests.fakes.recording_reporter import Event, RecordingPlanReporter


def test_records_method_name_and_args() -> None:
    r = RecordingPlanReporter()
    r.detect_disc(DiscType.BLURAY, "Matrix_BD")
    r.demux_disc_cached("OldMatrix_BD")
    r.demux_title_start(5)
    r.demux_title_substep("rip", has_progress=True)
    r.demux_title_progress(0.37)
    r.demux_title_done()
    r.scan_file("Inception.mkv")
    r.analyze_file_failed("HDR10+ not supported")
    r.plan_saved(tmp_path / "furnace-plan.json", 7)
    r.interrupted()
    r.pause()
    r.resume()

    assert r.events == [
        Event("detect_disc", (DiscType.BLURAY, "Matrix_BD")),
        Event("demux_disc_cached", ("OldMatrix_BD",)),
        Event("demux_title_start", (5,)),
        Event("demux_title_substep", ("rip",), (("has_progress", True),)),
        Event("demux_title_progress", (0.37,)),
        Event("demux_title_done", ()),
        Event("scan_file", ("Inception.mkv",)),
        Event("analyze_file_failed", ("HDR10+ not supported",)),
        Event("plan_saved", (tmp_path / "furnace-plan.json", 7)),
        Event("interrupted", ()),
        Event("pause", ()),
        Event("resume", ()),
    ]
```

> **Note (round-1 review):** `Event` now has three fields: `method: str`, `args: tuple[object, ...]`, `kwargs: tuple[tuple[str, object], ...] = ()`. The fake captures kwargs separately; `has_progress` is keyword-only on `demux_title_substep`, `analyze_microop`, `plan_microop`. The `RecordingPlanReporter` defines all 24 methods explicitly (not via `__getattr__`) so that `@runtime_checkable` `isinstance` works.

- [ ] **Step 2: Run test to verify it fails**

Run: `make test ARGS="tests/core/test_plan_reporter_fake.py -v"`
Expected: FAIL — `tests.fakes` and `furnace.core.ports.PlanReporter` not yet importable as needed.

(Note: existing `Makefile` likely doesn't pass ARGS through — if so, run `make test` and look for the new test name in the output. Either way, the test must run and fail.)

- [ ] **Step 3: Add `PlanReporter` Protocol to `furnace/core/ports.py`**

Append to the END of `furnace/core/ports.py`:

```python
@runtime_checkable
class PlanReporter(Protocol):
    """Structured terminal output for ``furnace plan``.

    State is implicit: after ``*_file_start(name)`` or ``demux_title_start(n)``,
    all subsequent micro-op / progress / done calls apply to that latest-started
    item. The ``plan`` pipeline is strictly serial — only one file/title is
    active at a time — so this is unambiguous.
    """

    # Detect
    def detect_disc(self, disc_type: DiscType, rel_path: str) -> None: ...

    # Demux
    def demux_disc_cached(self, label: str) -> None: ...
    def demux_disc_start(self, label: str) -> None: ...
    def demux_title_start(self, title_num: int) -> None: ...
    def demux_title_substep(self, label: str, *, has_progress: bool) -> None: ...
    def demux_title_progress(self, fraction: float) -> None: ...
    def demux_title_done(self) -> None: ...
    def demux_title_failed(self, reason: str) -> None: ...

    # Scan
    def scan_file(self, name: str) -> None: ...
    def scan_skipped(self, name: str, reason: str) -> None: ...

    # Analyze
    def analyze_file_start(self, name: str) -> None: ...
    def analyze_microop(self, label: str, *, has_progress: bool) -> None: ...
    def analyze_progress(self, fraction: float) -> None: ...
    def analyze_file_done(self, summary: str) -> None: ...
    def analyze_file_failed(self, reason: str) -> None: ...
    def analyze_file_skipped(self, reason: str) -> None: ...

    # Plan
    def plan_file_start(self, name: str) -> None: ...
    def plan_microop(self, label: str, *, has_progress: bool) -> None: ...
    def plan_progress(self, fraction: float) -> None: ...
    def plan_file_done(self, summary: str) -> None: ...

    # Final
    def plan_saved(self, path: Path, n_jobs: int) -> None: ...
    def interrupted(self) -> None: ...

    # Lifecycle (for interactive Textual TUI pauses)
    def pause(self) -> None: ...
    def resume(self) -> None: ...
```

This depends on `DiscType` (already imported via `models`).

- [ ] **Step 4: Create the fake**

Create `tests/fakes/__init__.py` as an empty file.

Create `tests/fakes/recording_reporter.py`:

```python
"""Test double for ``furnace.core.ports.PlanReporter``.

Captures every method call as ``Event(method_name, args_tuple)`` in a list,
in invocation order. Use to assert event sequences from services.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    method: str
    args: tuple[object, ...]


class RecordingPlanReporter:
    """Records every method call as an ``Event``. Returns ``None`` from all calls."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def __getattr__(self, name: str) -> Callable[..., None]:
        def _record(*args: object) -> None:
            self.events.append(Event(name, tuple(args)))

        return _record
```

- [ ] **Step 5: Run test to verify it passes**

Run: `make test`
Expected: the new test passes; all existing tests still pass.

- [ ] **Step 6: Run full quality gates**

Run: `make check`
Expected: ruff + mypy strict + pytest with 100% line+branch coverage on new files all pass clean.

- [ ] **Step 7: Commit**

```bash
git add furnace/core/ports.py tests/fakes/ tests/core/test_plan_reporter_fake.py
git commit -m "$(cat <<'EOF'
Add PlanReporter Protocol and RecordingPlanReporter test fake

Protocol declares typed events for each phase of `furnace plan`
(Detect, Demux, Scan, Analyze, Plan) plus lifecycle (pause/resume,
interrupted). State is implicit — the latest *_start call is the
active item — which works because the plan pipeline is serial.

The fake records (method, args) tuples for assertion in service tests.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: RichPlanReporter — skeleton + header + Detect rendering

**Files:**
- Create: `furnace/ui/plan_console.py`
- Create: `tests/ui/__init__.py` (empty)
- Create: `tests/ui/test_plan_console.py`

- [ ] **Step 1: Write the failing test**

Create `tests/ui/__init__.py` as an empty file.

Create `tests/ui/test_plan_console.py`:

```python
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from furnace.core.models import DiscType
from furnace.ui.plan_console import RichPlanReporter


def _make_reporter() -> tuple[RichPlanReporter, StringIO]:
    buf = StringIO()
    console = Console(
        file=buf,
        force_terminal=True,
        width=120,
        color_system=None,
        legacy_windows=False,
    )
    reporter = RichPlanReporter(
        source=Path("D:/Library"),
        output=Path("Z:/plans/library"),
        console=console,
        ascii_only=True,
    )
    return reporter, buf


def test_header_printed_once() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.stop()
    text = buf.getvalue()
    assert text.count("Source:") == 1
    assert "D:/Library" in text or "D:\\Library" in text
    assert "Z:/plans/library" in text or "Z:\\plans\\library" in text


def test_detect_renders_one_row_per_disc() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.detect_disc(DiscType.BLURAY, "Matrix_BD")
    reporter.detect_disc(DiscType.BLURAY, "OldMatrix_BD")
    reporter.detect_disc(DiscType.DVD, "DirtyHarry_DVD")
    reporter.stop()
    text = buf.getvalue()
    # Phase prefix exactly once
    assert text.count("Detect") == 1
    # All three rows present
    assert "BDMV" in text
    assert "DVD" in text
    assert "Matrix_BD" in text
    assert "OldMatrix_BD" in text
    assert "DirtyHarry_DVD" in text


def test_no_detect_block_when_no_discs() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.stop()
    text = buf.getvalue()
    assert "Detect" not in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test`
Expected: FAIL — `furnace.ui.plan_console` does not exist.

- [ ] **Step 3: Create the minimal implementation**

Create `furnace/ui/plan_console.py`:

```python
"""Structured terminal reporter for ``furnace plan``.

Owns stdout for the entire plan command. Renders phase headers, per-row
events, and a single floating progress bar at the bottom of the terminal.
Raw tool output never flows here; the reporter is fed structured events by
services and adapter progress parsers.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from furnace.core.models import DiscType

_PHASE_COL_WIDTH = 10  # "Detect    " etc.
_DISC_TYPE_NAMES: dict[DiscType, str] = {
    DiscType.BLURAY: "BDMV",
    DiscType.DVD: "DVD",
}


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
        self._console = console or Console()
        self._ascii_only = ascii_only
        self._detect_started = False

    def start(self) -> None:
        """Print the header. Called once at the beginning of ``plan``."""
        self._console.print(f"Source: {self._source}")
        self._console.print(f"Output: {self._output}")
        self._console.print()

    def stop(self) -> None:
        """Flush. Called once at end of ``plan``."""
        # No-op in the skeleton; later tasks may add Progress.stop() etc.

    # -- Detect ---------------------------------------------------------------

    def detect_disc(self, disc_type: DiscType, rel_path: str) -> None:
        prefix = "Detect    " if not self._detect_started else " " * _PHASE_COL_WIDTH
        type_name = _DISC_TYPE_NAMES[disc_type]
        self._console.print(f"{prefix}{type_name:<6} {rel_path}")
        self._detect_started = True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make test`
Expected: the three new tests pass.

- [ ] **Step 5: Run quality gates**

Run: `make check`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add furnace/ui/plan_console.py tests/ui/
git commit -m "$(cat <<'EOF'
Add RichPlanReporter skeleton with header + Detect rendering

Owns stdout for the plan command. Header prints Source/Output once;
Detect prints one row per disc with the phase prefix on the first row
only and aligned blank prefix on subsequent rows. ASCII-only mode is
forced for cmd.exe compatibility.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: RichPlanReporter — Demux rendering

**Files:**
- Modify: `furnace/ui/plan_console.py`
- Modify: `tests/ui/test_plan_console.py`

Adds: `demux_disc_cached`, `demux_disc_start`, `demux_title_start`, `demux_title_substep`, `demux_title_progress`, `demux_title_done`, `demux_title_failed`. Uses `rich.progress.Progress` for the active in-flight title row.

- [ ] **Step 1: Write the failing tests**

Append to `tests/ui/test_plan_console.py`:

```python
def test_demux_disc_cached_one_line() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.demux_disc_cached("OldMatrix_BD")
    reporter.stop()
    text = buf.getvalue()
    assert "Demux" in text
    assert "OldMatrix_BD" in text
    assert "cached" in text


def test_demux_fresh_disc_unfolds_titles() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.demux_disc_start("Matrix_BD")
    reporter.demux_title_start(3)
    reporter.demux_title_substep("rip", has_progress=True)
    reporter.demux_title_progress(0.5)
    reporter.demux_title_substep("remux", has_progress=True)
    reporter.demux_title_progress(0.5)
    reporter.demux_title_done()
    reporter.stop()
    text = buf.getvalue()
    assert "Matrix_BD" in text
    assert "title 3" in text
    # Only the final state survives the floating bar after done
    assert "done" in text


def test_demux_title_failed_renders_FAILED() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.demux_disc_start("DirtyHarry_DVD")
    reporter.demux_title_start(1)
    reporter.demux_title_substep("rip", has_progress=True)
    reporter.demux_title_failed("eac3to timeout")
    reporter.stop()
    text = buf.getvalue()
    assert "title 1" in text
    assert "FAILED" in text
    assert "eac3to timeout" in text


def test_demux_phase_prefix_appears_once() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.demux_disc_cached("A")
    reporter.demux_disc_cached("B")
    reporter.stop()
    text = buf.getvalue()
    assert text.count("Demux") == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `make test`
Expected: FAIL — Demux methods not implemented.

- [ ] **Step 3: Implement Demux rendering**

Modify `furnace/ui/plan_console.py`. Add Progress imports and Demux handling.

At the top, replace imports:

```python
from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from furnace.core.models import DiscType
```

Add bar column factory at module top (below `_DISC_TYPE_NAMES`):

```python
_ASCII_BAR = "#"
_ASCII_BG = "-"
_ASCII_SPINNER = "line"  # Rich built-in ASCII spinner: |/-\

_TITLE_INDENT = " " * (_PHASE_COL_WIDTH + 2)
_DISC_INDENT = " " * _PHASE_COL_WIDTH
```

Add to `RichPlanReporter.__init__` (replace its body):

```python
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
    self._console = console or Console()
    self._ascii_only = ascii_only
    self._detect_started = False
    self._demux_started = False
    self._current_disc_label: str | None = None
    self._current_title_num: int | None = None
    self._current_substep: str | None = None
    self._progress: Progress | None = None
    self._task_id: int | None = None
```

Add helper methods before `detect_disc`:

```python
def _ensure_progress(self) -> Progress:
    if self._progress is None:
        bar_col = BarColumn(complete_style="white", finished_style="white")
        self._progress = Progress(
            SpinnerColumn(spinner_name=_ASCII_SPINNER),
            TextColumn("{task.description}"),
            bar_col,
            TextColumn("{task.percentage:>3.0f}%"),
            console=self._console,
            transient=True,
            expand=False,
        )
        self._progress.start()
    return self._progress

def _stop_progress(self) -> None:
    if self._progress is not None:
        self._progress.stop()
        self._progress = None
        self._task_id = None
```

Replace `stop`:

```python
def stop(self) -> None:
    self._stop_progress()
```

Add Demux methods AFTER `detect_disc`:

```python
# -- Demux ---------------------------------------------------------------

def _demux_prefix(self) -> str:
    if not self._demux_started:
        self._demux_started = True
        return "Demux     "
    return _DISC_INDENT

def demux_disc_cached(self, label: str) -> None:
    self._stop_progress()
    self._console.print(f"{self._demux_prefix()}{label:<25} cached")

def demux_disc_start(self, label: str) -> None:
    self._stop_progress()
    self._console.print(f"{self._demux_prefix()}{label}")
    self._current_disc_label = label

def demux_title_start(self, title_num: int) -> None:
    self._stop_progress()
    self._current_title_num = title_num
    self._current_substep = None

def demux_title_substep(self, label: str, *, has_progress: bool) -> None:
    self._stop_progress()
    self._current_substep = label
    if self._current_title_num is None:
        return
    progress = self._ensure_progress()
    desc = f"{_TITLE_INDENT}title {self._current_title_num:<6} {label}"
    self._task_id = progress.add_task(desc, total=100 if has_progress else None)

def demux_title_progress(self, fraction: float) -> None:
    if self._progress is None or self._task_id is None:
        return
    self._progress.update(self._task_id, completed=fraction * 100)

def demux_title_done(self) -> None:
    self._stop_progress()
    if self._current_title_num is not None:
        self._console.print(
            f"{_TITLE_INDENT}title {self._current_title_num:<6} done"
        )
    self._current_title_num = None
    self._current_substep = None

def demux_title_failed(self, reason: str) -> None:
    self._stop_progress()
    if self._current_title_num is not None:
        self._console.print(
            f"{_TITLE_INDENT}title {self._current_title_num:<6} FAILED — {reason}"
        )
    self._current_title_num = None
    self._current_substep = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make test`
Expected: all four new Demux tests pass.

- [ ] **Step 5: Run quality gates**

Run: `make check`
Expected: all green. If branch coverage flags `_ensure_progress` second call without `start`, add a test that calls `demux_title_substep` twice in sequence.

- [ ] **Step 6: Commit**

```bash
git add furnace/ui/plan_console.py tests/ui/test_plan_console.py
git commit -m "$(cat <<'EOF'
Add Demux rendering to RichPlanReporter

One row per disc; cached discs collapse to a single 'cached' line;
fresh discs unfold per-title rows whose single line mutates through
sub-steps via a Rich Progress task. Done/failed transitions clear
the floating bar and emit a fixed-form row.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: RichPlanReporter — Scan + Analyze rendering

**Files:**
- Modify: `furnace/ui/plan_console.py`
- Modify: `tests/ui/test_plan_console.py`

Adds: `scan_file`, `scan_skipped`, `analyze_file_start`, `analyze_microop`, `analyze_progress`, `analyze_file_done`, `analyze_file_failed`, `analyze_file_skipped`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/ui/test_plan_console.py`:

```python
def test_scan_renders_per_file_with_phase_prefix_once() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.scan_file("Inception.mkv")
    reporter.scan_file("Tenet.mkv")
    reporter.scan_skipped("weird.mkv", "no video stream")
    reporter.stop()
    text = buf.getvalue()
    assert text.count("Scan") == 1
    assert "Inception.mkv" in text
    assert "Tenet.mkv" in text
    assert "weird.mkv" in text
    assert "SKIPPED" in text
    assert "no video stream" in text


def test_analyze_done_renders_summary() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.analyze_file_start("Inception.mkv")
    reporter.analyze_microop("probing", has_progress=False)
    reporter.analyze_file_done(
        "hevc 3840x2076 24fps HDR10 - 5 audio (rus,eng) - 12 subs"
    )
    reporter.stop()
    text = buf.getvalue()
    assert "Analyze" in text
    assert "Inception.mkv" in text
    assert "hevc 3840x2076" in text


def test_analyze_failed_inline() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.analyze_file_start("broken.mkv")
    reporter.analyze_file_failed("HDR10+ not supported")
    reporter.stop()
    text = buf.getvalue()
    assert "broken.mkv" in text
    assert "FAILED" in text
    assert "HDR10+ not supported" in text


def test_analyze_progress_with_microop_bar() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.analyze_file_start("interlaced.mkv")
    reporter.analyze_microop("idet", has_progress=True)
    reporter.analyze_progress(0.4)
    reporter.analyze_progress(1.0)
    reporter.analyze_file_done("h264 720x480 30fps SDR (interlaced) - 2 audio (eng) - 0 subs")
    reporter.stop()
    text = buf.getvalue()
    assert "interlaced.mkv" in text
    assert "h264 720x480" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `make test`
Expected: FAIL — Scan/Analyze methods not implemented.

- [ ] **Step 3: Implement Scan + Analyze rendering**

Add to `RichPlanReporter.__init__` body (after demux state):

```python
self._scan_started = False
self._analyze_started = False
self._current_file: str | None = None
```

Add methods AFTER the Demux block:

```python
# -- Scan -----------------------------------------------------------------

def _scan_prefix(self) -> str:
    if not self._scan_started:
        self._scan_started = True
        return "Scan      "
    return _DISC_INDENT

def scan_file(self, name: str) -> None:
    self._stop_progress()
    self._console.print(f"{self._scan_prefix()}{name}")

def scan_skipped(self, name: str, reason: str) -> None:
    self._stop_progress()
    self._console.print(f"{self._scan_prefix()}{name}  SKIPPED — {reason}")

# -- Analyze --------------------------------------------------------------

def _analyze_prefix(self) -> str:
    if not self._analyze_started:
        self._analyze_started = True
        return "Analyze   "
    return _DISC_INDENT

def analyze_file_start(self, name: str) -> None:
    self._stop_progress()
    self._current_file = name

def analyze_microop(self, label: str, *, has_progress: bool) -> None:
    self._stop_progress()
    if self._current_file is None:
        return
    progress = self._ensure_progress()
    desc = f"{self._analyze_prefix()}{self._current_file}  {label}"
    self._task_id = progress.add_task(desc, total=100 if has_progress else None)

def analyze_progress(self, fraction: float) -> None:
    if self._progress is None or self._task_id is None:
        return
    self._progress.update(self._task_id, completed=fraction * 100)

def analyze_file_done(self, summary: str) -> None:
    self._stop_progress()
    if self._current_file is not None:
        self._console.print(f"{self._analyze_prefix()}{self._current_file}  {summary}")
    self._current_file = None

def analyze_file_failed(self, reason: str) -> None:
    self._stop_progress()
    if self._current_file is not None:
        self._console.print(
            f"{self._analyze_prefix()}{self._current_file}  FAILED — {reason}"
        )
    self._current_file = None

def analyze_file_skipped(self, reason: str) -> None:
    self._stop_progress()
    if self._current_file is not None:
        self._console.print(
            f"{self._analyze_prefix()}{self._current_file}  SKIPPED — {reason}"
        )
    self._current_file = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make test`
Expected: all four new tests pass.

- [ ] **Step 5: Run quality gates**

Run: `make check`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add furnace/ui/plan_console.py tests/ui/test_plan_console.py
git commit -m "$(cat <<'EOF'
Add Scan + Analyze rendering to RichPlanReporter

Per-file rows with phase prefix on the first row only. Analyze rows
mutate through micro-operations via a single Rich Progress task and
finalize with a summary, FAILED, or SKIPPED line.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: RichPlanReporter — Plan rendering + final + interrupted + lifecycle

**Files:**
- Modify: `furnace/ui/plan_console.py`
- Modify: `tests/ui/test_plan_console.py`

Adds: `plan_file_start`, `plan_microop`, `plan_progress`, `plan_file_done`, `plan_saved`, `interrupted`, `pause`, `resume`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/ui/test_plan_console.py`:

```python
def test_plan_renders_per_file_and_final_summary() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.plan_file_start("Inception.mkv")
    reporter.plan_microop("cropdetect", has_progress=True)
    reporter.plan_progress(1.0)
    reporter.plan_file_done("cq 22, 3840x2076 -> 3840x1600")
    reporter.plan_saved(Path("Z:/plans/library/furnace-plan.json"), 7)
    reporter.stop()
    text = buf.getvalue()
    assert "Plan" in text
    assert "Inception.mkv" in text
    assert "cq 22" in text
    assert "furnace-plan.json" in text
    assert "(7 jobs)" in text


def test_interrupted_prints_final_line() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.plan_file_start("foo.mkv")
    reporter.plan_microop("cropdetect", has_progress=True)
    reporter.plan_progress(0.3)
    reporter.interrupted()
    text = buf.getvalue()
    assert "interrupted" in text


def test_pause_resume_stop_and_restart_progress() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.plan_file_start("foo.mkv")
    reporter.plan_microop("cropdetect", has_progress=True)
    reporter.plan_progress(0.3)
    reporter.pause()
    # After pause(), no live progress object is held
    assert reporter._progress is None  # noqa: SLF001 - reaching into impl
    reporter.resume()
    # resume() does not auto-restart Progress — next *_microop call recreates it
    reporter.plan_progress(0.6)
    reporter.plan_file_done("cq 22, 1920x1080 -> 1920x1080")
    reporter.stop()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `make test`
Expected: FAIL — Plan / interrupted / pause / resume not implemented.

- [ ] **Step 3: Implement remaining methods**

Add to `RichPlanReporter.__init__` body:

```python
self._plan_started = False
```

Add methods AFTER the Analyze block:

```python
# -- Plan -----------------------------------------------------------------

def _plan_prefix(self) -> str:
    if not self._plan_started:
        self._plan_started = True
        return "Plan      "
    return _DISC_INDENT

def plan_file_start(self, name: str) -> None:
    self._stop_progress()
    self._current_file = name

def plan_microop(self, label: str, *, has_progress: bool) -> None:
    self._stop_progress()
    if self._current_file is None:
        return
    progress = self._ensure_progress()
    desc = f"{self._plan_prefix()}{self._current_file}  {label}"
    self._task_id = progress.add_task(desc, total=100 if has_progress else None)

def plan_progress(self, fraction: float) -> None:
    if self._progress is None or self._task_id is None:
        return
    self._progress.update(self._task_id, completed=fraction * 100)

def plan_file_done(self, summary: str) -> None:
    self._stop_progress()
    if self._current_file is not None:
        self._console.print(f"{self._plan_prefix()}{self._current_file}  {summary}")
    self._current_file = None

# -- Final / lifecycle ---------------------------------------------------

def plan_saved(self, path: Path, n_jobs: int) -> None:
    self._stop_progress()
    self._console.print(f"{_DISC_INDENT}-> {path.name} ({n_jobs} jobs)")

def interrupted(self) -> None:
    self._stop_progress()
    self._console.print("interrupted")

def pause(self) -> None:
    self._stop_progress()

def resume(self) -> None:
    # Next *_microop call will recreate the Progress; nothing to do here
    return
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make test`
Expected: all three new tests pass.

- [ ] **Step 5: Run quality gates**

Run: `make check`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add furnace/ui/plan_console.py tests/ui/test_plan_console.py
git commit -m "$(cat <<'EOF'
Add Plan rendering, interrupted, and pause/resume to RichPlanReporter

Plan phase mirrors Analyze: per-file row with active cropdetect bar,
final saved-line shows just the basename + job count. interrupted()
clears the live Progress and prints a terminal 'interrupted' line.
pause()/resume() let the interactive Textual selectors take over the
terminal between phases without stale progress artifacts.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: RichPlanReporter — non-TTY mode + golden snapshot

**Files:**
- Modify: `tests/ui/test_plan_console.py`
- Create: `tests/ui/snapshots/canonical_plan.txt`

Verifies non-TTY behavior (no ANSI when piped) and adds a golden-file smoke snapshot.

- [ ] **Step 1: Write the failing tests**

Append to `tests/ui/test_plan_console.py`:

```python
def test_non_tty_output_has_no_ansi_escapes() -> None:
    buf = StringIO()
    console = Console(
        file=buf,
        force_terminal=False,
        width=120,
        color_system=None,
        legacy_windows=False,
    )
    reporter = RichPlanReporter(
        source=Path("D:/L"),
        output=Path("Z:/p"),
        console=console,
        ascii_only=True,
    )
    reporter.start()
    reporter.detect_disc(DiscType.BLURAY, "Matrix_BD")
    reporter.scan_file("Inception.mkv")
    reporter.analyze_file_start("Inception.mkv")
    reporter.analyze_microop("idet", has_progress=True)
    reporter.analyze_progress(0.5)
    reporter.analyze_file_done("hevc 3840x2076 24fps HDR10 - 5 audio (rus,eng) - 12 subs")
    reporter.plan_saved(Path("Z:/p/furnace-plan.json"), 1)
    reporter.stop()
    text = buf.getvalue()
    assert "\x1b[" not in text  # no ANSI escape sequences


def test_canonical_plan_golden_snapshot() -> None:
    buf = StringIO()
    console = Console(
        file=buf,
        force_terminal=False,
        width=120,
        color_system=None,
        legacy_windows=False,
    )
    reporter = RichPlanReporter(
        source=Path("D:/Library"),
        output=Path("Z:/plans/library"),
        console=console,
        ascii_only=True,
    )
    reporter.start()
    reporter.detect_disc(DiscType.BLURAY, "Matrix_BD")
    reporter.detect_disc(DiscType.DVD, "DirtyHarry_DVD")
    reporter.demux_disc_start("Matrix_BD")
    reporter.demux_title_start(3)
    reporter.demux_title_substep("rip", has_progress=True)
    reporter.demux_title_progress(1.0)
    reporter.demux_title_substep("remux", has_progress=True)
    reporter.demux_title_progress(1.0)
    reporter.demux_title_done()
    reporter.demux_disc_start("DirtyHarry_DVD")
    reporter.demux_title_start(1)
    reporter.demux_title_substep("rip", has_progress=True)
    reporter.demux_title_progress(1.0)
    reporter.demux_title_done()
    reporter.scan_file("Matrix_BD_title_3.mkv")
    reporter.scan_file("DirtyHarry_DVD_title_1.mkv")
    reporter.analyze_file_start("Matrix_BD_title_3.mkv")
    reporter.analyze_microop("probing", has_progress=False)
    reporter.analyze_file_done("hevc 3840x2160 24fps HDR10 - 4 audio (eng) - 0 subs")
    reporter.analyze_file_start("DirtyHarry_DVD_title_1.mkv")
    reporter.analyze_microop("idet", has_progress=True)
    reporter.analyze_progress(1.0)
    reporter.analyze_file_done("mpeg2video 720x480 30fps SDR (interlaced) - 2 audio (eng) - 1 subs")
    reporter.plan_file_start("Matrix_BD_title_3.mkv")
    reporter.plan_microop("cropdetect", has_progress=True)
    reporter.plan_progress(1.0)
    reporter.plan_file_done("cq 22, 3840x2160 -> 3840x2160")
    reporter.plan_file_start("DirtyHarry_DVD_title_1.mkv")
    reporter.plan_microop("cropdetect", has_progress=True)
    reporter.plan_progress(1.0)
    reporter.plan_file_done("cq 19, 720x480 -> 720x540, deinterlace")
    reporter.plan_saved(Path("Z:/plans/library/furnace-plan.json"), 2)
    reporter.stop()
    actual = buf.getvalue()
    expected = (Path(__file__).parent / "snapshots" / "canonical_plan.txt").read_text(encoding="utf-8")
    assert actual == expected
```

- [ ] **Step 2: Run the non-TTY test to verify it passes**

Run: `make test`
Expected: `test_non_tty_output_has_no_ansi_escapes` passes; `test_canonical_plan_golden_snapshot` FAILS (snapshot file missing).

- [ ] **Step 3: Capture the golden snapshot**

Add a one-shot helper that writes the snapshot. Easiest: temporarily change the assertion in `test_canonical_plan_golden_snapshot` to write `actual` to `snapshots/canonical_plan.txt`, run once, then revert the assertion. Concrete recipe — at the end of the test, replace the last two lines with:

```python
    snap = Path(__file__).parent / "snapshots" / "canonical_plan.txt"
    snap.parent.mkdir(parents=True, exist_ok=True)
    snap.write_text(actual, encoding="utf-8")
    raise AssertionError("snapshot written; remove this and re-run")
```

Run `make test`, watch the test fail with the message above, then restore the original assertion. Inspect `tests/ui/snapshots/canonical_plan.txt` — every line must look like the spec mockup. If anything is wrong (extra escapes, wrong indentation), fix `RichPlanReporter` before locking in the snapshot.

- [ ] **Step 4: Run all tests with the locked snapshot**

Run: `make test`
Expected: golden snapshot test passes; all other tests still pass.

- [ ] **Step 5: Run quality gates**

Run: `make check`
Expected: all green, 100% coverage on `furnace/ui/plan_console.py`.

- [ ] **Step 6: Commit**

```bash
git add furnace/ui/plan_console.py tests/ui/test_plan_console.py tests/ui/snapshots/
git commit -m "$(cat <<'EOF'
Verify RichPlanReporter non-TTY mode and add golden snapshot

Non-TTY output (force_terminal=False) emits no ANSI escapes —
verified explicitly. The golden snapshot at
tests/ui/snapshots/canonical_plan.txt covers the canonical happy-path
flow as a smoke regression check on overall format.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: MakeMKV PRGV progress parser

**Files:**
- Modify: `furnace/adapters/makemkv.py`
- Create: `tests/adapters/test_makemkv_progress.py`

Adds `_parse_makemkv_progress_line` modelled on `_parse_eac3to_progress_line` (`eac3to.py:26`). Wires `on_progress` through `run_tool` only for `demux_title` (not `list_titles` — the existing `_TITLE_ADDED_RE` parser would break with `-r`).

- [ ] **Step 1: Write the failing tests**

Create `tests/adapters/test_makemkv_progress.py`:

```python
from furnace.adapters.makemkv import _parse_makemkv_progress_line


def test_prgv_basic() -> None:
    sample = _parse_makemkv_progress_line("PRGV:5,10,100")
    assert sample is not None
    assert sample.fraction == 0.05


def test_prgv_complete() -> None:
    sample = _parse_makemkv_progress_line("PRGV:100,100,100")
    assert sample is not None
    assert sample.fraction == 1.0


def test_prgv_zero_max_returns_none() -> None:
    assert _parse_makemkv_progress_line("PRGV:5,10,0") is None


def test_prgt_returns_none() -> None:
    assert _parse_makemkv_progress_line('PRGT:0,0,"Saving to MKV"') is None


def test_prgc_returns_none() -> None:
    assert _parse_makemkv_progress_line('PRGC:0,0,"Backing up disc"') is None


def test_msg_returns_none() -> None:
    assert _parse_makemkv_progress_line('MSG:1004,0,1,"Some message"') is None


def test_garbage_returns_none() -> None:
    assert _parse_makemkv_progress_line("not a progress line") is None
    assert _parse_makemkv_progress_line("PRGV:abc,def,ghi") is None
    assert _parse_makemkv_progress_line("") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `make test`
Expected: FAIL — `_parse_makemkv_progress_line` does not exist.

- [ ] **Step 3: Implement the parser and wire it into `demux_title`**

Modify `furnace/adapters/makemkv.py`:

1. Add the parser function below `_parse_duration` (around line 28):

```python
_PRGV_RE = re.compile(r"^PRGV:(\d+),(\d+),(\d+)\s*$")


def _parse_makemkv_progress_line(line: str) -> ProgressSample | None:
    """Parse a makemkvcon ``-r`` (robot mode) ``PRGV:current,total,max`` line.

    ``max`` is the total work for the overall task; ``current`` is overall
    progress against ``max``. ``total`` describes the current sub-task scale
    and is ignored here. Returns ``None`` for any other line shape (PRGT,
    PRGC, MSG, malformed, empty).
    """
    m = _PRGV_RE.match(line.strip())
    if not m:
        return None
    current = int(m.group(1))
    max_val = int(m.group(3))
    if max_val == 0:
        return None
    return ProgressSample(fraction=current / max_val)
```

2. Update `demux_title` to add `-r` and wire `on_progress`. Replace the existing method body (`makemkv.py:64-119`):

```python
def demux_title(
    self,
    disc_path: Path,
    title_num: int,
    output_dir: Path,
    on_progress: Callable[[ProgressSample], None] | None = None,
) -> list[Path]:
    """Demux one DVD title to MKV via makemkvcon -r mkv.

    ``-r`` (robot mode) emits structured PRGV/PRGT/PRGC/MSG lines; the
    progress parser consumes PRGV and feeds ``on_progress``. Other lines
    flow to the per-tool log file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    titles = self.list_titles(disc_path)
    index = None
    for i, t in enumerate(titles):
        if t.number == title_num:
            index = i
            break
    if index is None:
        raise RuntimeError(f"Title {title_num} not found in makemkvcon listing for {disc_path}")

    before = set(output_dir.iterdir())

    cmd = [
        str(self._makemkvcon),
        "-r",
        "--noscan",
        "mkv",
        f"file:{disc_path}",
        str(index),
        str(output_dir),
    ]

    def _on_progress_line(line: str) -> bool:
        sample = _parse_makemkv_progress_line(line)
        if sample is None:
            return False
        if on_progress is not None:
            on_progress(sample)
        return True

    rc, _output = run_tool(
        cmd,
        on_output=self._on_output,
        on_progress_line=_on_progress_line,
        log_path=self._log_path(f"demux_t{title_num}"),
    )
    if rc != 0:
        raise RuntimeError(f"makemkvcon demux failed for {disc_path} title {title_num} (rc={rc})")

    after = set(output_dir.iterdir())
    new_files = sorted(p for p in (after - before) if p.is_file() and p.suffix.lower() == ".mkv")
    if not new_files:
        raise RuntimeError(f"makemkvcon produced no MKV files for {disc_path} title {title_num}")
    return new_files
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make test`
Expected: parser tests pass; existing `tests/services/test_disc_demuxer*.py` (if any) still pass.

- [ ] **Step 5: Run quality gates**

Run: `make check`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add furnace/adapters/makemkv.py tests/adapters/test_makemkv_progress.py
git commit -m "$(cat <<'EOF'
Add MakeMKV PRGV progress parser; wire -r into demux_title

makemkvcon in -r (robot) mode emits PRGV:cur,tot,max lines; we parse
the cur/max fraction and surface it as ProgressSample. PRGT/PRGC/MSG
return None — only PRGV is structured progress.

Only demux_title gets -r; list_titles keeps human-readable mode so
the existing _TITLE_ADDED_RE parser still works.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: ffmpeg per-sample-point progress for run_idet

**Files:**
- Modify: `furnace/core/ports.py` (add `on_progress` param to `Prober.run_idet`)
- Modify: `furnace/adapters/ffmpeg.py` (`run_idet` body)
- Create: `tests/adapters/test_ffmpeg_idet_progress.py`

`run_idet` runs `ffmpeg` 5 times (one per sample point). Each invocation is short enough that we just emit `ProgressSample(fraction=points_done/total)` after each one — no need for `-progress pipe:1` parsing.

- [ ] **Step 1: Write the failing test**

Create `tests/adapters/test_ffmpeg_idet_progress.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

from furnace.adapters.ffmpeg import FFmpegAdapter
from furnace.core.progress import ProgressSample


def test_run_idet_calls_on_progress_after_each_sample_point() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    samples: list[ProgressSample] = []

    fake_result = MagicMock()
    fake_result.stderr = "Multi frame detection: TFF: 0 BFF: 0 Progressive: 1000\n"
    fake_result.returncode = 0

    with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=fake_result):
        adapter.run_idet(Path("/tmp/x.mkv"), duration_s=1000.0, on_progress=samples.append)

    # 5 sample points: 10/30/50/70/90 — fractions 0.2, 0.4, 0.6, 0.8, 1.0
    assert [round(s.fraction or 0, 1) for s in samples] == [0.2, 0.4, 0.6, 0.8, 1.0]


def test_run_idet_works_without_on_progress() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    fake_result = MagicMock()
    fake_result.stderr = "Multi frame detection: TFF: 0 BFF: 0 Progressive: 1000\n"
    fake_result.returncode = 0
    with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=fake_result):
        ratio = adapter.run_idet(Path("/tmp/x.mkv"), duration_s=1000.0)
    assert ratio == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test`
Expected: FAIL — `on_progress` param does not exist.

- [ ] **Step 3: Add `on_progress` to `Prober.run_idet` Protocol**

In `furnace/core/ports.py`, replace the `run_idet` Protocol method with:

```python
def run_idet(
    self,
    path: Path,
    duration_s: float,
    on_progress: Callable[[ProgressSample], None] | None = None,
) -> float:
    """Run idet analysis. Returns interlaced frame ratio (0.0 to 1.0).

    ``on_progress`` is called after each sample point with a fraction
    (``points_done / total_points``).
    """
    ...
```

- [ ] **Step 4: Update `FFmpegAdapter.run_idet`**

Modify `furnace/adapters/ffmpeg.py`. Replace `run_idet` (current `ffmpeg.py:261-306`):

```python
def run_idet(
    self,
    path: Path,
    duration_s: float,
    on_progress: Callable[[ProgressSample], None] | None = None,
) -> float:
    """Run idet filter at multiple points across the timeline.

    Samples 1000 frames at 10%, 30%, 50%, 70%, 90% of duration.
    Returns the ratio of interlaced frames (0.0 to 1.0). After each
    sample point, calls ``on_progress`` with a fraction.
    """
    points = (0.10, 0.30, 0.50, 0.70, 0.90)
    total_interlaced = 0
    total_prog = 0

    for i, pct in enumerate(points, start=1):
        seek = duration_s * pct
        cmd = [
            str(self._ffmpeg),
            "-hide_banner",
            "-ss",
            f"{seek:.2f}",
            "-i",
            str(path),
            "-vf",
            "idet",
            "-frames:v",
            "1000",
            "-f",
            "null",
            "-",
        ]
        logger.debug("run_idet cmd: %s", cmd)
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
        )

        for line in result.stderr.splitlines():
            m = re.search(
                r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*Progressive:\s*(\d+)",
                line,
            )
            if m:
                total_interlaced += int(m.group(1)) + int(m.group(2))
                total_prog += int(m.group(3))

        if on_progress is not None:
            on_progress(ProgressSample(fraction=i / len(points)))

    total = total_interlaced + total_prog
    if total == 0:
        return 0.0

    return total_interlaced / total
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `make test`
Expected: new tests pass; existing `tests/adapters/test_ffmpeg*.py` and `tests/services/test_analyzer*.py` still pass (they call `run_idet` without `on_progress`, default is `None`).

- [ ] **Step 6: Run quality gates**

Run: `make check`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add furnace/core/ports.py furnace/adapters/ffmpeg.py tests/adapters/test_ffmpeg_idet_progress.py
git commit -m "$(cat <<'EOF'
Add per-sample-point on_progress callback to Prober.run_idet

run_idet runs ffmpeg 5 times across the timeline; emit
ProgressSample(fraction=i/5) after each iteration so the reporter can
show a smooth bar. Default on_progress=None is a no-op for callers
that don't care.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: ffmpeg per-sample-point progress for detect_crop

**Files:**
- Modify: `furnace/core/ports.py` (`Prober.detect_crop`)
- Modify: `furnace/adapters/ffmpeg.py` (`detect_crop`)
- Create: `tests/adapters/test_ffmpeg_cropdetect_progress.py`

Same pattern as Task 8. `detect_crop` iterates over 10 (HD) or 15 (DVD) sample points.

- [ ] **Step 1: Write the failing test**

Create `tests/adapters/test_ffmpeg_cropdetect_progress.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

from furnace.adapters.ffmpeg import FFmpegAdapter
from furnace.core.progress import ProgressSample


def test_detect_crop_calls_on_progress_per_point_hd() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    samples: list[ProgressSample] = []
    fake_result = MagicMock()
    fake_result.stderr = "[Parsed_cropdetect_0 @ 0x0] crop=3840:1600:0:280\n"
    fake_result.returncode = 0
    with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=fake_result):
        adapter.detect_crop(
            Path("/tmp/x.mkv"),
            duration_s=1000.0,
            interlaced=False,
            is_dvd=False,
            on_progress=samples.append,
        )
    # HD has 10 sample points → 10 progress events ending at 1.0
    assert len(samples) == 10
    assert samples[-1].fraction == 1.0


def test_detect_crop_calls_on_progress_per_point_dvd() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    samples: list[ProgressSample] = []
    fake_result = MagicMock()
    fake_result.stderr = "[Parsed_cropdetect_0 @ 0x0] crop=720:480:0:0\n"
    fake_result.returncode = 0
    with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=fake_result):
        adapter.detect_crop(
            Path("/tmp/x.mkv"),
            duration_s=1000.0,
            interlaced=False,
            is_dvd=True,
            on_progress=samples.append,
        )
    # DVD has 15 sample points → 15 progress events
    assert len(samples) == 15
    assert samples[-1].fraction == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `make test`
Expected: FAIL — `detect_crop` does not accept `on_progress`.

- [ ] **Step 3: Update `Prober.detect_crop` Protocol**

In `furnace/core/ports.py`, replace `detect_crop`:

```python
def detect_crop(
    self,
    path: Path,
    duration_s: float,
    *,
    interlaced: bool = False,
    is_dvd: bool = False,
    on_progress: Callable[[ProgressSample], None] | None = None,
) -> CropRect | None:
    """Run cropdetect, return detected values (before alignment).

    ``on_progress`` is called after each sample point.
    """
    ...
```

- [ ] **Step 4: Update `FFmpegAdapter.detect_crop`**

Modify `furnace/adapters/ffmpeg.py`. Replace the `detect_crop` signature line and add the `on_progress` call. The exact change: after the inner loop body where each `last_crop` is captured (still inside `for pct in points`), add at the end of the loop iteration:

```python
        if on_progress is not None:
            on_progress(ProgressSample(fraction=(point_idx + 1) / len(points)))
```

Replace the `for pct in points:` line with `for point_idx, pct in enumerate(points):` and add the new signature parameter:

```python
def detect_crop(
    self,
    path: Path,
    duration_s: float,
    *,
    interlaced: bool = False,
    is_dvd: bool = False,
    on_progress: Callable[[ProgressSample], None] | None = None,
) -> CropRect | None:
```

(All other body lines unchanged.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `make test`
Expected: new tests pass; existing planner tests still pass.

- [ ] **Step 6: Run quality gates**

Run: `make check`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add furnace/core/ports.py furnace/adapters/ffmpeg.py tests/adapters/test_ffmpeg_cropdetect_progress.py
git commit -m "$(cat <<'EOF'
Add per-sample-point on_progress callback to Prober.detect_crop

detect_crop iterates 10 (HD) or 15 (DVD) sample points; emit
ProgressSample(fraction=i/N) after each. Lets the reporter render a
real progress bar during cropdetect — the slowest part of the Plan
phase.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: ffmpeg per-sample-point progress for profile_audio_track

**Files:**
- Modify: `furnace/core/ports.py` (`Prober.profile_audio_track`)
- Modify: `furnace/adapters/ffmpeg.py` (`profile_audio_track`)
- Create: `tests/adapters/test_ffmpeg_audio_profile_progress.py`

Same pattern. Stereo: 2 points, 5.1/7.1: 4 points.

- [ ] **Step 1: Write the failing test**

Create `tests/adapters/test_ffmpeg_audio_profile_progress.py`:

```python
from pathlib import Path
from unittest.mock import patch

import numpy as np

from furnace.adapters.ffmpeg import FFmpegAdapter
from furnace.core.progress import ProgressSample


def _fake_window(channels: int) -> np.ndarray:
    return np.zeros((48000, channels), dtype=np.float32)


def test_profile_audio_track_stereo_emits_2_progress_events() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    samples: list[ProgressSample] = []
    with patch.object(
        FFmpegAdapter,
        "_decode_pcm_window",
        side_effect=lambda *a, **k: _fake_window(2),
    ):
        adapter.profile_audio_track(
            Path("/tmp/x.mkv"),
            stream_index=1,
            channels=2,
            duration_s=1000.0,
            on_progress=samples.append,
        )
    assert len(samples) == 2
    assert samples[-1].fraction == 1.0


def test_profile_audio_track_5_1_emits_4_progress_events() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    samples: list[ProgressSample] = []
    with patch.object(
        FFmpegAdapter,
        "_decode_pcm_window",
        side_effect=lambda *a, **k: _fake_window(6),
    ):
        adapter.profile_audio_track(
            Path("/tmp/x.mkv"),
            stream_index=1,
            channels=6,
            duration_s=1000.0,
            on_progress=samples.append,
        )
    assert len(samples) == 4
    assert samples[-1].fraction == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `make test`
Expected: FAIL — `profile_audio_track` does not accept `on_progress`.

- [ ] **Step 3: Update `Prober.profile_audio_track` Protocol**

In `furnace/core/ports.py`, replace `profile_audio_track`:

```python
def profile_audio_track(
    self,
    path: Path,
    stream_index: int,
    channels: int,
    duration_s: float,
    on_progress: Callable[[ProgressSample], None] | None = None,
) -> AudioMetrics:
    """Sample PCM windows from an audio stream, compute per-channel RMS
    and pairwise correlations, and return raw measurements.

    channels must be 2, 6, or 8; other counts raise ValueError.
    duration_s is used to pick sample offsets.
    ``on_progress`` is called after each window decode with a fraction.

    Raises RuntimeError if no windows decoded successfully.
    """
    ...
```

- [ ] **Step 4: Update `FFmpegAdapter.profile_audio_track`**

In `furnace/adapters/ffmpeg.py`, modify `profile_audio_track` (current `ffmpeg.py:484-562`):

1. Add `on_progress` parameter to signature.
2. After each `chunks.append(window)` (or whether or not it appended), call `on_progress(ProgressSample(fraction=(i+1)/len(points)))`.

Replace the loop:

```python
    chunks: list[np.ndarray] = []
    for i, frac in enumerate(points):
        start = max(0.0, duration_s * frac - _PROFILE_WINDOW_SEC / 2)
        window = self._decode_pcm_window(
            path, stream_index, channels, layout, start, _PROFILE_WINDOW_SEC,
        )
        if window.size > 0:
            chunks.append(window)
        if on_progress is not None:
            on_progress(ProgressSample(fraction=(i + 1) / len(points)))
```

Add `on_progress: Callable[[ProgressSample], None] | None = None,` to the method signature.

- [ ] **Step 5: Run tests to verify they pass**

Run: `make test`
Expected: new tests pass; existing audio-profile tests still pass.

- [ ] **Step 6: Run quality gates**

Run: `make check`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add furnace/core/ports.py furnace/adapters/ffmpeg.py tests/adapters/test_ffmpeg_audio_profile_progress.py
git commit -m "$(cat <<'EOF'
Add per-sample-point on_progress callback to Prober.profile_audio_track

Emits ProgressSample(fraction=i/N) after each window decode (2 for
stereo, 4 for 5.1/7.1). Reporter shows a real bar during the slow
'audio profile track N' micro-op of Analyze.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Subprocess cancel_event for plan-phase Ctrl+C

**Files:**
- Modify: `furnace/adapters/_subprocess.py`
- Create: `tests/adapters/test_subprocess_cancel.py`

Adds a `cancel_event: threading.Event | None` parameter to `run_tool`. When set, the running child process is killed and `run_tool` returns. Used by `cli.plan` to wire `KeyboardInterrupt` cleanly.

- [ ] **Step 1: Write the failing test**

Create `tests/adapters/test_subprocess_cancel.py`:

```python
import sys
import threading
import time

from furnace.adapters._subprocess import run_tool


def test_cancel_event_kills_child_promptly() -> None:
    cancel = threading.Event()

    def _trigger() -> None:
        time.sleep(0.2)
        cancel.set()

    threading.Thread(target=_trigger, daemon=True).start()

    start = time.monotonic()
    rc, _ = run_tool(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        cancel_event=cancel,
    )
    elapsed = time.monotonic() - start
    assert elapsed < 5.0
    assert rc != 0


def test_no_cancel_event_runs_to_completion() -> None:
    rc, output = run_tool(
        [sys.executable, "-c", "print('hello')"],
    )
    assert rc == 0
    assert "hello" in output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `make test`
Expected: FAIL — `cancel_event` parameter does not exist.

- [ ] **Step 3: Add cancel support to `run_tool`**

Modify `furnace/adapters/_subprocess.py`. Update `run_tool` signature:

```python
def run_tool(
    cmd: Sequence[str | Path],
    on_output: OutputCallback = None,
    on_progress_line: Callable[[str], bool] | None = None,
    log_path: Path | None = None,
    cwd: Path | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[int, str]:
```

Replace the `process.wait()` block (around line 124) with a polled wait that checks `cancel_event`:

```python
        # Polled wait so we can react to cancel_event.
        if cancel_event is not None:
            while process.poll() is None:
                if cancel_event.is_set():
                    process.kill()
                    break
                cancel_event.wait(timeout=0.1)
        process.wait()
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make test`
Expected: both new tests pass; all existing `tests/adapters/*` still pass.

- [ ] **Step 5: Run quality gates**

Run: `make check`
Expected: all green. If branch coverage flags the cancel_event path, the second test (`test_no_cancel_event_runs_to_completion`) covers the None branch.

- [ ] **Step 6: Commit**

```bash
git add furnace/adapters/_subprocess.py tests/adapters/test_subprocess_cancel.py
git commit -m "$(cat <<'EOF'
Add cancel_event parameter to run_tool

A threading.Event consumers can set to kill the running child
subprocess promptly. Used by cli.plan's KeyboardInterrupt handler;
default None preserves existing behavior for everyone else.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Scanner emits PlanReporter events

**Files:**
- Modify: `furnace/services/scanner.py`
- Create: `tests/services/test_scanner_reports.py`

`Scanner.scan` walks the filesystem; we add per-file `scan_file(name)` events. The reporter is optional (`None` = silent, preserves headless behavior).

- [ ] **Step 1: Write the failing test**

Create `tests/services/test_scanner_reports.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock

from furnace.services.scanner import Scanner
from tests.fakes.recording_reporter import Event, RecordingPlanReporter


def test_scan_emits_one_scan_file_per_video(tmp_path: Path) -> None:
    (tmp_path / "Inception.mkv").touch()
    (tmp_path / "Tenet.mkv").touch()
    (tmp_path / "notes.txt").touch()  # not a video, must be skipped

    reporter = RecordingPlanReporter()
    scanner = Scanner(prober=MagicMock(), reporter=reporter)
    results = scanner.scan(tmp_path, tmp_path / "out")

    assert len(results) == 2
    file_events = [e for e in reporter.events if e.method == "scan_file"]
    names = sorted(e.args[0] for e in file_events)
    assert names == ["Inception.mkv", "Tenet.mkv"]


def test_scan_without_reporter_is_silent(tmp_path: Path) -> None:
    (tmp_path / "x.mkv").touch()
    scanner = Scanner(prober=MagicMock())  # no reporter
    results = scanner.scan(tmp_path, tmp_path / "out")
    assert len(results) == 1  # still works
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `make test`
Expected: FAIL — `Scanner` does not accept `reporter`.

- [ ] **Step 3: Update Scanner**

Modify `furnace/services/scanner.py`. Update imports:

```python
from furnace.core.models import ScanResult
from furnace.core.ports import PlanReporter, Prober
```

Update `__init__` and `scan`:

```python
class Scanner:
    def __init__(self, prober: Prober, reporter: PlanReporter | None = None) -> None:
        self._prober = prober
        self._reporter = reporter

    def scan(
        self,
        source: Path,
        dest: Path,
        names_map: dict[str, str] | None = None,
    ) -> list[ScanResult]:
        results: list[ScanResult] = []

        if source.is_file():
            if source.suffix.lower() in VIDEO_EXTENSIONS:
                satellites = self.find_satellites(source)
                output_path = self.build_output_path(source, source.parent, dest, names_map)
                results.append(
                    ScanResult(
                        main_file=source,
                        satellite_files=satellites,
                        output_path=output_path,
                    )
                )
                if self._reporter is not None:
                    self._reporter.scan_file(source.name)
            return results

        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            if ".furnace_demux" in path.parts:
                continue
            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            satellites = self.find_satellites(path)
            output_path = self.build_output_path(path, source, dest, names_map)
            results.append(
                ScanResult(
                    main_file=path,
                    satellite_files=satellites,
                    output_path=output_path,
                )
            )
            if self._reporter is not None:
                rel = path.relative_to(source)
                self._reporter.scan_file(str(rel))
            logger.debug("Scanned %s -> %s (%d satellites)", path, output_path, len(satellites))

        logger.debug("Scan complete: %d video files found in %s", len(results), source)
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make test`
Expected: new tests pass; existing scanner tests still pass.

- [ ] **Step 5: Run quality gates**

Run: `make check`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add furnace/services/scanner.py tests/services/test_scanner_reports.py
git commit -m "$(cat <<'EOF'
Scanner accepts PlanReporter and emits scan_file per video

reporter is optional; default None preserves headless behavior. Path
in event is relative to source, matching the spec ('paths relative
to source').

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Analyzer emits PlanReporter events

**Files:**
- Modify: `furnace/services/analyzer.py`
- Create: `tests/services/test_analyzer_reports.py`

`Analyzer.analyze` orchestrates ffprobe + idet + audio profile. We emit:
- `analyze_file_start(name)` at entry.
- `analyze_microop("probing", has_progress=False)` before ffprobe.
- `analyze_microop("HDR side data", has_progress=False)` if PQ/HLG transfer.
- `analyze_microop("idet", has_progress=True)` if needs_idet, with progress passed through.
- `analyze_microop("audio profile track N", has_progress=True)` per profileable audio track, with progress passed through.
- `analyze_file_done(summary)` / `analyze_file_failed(reason)` / `analyze_file_skipped(reason)` at exit.

- [ ] **Step 1: Write the failing tests**

Create `tests/services/test_analyzer_reports.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock

from furnace.core.models import ScanResult
from furnace.services.analyzer import Analyzer
from tests.fakes.recording_reporter import Event, RecordingPlanReporter


def _make_scan(name: str = "x.mkv") -> ScanResult:
    return ScanResult(
        main_file=Path(f"/tmp/{name}"),
        satellite_files=[],
        output_path=Path(f"/out/{name}"),
    )


def _make_prober_simple_sdr() -> MagicMock:
    prober = MagicMock()
    prober.get_encoder_tag.return_value = None
    prober.probe.return_value = {
        "streams": [
            {
                "codec_type": "video",
                "index": 0,
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "24/1",
                "r_frame_rate": "24/1",
                "duration": "100",
                "color_primaries": "bt709",
                "color_transfer": "bt709",
                "color_space": "bt709",
                "pix_fmt": "yuv420p",
                "field_order": "progressive",
                "sample_aspect_ratio": "1:1",
                "side_data_list": [],
            }
        ],
        "format": {},
        "chapters": [],
    }
    return prober


def test_simple_sdr_emits_start_probing_done() -> None:
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=_make_prober_simple_sdr(), reporter=reporter)
    movie = analyzer.analyze(_make_scan("Inception.mkv"))
    assert movie is not None
    methods = [e.method for e in reporter.events]
    assert methods[0] == "analyze_file_start"
    assert reporter.events[0].args == ("Inception.mkv",)
    assert ("analyze_microop", ("probing",), (("has_progress", False),)) in [(e.method, e.args, e.kwargs) for e in reporter.events]
    assert methods[-1] == "analyze_file_done"


def test_skip_already_encoded_emits_skipped() -> None:
    prober = _make_prober_simple_sdr()
    prober.get_encoder_tag.return_value = "furnace 1.13.2"
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    movie = analyzer.analyze(_make_scan("foo.mkv"))
    assert movie is None
    assert reporter.events[-1].method == "analyze_file_skipped"


def test_no_video_stream_emits_skipped() -> None:
    prober = _make_prober_simple_sdr()
    prober.probe.return_value["streams"] = [{"codec_type": "audio"}]
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    movie = analyzer.analyze(_make_scan("foo.mkv"))
    assert movie is None
    assert reporter.events[-1].method == "analyze_file_skipped"
    assert "no video stream" in reporter.events[-1].args[0]


def test_hdr10_emits_hdr_side_data_microop() -> None:
    prober = _make_prober_simple_sdr()
    prober.probe.return_value["streams"][0]["color_transfer"] = "smpte2084"
    prober.probe_hdr_side_data.return_value = []
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    analyzer.analyze(_make_scan("hdr.mkv"))
    labels = [e.args[0] for e in reporter.events if e.method == "analyze_microop"]
    assert "HDR side data" in labels


def test_analyze_without_reporter_is_silent() -> None:
    analyzer = Analyzer(prober=_make_prober_simple_sdr())  # no reporter
    movie = analyzer.analyze(_make_scan("x.mkv"))
    assert movie is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `make test`
Expected: FAIL — `Analyzer` does not accept `reporter`.

- [ ] **Step 3: Update Analyzer**

Modify `furnace/services/analyzer.py`. Update import:

```python
from furnace.core.ports import PlanReporter, Prober
```

Update `__init__`:

```python
class Analyzer:
    def __init__(self, prober: Prober, reporter: PlanReporter | None = None) -> None:
        self._prober = prober
        self._reporter = reporter
```

In `analyze()`, instrument the body. At the very top:

```python
def analyze(self, scan_result: ScanResult) -> Movie | None:
    main_file = scan_result.main_file
    output_path = scan_result.output_path
    name = main_file.name
    if self._reporter is not None:
        self._reporter.analyze_file_start(name)

    # Skip check
    encoder_tag = self._prober.get_encoder_tag(main_file)
    skip, reason = should_skip_file(output_path, encoder_tag)
    if skip:
        logger.info("Skipping %s: %s", name, reason)
        if self._reporter is not None:
            self._reporter.analyze_file_skipped(reason)
        return None
```

Before `probe_data = self._prober.probe(main_file)` add:

```python
    if self._reporter is not None:
        self._reporter.analyze_microop("probing", has_progress=False)
```

Wrap that probe in try/except so we report failure:

```python
    try:
        probe_data = self._prober.probe(main_file)
    except (OSError, RuntimeError, ValueError):
        logger.exception("Failed to probe %s", main_file)
        if self._reporter is not None:
            self._reporter.analyze_file_failed("probe failed")
        return None
```

After `if not video_streams:` block add reporter call before `return None`:

```python
    if not video_streams:
        logger.warning("No video stream found in %s, skipping", name)
        if self._reporter is not None:
            self._reporter.analyze_file_skipped("no video stream")
        return None
```

In `_parse_video_info` block — wrap so we report `analyze_file_failed` on parse failure:

```python
    try:
        video_info = self._parse_video_info(video_stream, format_data, main_file)
    except (KeyError, ValueError, IndexError, TypeError):
        logger.exception("Failed to parse video info for %s", main_file)
        if self._reporter is not None:
            self._reporter.analyze_file_failed("parse failed")
        return None
```

Before the `if video_info.hdr.is_hdr10_plus:` raise, surface as failed:

```python
    if video_info.hdr.is_hdr10_plus:
        if self._reporter is not None:
            self._reporter.analyze_file_failed("HDR10+ not supported")
        raise ValueError(f"HDR10+ not supported: {name}")
```

For HDR side data, the existing logic is implicit in `_parse_video_info` (line 233 calls `probe_hdr_side_data` for PQ/HLG). To surface this as a microop, we need to know in `analyze` whether HDR side data WILL be probed. The cheapest fix: read the raw transfer BEFORE calling `_parse_video_info` and emit the microop:

Right before the `_parse_video_info` try block above, add:

```python
    color_transfer_raw = video_stream.get("color_transfer")
    if color_transfer_raw in ("smpte2084", "arib-std-b67") and self._reporter is not None:
        self._reporter.analyze_microop("HDR side data", has_progress=False)
```

For idet — the existing `if needs_idet(...)` block (line 125) is where idet runs. Wrap it:

```python
    if needs_idet(field_order_raw, fps):
        if self._reporter is not None:
            self._reporter.analyze_microop("idet", has_progress=True)
        try:
            on_progress = (
                lambda s: self._reporter.analyze_progress(s.fraction or 0.0)
                if self._reporter is not None and s.fraction is not None
                else None
            )
            idet_ratio = self._prober.run_idet(main_file, video_info.duration_s, on_progress)
            logger.debug("%s: idet ratio %.3f", name, idet_ratio)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("idet failed for %s: %s", name, exc)
```

For each audio profile loop (line 142), wrap each iteration:

```python
    for track in audio_tracks:
        if track.channels not in _PROFILEABLE_CHANNEL_COUNTS:
            continue
        if self._reporter is not None:
            self._reporter.analyze_microop(
                f"audio profile track {track.index}", has_progress=True,
            )
        logger.info(
            "Profiling audio track %d (%s %s %dch)",
            track.index, track.codec_name, track.language, track.channels,
        )
        try:
            on_progress = (
                lambda s: self._reporter.analyze_progress(s.fraction or 0.0)
                if self._reporter is not None and s.fraction is not None
                else None
            )
            metrics = self._prober.profile_audio_track(
                path=main_file,
                stream_index=track.index,
                channels=track.channels,
                duration_s=video_info.duration_s,
                on_progress=on_progress,
            )
            track.audio_profile = classify_audio(metrics)
        except Exception as exc:
            logger.warning(
                "profile_audio_track failed for track %d: %s", track.index, exc,
            )
            continue
        logger.info(
            "Profiled track %d: %s (score %d)",
            track.index, track.audio_profile.verdict.value, track.audio_profile.score,
        )
```

Right before `return Movie(...)` at the end of `analyze()`, build the summary and emit done:

```python
    if self._reporter is not None:
        summary = _format_analyze_summary(video_info, audio_tracks, subtitle_tracks)
        self._reporter.analyze_file_done(summary)

    return Movie(...)
```

Add `_format_analyze_summary` as a module-level helper (below imports, near top of file):

```python
def _format_analyze_summary(
    video: VideoInfo,
    audio_tracks: list[Track],
    subtitle_tracks: list[Track],
) -> str:
    """Build the one-line summary shown for a successfully analyzed file.

    Format: ``codec WxH FPSfps HDR-CLASS [interlaced] - N audio (langs) - N subs``
    """
    fps = video.fps_num // video.fps_den if video.fps_den else 0
    hdr = _hdr_class(video)
    parts = [
        video.codec_name,
        f"{video.width}x{video.height}",
        f"{fps}fps",
        hdr,
    ]
    if video.interlaced:
        parts.append("(interlaced)")
    head = " ".join(parts)

    audio_langs = sorted({t.language for t in audio_tracks if t.language})
    audio = f"{len(audio_tracks)} audio ({','.join(audio_langs)})" if audio_langs else f"{len(audio_tracks)} audio"
    subs = f"{len(subtitle_tracks)} subs"
    return f"{head} - {audio} - {subs}"


def _hdr_class(video: VideoInfo) -> str:
    if video.hdr.is_dolby_vision:
        bl_map = {1: "HDR10", 2: "SDR", 4: "HLG"}
        bl = bl_map.get(int(video.hdr.dv_bl_compatibility) if video.hdr.dv_bl_compatibility else 0, "none")
        prof = video.hdr.dv_profile or "?"
        return f"DV P{prof} (BL={bl})"
    if video.color_transfer == "smpte2084":
        return "HDR10"
    if video.color_transfer == "arib-std-b67":
        return "HLG"
    return "SDR"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make test`
Expected: new tests pass; existing analyzer tests still pass.

- [ ] **Step 5: Run quality gates**

Run: `make check`
Expected: all green. If branch coverage flags either reporter-None branch, ensure each `if self._reporter is not None:` block is exercised by both the recording-reporter and silent tests above; add small targeted tests if gaps remain.

- [ ] **Step 6: Commit**

```bash
git add furnace/services/analyzer.py tests/services/test_analyzer_reports.py
git commit -m "$(cat <<'EOF'
Analyzer emits PlanReporter events

Threads the reporter through analyze() so each file emits start ->
microop transitions (probing, optional 'HDR side data', optional
'idet', per-track 'audio profile track N') -> done/failed/skipped.
Progress fractions from the new ffmpeg per-point callbacks are
forwarded to analyze_progress.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: DiscDemuxer emits PlanReporter events

**Files:**
- Modify: `furnace/services/disc_demuxer.py`
- Create: `tests/services/test_disc_demuxer_reports.py`

Adds reporter parameter to `demux()` and emits per-disc + per-title events with sub-step transitions.

- [ ] **Step 1: Write the failing tests**

Create `tests/services/test_disc_demuxer_reports.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from furnace.core.models import DiscSource, DiscTitle, DiscType
from furnace.services.disc_demuxer import DiscDemuxer
from tests.fakes.recording_reporter import Event, RecordingPlanReporter


def _make_disc(tmp_path: Path, name: str, dtype: DiscType) -> DiscSource:
    leaf = "BDMV" if dtype == DiscType.BLURAY else "VIDEO_TS"
    p = tmp_path / name / leaf
    p.mkdir(parents=True)
    return DiscSource(path=p, disc_type=dtype)


def test_cached_disc_emits_one_cached_event(tmp_path: Path) -> None:
    disc = _make_disc(tmp_path, "OldMatrix_BD", DiscType.BLURAY)
    title = DiscTitle(number=2, duration_s=3600.0, raw_label="2) 1:00:00")

    demux_dir = tmp_path / "demux"
    demux_dir.mkdir()
    # Pre-create the .done marker and the resulting MKV
    (demux_dir / "OldMatrix_BD_title_2.done").touch()
    (demux_dir / "OldMatrix_BD_title_2.mkv").write_bytes(b"\x00")

    bd_port = MagicMock()
    dvd_port = MagicMock()
    reporter = RecordingPlanReporter()
    demuxer = DiscDemuxer(
        bd_port=bd_port,
        dvd_port=dvd_port,
        mkvmerge_path=Path("mkvmerge"),
    )
    demuxer.demux(
        discs=[disc],
        selected_titles={disc: [title]},
        demux_dir=demux_dir,
        reporter=reporter,
    )
    methods = [e.method for e in reporter.events]
    assert methods == ["demux_disc_cached"]
    assert reporter.events[0].args == ("OldMatrix_BD",)


def test_fresh_dvd_title_emits_rip_only(tmp_path: Path) -> None:
    disc = _make_disc(tmp_path, "DirtyHarry_DVD", DiscType.DVD)
    title = DiscTitle(number=1, duration_s=3600.0, raw_label="1) 1:00:00")

    demux_dir = tmp_path / "demux"
    demux_dir.mkdir()

    dvd_port = MagicMock()

    def _fake_demux(disc_path: Path, title_num: int, output_dir: Path,
                    on_progress=None) -> list[Path]:
        out = output_dir / "title.mkv"
        out.write_bytes(b"\x00")
        if on_progress is not None:
            from furnace.core.progress import ProgressSample
            on_progress(ProgressSample(fraction=0.5))
            on_progress(ProgressSample(fraction=1.0))
        return [out]

    dvd_port.demux_title.side_effect = _fake_demux

    reporter = RecordingPlanReporter()
    demuxer = DiscDemuxer(
        bd_port=MagicMock(),
        dvd_port=dvd_port,
        mkvmerge_path=Path("mkvmerge"),
    )
    demuxer.demux(
        discs=[disc],
        selected_titles={disc: [title]},
        demux_dir=demux_dir,
        reporter=reporter,
    )
    methods = [e.method for e in reporter.events]
    assert methods[0] == "demux_disc_start"
    assert methods[1] == "demux_title_start"
    assert ("demux_title_substep", ("rip",), (("has_progress", True),)) in [(e.method, e.args, e.kwargs) for e in reporter.events]
    assert methods[-1] == "demux_title_done"
    # No remux for DVD (single MKV output)
    assert ("demux_title_substep", ("remux",), (("has_progress", True),)) not in [(e.method, e.args, e.kwargs) for e in reporter.events]


def test_demux_failure_emits_failed_then_propagates(tmp_path: Path) -> None:
    disc = _make_disc(tmp_path, "Broken_BD", DiscType.BLURAY)
    title = DiscTitle(number=5, duration_s=3600.0, raw_label="5) 1:00:00")
    demux_dir = tmp_path / "demux"
    demux_dir.mkdir()

    bd_port = MagicMock()
    bd_port.demux_title.side_effect = RuntimeError("eac3to crashed")

    reporter = RecordingPlanReporter()
    demuxer = DiscDemuxer(
        bd_port=bd_port,
        dvd_port=MagicMock(),
        mkvmerge_path=Path("mkvmerge"),
    )
    with pytest.raises(RuntimeError, match="eac3to crashed"):
        demuxer.demux(
            discs=[disc],
            selected_titles={disc: [title]},
            demux_dir=demux_dir,
            reporter=reporter,
        )
    methods = [e.method for e in reporter.events]
    assert "demux_title_failed" in methods


def test_demux_without_reporter_is_silent(tmp_path: Path) -> None:
    """Reporter is optional — pipeline still works without it."""
    disc = _make_disc(tmp_path, "X", DiscType.DVD)
    title = DiscTitle(number=1, duration_s=3600.0, raw_label="1) 1:00:00")
    demux_dir = tmp_path / "demux"
    demux_dir.mkdir()

    dvd_port = MagicMock()
    def _fake_demux(disc_path, title_num, output_dir, on_progress=None) -> list[Path]:
        out = output_dir / "x.mkv"
        out.write_bytes(b"\x00")
        return [out]
    dvd_port.demux_title.side_effect = _fake_demux

    demuxer = DiscDemuxer(
        bd_port=MagicMock(),
        dvd_port=dvd_port,
        mkvmerge_path=Path("mkvmerge"),
    )
    # Just don't crash
    demuxer.demux(
        discs=[disc],
        selected_titles={disc: [title]},
        demux_dir=demux_dir,
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `make test`
Expected: FAIL — `demux()` does not accept `reporter`.

- [ ] **Step 3: Update DiscDemuxer**

Modify `furnace/services/disc_demuxer.py`. Update imports:

```python
from furnace.core.ports import DiscDemuxerPort, PcmTranscoder, PlanReporter
from furnace.core.progress import ProgressSample
```

Update `demux()` signature:

```python
def demux(
    self,
    discs: list[DiscSource],
    selected_titles: dict[DiscSource, list[DiscTitle]],
    demux_dir: Path,
    on_output: Callable[[str], None] | None = None,
    reporter: PlanReporter | None = None,
) -> list[Path]:
```

Then instrument the body. Replace the per-disc / per-title loop:

```python
    demux_dir.mkdir(parents=True, exist_ok=True)
    result_paths: list[Path] = []

    for disc in discs:
        titles = selected_titles.get(disc, [])
        disc_label = disc.path.parent.name
        port = self._port_for(disc)

        # Determine cached-ness up front: a disc is cached only if EVERY
        # selected title already has a .done marker.
        all_cached = all(
            (demux_dir / f"{disc_label}_title_{t.number}.done").exists()
            for t in titles
        )
        if titles and all_cached:
            if reporter is not None:
                reporter.demux_disc_cached(disc_label)
            for title in titles:
                existing = self._find_done_files(demux_dir, disc_label, title.number)
                result_paths.extend(existing)
            continue

        if reporter is not None:
            reporter.demux_disc_start(disc_label)

        for title in titles:
            done_name = f"{disc_label}_title_{title.number}.done"
            done_marker = demux_dir / done_name

            if done_marker.exists():
                existing = self._find_done_files(demux_dir, disc_label, title.number)
                if existing:
                    logger.info("Already demuxed, skipping: title %d", title.number)
                    result_paths.extend(existing)
                    continue

            if reporter is not None:
                reporter.demux_title_start(title.number)

            self._clean_partial(demux_dir, disc_label, title.number)
            logger.info("Demuxing title %d from %s", title.number, disc.path)

            title_dir = demux_dir / f"{disc_label}_title_{title.number}"
            if title_dir.exists():
                shutil.rmtree(title_dir)
            title_dir.mkdir()

            def _rip_progress(s: ProgressSample) -> None:
                if reporter is not None and s.fraction is not None:
                    reporter.demux_title_progress(s.fraction)

            try:
                if reporter is not None:
                    reporter.demux_title_substep("rip", has_progress=True)
                created_files = port.demux_title(
                    disc.path, title.number, title_dir, on_progress=_rip_progress,
                )
                created_files = self._transcode_w64_files(
                    created_files, reporter=reporter,
                )

                final_mkv = demux_dir / f"{disc_label}_title_{title.number}.mkv"
                if self._needs_muxing(created_files):
                    if reporter is not None:
                        reporter.demux_title_substep("remux", has_progress=True)
                    self._mux_to_mkv(created_files, final_mkv, on_output)
                else:
                    src_mkv = next(f for f in created_files if f.suffix.lower() == ".mkv")
                    shutil.move(str(src_mkv), str(final_mkv))

                shutil.rmtree(title_dir, ignore_errors=True)
                done_marker.touch()
                result_paths.append(final_mkv)

                if reporter is not None:
                    reporter.demux_title_done()
            except Exception as exc:
                if reporter is not None:
                    reporter.demux_title_failed(str(exc))
                raise

    return result_paths
```

Update `_transcode_w64_files` to accept reporter and emit substep events:

```python
def _transcode_w64_files(
    self,
    files: list[Path],
    reporter: PlanReporter | None = None,
) -> list[Path]:
    if not any(f.suffix.lower() == ".w64" for f in files):
        return files
    if self._pcm_transcoder is None:
        w64_names = [f.name for f in files if f.suffix.lower() == ".w64"]
        msg = (
            "pcm_transcoder not configured; cannot handle Wave64 demux "
            f"output: {w64_names}"
        )
        raise RuntimeError(msg)

    w64_files = [f for f in files if f.suffix.lower() == ".w64"]
    total = len(w64_files)
    result: list[Path] = []
    w64_seen = 0
    for f in files:
        if f.suffix.lower() != ".w64":
            result.append(f)
            continue
        w64_seen += 1
        if reporter is not None:
            reporter.demux_title_substep(f"transcode {w64_seen}/{total}", has_progress=True)

        def _tr_progress(s: ProgressSample) -> None:
            if reporter is not None and s.fraction is not None:
                reporter.demux_title_progress(s.fraction)

        flac_path = f.with_suffix(".flac")
        logger.info("Transcoding Wave64 to FLAC: %s -> %s", f.name, flac_path.name)
        rc = self._pcm_transcoder.transcode_to_flac(
            f, flac_path, on_progress=_tr_progress,
        )
        if rc != 0:
            msg = f"eac3to transcode of {f.name} to FLAC failed (rc={rc})"
            raise RuntimeError(msg)
        f.unlink()
        result.append(flac_path)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make test`
Expected: new tests pass; existing `test_disc_demux*.py` still pass.

- [ ] **Step 5: Run quality gates**

Run: `make check`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add furnace/services/disc_demuxer.py tests/services/test_disc_demuxer_reports.py
git commit -m "$(cat <<'EOF'
DiscDemuxer accepts and emits PlanReporter events

demux() takes an optional reporter; per-disc cached/start, per-title
start/substep(rip|transcode N/M|remux)/progress/done/failed events
are emitted in the order the spec requires. Failure inside a title
emits demux_title_failed before re-raising.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 15: PlannerService emits PlanReporter events

**Files:**
- Modify: `furnace/services/planner.py`
- Create: `tests/services/test_planner_reports.py`

Adds reporter param. For each movie:
- `plan_file_start(name)`
- `plan_microop("cropdetect", has_progress=True)` + `plan_progress(...)` (only if `not dry_run`)
- `plan_file_done(summary)` with the input→output resolution + cq + optional deinterlace.

- [ ] **Step 1: Write the failing tests**

Create `tests/services/test_planner_reports.py`:

```python
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

from furnace.core.models import (
    Attachment,
    HdrMetadata,
    Movie,
    VideoInfo,
)
from furnace.services.planner import PlannerService
from tests.fakes.recording_reporter import Event, RecordingPlanReporter


def _make_video_info() -> VideoInfo:
    return VideoInfo(
        index=0, codec_name="hevc", width=1920, height=1080,
        pixel_area=1920 * 1080, fps_num=24, fps_den=1, duration_s=100.0,
        interlaced=False,
        color_matrix_raw="bt709", color_range="tv",
        color_transfer="bt709", color_primaries="bt709",
        pix_fmt="yuv420p10le",
        hdr=HdrMetadata(),
        source_file=Path("/in/x.mkv"),
        bitrate=8_000_000,
        sar_num=1, sar_den=1,
    )


def _make_movie() -> Movie:
    return Movie(
        main_file=Path("/in/x.mkv"),
        satellite_files=[],
        video=_make_video_info(),
        audio_tracks=[],
        subtitle_tracks=[],
        attachments=[],
        has_chapters=False,
        file_size=1_000_000,
    )


def test_plan_file_emits_start_microop_done() -> None:
    prober = MagicMock()
    prober.detect_crop.return_value = None
    reporter = RecordingPlanReporter()
    planner = PlannerService(prober=prober, previewer=None, reporter=reporter)
    planner.create_plan(
        movies=[(_make_movie(), Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        vmaf_enabled=False,
        dry_run=False,
    )
    methods = [e.method for e in reporter.events]
    assert "plan_file_start" in methods
    assert ("plan_microop", ("cropdetect",), (("has_progress", True),)) in [(e.method, e.args, e.kwargs) for e in reporter.events]
    assert methods[-2] == "plan_file_done"  # last is plan_saved
    assert methods[-1] == "plan_saved"


def test_plan_dry_run_skips_cropdetect_microop() -> None:
    prober = MagicMock()
    reporter = RecordingPlanReporter()
    planner = PlannerService(prober=prober, previewer=None, reporter=reporter)
    planner.create_plan(
        movies=[(_make_movie(), Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        vmaf_enabled=False,
        dry_run=True,
    )
    labels = [e.args[0] for e in reporter.events if e.method == "plan_microop"]
    assert "cropdetect" not in labels  # args[0] is the label; has_progress is in kwargs


def test_plan_without_reporter_is_silent() -> None:
    prober = MagicMock()
    prober.detect_crop.return_value = None
    planner = PlannerService(prober=prober, previewer=None)  # no reporter
    plan = planner.create_plan(
        movies=[(_make_movie(), Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        vmaf_enabled=False,
        dry_run=False,
    )
    assert len(plan.jobs) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `make test`
Expected: FAIL — `PlannerService` does not accept `reporter`.

- [ ] **Step 3: Update PlannerService**

Modify `furnace/services/planner.py`. Update import:

```python
from furnace.core.ports import PlanReporter, Previewer, Prober
from furnace.core.progress import ProgressSample
```

Update `__init__`:

```python
def __init__(
    self,
    prober: Prober,
    previewer: Previewer | None,
    track_selector: TrackSelectorFn | None = None,
    und_resolver: UndLanguageResolverFn | None = None,
    reporter: PlanReporter | None = None,
) -> None:
    self._prober = prober
    self._previewer = previewer
    self._track_selector = track_selector
    self._und_resolver = und_resolver
    self._reporter = reporter
```

In `create_plan`, instrument the per-movie loop. Replace the existing loop:

```python
    for movie, output_path in movies:
        if self._reporter is not None:
            self._reporter.plan_file_start(movie.main_file.name)
        job = self._build_job(
            movie,
            output_path,
            audio_lang_filter,
            sub_lang_filter,
            dry_run=dry_run,
            sar_overrides=effective_sar_overrides,
            downmix_overrides=effective_overrides,
        )
        if self._reporter is not None:
            summary = _format_plan_summary(movie, job)
            self._reporter.plan_file_done(summary)
        jobs.append(job)
```

Inside `_build_job`, instrument the cropdetect block. Replace the cropdetect call with:

```python
    crop: CropRect | None = None
    if not dry_run:
        try:
            is_dvd = is_dvd_resolution(movie.video.width, movie.video.height)
            if self._reporter is not None:
                self._reporter.plan_microop("cropdetect", has_progress=True)
            on_progress = (
                lambda s: self._reporter.plan_progress(s.fraction or 0.0)
                if self._reporter is not None and s.fraction is not None
                else None
            )
            raw_crop = self._prober.detect_crop(
                movie.main_file,
                movie.video.duration_s,
                interlaced=movie.video.interlaced,
                is_dvd=is_dvd,
                on_progress=on_progress,
            )
            # ...rest unchanged
```

After the loop, add the `plan_saved` emission. The current code does NOT save here — `save_plan` is called from `cli.plan`. So `plan_saved` must be emitted by the CLI, not by the planner. **Adjust the test:** `plan_saved` is the responsibility of `cli.plan`, not the planner. Update the first test's last assertion:

```python
    methods = [e.method for e in reporter.events]
    assert "plan_file_start" in methods
    assert ("plan_microop", ("cropdetect",), (("has_progress", True),)) in [(e.method, e.args, e.kwargs) for e in reporter.events]
    assert methods[-1] == "plan_file_done"  # last event from PlannerService
```

(Remove the `plan_saved` assertion from this test — that goes in Task 16's integration test.)

Add `_format_plan_summary` as a module-level helper at the top of planner.py (after the imports):

```python
def _format_plan_summary(movie: Movie, job: Job) -> str:
    """One-line per-movie summary shown after Plan completes for that movie.

    Format: ``cq <CQ>, <SrcW>x<SrcH> -> <DstW>x<DstH>[, deinterlace]``
    """
    src_w = movie.video.width
    src_h = movie.video.height
    if job.video_params.crop is not None:
        dst_w = job.video_params.crop.w
        dst_h = job.video_params.crop.h
    else:
        dst_w, dst_h = src_w, src_h
    parts = [
        f"cq {job.video_params.cq}",
        f"{src_w}x{src_h} -> {dst_w}x{dst_h}",
    ]
    if job.video_params.deinterlace:
        parts.append("deinterlace")
    return ", ".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make test`
Expected: new tests pass; existing planner tests still pass.

- [ ] **Step 5: Run quality gates**

Run: `make check`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add furnace/services/planner.py tests/services/test_planner_reports.py
git commit -m "$(cat <<'EOF'
PlannerService emits PlanReporter events

Per-movie plan_file_start -> plan_microop('cropdetect', True) +
plan_progress -> plan_file_done(summary). The summary format
matches the spec: cq N, src -> dst[, deinterlace], where dst
reflects crop (if any) or source dims. dry_run skips the cropdetect
microop entirely.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 16: cli.plan integration + Ctrl+C handling + version bump + integration test

**Files:**
- Modify: `furnace/cli.py`
- Modify: `furnace/__init__.py` (version bump)
- Modify: `pyproject.toml` (version bump)
- Create: `tests/test_plan_output_integration.py`

This is the user-visible flip. After this commit, `furnace plan` prints the new structured output. Bump to `1.14.0` (MINOR — new feature).

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_plan_output_integration.py`:

```python
"""Smoke integration test: real services + recording reporter, fake adapters.

Bypasses typer; calls the reporter-aware portion of cli.plan directly
through the same wiring it uses, with all external tools stubbed.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from furnace.core.models import (
    DiscSource,
    DiscTitle,
    DiscType,
    HdrMetadata,
    Movie,
    Plan,
    ScanResult,
    VideoInfo,
)
from furnace.services.analyzer import Analyzer
from furnace.services.disc_demuxer import DiscDemuxer
from furnace.services.planner import PlannerService
from furnace.services.scanner import Scanner
from tests.fakes.recording_reporter import RecordingPlanReporter


def _stub_prober() -> MagicMock:
    p = MagicMock()
    p.get_encoder_tag.return_value = None
    p.probe.return_value = {
        "streams": [{
            "codec_type": "video", "index": 0, "codec_name": "h264",
            "width": 1920, "height": 1080,
            "avg_frame_rate": "24/1", "r_frame_rate": "24/1",
            "duration": "100",
            "color_primaries": "bt709", "color_transfer": "bt709",
            "color_space": "bt709", "pix_fmt": "yuv420p",
            "field_order": "progressive", "sample_aspect_ratio": "1:1",
            "side_data_list": [],
        }],
        "format": {},
        "chapters": [],
    }
    p.detect_crop.return_value = None
    return p


def test_plan_emits_full_event_sequence(tmp_path: Path) -> None:
    # Source dir: one MKV file (no discs).
    src = tmp_path / "src"
    src.mkdir()
    (src / "Inception.mkv").touch()
    out = tmp_path / "out"

    reporter = RecordingPlanReporter()
    prober = _stub_prober()

    scanner = Scanner(prober=prober, reporter=reporter)
    scan_results = scanner.scan(src, out)
    assert len(scan_results) == 1

    analyzer = Analyzer(prober=prober, reporter=reporter)
    movies = []
    for sr in scan_results:
        m = analyzer.analyze(sr)
        if m is not None:
            movies.append((m, sr.output_path))

    planner = PlannerService(prober=prober, previewer=None, reporter=reporter)
    plan = planner.create_plan(
        movies=movies,
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        vmaf_enabled=False,
        dry_run=False,
    )

    methods = [e.method for e in reporter.events]
    assert methods[0] == "scan_file"
    assert "analyze_file_start" in methods
    assert "analyze_file_done" in methods
    assert "plan_file_start" in methods
    assert ("plan_microop", ("cropdetect",), (("has_progress", True),)) in [(e.method, e.args, e.kwargs) for e in reporter.events]
    assert "plan_file_done" in methods
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test`
Expected: ALL referenced services accept `reporter` (from prior tasks), so the test should already pass on the wiring side. If it doesn't, fix whatever wiring step you missed before continuing.

- [ ] **Step 3: Wire RichPlanReporter into `cli.plan`**

Modify `furnace/cli.py`. The changes form a coherent diff against the current `plan` function (`cli.py:369-494`):

1. Add import at top:

```python
from .ui.plan_console import RichPlanReporter
```

2. Delete `_console_output` (lines 96-99):

```python
# DELETE this entire function:
# def _console_output(line: str) -> None:
#     """Echo a line of tool output to stderr."""
#     typer.echo(line, err=True)
```

3. Update `_collect_selected_titles` to remove the `typer.echo("[furnace] Listing titles ...")` call (line 198):

```python
def _collect_selected_titles(
    detected_discs: list[DiscSource],
    disc_demuxer: DiscDemuxer,
    *,
    playlist_app_runner: Callable[
        [Callable[[], Screen[list[DiscTitle]]]], list[DiscTitle] | None
    ] = _run_screen_app,
) -> dict[DiscSource, list[DiscTitle]]:
    selected_titles: dict[DiscSource, list[DiscTitle]] = {}
    for disc in detected_discs:
        playlists = disc_demuxer.list_titles(disc)
        if not playlists:
            logger.warning("No playlists found for disc at %s", disc.path)
            continue
        if len(playlists) == 1:
            selected_titles[disc] = playlists
            continue
        disc_label = disc.path.parent.name

        def _factory(
            _disc_label: str = disc_label,
            _playlists: list[DiscTitle] = playlists,
        ) -> Screen[list[DiscTitle]]:
            return PlaylistSelectorScreen(disc_label=_disc_label, playlists=_playlists)

        picked = playlist_app_runner(_factory)
        if picked:
            selected_titles[disc] = picked
    return selected_titles
```

4. Update `_run_disc_demux_interactive` to remove `typer.echo` calls (lines 270, 282) and to NOT pass `on_output`:

```python
def _run_disc_demux_interactive(
    *,
    source: Path,
    detected_discs: list[DiscSource],
    disc_demuxer: DiscDemuxer,
    ffmpeg_adapter: FFmpegAdapter,
    mpv_adapter: MpvAdapter,
    reporter: RichPlanReporter | None = None,
    playlist_app_runner: Callable[
        [Callable[[], Screen[list[DiscTitle]]]], list[DiscTitle] | None
    ] = _run_screen_app,
    file_app_runner: Callable[
        [Callable[[], Screen[FileSelection]]], FileSelection | None
    ] = _run_screen_app,
) -> tuple[Path | None, list[Path], set[Path]]:
    if not detected_discs:
        return None, [], set()

    if reporter is not None:
        reporter.pause()
    selected_titles = _collect_selected_titles(
        detected_discs,
        disc_demuxer,
        playlist_app_runner=playlist_app_runner,
    )
    if reporter is not None:
        reporter.resume()

    if not selected_titles:
        return None, [], set()

    demux_dir = source / ".furnace_demux"
    demuxed_paths = disc_demuxer.demux(
        discs=detected_discs,
        selected_titles=selected_titles,
        demux_dir=demux_dir,
        reporter=reporter,
    )

    dvd_demuxed = _dvd_demuxed_paths(detected_discs, selected_titles, demuxed_paths)
    sar_override_paths: set[Path] = set()

    if dvd_demuxed or len(demuxed_paths) > 1:
        if reporter is not None:
            reporter.pause()
        file_infos = _probe_file_infos(demuxed_paths, ffmpeg_adapter)

        def _factory(
            _file_infos: list[tuple[Path, float, int]] = file_infos,
            _dvd: set[Path] = dvd_demuxed,
        ) -> Screen[FileSelection]:
            return FileSelectorScreen(
                files=_file_infos,
                dvd_files=_dvd,
                preview_cb=lambda p, a: mpv_adapter.preview_file(p, aspect_override=a),
            )

        file_selection = file_app_runner(_factory)
        if reporter is not None:
            reporter.resume()
        if file_selection is not None:
            demuxed_paths = file_selection.selected
            sar_override_paths = file_selection.sar_override

    return demux_dir, demuxed_paths, sar_override_paths
```

5. Replace the `plan` function body. The new structure:

```python
@app.command()
def plan(
    source: Path = typer.Argument(..., help="Video file or directory"),
    output: Path = typer.Option(..., "-o", help="Output directory"),
    audio_lang: str = typer.Option(
        ..., "--audio-lang", "-al", help="Audio languages, comma-separated (e.g. jpn or rus,eng)"
    ),
    sub_lang: str = typer.Option(..., "--sub-lang", "-sl", help="Subtitle languages, comma-separated (e.g. rus,eng)"),
    names: Path | None = typer.Option(None, "--names", help="Rename map file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan without saving"),
    vmaf: bool = typer.Option(False, "--vmaf", help="Enable VMAF"),
    config: Path | None = typer.Option(None, "--config", help="Path to config file"),
) -> None:
    """Scan source, show TUI for track selection, save JSON plan."""
    audio_lang_list = [x.strip() for x in audio_lang.split(",") if x.strip()]
    sub_lang_list = [x.strip() for x in sub_lang.split(",") if x.strip()]

    cfg = load_config(config)

    output.mkdir(parents=True, exist_ok=True)
    _setup_logging(output, console=False)  # console handler removed; reporter owns terminal

    logger.debug(
        "plan command started: source=%s output=%s audio_lang=%s sub_lang=%s names=%s dry_run=%s vmaf=%s",
        source, output, audio_lang, sub_lang, names, dry_run, vmaf,
    )

    reporter = RichPlanReporter(source=source, output=output)
    reporter.start()

    try:
        ffmpeg_adapter = FFmpegAdapter(cfg.ffmpeg, cfg.ffprobe)
        mpv_adapter = MpvAdapter(cfg.mpv)
        eac3to_adapter = Eac3toAdapter(cfg.eac3to)
        makemkv_adapter = MakemkvAdapter(cfg.makemkvcon)

        disc_demuxer = DiscDemuxer(
            bd_port=eac3to_adapter,
            dvd_port=makemkv_adapter,
            mkvmerge_path=cfg.mkvmerge,
            pcm_transcoder=eac3to_adapter,
        )

        detected_discs = disc_demuxer.detect(source)
        for disc in detected_discs:
            try:
                rel = disc.path.parent.relative_to(source)
                rel_str = str(rel) if str(rel) != "." else disc.path.parent.name
            except ValueError:
                rel_str = disc.path.parent.name
            reporter.detect_disc(disc.disc_type, rel_str)

        demux_dir: Path | None = None
        demuxed_paths: list[Path] = []
        sar_override_paths: set[Path] = set()

        if not dry_run:
            demux_dir, demuxed_paths, sar_override_paths = _run_disc_demux_interactive(
                source=source,
                detected_discs=detected_discs,
                disc_demuxer=disc_demuxer,
                ffmpeg_adapter=ffmpeg_adapter,
                mpv_adapter=mpv_adapter,
                reporter=reporter,
            )

        names_map: dict[str, str] | None = None
        if names is not None:
            with names.open("r", encoding="utf-8") as f:
                names_map = json.load(f)

        scanner = Scanner(prober=ffmpeg_adapter, reporter=reporter)
        scan_results = scanner.scan(source, output, names_map)
        _append_demuxed_scan_results(scan_results, demuxed_paths, output)
        # The appended demuxed entries also deserve scan_file events
        for mkv_path in demuxed_paths:
            reporter.scan_file(mkv_path.name)

        analyzer = Analyzer(prober=ffmpeg_adapter, reporter=reporter)
        movies_with_paths: list[tuple[Movie, Path]] = []
        for sr in scan_results:
            try:
                movie = analyzer.analyze(sr)
            except ValueError as exc:
                # analyze() raises for HDR10+; reporter already saw analyze_file_failed
                logger.warning("analyze raised: %s", exc)
                continue
            if movie is not None:
                movies_with_paths.append((movie, sr.output_path))

        downmix_overrides: dict[tuple[Path, int], DownmixMode] = {}

        def _track_selector(movie: Movie, candidates: list[Track], track_type: TrackType) -> list[Track]:
            return _select_tracks_tui_for_planner(movie, candidates, track_type, mpv_adapter, downmix_overrides)

        def _und_resolver(movie: Movie, track: Track, lang_list: list[str]) -> str:
            return _resolve_und_language_tui(movie, track, lang_list, mpv_adapter)

        if not dry_run:
            reporter.pause()
        planner = PlannerService(
            prober=ffmpeg_adapter,
            previewer=mpv_adapter,
            track_selector=_track_selector if not dry_run else None,
            und_resolver=_und_resolver if not dry_run else None,
            reporter=reporter,
        )
        if not dry_run:
            reporter.resume()

        plan_obj = planner.create_plan(
            movies=movies_with_paths,
            audio_lang_filter=audio_lang_list,
            sub_lang_filter=sub_lang_list,
            vmaf_enabled=vmaf,
            dry_run=dry_run,
            sar_overrides=sar_override_paths,
            downmix_overrides=downmix_overrides,
        )
        _apply_demux_dir_to_plan(plan_obj, demux_dir)

        if dry_run:
            reporter.plan_saved(output / "furnace-plan.json", len(plan_obj.jobs))
        else:
            plan_path = output / "furnace-plan.json"
            save_plan(plan_obj, plan_path)
            reporter.plan_saved(plan_path, len(plan_obj.jobs))

        logger.debug("plan command finished: jobs=%d", len(plan_obj.jobs))
    except KeyboardInterrupt:
        reporter.interrupted()
        raise typer.Exit(code=130)
    finally:
        reporter.stop()
```

6. Bump version. Modify `furnace/__init__.py` (current `VERSION = "1.13.2"` per git log of last commit):

```python
VERSION = "1.14.0"
```

7. Modify `pyproject.toml`:

```toml
version = "1.14.0"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make test`
Expected: integration test passes; all prior tests still pass.

- [ ] **Step 5: Run quality gates**

Run: `make check`
Expected: all green, 100% line+branch coverage. If `cli.py` coverage drops because the `KeyboardInterrupt` block is unreachable in tests, add a small unit test that monkeypatches `disc_demuxer.detect` to raise `KeyboardInterrupt` and asserts `typer.Exit(130)` propagates.

- [ ] **Step 6: Manual smoke test**

The reporter changes are user-visible — verify by hand. Run from a real terminal:

```
furnace plan <some-source-dir> -o /tmp/plan-test --audio-lang eng --sub-lang eng --dry-run
```

Confirm:
- No `[furnace]` prefix anywhere.
- Phase prefix appears once per phase.
- No raw `eac3to`/`makemkvcon`/`mkvmerge` lines bleed through.
- Active operation shows a bar (or spinner) that updates in place.
- `--dry-run` skips Demux and `cropdetect`; Plan section shows the file with no microop.

If any of these are wrong, fix the reporter / wiring before committing.

- [ ] **Step 7: Commit**

```bash
git add furnace/cli.py furnace/__init__.py pyproject.toml tests/test_plan_output_integration.py
git commit -m "$(cat <<'EOF'
Bump to 1.14.0: structured CLI output for furnace plan

Replaces the noisy interleaved stream (raw makemkvcon/eac3to/mkvmerge
output + [furnace] typer.echo + [furnace] INFO log handler) with a
single Rich-based PlanReporter that owns stdout for the plan command.

Each phase (Detect, Demux, Scan, Analyze, Plan) prints structured
rows; the active long operation floats a progress bar (or spinner
where the tool exposes no %) at the bottom of the terminal. Errors
and skips render inline. Ctrl+C clears the live region and exits 130.

The console logging handler is removed for plan; all logger.* calls
keep going to furnace.log. Per-tool log files in <output>/logs/ are
unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

### Spec coverage

Walked the spec section-by-section. Every requirement has a task:

- "No raw tool output in the terminal" → Tasks 12-15 wire all services through reporter; Task 16 deletes `_console_output` and stops passing it to adapters.
- "No Textual TUI for non-interactive parts" → Task 2 builds RichPlanReporter on plain `Console` + `Progress`, no Textual.
- "Show what phase / file / progress" → Tasks 2-5 cover phase rendering; Tasks 7-10 surface real progress fractions.
- "Readable when piped to a file" → Task 6 explicit non-TTY test.
- "Windows cmd.exe compatibility" → ASCII-only mode is hard-wired in Task 2 (`ascii_only=True`, `_ASCII_BAR='#'`, `_ASCII_SPINNER='line'`).
- Phase formats (Detect / Demux / Scan / Analyze / Plan / Final / Interrupt) → Tasks 2-5 implement; Task 6 golden snapshot locks them.
- Per-file failures (HDR10+, unknown codec, probe failed, skipped) → Task 13 emits all of them inline.
- Phase-fatal failures (demux crash, transcode fail, missing pcm_transcoder) → Task 14 emits `demux_title_failed` then re-raises.
- KeyboardInterrupt handling → Task 11 + Task 16 (cancel_event plumbing + try/except in cli.plan).
- MakeMKV PRGV parser → Task 7.
- ffmpeg per-point progress (idet / cropdetect / audio profile) → Tasks 8-10.
- "Paths relative to source" → Task 12 (Scanner uses `relative_to`); Task 16 (cli.py uses `parent.relative_to(source)` for Detect rows).
- "Drop `.furnace_demux/` prefix from demuxed file names" → Task 16 emits `mkv_path.name` (basename only) for demuxed entries.
- "Drop leaf BDMV/VIDEO_TS from disc rel_path" → Task 16 uses `disc.path.parent.relative_to(source)` (parent strips the leaf).

The two **Future Work** items in the spec (source = VIDEO_TS directly; source = drive root) are explicitly OUT OF SCOPE and not implemented in this plan. They live as separate small tasks for later.

### Placeholder scan

Searched the plan for "TBD", "TODO", "implement later", "fill in details", "add appropriate error handling", "similar to Task N", "...". None found in step bodies. Each step has either concrete code or a concrete shell command with expected outcome.

### Type consistency

- `PlanReporter` method signatures defined in Task 1 are called consistently in Tasks 2-15.
- `DiscType` import matches actual enum (`DiscType.BLURAY`, `DiscType.DVD`).
- `ProgressSample` import path consistent (`furnace.core.progress`).
- `RecordingPlanReporter` uses `__getattr__` so any new method can be called on it without code changes — robust to incremental additions.
- `on_progress: Callable[[ProgressSample], None] | None = None` signature consistent across all three new ffmpeg-prober additions.
- Service constructors all use `reporter: PlanReporter | None = None` consistently.

No mismatches found.
