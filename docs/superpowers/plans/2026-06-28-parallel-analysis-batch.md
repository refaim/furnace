# Parallel up-front analysis batch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `cropdetect` (and the whole detector batch) ahead of the track-selection TUI and run it in parallel across files, so the Plan phase has zero waits between files.

**Architecture:** A new `AnalysisPipeline` service runs per-file analysis (`analyze` + `cropdetect`) on a `ThreadPoolExecutor`; workers do I/O only and the reporter is touched solely from the main thread. `Analyzer` becomes reporter-free and returns a structured `AnalysisOutcome`. The planner stops detecting crop and consumes a precomputed `dict[Path, CropRect]`.

**Tech Stack:** Python 3.13, `concurrent.futures.ThreadPoolExecutor`, typer CLI, Rich reporter, pytest + mypy --strict + ruff (ALL), 100% line+branch coverage.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-06-28-parallel-analysis-batch-design.md` — the authority for every task.
- **Version:** bump `2.0.0 → 2.1.0` in BOTH `furnace/__init__.py` (`VERSION`) and `pyproject.toml` (`version`) — in the final task only.
- **NO COMMITS** until the entire plan is done AND the user explicitly approves. No per-task commits.
- **Tests run ONLY via the Makefile** (`make check` = ruff + mypy --strict + pytest with `--cov-fail-under=100`). Do **not** run `uv run pytest`/`ruff`/`mypy` directly.
- **This refactor is interdependent: the full suite is RED until the final integration task.** Per-task agents write test + implementation together and do **not** run `make check` mid-refactor (it will be red from other tasks' pending work). The single authoritative `make check` runs in Task 8.
- **Subagents: Opus only. No git worktrees — work in the main checkout.**
- **Core stays pure** (no I/O, no imports from services/adapters/ui). Adapters/reporter unchanged in behavior except where specified.
- Reporter `RichPlanReporter` is ASCII-only (cmd.exe). Keep existing non-TTY handling (suppress floating bar, still print persistent lines).

## File map

| File | Change |
|---|---|
| `furnace/core/models.py` | + `AnalyzeStatus`, `AnalysisOutcome`; `__all__` |
| `furnace/core/detect.py` | + `DV_PROFILE_FEL`, `classify_passthrough`; import `VideoInfo` |
| `furnace/core/ports.py` | `PlanReporter`: remove analyze per-file + `plan_microop`/`plan_progress`; add `analyze_batch_*` |
| `furnace/ui/plan_console.py` | `RichPlanReporter`: implement `analyze_batch_*`; remove old methods |
| `furnace/services/analyzer.py` | reporter-free; `analyze` returns `AnalysisOutcome` |
| `furnace/services/analysis_pipeline.py` | **new** — `AnalysisPipeline`, `AnalysisBatchResult`, `_detect_crop` |
| `furnace/services/planner.py` | drop `prober` + crop detection; consume `precomputed_crops`; use core `classify_passthrough` |
| `furnace/cli.py` | wire pipeline; `--jobs/-j`; drop planner `prober`; remove `ValueError` path |
| `furnace/__init__.py`, `pyproject.toml` | version `2.1.0` |
| `tests/fakes/recording_reporter.py` | update to new Protocol |
| `tests/...` | update per task |

## Dependency / execution order

```
T1 models ─┐                 (parallel)
T2 detect ─┘
              ├─ T3 reporter (needs T1) ┐
              ├─ T4 analyzer (needs T1) ┤  (parallel — disjoint files)
              └─ T5 planner  (needs T2) ┘
T6 pipeline (needs T1,T2,T3,T4)
T7 cli      (needs T5,T6,T3)
T8 integration: version bump + `make check` to green + review-to-zero
```

---

### Task 1: Analysis result models

**Files:**
- Modify: `furnace/core/models.py`
- Test: `tests/core/test_models.py`

**Interfaces:**
- Produces: `AnalyzeStatus(enum.Enum)` with `DONE="done"`, `SKIPPED="skipped"`, `FAILED="failed"`; `AnalysisOutcome(frozen dataclass)` with fields `movie: Movie | None`, `status: AnalyzeStatus`, `detail: str`.

- [ ] **Step 1: Write the failing test** — append to `tests/core/test_models.py`:

```python
def test_analyze_status_values():
    from furnace.core.models import AnalyzeStatus
    assert AnalyzeStatus.DONE.value == "done"
    assert AnalyzeStatus.SKIPPED.value == "skipped"
    assert AnalyzeStatus.FAILED.value == "failed"


def test_analysis_outcome_done_carries_movie():
    from furnace.core.models import AnalysisOutcome, AnalyzeStatus
    movie = object()  # placeholder; real Movie not needed for the dataclass test
    outcome = AnalysisOutcome(movie=movie, status=AnalyzeStatus.DONE, detail="summary")
    assert outcome.movie is movie
    assert outcome.status is AnalyzeStatus.DONE
    assert outcome.detail == "summary"


def test_analysis_outcome_is_frozen():
    import dataclasses
    import pytest
    from furnace.core.models import AnalysisOutcome, AnalyzeStatus
    outcome = AnalysisOutcome(movie=None, status=AnalyzeStatus.FAILED, detail="boom")
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.detail = "x"  # type: ignore[misc]
```

- [ ] **Step 2: Implement** in `furnace/core/models.py`:
  - Add after `JobStatus` enum:
    ```python
    class AnalyzeStatus(enum.Enum):
        DONE = "done"
        SKIPPED = "skipped"
        FAILED = "failed"
    ```
  - Add after the `Movie` dataclass:
    ```python
    @dataclass(frozen=True)
    class AnalysisOutcome:
        movie: Movie | None
        status: AnalyzeStatus
        detail: str  # DONE: summary line; SKIPPED/FAILED: reason
    ```
  - Add `"AnalysisOutcome"` and `"AnalyzeStatus"` to `__all__` (keep it sorted).

- [ ] **Step 3: Verify** — deferred to Task 8 `make check`. Do not run pytest directly.

---

### Task 2: `classify_passthrough` in core

**Files:**
- Modify: `furnace/core/detect.py`
- Test: `tests/core/test_detect.py`

**Interfaces:**
- Produces: `DV_PROFILE_FEL = 7` (int constant); `classify_passthrough(video: VideoInfo, *, copy_video: bool) -> tuple[bool, str | None]`.

- [ ] **Step 1: Write the failing test** — append to `tests/core/test_detect.py` (use the existing VideoInfo construction helpers in that file if present; otherwise build a minimal `VideoInfo`):

```python
def _vi(*, interlaced=False, dv=False, dv_profile=None):
    from furnace.core.models import HdrMetadata, VideoInfo
    from pathlib import Path
    return VideoInfo(
        index=0, codec_name="hevc", width=1920, height=1080, pixel_area=1920 * 1080,
        fps_num=24, fps_den=1, duration_s=10.0, interlaced=interlaced,
        color_matrix_raw=None, color_range=None, color_transfer=None,
        color_primaries=None, pix_fmt="yuv420p10le",
        hdr=HdrMetadata(is_dolby_vision=dv, dv_profile=dv_profile),
        source_file=Path("x.mkv"),
    )


def test_classify_passthrough_disabled_returns_encode():
    from furnace.core.detect import classify_passthrough
    assert classify_passthrough(_vi(), copy_video=False) == (False, None)


def test_classify_passthrough_interlaced_falls_back():
    from furnace.core.detect import classify_passthrough
    assert classify_passthrough(_vi(interlaced=True), copy_video=True) == (False, "interlaced")


def test_classify_passthrough_dv_p7_fel_falls_back():
    from furnace.core.detect import classify_passthrough, DV_PROFILE_FEL
    assert DV_PROFILE_FEL == 7
    assert classify_passthrough(_vi(dv=True, dv_profile=7), copy_video=True) == (False, "DV P7 FEL")


def test_classify_passthrough_eligible_passes_through():
    from furnace.core.detect import classify_passthrough
    assert classify_passthrough(_vi(dv=True, dv_profile=8), copy_video=True) == (True, None)
```

- [ ] **Step 2: Implement** in `furnace/core/detect.py`:
  - Add `VideoInfo` to the `from .models import (...)` line.
  - Add near the top (after imports):
    ```python
    DV_PROFILE_FEL = 7  # Dolby Vision FEL — needs a P7 → P8.1 re-encode (no passthrough)


    def classify_passthrough(video: VideoInfo, *, copy_video: bool) -> tuple[bool, str | None]:
        """Decide whether a source video can be copied verbatim.

        (False, None)         -> copy_video not requested (normal encode)
        (False, "interlaced") -> must deinterlace
        (False, "DV P7 FEL")  -> P7 FEL needs the P7 -> P8.1 conversion
        (True, None)          -> copy the stream verbatim
        """
        if not copy_video:
            return False, None
        if video.interlaced:
            return False, "interlaced"
        if video.hdr.is_dolby_vision and video.hdr.dv_profile == DV_PROFILE_FEL:
            return False, "DV P7 FEL"
        return True, None
    ```

- [ ] **Step 3: Verify** — deferred to Task 8.

---

### Task 3: Reporter — parallel-batch Protocol + RichPlanReporter

**Files:**
- Modify: `furnace/core/ports.py`, `furnace/ui/plan_console.py`, `tests/fakes/recording_reporter.py`
- Test: `tests/ui/test_plan_console.py`, `tests/core/test_plan_reporter_fake.py`, `tests/core/test_ports.py`

**Interfaces:**
- Consumes: `AnalyzeStatus` (Task 1).
- Produces (PlanReporter Protocol, RichPlanReporter, fake):
  - `analyze_batch_start(self, total: int) -> None`
  - `analyze_batch_item(self, name: str, detail: str, *, status: AnalyzeStatus) -> None`
  - `analyze_batch_finish(self) -> None`
  Removed from the Protocol/impl/fake: `analyze_file_start`, `analyze_microop`, `analyze_progress`, `analyze_file_done`, `analyze_file_failed`, `analyze_file_skipped`, `plan_microop`, `plan_progress`.

- [ ] **Step 1: ports.py** — in `furnace/core/ports.py`:
  - Add to the `models` import: `AnalyzeStatus`.
  - In `PlanReporter`, replace the `# Analyze` block and remove the two `# Plan` micro-op lines so the section reads:
    ```python
        # Analyze (parallel batch — called only from the orchestrator's main thread)
        def analyze_batch_start(self, total: int) -> None: ...
        def analyze_batch_item(self, name: str, detail: str, *, status: AnalyzeStatus) -> None: ...
        def analyze_batch_finish(self) -> None: ...

        # Plan
        def plan_file_start(self, name: str) -> None: ...
        def plan_file_done(self, summary: str) -> None: ...
    ```
    (Delete `plan_microop` and `plan_progress` lines.)

- [ ] **Step 2: RichPlanReporter** — in `furnace/ui/plan_console.py`:
  - Import `AnalyzeStatus` from `furnace.core.models`.
  - Delete methods: `analyze_file_start`, `analyze_microop`, `analyze_progress`, `analyze_file_done`, `analyze_file_failed`, `analyze_file_skipped`, `plan_microop`, `plan_progress`. Keep `_ensure_analyze_header` (reused below) and `_current_file` may be removed if now unused by analyze (still used by plan_file_start/done — keep).
  - Add a constant near the top:
    ```python
    _BATCH_STATUS_PREFIX: dict[AnalyzeStatus, str] = {
        AnalyzeStatus.DONE: "",
        AnalyzeStatus.SKIPPED: "SKIPPED — ",
        AnalyzeStatus.FAILED: "FAILED — ",
    }
    ```
  - Add methods:
    ```python
    def analyze_batch_start(self, total: int) -> None:
        """Begin the parallel analyze batch: header + a persistent count bar.

        Called only from the pipeline's main thread; no locking needed.
        """
        self._stop_progress()
        self._ensure_analyze_header()
        if not self._console.is_terminal:
            self._progress = None
            return
        progress = Progress(
            TextColumn("Analyzing"),
            _ChunkBarColumn(),
            TextColumn("{task.completed:>2.0f}/{task.total:.0f}"),
            console=self._console,
            transient=True,
            expand=False,
        )
        progress.start()
        self._progress = progress
        self._task_id = progress.add_task("", total=total)

    def analyze_batch_item(self, name: str, detail: str, *, status: AnalyzeStatus) -> None:
        """Print one file's result line above the bar, then advance the counter."""
        line = f"{name} -> {_BATCH_STATUS_PREFIX[status]}{detail}"
        if self._progress is not None and self._task_id is not None:
            self._progress.console.print(line, highlight=False)
            self._progress.advance(self._task_id)
        else:
            self._console.print(line, highlight=False)

    def analyze_batch_finish(self) -> None:
        self._stop_progress()
    ```
  - Note `_ensure_analyze_header` already emits the `Analyze` header once.

- [ ] **Step 3: fake** — in `tests/fakes/recording_reporter.py`, mirror the Protocol change: drop the removed methods, add `analyze_batch_start`/`analyze_batch_item`/`analyze_batch_finish` recording calls (append a tuple to the recorded-events list, matching the file's existing recording style).

- [ ] **Step 4: tests** — update:
  - `tests/core/test_ports.py`: any structural assertions about `PlanReporter` method names → new set.
  - `tests/core/test_plan_reporter_fake.py`: exercise the fake's new methods; drop old.
  - `tests/ui/test_plan_console.py`: replace analyze per-file/`plan_microop`/`plan_progress` tests with:
    ```python
    def test_analyze_batch_tty_streams_lines_and_counts(capsys, ...):
        # console forced to a terminal (use the existing test helper/Console(force_terminal=True))
        r = RichPlanReporter(source=..., output=..., console=<terminal console>)
        r.analyze_batch_start(2)
        r.analyze_batch_item("a.mkv", "h264 1920x1080 24fps SDR, 1 audio (eng), 0 subs", status=AnalyzeStatus.DONE)
        r.analyze_batch_item("b.mkv", "HDR10+ not supported", status=AnalyzeStatus.FAILED)
        r.analyze_batch_finish()
        out = capsys.readouterr().out
        assert "Analyze" in out
        assert "a.mkv -> h264" in out
        assert "b.mkv -> FAILED — HDR10+ not supported" in out

    def test_analyze_batch_non_tty_prints_lines_without_bar(...):
        # console NOT a terminal -> no bar, lines still printed
        r = RichPlanReporter(source=..., output=..., console=<non-terminal console>)
        r.analyze_batch_start(1)
        r.analyze_batch_item("a.mkv", "SKIP reason", status=AnalyzeStatus.SKIPPED)
        r.analyze_batch_finish()
        out = capsys.readouterr().out
        assert "a.mkv -> SKIPPED — SKIP reason" in out
    ```
    Match the file's existing pattern for building TTY vs non-TTY `Console` instances (search the file for `is_terminal` / `force_terminal`).

- [ ] **Step 5: Verify** — deferred to Task 8.

---

### Task 4: Analyzer — reporter-free, returns `AnalysisOutcome`

**Files:**
- Modify: `furnace/services/analyzer.py`
- Test: `tests/services/test_analyzer.py`; delete `tests/services/test_analyzer_reports.py` (its reporter coverage moves to Task 6).

**Interfaces:**
- Consumes: `AnalysisOutcome`, `AnalyzeStatus` (Task 1).
- Produces: `Analyzer.__init__(self, prober: Prober, *, force: bool = False)` (no `reporter`); `Analyzer.analyze(self, scan_result: ScanResult) -> AnalysisOutcome`.

- [ ] **Step 1: Implementation** — in `furnace/services/analyzer.py`:
  - Imports: add `AnalysisOutcome, AnalyzeStatus` from `furnace.core.models`; drop `PlanReporter` from the ports import; drop `ProgressSample` import if now unused.
  - `__init__`: `def __init__(self, prober: Prober, *, force: bool = False) -> None:` — set `self._prober`, `self._force`. Remove `self._reporter`. Delete `_forward_progress`.
  - `analyze` returns `AnalysisOutcome` at every exit. Remove every `if self._reporter is not None: ...` block and every `on_progress=self._forward_progress` argument (call `run_idet`/`profile_audio_track` with no `on_progress`). Map exits:
    - skip: `return AnalysisOutcome(None, AnalyzeStatus.SKIPPED, reason)`
    - probe except: `return AnalysisOutcome(None, AnalyzeStatus.FAILED, "probe failed")`
    - no video stream: `return AnalysisOutcome(None, AnalyzeStatus.SKIPPED, "no video stream")`
    - parse video except: `return AnalysisOutcome(None, AnalyzeStatus.FAILED, "parse failed")`
    - HDR10+: `return AnalysisOutcome(None, AnalyzeStatus.FAILED, "HDR10+ not supported")` (delete the `raise ValueError`)
    - unsupported codecs: `return AnalysisOutcome(None, AnalyzeStatus.SKIPPED, codec_warning)` (was a bare `return None`)
    - success: build `movie` as today, then
      `summary = _format_analyze_summary(video_info, audio_tracks, subtitle_tracks)`;
      `return AnalysisOutcome(movie, AnalyzeStatus.DONE, summary)`.
  - Keep `_format_analyze_summary` and all parsing helpers unchanged.

- [ ] **Step 2: tests** — rewrite `tests/services/test_analyzer.py`:
  - Construct `Analyzer(prober=fake_prober, force=...)` (drop `reporter=`).
  - Replace assertions on reporter calls with assertions on the returned `AnalysisOutcome`:
    - success → `out.status is AnalyzeStatus.DONE and out.movie is not None`; assert `out.detail` equals the expected summary string (reuse the strings the old `analyze_file_done` tests asserted).
    - skip/fail cases → `out.movie is None`, `out.status`, `out.detail`.
    - HDR10+: `out == AnalysisOutcome(None, FAILED, "HDR10+ not supported")` (no `pytest.raises`).
    - unsupported codecs: now `SKIPPED` with the `check_unsupported_codecs(...)` string.
  - Delete `tests/services/test_analyzer_reports.py`.

- [ ] **Step 3: Verify** — deferred to Task 8.

---

### Task 5: Planner — drop prober + crop detection; consume `precomputed_crops`

**Files:**
- Modify: `furnace/services/planner.py`
- Test: all `tests/services/test_planner_*.py`

**Interfaces:**
- Consumes: `classify_passthrough`, `DV_PROFILE_FEL` (Task 2); `precomputed_crops: dict[Path, CropRect] | None`.
- Produces: `PlannerService.__init__(self, previewer, track_selector=None, und_resolver=None, reporter=None)` (no `prober`); `create_plan(..., precomputed_crops: dict[Path, CropRect] | None = None)`.

- [ ] **Step 1: Implementation** — in `furnace/services/planner.py`:
  - Imports: from `furnace.core.detect` add `classify_passthrough, DV_PROFILE_FEL`. Remove the local `_DV_PROFILE_FEL` constant.
  - `__init__`: remove the `prober` parameter and `self._prober`. Delete `_on_crop_progress`.
  - `create_plan`: add `precomputed_crops: dict[Path, CropRect] | None = None` (keyword). Inside, mirror the existing override-dict pattern:
    ```python
    effective_crops: dict[Path, CropRect] = precomputed_crops if precomputed_crops is not None else {}
    ```
    Pass `precomputed_crops=effective_crops` into `_build_job`.
  - `_build_job`: add param `precomputed_crops: dict[Path, CropRect]`. Replace the whole `# Detect crop` block (the `crop: CropRect | None = None` + `if not dry_run and not passthrough:` try/except) with:
    ```python
    crop = precomputed_crops.get(movie.main_file)
    ```
    Replace `passthrough, fallback_reason = self._classify_passthrough(movie.video, copy_video=copy_video)` with `passthrough, fallback_reason = classify_passthrough(movie.video, copy_video=copy_video)`.
  - Delete the `_classify_passthrough` method.
  - `_build_video_params`: change `video.hdr.dv_profile == _DV_PROFILE_FEL` to `... == DV_PROFILE_FEL`.
  - (Result: `planner.py` no longer references `self._prober`, `ProgressSample` for crop, or `plan_microop`/`plan_progress`. Remove now-unused imports — `ProgressSample`, `hdr_transfer_for_cropdetect`, `is_dvd_resolution` if they were only used by the deleted crop block. Keep `final_output_dimensions` etc.)

- [ ] **Step 2: tests** — across `tests/services/test_planner_*.py`:
  - Every `PlannerService(...)` construction: drop the `prober=...` argument.
  - `test_planner_crop.py` + `test_planner_crop_detect.py`: stop mocking `prober.detect_crop`; instead pass `precomputed_crops={movie.main_file: CropRect(...)}` to `create_plan` and assert `job.video_params.crop`. Add a case where the key is absent → `crop is None`. Add a case where `precomputed_crops=None` → `crop is None`.
  - `test_planner_passthrough.py`: passthrough still forces `crop=None` even if a crop is supplied in `precomputed_crops` (assert this); DV/interlaced fallback reasons unchanged (now sourced from core `classify_passthrough`).
  - `test_planner_reports.py`: remove assertions for `plan_microop("cropdetect")` / `plan_progress`; keep `plan_file_start`/`plan_file_done`.
  - Any planner test that depended on `_DV_PROFILE_FEL`/`_classify_passthrough` internals: retarget to the public `classify_passthrough`/`DV_PROFILE_FEL` (now tested in Task 2) or remove if redundant.

- [ ] **Step 3: Verify** — deferred to Task 8.

---

### Task 6: `AnalysisPipeline` service

**Files:**
- Create: `furnace/services/analysis_pipeline.py`
- Test: `tests/services/test_analysis_pipeline.py`, `tests/services/test_analysis_pipeline_reports.py`

**Interfaces:**
- Consumes: `Analyzer.analyze -> AnalysisOutcome` (Task 4); `classify_passthrough` (Task 2); reporter `analyze_batch_*` (Task 3); `Prober.detect_crop`, `is_dvd_resolution`, `hdr_transfer_for_cropdetect`.
- Produces:
  - `AnalysisBatchResult(frozen dataclass)`: `movies: list[tuple[Movie, Path]]`, `crops: dict[Path, CropRect]`.
  - `AnalysisPipeline(analyzer, prober, reporter, *, max_workers)`; `run(self, scan_results, *, copy_video, dry_run) -> AnalysisBatchResult`.

- [ ] **Step 1: Implementation** — `furnace/services/analysis_pipeline.py`:

```python
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from furnace.core.detect import (
    classify_passthrough,
    hdr_transfer_for_cropdetect,
    is_dvd_resolution,
)
from furnace.core.models import (
    AnalysisOutcome,
    CropRect,
    Movie,
    ScanResult,
)
from furnace.core.ports import PlanReporter, Prober
from furnace.services.analyzer import Analyzer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisBatchResult:
    movies: list[tuple[Movie, Path]]   # input order
    crops: dict[Path, CropRect]        # main_file -> detected crop (non-None only)


class AnalysisPipeline:
    """Run per-file analysis (analyze + cropdetect) in parallel.

    Workers do I/O only and NEVER touch the reporter; the reporter is driven
    exclusively from the main thread in ``run``.
    """

    def __init__(
        self,
        analyzer: Analyzer,
        prober: Prober,
        reporter: PlanReporter | None,
        *,
        max_workers: int,
    ) -> None:
        self._analyzer = analyzer
        self._prober = prober
        self._reporter = reporter
        self._max_workers = max_workers

    def run(
        self,
        scan_results: list[ScanResult],
        *,
        copy_video: bool,
        dry_run: bool,
    ) -> AnalysisBatchResult:
        if self._reporter is not None:
            self._reporter.analyze_batch_start(len(scan_results))

        results: dict[int, tuple[AnalysisOutcome, CropRect | None]] = {}
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = [
                pool.submit(self._process, i, sr, copy_video=copy_video, dry_run=dry_run)
                for i, sr in enumerate(scan_results)
            ]
            for fut in as_completed(futures):
                i, outcome, crop = fut.result()
                results[i] = (outcome, crop)
                if self._reporter is not None:
                    self._reporter.analyze_batch_item(
                        scan_results[i].main_file.name,
                        outcome.detail,
                        status=outcome.status,
                    )

        if self._reporter is not None:
            self._reporter.analyze_batch_finish()

        movies: list[tuple[Movie, Path]] = []
        crops: dict[Path, CropRect] = {}
        for i, sr in enumerate(scan_results):
            outcome, crop = results[i]
            if outcome.movie is not None:
                movies.append((outcome.movie, sr.output_path))
                if crop is not None:
                    crops[outcome.movie.main_file] = crop
        return AnalysisBatchResult(movies=movies, crops=crops)

    def _process(
        self,
        index: int,
        scan_result: ScanResult,
        *,
        copy_video: bool,
        dry_run: bool,
    ) -> tuple[int, AnalysisOutcome, CropRect | None]:
        outcome = self._analyzer.analyze(scan_result)
        crop: CropRect | None = None
        if outcome.movie is not None and not dry_run:
            passthrough, _reason = classify_passthrough(outcome.movie.video, copy_video=copy_video)
            if not passthrough:
                crop = self._detect_crop(outcome.movie)
        return index, outcome, crop

    def _detect_crop(self, movie: Movie) -> CropRect | None:
        """Run cropdetect; map full-frame -> None; fail-soft to None. No reporter."""
        try:
            is_dvd = is_dvd_resolution(movie.video.width, movie.video.height)
            raw_crop = self._prober.detect_crop(
                movie.main_file,
                movie.video.duration_s,
                interlaced=movie.video.interlaced,
                is_dvd=is_dvd,
                hdr_transfer=hdr_transfer_for_cropdetect(movie.video.color_transfer),
                on_progress=None,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Crop detection failed for %s: %s", movie.main_file.name, exc)
            return None
        if raw_crop is None:
            logger.warning("%s: cropdetect unable to determine crop", movie.main_file.name)
            return None
        if raw_crop.w == movie.video.width and raw_crop.h == movie.video.height:
            logger.info(
                "%s: no black bars detected (crop equals full frame %dx%d)",
                movie.main_file.name, movie.video.width, movie.video.height,
            )
            return None
        logger.info(
            "%s: crop detected %d:%d:%d:%d (source %dx%d)",
            movie.main_file.name, raw_crop.w, raw_crop.h, raw_crop.x, raw_crop.y,
            movie.video.width, movie.video.height,
        )
        return raw_crop
```

- [ ] **Step 2: tests** — `tests/services/test_analysis_pipeline.py`:
  - Build a fake `Analyzer` (or real `Analyzer` with a fake `Prober`) and a fake `Prober.detect_crop`. Use `tests/fakes/recording_reporter.py` for the reporter.
  - With `max_workers=1` (deterministic):
    - Two scan results, both `DONE`, one with a crop, one full-frame → `result.movies` has both in **input order**; `result.crops` has only the cropped file's `main_file`.
    - Reporter recorded: `analyze_batch_start(2)`, two `analyze_batch_item(...)` calls (assert names/status/detail), `analyze_batch_finish()`.
    - `dry_run=True` → `detect_crop` never called; `crops == {}`.
    - `copy_video=True` + passthrough-eligible movie → `detect_crop` never called for it.
    - A `SKIPPED`/`FAILED` outcome (movie None) → excluded from `movies`, no crop, still produces a `analyze_batch_item` line.
    - `_detect_crop` fail-soft: `detect_crop` raises `RuntimeError` → crop `None`, file still in `movies`.
  - With `max_workers=3` (correctness, no line-order asserts): N scan results → `movies` in input order, all processed, `crops` keyed correctly.
  - `tests/services/test_analysis_pipeline_reports.py`: port the reporter-sequence expectations that used to live in `test_analyzer_reports.py` (start/item/finish, per-status lines), driving them through `AnalysisPipeline.run` with `max_workers=1`.

- [ ] **Step 3: Verify** — deferred to Task 8.

---

### Task 7: CLI wiring + `--jobs` flag

**Files:**
- Modify: `furnace/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `AnalysisPipeline`, `AnalysisBatchResult` (Task 6); `PlannerService` without `prober` (Task 5).

- [ ] **Step 1: Implementation** — in `furnace/cli.py`:
  - Add `import os` (if absent — it is already imported).
  - Imports: `from .services.analysis_pipeline import AnalysisPipeline`.
  - `plan(...)` signature: add (place near other options)
    ```python
    jobs: int | None = typer.Option(
        None, "--jobs", "-j", help="Parallel analysis workers (default: CPU cores - 2)"
    ),
    ```
  - Compute workers (after `cfg = load_config(config)` or near where the pipeline is built):
    ```python
    workers = max(1, jobs) if jobs is not None else max(1, (os.cpu_count() or 1) - 2)
    ```
  - Replace the analyze loop (the `analyzer = Analyzer(...)` block + the `for sr in scan_results:` try/except building `movies_with_paths`) with:
    ```python
    analyzer = Analyzer(prober=ffmpeg_adapter, force=force)
    pipeline = AnalysisPipeline(
        analyzer=analyzer,
        prober=ffmpeg_adapter,
        reporter=reporter,
        max_workers=workers,
    )
    batch = pipeline.run(scan_results, copy_video=copy_video, dry_run=dry_run)
    movies_with_paths = batch.movies
    precomputed_crops = batch.crops
    ```
  - `PlannerService(...)` construction: remove `prober=ffmpeg_adapter`.
  - `planner.create_plan(...)`: add `precomputed_crops=precomputed_crops`.
  - Remove the now-unused `ValueError` handling that wrapped `analyzer.analyze`.

- [ ] **Step 2: tests** — in `tests/test_cli.py`:
  - Update the `plan` command tests: the analyze step now goes through `AnalysisPipeline`. Where tests patched `Analyzer.analyze` to return a `Movie`, change to return `AnalysisOutcome(movie, AnalyzeStatus.DONE, "...")`. Where they asserted the planner was built with `prober=`, drop it.
  - Add: `--jobs 1` is accepted and produces the same plan as default (patch `os.cpu_count` for the default path); assert `max_workers` reaches the pipeline (patch `AnalysisPipeline` to capture the kwarg, or assert via a recorded fake).
  - HDR10+ file: assert it is skipped (no job) and a `FAILED` line is reported — no exception escapes.

- [ ] **Step 3: Verify** — deferred to Task 8.

---

### Task 8: Integration — version bump, green `make check`, review-to-zero

**Files:**
- Modify: `furnace/__init__.py`, `pyproject.toml`

- [ ] **Step 1: Version bump** — `furnace/__init__.py` → `VERSION = "2.1.0"`; `pyproject.toml` → `version = "2.1.0"`.

- [ ] **Step 2: Run the authoritative gate**

Run: `make check`
Expected: ruff clean, mypy --strict clean, pytest 100% line+branch. Fix any failures (most likely: leftover imports of removed symbols, `recording_reporter` Protocol drift, coverage gaps on new branches — add tests for any uncovered branch in `analysis_pipeline.py`/`plan_console.py`).

- [ ] **Step 3: Coverage sweep** — confirm 100% on `analysis_pipeline.py`, the new reporter methods, `classify_passthrough`, `analyze`'s new return points. Add targeted tests for any `term-missing` line/branch.

- [ ] **Step 4: Code review to zero** — dispatch a separate code-reviewer subagent (Opus, never self-review). Address every comment, re-run `make check`, re-dispatch review. Repeat until zero comments.

- [ ] **Step 5: STOP** — do NOT commit. Report completion and the proposed `2.1.0` commit to the user; commit only after explicit approval.

## Self-review (plan vs spec)

- **Spec coverage:** models (T1), classify_passthrough (T2), reporter batch (T3), analyzer outcome (T4), planner precomputed_crops + prober drop (T5), AnalysisPipeline + crop move + parallelism + input-order + non-None crops (T6), CLI pipeline + `--jobs` + ValueError removal (T7), version + gate + review (T8). All spec sections mapped.
- **Determinism:** T6 assembles `movies`/`crops` by input index — covered.
- **Threading:** workers never call the reporter; reporter only in `run`'s main thread — stated in T6 code + T3 docstring.
- **Types:** `AnalysisOutcome(movie, status, detail)`, `classify_passthrough -> tuple[bool, str|None]`, `AnalysisBatchResult(movies, crops)`, reporter `analyze_batch_item(name, detail, *, status)` — consistent across T1/T2/T3/T6/T7.
- **No per-task commits / Makefile-only / no worktrees / Opus-only:** in Global Constraints.
