# Parallel up-front analysis batch (crop + detectors before the TUI)

**Date:** 2026-06-28
**Status:** Approved (design), pending implementation
**Version target:** `2.0.0 → 2.1.0` (MINOR — additive CLI flag, phase reorder, plan JSON unchanged)

## Problem

`furnace plan` runs the heavy non-interactive detectors in two different places:

- **Analyze phase** (up front, all files): probe, interlace detection (`idet`),
  fake-surround audio profiling. The user happily waits through this batch.
- **Plan phase** (per file, in the loop): `cropdetect` runs inside
  `PlannerService._build_job` *immediately before* the track-selection TUI for
  that file.

Result: while clicking through track selection file-by-file, the user waits for
the *next* file's `cropdetect` between every file. `cropdetect` is the **only**
blocking operation in the Plan phase — everything else is either the interactive
TUI or instant computation from already-probed data. Crop geometry does not
depend on track selection, so it can be computed ahead of time.

## Goal

1. Move `cropdetect` out of the Plan phase into an up-front batch, so the Plan
   phase is pure "click-click-click" with **zero waits between files**.
2. **Parallelise** the whole up-front detector batch (probe + idet + audio
   profile + cropdetect) across files, so the batch itself is shorter.

## Key facts that make this safe

- All adapter detect operations (`probe`, `run_idet`, `detect_crop`,
  `probe_hdr_side_data`, `profile_audio_track`) use `subprocess.run(...,
  capture_output=True)` **directly** — no shared log files, no shared mutable
  state (the only instance state is a benign ffmpeg-version cache not touched on
  these paths). They are safe to call concurrently as-is.
- The **only** thread-unsafe component is the reporter (single Rich progress
  bar, implicit `_current_file` state). **Therefore: workers do I/O only; the
  reporter is touched exclusively from the main thread.**
- Heavy work is in subprocesses (`subprocess.run` releases the GIL while
  waiting) and numpy (releases the GIL), so a `ThreadPoolExecutor` is the right
  tool — no multiprocessing needed.

## Design decisions (locked)

- **Progress UX:** a single aggregated counter bar
  (`Analyzing [####----] 7/20`) at the bottom; per-file result lines stream
  above it in completion order. No per-file sub-bars during the parallel batch.
- **Worker count:** default `max(1, os.cpu_count() - 2)`. Overridable via a CLI
  flag `--jobs / -j N` (`N >= 1`; `1` = sequential, useful for debugging/tests).
- **Determinism:** the resulting plan is assembled in **input order**
  regardless of completion order. Reporter result lines stream in completion
  order (live progress).

## Architecture

New phase flow:

```
Detect → Demux → Scan → Analyze (PARALLEL, includes cropdetect) → Plan (pure TUI)
```

### New / changed components

#### `core/models.py` — new result types

```python
class AnalyzeStatus(enum.Enum):
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"

@dataclass(frozen=True)
class AnalysisOutcome:
    movie: Movie | None
    status: AnalyzeStatus
    detail: str   # DONE: the summary line; SKIPPED/FAILED: the reason
```

Invariants (asserted in tests): `status == DONE ⇒ movie is not None`;
`status in {SKIPPED, FAILED} ⇒ movie is None`. Add both names to `__all__`.

#### `core/detect.py` — `classify_passthrough` (moved from planner)

```python
DV_PROFILE_FEL = 7  # Dolby Vision FEL — needs P7 → P8.1 re-encode

def classify_passthrough(video: VideoInfo, *, copy_video: bool) -> tuple[bool, str | None]:
    if not copy_video:
        return False, None
    if video.interlaced:
        return False, "interlaced"
    if video.hdr.is_dolby_vision and video.hdr.dv_profile == DV_PROFILE_FEL:
        return False, "DV P7 FEL"
    return True, None
```

Pure function. Used by both the pipeline (decide whether crop is needed) and the
planner (video params). `detect.py` adds `VideoInfo` to its `models` import.
The planner deletes its private `_classify_passthrough` and `_DV_PROFILE_FEL`
and imports `classify_passthrough` + `DV_PROFILE_FEL` from `core.detect`.

#### `services/analyzer.py` — reporter-free, returns `AnalysisOutcome`

- `Analyzer.__init__(self, prober, *, force=False)` — **drop the `reporter`
  parameter**. Drop `_forward_progress`.
- `analyze(self, scan_result) -> AnalysisOutcome` — **no reporter calls at
  all**. idet and audio profiling still run (the actual work), but with no
  `on_progress` (no per-file sub-progress in the parallel model).
- Exit-point → outcome mapping:
  | condition | outcome |
  |---|---|
  | `should_skip_file` | `SKIPPED(reason)` |
  | probe raises | `FAILED("probe failed")` |
  | no video stream | `SKIPPED("no video stream")` |
  | video parse raises | `FAILED("parse failed")` |
  | HDR10+ | `FAILED("HDR10+ not supported")` — **return, do not raise** |
  | unsupported codecs | `SKIPPED(codec_warning)` — previously a silent `None`, now surfaced |
  | success | `DONE(summary)` where `summary = _format_analyze_summary(...)` |
- `_format_analyze_summary` stays in the analyzer (produces the DONE detail).

#### `services/analysis_pipeline.py` — NEW service (owns concurrency)

```python
@dataclass(frozen=True)
class AnalysisBatchResult:
    movies: list[tuple[Movie, Path]]        # input order
    crops: dict[Path, CropRect]             # main_file → detected crop (non-None only)

class AnalysisPipeline:
    def __init__(self, analyzer, prober, reporter, *, max_workers): ...
    def run(self, scan_results, *, copy_video, dry_run) -> AnalysisBatchResult: ...
```

Per-file work unit (`_process`, runs in a worker — **never touches the
reporter**):

```python
outcome = self._analyzer.analyze(scan_result)
crop = None
if outcome.movie is not None and not dry_run:
    passthrough, _ = classify_passthrough(outcome.movie.video, copy_video=copy_video)
    if not passthrough:
        crop = self._detect_crop(outcome.movie)
return index, outcome, crop
```

`_detect_crop(movie) -> CropRect | None` — the crop logic **moved out of the
planner**: builds `is_dvd` / `hdr_transfer`, calls `prober.detect_crop(...,
on_progress=None)`, maps full-frame → `None`, logs, and catches
`(OSError, RuntimeError, ValueError)` → `None` (same fail-soft as the planner
does today).

`run()`:
1. `reporter.analyze_batch_start(len(scan_results))`.
2. Submit one `_process` per scan result to `ThreadPoolExecutor(max_workers)`.
3. Main thread iterates `as_completed`: for each, call
   `reporter.analyze_batch_item(name, outcome.detail, status=outcome.status)`
   (advances the counter) and stash the result by index.
4. `reporter.analyze_batch_finish()`.
5. Assemble `movies` in **input order**; build `crops` from non-None detected
   crops keyed by `movie.main_file`.

`crops` stores only non-None values; the planner reads it with `.get(...)` so a
missing key (passthrough / dry-run / no-bars / not analysed) yields `None`
uniformly.

#### `services/planner.py` — drop crop detection and `prober`

- `__init__` loses the `prober` parameter (crop was its only use). Signature:
  `(self, previewer, track_selector=None, und_resolver=None, reporter=None)`.
- Delete `_on_crop_progress`, the inline `cropdetect` block in `_build_job`, the
  `plan_microop("cropdetect")` / `plan_progress` calls, and the private
  `_classify_passthrough` / `_DV_PROFILE_FEL`.
- `create_plan(..., precomputed_crops: dict[Path, CropRect] | None = None)`.
  `_build_job` reads `crop = effective_crops.get(movie.main_file)` (default
  `None`). `_build_video_params` still forces `crop=None` for passthrough, so
  passthrough/dry-run need no special handling here.
- Passthrough + DV mode use the imported `classify_passthrough` /
  `DV_PROFILE_FEL`.
- `plan_file_start` / `plan_file_done` stay (instant Plan-phase summary lines).

#### `core/ports.py` — `PlanReporter` Protocol

- **Remove:** `analyze_file_start`, `analyze_microop`, `analyze_progress`,
  `analyze_file_done`, `analyze_file_failed`, `analyze_file_skipped`,
  `plan_microop`, `plan_progress`.
- **Add:**
  ```python
  def analyze_batch_start(self, total: int) -> None: ...
  def analyze_batch_item(self, name: str, detail: str, *, status: AnalyzeStatus) -> None: ...
  def analyze_batch_finish(self) -> None: ...
  ```
  (imports `AnalyzeStatus` from `models`).

#### `ui/plan_console.py` — `RichPlanReporter`

- Remove the old analyze per-file methods and `plan_microop` / `plan_progress`.
- `analyze_batch_start(total)`: emit the `Analyze` header; start a **persistent**
  count `Progress` (`[TextColumn("Analyzing"), _ChunkBarColumn(),
  TextColumn("{task.completed:>2.0f}/{task.total}")]`) with `total=total`. On a
  non-TTY console, skip the bar (header only) — mirrors the existing non-TTY
  handling.
- `analyze_batch_item(name, detail, *, status)`: format the line
  (`DONE → "{name} -> {detail}"`, `SKIPPED → "{name} -> SKIPPED — {detail}"`,
  `FAILED → "{name} -> FAILED — {detail}"`), print it **above** the live bar
  (via the progress's console when active, else the plain console), then advance
  the task by 1.
- `analyze_batch_finish()`: stop the progress.
- **Threading contract:** these methods are called only from the pipeline's main
  thread; no locking required. Document this on the methods.

#### `furnace/cli.py` — wire the pipeline, add the flag

- New typer option: `jobs: int | None = typer.Option(None, "--jobs", "-j",
  help="Parallel analysis workers (default: CPU cores - 2)")`.
- `workers = max(1, jobs) if jobs is not None else max(1, (os.cpu_count() or 1) - 2)`.
- Replace the per-file analyze loop **and** the planner-side crop with:
  ```python
  analyzer = Analyzer(prober=ffmpeg_adapter, force=force)
  pipeline = AnalysisPipeline(analyzer=analyzer, prober=ffmpeg_adapter,
                              reporter=reporter, max_workers=workers)
  batch = pipeline.run(scan_results, copy_video=copy_video, dry_run=dry_run)
  movies_with_paths = batch.movies
  precomputed_crops = batch.crops
  ```
- Remove the `try/except ValueError` around analyze (HDR10+ no longer raises).
- Construct `PlannerService` **without** `prober`; pass
  `precomputed_crops=precomputed_crops` to `create_plan`.
- Keep the existing `reporter.scan_file(...)` calls for demuxed paths (Scan
  phase) and the pause/resume around the planner construction unchanged.

#### Version

`furnace/__init__.py` `VERSION = "2.1.0"` and `pyproject.toml`
`version = "2.1.0"` (bumped together).

## Testing strategy (TDD, 100% line+branch on touched code, via `make check`)

- **`core` (new, pure):** `AnalyzeStatus` / `AnalysisOutcome` in
  `test_models.py`; `classify_passthrough` table in `test_detect.py` (copy/move
  the cases from the planner passthrough tests).
- **`Analyzer`:** rewrite `test_analyzer.py` to assert on the returned
  `AnalysisOutcome` (status + detail + movie) instead of reporter calls. The
  reporter-interaction tests in `test_analyzer_reports.py` move to a new
  `test_analysis_pipeline_reports.py` (the pipeline now owns reporting).
- **`AnalysisPipeline` (new):** `test_analysis_pipeline.py` — with
  `max_workers=1` for deterministic reporter/order assertions; a separate
  `max_workers>1` test asserting *correctness* (all files processed, plan in
  input order, crops keyed correctly) **without** asserting line order; crop
  skipped for passthrough and for dry-run; crop full-frame → absent from dict;
  fail-soft on `detect_crop` raising.
- **`PlannerService`:** update every `test_planner_*.py` constructor (drop
  `prober`); `test_planner_crop*.py` now feed `precomputed_crops` instead of
  mocking `detect_crop`; `test_planner_reports.py` drops
  `plan_microop`/`plan_progress` assertions; passthrough cases still pass via
  `classify_passthrough`.
- **Reporter:** `test_plan_console.py` covers the new batch methods (TTY +
  non-TTY); `tests/fakes/recording_reporter.py`, `test_plan_reporter_fake.py`
  and `test_ports.py` updated to the new Protocol surface.
- **CLI:** `test_cli.py` updated for the pipeline wiring, the `--jobs` flag
  (default auto, explicit value, sequential), and removal of the analyze loop /
  `ValueError` path.

## Out of scope (YAGNI)

- Per-file live sub-progress bars during the parallel batch (multi-line render).
- Reading worker count from `furnace.toml` (CLI flag + auto default is enough).
- Parallelism finer than per-file (e.g. concurrent audio tracks within a file).
- Any change to the plan JSON format or the `run` command.
