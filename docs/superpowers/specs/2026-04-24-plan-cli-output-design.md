# Plan-phase CLI Output Redesign

**Date:** 2026-04-24
**Scope:** `furnace plan` command terminal output. `furnace run` is **not** affected.

## Problem

In the `plan` phase the terminal currently mixes three uncoordinated output streams:
1. Raw stdout/stderr from external tools (`makemkvcon`, `eac3to`, `mkvmerge`) routed via the `_console_output` callback in `cli.py:96-98`.
2. `typer.echo("[furnace] ...")` lines from `cli.plan` itself.
3. The console `StreamHandler` installed by `_setup_logging` (`cli.py:362-365`), which prints `[furnace] %(levelname)s` for every `INFO+` log call from any module.

The result is unreadable: a user cannot tell which movie is being processed, which phase is active, or how far along it is. Tool banners (eac3to title-listing tables, makemkvcon progress messages) leak into the same stream as Furnace's own status messages.

The `run` phase already solved this with the Textual `RunApp` (it owns the terminal; tool stdout is captured into a `RichLog` widget). The `plan` phase has no equivalent.

## Goal

Replace all three output streams in `plan` with a single, structured, phase-aware reporter. Constraints:

- **Zero raw tool output** in the terminal. Tool stdout/stderr keeps being written to per-tool log files (`<output>/logs/...`) and to `furnace.log` — only the terminal is cleaned up.
- **No Textual TUI** for the non-interactive parts of `plan`. We use plain `Rich` printing + `Rich.Progress` bars rendered inline. The interactive Textual screens (`PlaylistSelector`, `FileSelector`, `TrackSelector`, `LanguageSelector`) are untouched and continue to take over the terminal between phases.
- Output must show **what phase is active**, **what file/disc/title is being processed**, and **how far along** the active long operation is (real progress bar where the tool exposes %, spinner where it does not).
- Final output must remain readable when piped to a file or run on Windows `cmd.exe`.

## Non-goals

- Changing the `furnace run` UI.
- Refactoring the interactive Textual selectors.
- Fixing the unrelated bugs uncovered during this design (see Future Work).

## User-facing output specification

### Rendering rules

- **No prefix** on Furnace's own messages (the existing `[furnace] ` prefix is gone).
- **Header** at the top of the run, printed once:
  ```
  Source: <source path>
  Output: <output path>
  ```
- **Phase rows** are left-aligned with the phase name in the first column; subsequent rows of the same phase have the phase column blank and the content indented to align with the first row's content column.
- **Per-phase formats** (see "Per-phase content" below).
- **Active in-flight row** is the only line that updates in place. It is rendered via `Rich.Progress` so it floats at the bottom of the terminal while accumulated history scrolls above. When the operation completes, the floating row is replaced by a final fixed-form row of the same content, and a new floating row appears for the next active operation.
- **Errors and skips** are rendered inline as part of the phase, not in a separate block. Format: `name  FAILED — <reason>` or `name  SKIPPED — <reason>`. Failures of per-file operations do not abort the pipeline; failures of demux abort the whole `plan` command.
- **Time is not displayed** anywhere — neither elapsed durations nor ETA.
- **Paths are shown relative to `source`**, with the leaf `BDMV` / `VIDEO_TS` segment dropped from disc paths (e.g. `Matrix_BD/BDMV/` → `Matrix_BD`). Demuxed-file paths inside `.furnace_demux/` are shown without the `.furnace_demux/` prefix.
- **Bar vs spinner**: when the underlying tool exposes a numeric progress, render a bar with `NN%`. When it does not, render a spinner with the operation label. Never both at once.
- **ASCII-only rendering** by default (consistent with the project's existing cmd.exe compatibility policy). Bar uses `#` and `-`; spinner uses `|/-\`.

### Phases

The `plan` command runs these phases in order:

1. `Detect` — `disc_demuxer.detect` filesystem walk for `BDMV/` and `VIDEO_TS/` directories.
2. *(interactive Textual screen — out of scope)* — title selection per disc.
3. `Demux` — `disc_demuxer.demux` over the user-selected titles.
4. `Scan` — `Scanner.scan` filesystem walk for video files.
5. `Analyze` — `Analyzer.analyze` per file (ffprobe + idet + audio profiling).
6. *(interactive Textual screen — out of scope)* — track selection per movie.
7. `Plan` — `PlannerService.create_plan` per movie (cropdetect + parameter assembly), then `save_plan`.

Phases that produce zero rows (e.g. `Detect` on a file-only source, `Demux` on a source with no discs) emit no header at all — they are simply absent from the output.

### Per-phase content

#### Detect

One row per disc found. Format: `<disc-type>  <relative-path>`.
```
Detect    BDMV   Matrix_BD
          BDMV   OldMatrix_BD
          DVD    DirtyHarry_DVD
```

#### Demux

One row per disc, with per-title rows indented underneath. Cached discs (those whose every selected title already has a `.done` marker) collapse to a single row. Per-title rows are **one line each**: the same line mutates through `rip` → `transcode N/M` (only if any `.w64` was produced) → `remux` (BD only) → `done`. Sub-step transitions are driven by the service; the active sub-step is rendered as either a progress bar or a spinner depending on whether the tool exposes a `%`.

```
Demux     OldMatrix_BD              cached
          DirtyHarry_DVD
            title 1                 done
          Matrix_BD
            title 3                 done
            title 5                 transcode 2/2 ####----  37%
```

`list_titles` is **not** rendered here — it happens earlier as part of the interactive title-selection TUI.

#### Scan

One row per video file found. The file name is shown relative to `source`. No bar (each file is microseconds).

```
Scan      Inception.mkv
          Tenet.mkv
          Matrix_BD_title_3.mkv
```

#### Analyze

One row per file. Each row mutates through micro-operations (`probing`, `HDR side data`, `idet`, `audio profile track N`) and finalizes with a summary. The summary format is:

```
<codec> <width>x<height> <fps>fps <hdr-class> [• interlaced] • <N> audio (<langs>) • <N> subs
```

where `<hdr-class>` is one of `SDR`, `HDR10`, `HLG`, or `DV P<n> (BL=<HDR10|SDR|HLG|none>)`. The `interlaced` token is appended only when `should_deinterlace()` returns True. `<langs>` is a comma-separated, deduplicated list of audio track ISO 639-3 codes.

```
Analyze   Inception.mkv             hevc 3840x2076 24fps HDR10 • 5 audio (rus,eng) • 12 subs
          broken_hdr10plus.mkv      FAILED — HDR10+ not supported
          some_finished.mkv         SKIPPED — already encoded by furnace
          DirtyHarry_DVD_title_1.mkv  idet ########--  72%
```

#### Plan

One row per movie. Active row shows `cropdetect ####----  37%` while cropdetect runs; finalized row shows the resolved parameters in the form:

```
cq <CQ>, <source-w>x<source-h> -> <output-w>x<output-h>[, deinterlace]
```

The `<output-w>x<output-h>` reflects both crop and any SAR-driven resize; if neither applies it equals the source dimensions.

Final line of the whole command:

```
          -> furnace-plan.json (7 jobs)
```

#### Interrupt

If the user hits Ctrl+C, the active floating row is cleared, and the very last printed line is `interrupted`. The exit code is 130.

## Architecture

### Component map

```
furnace/core/ports.py                ← + PlanReporter Protocol
furnace/ui/plan_console.py           ← NEW: RichPlanReporter (real terminal output)
tests/fakes/recording_reporter.py    ← NEW: RecordingPlanReporter (test fake)
furnace/adapters/makemkv.py          ← + PRGV/PRGT/PRGC parser; ensure -r flag
furnace/adapters/ffmpeg.py           ← + -progress pipe wiring for idet/cropdetect/audio profile
furnace/services/disc_demuxer.py     ← takes PlanReporter, emits demux events
furnace/services/scanner.py          ← takes PlanReporter, emits scan events
furnace/services/analyzer.py         ← takes PlanReporter, emits analyze events
furnace/services/planner.py          ← takes PlanReporter, emits plan events
furnace/cli.py                       ← creates RichPlanReporter; removes _console_output,
                                       typer.echo plan messages, and _setup_logging console handler
```

The reporter is a `Protocol` in `core/ports.py`, satisfying the hexagonal rule that core defines the interface and adapters/UI implement it. Services receive the reporter via constructor or method argument (DI).

### Reporter Protocol

State is **implicit**: after `*_file_start(name)` or `demux_title_start(num)`, all subsequent micro-op / progress / done calls apply to that latest-started item. This works because the `plan` pipeline is strictly serial — only one file/title is ever active at a time.

```python
# furnace/core/ports.py

class PlanReporter(Protocol):
    # Detect
    def detect_disc(self, disc_type: DiscType, rel_path: str) -> None: ...

    # Demux
    def demux_disc_cached(self, label: str) -> None: ...
    def demux_disc_start(self, label: str) -> None: ...
    def demux_title_start(self, title_num: int) -> None: ...
    def demux_title_substep(self, label: str, has_progress: bool) -> None: ...
    def demux_title_progress(self, fraction: float) -> None: ...
    def demux_title_done(self) -> None: ...
    def demux_title_failed(self, reason: str) -> None: ...

    # Scan
    def scan_file(self, name: str) -> None: ...
    def scan_skipped(self, name: str, reason: str) -> None: ...

    # Analyze
    def analyze_file_start(self, name: str) -> None: ...
    def analyze_microop(self, label: str, has_progress: bool) -> None: ...
    def analyze_progress(self, fraction: float) -> None: ...
    def analyze_file_done(self, summary: str) -> None: ...
    def analyze_file_failed(self, reason: str) -> None: ...
    def analyze_file_skipped(self, reason: str) -> None: ...

    # Plan
    def plan_file_start(self, name: str) -> None: ...
    def plan_microop(self, label: str, has_progress: bool) -> None: ...
    def plan_progress(self, fraction: float) -> None: ...
    def plan_file_done(self, summary: str) -> None: ...

    # Final / lifecycle
    def plan_saved(self, path: Path, n_jobs: int) -> None: ...
    def interrupted(self) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
```

### Data flow

```
cli.plan(...)
  reporter = RichPlanReporter()
  _setup_logging(output, console=False)              # console handler removed
  build adapters (no on_output -> console)
  disc_demuxer = DiscDemuxer(..., reporter=reporter)

  disc_demuxer.detect(source)                        # -> reporter.detect_disc(...) per disc

  reporter.pause()
  _run_disc_demux_interactive(...)                    # Textual title-selection screens
  reporter.resume()

  disc_demuxer.demux(selected_titles, demux_dir)     # -> reporter.demux_*(...)
    adapters call reporter.demux_title_progress(fraction) directly from progress parsers

  scanner.scan(source, output)                        # -> reporter.scan_file(...)

  for sr in scan_results:
      analyzer.analyze(sr)                            # -> reporter.analyze_*(...)
        analyzer triggers analyze_microop before each ffprobe/idet/profile call
        analyze_progress is fed by the -progress pipe parser for idet / audio profile

  reporter.pause()
  (track-selector Textual screens)
  reporter.resume()

  planner.create_plan(movies, ...)                    # -> reporter.plan_*(...)
    cropdetect drives plan_microop("cropdetect", True) + plan_progress(...)

  save_plan(plan_obj, output / "furnace-plan.json")
  reporter.plan_saved(path, n_jobs)
```

### Adapter integration points

The existing `run_tool(cmd, on_progress_line=...)` contract in `adapters/_subprocess.py` already returns parsed `ProgressSample` objects to the caller. We change only the consumer: instead of feeding a `ProgressTracker` (as in the run phase), the parser callback calls the appropriate reporter method directly. The service is responsible for choosing which reporter method to call (it knows which phase is active); the adapter stays phase-agnostic.

| Adapter call | Progress source | Reporter method called by service |
|---|---|---|
| `eac3to.demux_title` | `process: NN%` (parser exists) | `demux_title_progress` |
| `eac3to.transcode_to_flac` | `process: NN%` | `demux_title_progress` |
| `mkvmerge` (in `disc_demuxer._mux_to_mkv`) | `Progress: NN%` (parser exists) | `demux_title_progress` |
| `makemkv.demux_title` | PRGV / PRGT / PRGC (NEW parser) | `demux_title_progress` |
| `ffmpeg` idet (in `Analyzer`) | `-progress pipe` (NEW wiring) | `analyze_progress` |
| `ffmpeg` audio profile (in `Analyzer`) | `-progress pipe` (NEW wiring) | `analyze_progress` |
| `ffmpeg` cropdetect (in `Planner`) | `-progress pipe` (NEW wiring) | `plan_progress` |

### MakeMKV progress parser

`makemkvcon` in `-r` (robot) mode emits structured lines. The parser handles:

- `PRGV:current,total,max` → `ProgressSample(fraction=current/max)`. `total` reflects the current sub-task; `max` reflects the overall task. Use `max` for the overall fraction.
- `PRGT:code,id,"name"` → ignored (we use the substep label set by the service, not MakeMKV's internal task names).
- `PRGC:code,id,"name"` → ignored, same reason.
- Any other line → `None` (not a progress line).

The adapter must run `makemkvcon` with `-r` (already done? — verify in implementation; add if missing). Raw lines that are not consumed by the progress parser still flow to per-tool log file via `run_tool`'s normal capture path; nothing changes there.

### ffmpeg `-progress pipe` wiring

Three current ffmpeg invocations in `Analyzer`/`Planner` do not pass `-progress pipe:1`:
- `Prober.run_idet`
- `Prober.profile_audio_track`
- `FFmpegAdapter.cropdetect` (`ffmpeg.py:181`), called from `PlannerService` (`planner.py:173`)

We add `-progress pipe:1` to each, parse the resulting `key=value` blocks (parser already exists in `ffmpeg.py:57-78` for the encode path — reuse it), and feed `processed_s / duration_s` as the fraction. The duration of the input is already known to the caller.

### Logging changes

In `_setup_logging` for the `plan` command, pass `console=False` (already supported — used by `run`). All `logger.info/warning/exception` calls keep going to `furnace.log` as before; nothing reaches the terminal except via the reporter.

The `_console_output` callback in `cli.py:96-98` is deleted. All adapter constructors that currently take `on_output=_console_output` (eac3to, makemkv) instead pass `on_output=None` in the `plan` flow. Per-tool log files are still produced (handled by `run_tool` via `set_log_dir`), so all raw tool output remains inspectable on disk.

## Error handling

### Per-file (soft — pipeline continues)

| Trigger | Reporter call | Render |
|---|---|---|
| `Analyzer` raises `ValueError("HDR10+ not supported")` | `analyze_file_failed("HDR10+ not supported")` | `name  FAILED — HDR10+ not supported` |
| `Analyzer` returns `None` for unknown codec | `analyze_file_failed(reason)` | `name  FAILED — <reason>` |
| `Analyzer` returns `None` because `should_skip_file` | `analyze_file_skipped(reason)` | `name  SKIPPED — <reason>` |
| `Analyzer` returns `None` because no video stream | `analyze_file_skipped("no video stream")` | `name  SKIPPED — no video stream` |
| ffprobe fails on a file | `analyze_file_failed("probe failed")` | `name  FAILED — probe failed` |

After each `failed` / `skipped`, the loop continues with the next file. The final `Plan` only includes successful files.

### Phase-fatal (hard — command aborts)

| Trigger | Reporter call | Caller behavior |
|---|---|---|
| eac3to / MakeMKV / mkvmerge crash during demux | `demux_title_failed(reason)` | `RuntimeError` propagates; `cli.plan` catches at top level, exits non-zero |
| `.w64 → .flac` transcode fails | `demux_title_failed("transcode to FLAC failed: <details>")` | same |
| `pcm_transcoder=None` but `.w64` was produced | `demux_title_failed("pcm_transcoder not configured")` | same |
| Configuration error (missing tool path, unreadable config) | reporter not created yet — fall back to typer error | exit 2 |

### Interrupt

`KeyboardInterrupt` is caught in `cli.plan`'s top-level try block:
1. `reporter.interrupted()` — clears the floating progress row, prints the final line `interrupted`.
2. Any in-flight subprocess is terminated. The `run` phase already does this via `Executor._shutdown_event` (`executor.py:123, 723, 728`) — a `threading.Event` checked between jobs and used to `child.kill()` running subprocesses. The `plan` phase has no equivalent today; we add a small mirror: a `threading.Event` owned by `cli.plan`, set on `KeyboardInterrupt`, passed to `run_tool` so the subprocess reader threads can check it and kill the child. This is a small new piece of plumbing in `adapters/_subprocess.py`, not a reuse.
3. Exit code 130.

Partial demux state is preserved by the existing `.done` marker mechanism in `disc_demuxer` — re-running the command picks up where it left off.

## Edge cases

| Case | Behavior |
|---|---|
| Detect finds no discs | `Detect` section absent. Continue to `Scan`. |
| Scan finds no video files | `Scan` section shows a single line `(no video files found)`. `Analyze` and `Plan` sections absent. The empty plan is still saved (preserves current behavior). |
| Source is `D:\foo\VIDEO_TS` directly | Detect returns nothing because `rglob("*")` does not include `source` itself. Renders empty `Detect`. **Out of scope — see Future Work.** |
| Source is a drive root (`C:\`) with `C:\VIDEO_TS` inside | Detect finds it, but `disc_label = ""` produces output filenames like `_title_1.mkv`. **Out of scope — see Future Work.** |
| File names with non-ASCII characters | Rich is Unicode-safe. On cmd.exe, ASCII-only mode is forced (see Rendering rules). |
| `stdout` not a TTY (piped/redirected) | Rich auto-detects and disables animation: bars do not redraw, only final-state lines are emitted. Text content is identical. Verified by an explicit test. |

## Testing

All tests run via `make check` (per CLAUDE.md). TDD strict — failing test before implementation, every change. 100% line + branch coverage on new/changed code.

### Layout

```
tests/
  fakes/
    recording_reporter.py            # RecordingPlanReporter
  adapters/
    test_makemkv_progress.py
    test_ffmpeg_idet_progress.py
    test_ffmpeg_cropdetect_progress.py
    test_ffmpeg_audio_profile_progress.py
  services/
    test_disc_demuxer_reports.py
    test_scanner_reports.py
    test_analyzer_reports.py
    test_planner_reports.py
  ui/
    test_plan_console.py
    snapshots/
      canonical_plan.txt
  test_plan_output_integration.py
```

### Recording fake

```python
# tests/fakes/recording_reporter.py
@dataclass
class Event:
    method: str
    args: tuple

class RecordingPlanReporter:
    def __init__(self) -> None:
        self.events: list[Event] = []
    def __getattr__(self, name: str) -> Callable[..., None]:
        def _record(*args: object) -> None:
            self.events.append(Event(name, args))
        return _record
```

Service tests assert event sequences against this list.

### RichPlanReporter rendering tests

Render via `Console(file=StringIO(), force_terminal=True, width=120, color_system=None, legacy_windows=False)`. Avoid full snapshot comparisons (fragile) — test invariants instead:

- Header `Source:` / `Output:` printed exactly once.
- Phase prefix appears on the first row of each phase only.
- `cached`, `done`, `FAILED`, `SKIPPED` markers render correctly.
- `.furnace_demux/` prefix is stripped from demuxed file names.
- A series of `*_progress` events does not append new lines (in-place update).
- `pause()` / `resume()` start/stop the underlying `Rich.Progress`.
- `interrupted()` clears the live region and emits `interrupted` as the final line.
- Forced ASCII mode renders `#`/`-` (no `█`) and `|/-\` spinner (no Braille).
- Non-TTY mode (`force_terminal=False`) emits no ANSI codes and no partial-update characters.

One **golden-file snapshot test** for the canonical happy-path flow, stored at `tests/ui/snapshots/canonical_plan.txt`. Acts as a smoke regression check on overall format.

### Adapter parser tests

| Test | Asserts |
|---|---|
| `test_makemkv_progress::test_prgv_basic` | `_parse_makemkv_progress_line("PRGV:5,10,100")` → `ProgressSample(fraction=0.05)` |
| `test_makemkv_progress::test_prgv_complete` | `PRGV:100,100,100` → `ProgressSample(fraction=1.0)` |
| `test_makemkv_progress::test_prgt_returns_none` | non-progress lines return `None` |
| `test_makemkv_progress::test_invalid_format` | malformed input returns `None` |
| `test_ffmpeg_idet_progress::test_command_includes_progress_pipe` | `run_idet` cmd contains `-progress pipe:1` |
| `test_ffmpeg_idet_progress::test_progress_block_parses_to_fraction` | `out_time_ms=12345000\nprogress=continue` → `ProgressSample(processed_s=12.345)` (fraction computed by caller using known duration) |
| `test_ffmpeg_idet_progress::test_progress_end_completes` | `progress=end` → fraction=1.0 |
| `test_ffmpeg_cropdetect_progress` | mirror of idet tests for cropdetect |
| `test_ffmpeg_audio_profile_progress` | mirror for `profile_audio_track` |

Reuse the existing `-progress` block parser from `ffmpeg.py:57-78` (encode path).

### Service tests

| Test | Scenario | Asserted event sequence |
|---|---|---|
| `test_disc_demuxer_reports::test_cached_disc` | every selected title has a `.done` marker | `[demux_disc_cached("OldMatrix_BD")]` |
| `::test_fresh_dvd_title` | DVD, MakeMKV outputs single `.mkv` | `demux_disc_start, demux_title_start, demux_title_substep("rip", True), <progress>, demux_title_done` |
| `::test_fresh_bd_title_no_w64` | BD, eac3to + mkvmerge, no `.w64` | substeps `rip`, `remux` (no `transcode`) |
| `::test_fresh_bd_title_with_w64` | BD, eac3to with two `.w64` files | substeps `rip`, `transcode 1/2`, `transcode 2/2`, `remux` |
| `::test_demux_fails_propagates` | adapter raises `RuntimeError` | `demux_title_failed(reason)`, `RuntimeError` re-raised |
| `test_analyzer_reports::test_simple_sdr` | SDR, no idet, no profileable audio | `analyze_file_start, analyze_microop("probing", False), analyze_file_done(summary)` |
| `::test_hdr10_with_side_data` | PQ transfer | extra `analyze_microop("HDR side data", False)` |
| `::test_interlaced_idet` | ambiguous `field_order` | `analyze_microop("idet", True)` + progress |
| `::test_5_1_audio_profile` | 6-channel audio | `analyze_microop("audio profile track 1", True)` per track |
| `::test_hdr10plus_fails` | raises `ValueError` | `analyze_file_failed("HDR10+ not supported")` |
| `::test_already_encoded_skipped` | `should_skip_file` True | `analyze_file_skipped(reason)` |
| `::test_no_video_stream` | streams list lacks video | `analyze_file_skipped("no video stream")` |
| `test_planner_reports::test_cropdetect_emits_progress` | planner runs cropdetect | `plan_microop("cropdetect", True)` + progress |
| `test_scanner_reports::test_emits_per_file` | rglob over fake FS | one `scan_file(name)` per video file |

### Integration test

`test_plan_output_integration::test_full_plan_flow` wires:
- Fake filesystem with one BD (one title with `.w64`), one DVD, two regular files, one HDR10+ file (broken).
- Real `Scanner`, `Analyzer`, `PlannerService`, `DiscDemuxer`.
- Fake adapters (ffprobe / eac3to / makemkv / mkvmerge / cropdetect — return prepared canned data).
- `RecordingPlanReporter`.
- Direct call to the reporter-aware portion of `cli.plan` (bypassing typer).

Asserts the full event stream matches the expected sequence (Detect → all Demux events including `transcode`, → Scan → Analyze with one `analyze_file_failed` for the HDR10+ file → Plan → `plan_saved`).

### Coverage targets

100% line + branch on all new/changed files. Branch coverage matters most for:
- HDR transfer check (`color_transfer in {smpte2084, arib-std-b67}`)
- `needs_idet()` trigger
- Audio-profile channel filter (`channels in {2, 6, 8}`)
- `.w64` presence check in `_transcode_w64_files`
- Demux per-disc cached vs fresh
- `analyze_file_failed` vs `analyze_file_skipped` vs `analyze_file_done`
- TTY vs non-TTY rendering
- ASCII vs Unicode rendering

## Future Work (out of scope)

These bugs were uncovered during this design and are deliberately **not** addressed by this work:

1. **`disc_demuxer.detect` does not match `source` itself.** When `source` points directly at a `BDMV/` or `VIDEO_TS/` directory, `rglob("*")` does not include `source` in its results, so detection returns empty. Fix: also check `source.name` itself.

2. **Empty `disc_label` for drive-root sources.** When a disc directory sits directly under a drive root (e.g. `C:\VIDEO_TS`), `disc.path.parent.name` is `""`, producing output filenames like `_title_1.mkv`. Fix: fall back to drive letter or a sentinel name.

Each should be its own small task with its own spec when prioritized.
