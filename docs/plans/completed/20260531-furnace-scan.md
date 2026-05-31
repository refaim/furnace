# furnace scan — folder inventory & encode-status filter

## Overview

Add a read-only `furnace scan` CLI command that walks a folder (or single file)
for video files and prints a redirect-safe table describing each one: its
Furnace-encode status plus its video, audio, and subtitle tracks. The table can
optionally be filtered by encode status, so the user can find files that need
(re)encoding — never touched by Furnace, or encoded by an old Furnace version.

`scan` only reports. It never modifies files and never builds or runs a plan.

## Context

- Adopted from design spec `docs/designs/2026-05-31-furnace-scan-design.md`.
- New `scan` vertical in the existing hexagonal layout (core → services → ui/cli).
- Reuses `Prober.probe()` (one ffprobe call yields both the `ENCODER` tag and all
  stream detail) and `Scanner`'s `VIDEO_EXTENSIONS` + recursive discovery. Does
  NOT use the heavy `Analyzer` (HDR/idet/crop/fps work scan does not need).
- Furnace stamps its output's MKV `ENCODER` tag as `Furnace v{VERSION}`.
- Architecture rules: core is pure (no I/O); adapters behind ports; services take
  adapters via constructor injection; TUI/table output ASCII-only (Windows cmd).
- Quality gate: `make check` (ruff + mypy --strict + pytest, 100% line+branch on
  furnace/ and tests/). Linters/tests are invoked ONLY via the Makefile.

## Development Approach

- Testing approach: TDD — write the failing test before the implementation for
  every feature, per project rules. 100% line+branch coverage on new/touched code.
- Build bottom-up: pure core units first, then service, then ui, then CLI wiring.
- Complete each task fully (`make check` green) before moving to the next.
- Update this plan if scope changes during implementation.

## Testing Strategy

- `tests/core/` — pure unit tests, no mocks.
- `tests/services/` — service tests with a mocked `Prober` (Protocol).
- `tests/ui/` — table-renderer tests asserting redirect-safe output.
- Run `make check` after each task; it must pass before proceeding.

## Progress Tracking

- Mark completed items with `[x]` immediately when done.
- Update plan if implementation deviates from original scope.

## Technical Details

Detection (strict): only an `ENCODER` tag matching `^Furnace v(\d+)\.(\d+)\.(\d+)$`
counts as Furnace, yielding `(major, minor, patch)`. Anything else (no tag,
foreign encoder, malformed Furnace tag) → "not encoded". Foreign encoder names
are never surfaced.

CLI:

```
furnace scan SRC [--not-encoded] [--encoded] [--max-version X.Y.Z] [--config PATH]
```

- No flags → all video files shown.
- Predicates union (OR), all on the encode-status dimension:
  - `--not-encoded` → no parseable Furnace tag
  - `--encoded` → any Furnace version
  - `--max-version X.Y.Z` → Furnace version ≤ X.Y.Z (arg must be full `X.Y.Z`)

Output: table to **stdout**, redirect-safe — no ANSI when stdout is not a TTY,
ASCII box, columns sized to content (no truncation). Summary (`12 of 80 shown`),
warnings, and "no video files found" go to **stderr**. Columns: File (path
relative to SRC) · Status (`Furnace v1.19.3` / `not encoded` / `unreadable`) ·
Video (codec or `—`) · Audio (one line per track `lang codec Nch`, `und` when
language missing) · Subs (one line per track `lang codec`).

Core API (`furnace/core/scan.py`, pure):

- `parse_furnace_version(encoder_tag: str | None) -> tuple[int,int,int] | None`
- `parse_version_arg(s: str) -> tuple[int,int,int]` (raises `ValueError`)
- `summarize_streams(probe_json) -> (video_codec, audio_tuple, sub_tuple)`
- dataclasses `AudioTrackSummary`, `SubtitleTrackSummary`, `ScanRow`
- `row_matches(version, *, not_encoded, encoded, max_version) -> bool`
  (no predicate set → True)

## Implementation Steps

### Task 1: Core — Furnace version parsing

- [x] add `furnace/core/scan.py` with `parse_furnace_version(encoder_tag)` —
      strict `^Furnace v(\d+)\.(\d+)\.(\d+)$` → `(maj,min,patch)`, else `None`
- [x] add `parse_version_arg(s)` for the `--max-version` argument; raises
      `ValueError` on anything not a full `X.Y.Z`
- [x] write failing unit tests first: valid tag, foreign tag, malformed Furnace
      tag, missing tag/None; arg parse valid + raises (in `tests/core/test_scan.py`)
- [x] run `make check` — must pass (100% line+branch) before next task

### Task 2: Core — stream summary and row model

- [x] add dataclasses `AudioTrackSummary(language, codec, channels)`,
      `SubtitleTrackSummary(language, codec)`, `ScanRow(path, furnace_version,
      video_codec, audio, subtitles, unreadable=False)` to `furnace/core/scan.py`
- [x] add `summarize_streams(probe_json)` — extract video codec, per-audio
      (language/codec/channels), per-subtitle (language/codec) from ffprobe JSON
- [x] write failing unit tests first: multi audio/sub, missing language → `und`,
      no video stream → `None`, channel counts, empty streams
- [x] run `make check` — must pass (100% line+branch) before next task

### Task 3: Core — encode-status filter predicate

- [x] add `row_matches(version, *, not_encoded, encoded, max_version)` —
      no predicate set → True; otherwise OR of `not_encoded and version is None`,
      `encoded and version is not None`, `max_version and version <= max_version`
- [x] write failing unit tests first: each predicate alone, OR combinations,
      empty filter → all match, boundary (== max_version)
- [x] run `make check` — must pass before next task

### Task 4: Service — ScanService

- [x] add `furnace/services/scan_service.py` with a `ScanService` taking a
      `Prober` (and reusing `Scanner` `VIDEO_EXTENSIONS` + sorted recursive
      discovery; single-file root yields one entry)
- [x] per file: `prober.probe()` → read `format.tags.ENCODER` →
      `parse_furnace_version`; `summarize_streams`; build `ScanRow`; filter via
      `row_matches`; return survivors in discovery order
- [x] probe failure (`OSError`/`RuntimeError`/`ValueError`) → `ScanRow(unreadable=True)`
- [x] write failing service tests first (mocked `Prober`): row building,
      recursion + single-file root, each filter, unreadable handling, ordering
- [x] run `make check` — must pass (100% line+branch) before next task

### Task 5: UI — redirect-safe table renderer

- [x] add `furnace/ui/scan_table.py` with `render_scan_table(rows, *, root,
      file=...)` — Rich `Table` (`box=ASCII`) via a `Console` that disables
      styling for non-TTY and does not truncate columns; one row per file with
      multi-line Audio/Subs cells; `unreadable` row shows Status `unreadable`,
      `—` elsewhere
- [x] emit summary (`N of M shown`), warnings, and "no video files found" to
      stderr (never stdout)
- [x] write failing UI tests first: ASCII only (no Unicode box), no ANSI when
      non-TTY, multi-line track cells, long path not truncated, unreadable row,
      summary goes to stderr not stdout
- [x] run `make check` — must pass before next task

### Task 6: CLI — `scan` command and version bump

- [x] add a `scan` command to `furnace/cli.py`: arg `SRC`, options
      `--not-encoded`, `--encoded`, `--max-version`, `--config`; build the
      ffprobe-backed `Prober` from config; invalid `--max-version` → typer
      `BadParameter`; run `ScanService`; render via `render_scan_table`
- [x] bump version to `1.20.0` in BOTH `furnace/__init__.py` and `pyproject.toml`
- [x] write failing tests first for the CLI command (flag parsing, bad
      `--max-version` error, wiring to service/renderer)
- [x] run `make check` — must pass before next task

### Task 7: Verify acceptance criteria

- [x] verify all requirements from Overview are implemented: `furnace scan DIR`
      lists all video files; `--not-encoded` / `--encoded` / `--max-version`
      filter (and union) correctly; output is redirect-safe (no ANSI/no
      truncation/ASCII) on stdout with summary+warnings on stderr
- [x] run the full project test suite via `make check`
- [x] run the project linter (part of `make check`) — all issues must be fixed

## Post-Completion

*Items requiring manual intervention - no checkboxes, informational only*

- Smoke-test on a real folder, including redirecting to a file
  (`furnace scan DIR --max-version 1.19.3 > out.txt`) to confirm the file is
  clean plain text.
- Dispatch a separate code-reviewer pass to zero comments (project workflow rule).
