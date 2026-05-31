# `furnace scan` — design

Date: 2026-05-31
Status: approved (pending spec review)

## Purpose

A read-only inventory command. Given a folder (or single file), walk it for
video files and print a table describing each one — its Furnace-encode status
plus its video/audio/subtitle tracks — optionally filtered by encode status.

Primary motivating use: find files that need (re)encoding — never touched by
Furnace, or encoded by an old Furnace version (e.g. before a fix). General
use: "what's in this folder and what has Furnace already done to it."

Non-goal: `scan` never modifies files and never builds or runs a plan. It only
reports.

## CLI surface

```
furnace scan SRC [--not-encoded] [--encoded] [--max-version X.Y.Z] [--config PATH]
```

- `SRC` — a directory (scanned recursively) or a single video file.
- `--config PATH` — config file for tool paths (ffprobe), same as `plan`/`run`.
- No filter flags → every video file is shown.
- Filter flags select by **encode status** only (the table still *displays*
  track detail; it is not filtered on track detail in v1).

### Filter predicates

- `--not-encoded` — files with no parseable Furnace tag.
- `--encoded` — files encoded by any Furnace version.
- `--max-version X.Y.Z` — files encoded by Furnace at version ≤ X.Y.Z.

All three are on the same dimension (encode status), so multiple flags
**union (OR)**: a file is shown if it matches *any* supplied predicate. This
makes the redo-hunt combo natural:

```
furnace scan DIR --not-encoded --max-version 1.19.3
# everything needing (re)encode: never touched OR encoded by an old Furnace
```

A `--max-version` whose argument does not parse as `X.Y.Z` is a CLI error.

## Furnace detection (strict)

Furnace stamps its output's MKV `ENCODER` tag as `Furnace v{VERSION}`
(e.g. `Furnace v1.19.4`). Detection is **strict**: only a tag matching
`^Furnace v(\d+)\.(\d+)\.(\d+)$` counts as Furnace, yielding the version
tuple `(major, minor, patch)`.

Everything else — no tag, a foreign encoder's tag, or a malformed Furnace tag
— is treated as **not encoded**. Foreign encoder names are not surfaced: the
status is simply `not encoded`. Consequence accepted by design: a malformed
Furnace tag will not match `--max-version` (it reads as not-encoded).

## Output

The **table is written to stdout** and must be redirect-safe
(`furnace scan DIR > out.txt` yields a clean file):

- No ANSI color/control codes when stdout is not a TTY.
- ASCII box-drawing (consistent with the project's Windows-cmd rule).
- Columns sized to content — long paths are not truncated.

All non-table output — a one-line summary (`12 of 80 shown`), warnings, and
"no video files found" — goes to **stderr**, so the redirected file is pure
table.

One row per file, columns:

| Column | Content |
|--------|---------|
| File   | Path relative to `SRC` |
| Status | `Furnace v1.19.3` or `not encoded` |
| Video  | Video stream codec (`hevc`, `h264`, `mpeg2video`, …) or `—` if none |
| Audio  | One line per audio track: `<lang> <codec> <N>ch` (e.g. `rus ac3 2ch`); `und` when language is missing |
| Subs   | One line per subtitle track: `<lang> <codec>` (e.g. `eng subrip`) |

## Architecture

Hexagonal, a new self-contained `scan` vertical. Reuses `Prober.probe()`
(one ffprobe call per file yields both the `ENCODER` tag and all stream
detail) and `Scanner`'s video-file discovery. Does **not** use the heavy
`Analyzer` (HDR/idet/crop/fps work scan does not need).

### Core — `furnace/core/scan.py` (pure, no I/O)

```python
@dataclass(frozen=True)
class AudioTrackSummary:
    language: str | None
    codec: str
    channels: int | None

@dataclass(frozen=True)
class SubtitleTrackSummary:
    language: str | None
    codec: str

@dataclass(frozen=True)
class ScanRow:
    path: Path
    furnace_version: tuple[int, int, int] | None
    video_codec: str | None
    audio: tuple[AudioTrackSummary, ...]
    subtitles: tuple[SubtitleTrackSummary, ...]
    unreadable: bool = False

def parse_furnace_version(encoder_tag: str | None) -> tuple[int, int, int] | None: ...
def parse_version_arg(s: str) -> tuple[int, int, int]: ...   # raises ValueError
def summarize_streams(probe_json: dict) -> tuple[
    str | None,
    tuple[AudioTrackSummary, ...],
    tuple[SubtitleTrackSummary, ...],
]: ...
def row_matches(
    version: tuple[int, int, int] | None,
    *,
    not_encoded: bool,
    encoded: bool,
    max_version: tuple[int, int, int] | None,
) -> bool: ...   # no predicate set -> True
```

### Service — `furnace/services/scan_service.py`

- `scan(root, *, not_encoded, encoded, max_version) -> list[ScanRow]`
- Discover video files recursively (reuse `Scanner` `VIDEO_EXTENSIONS` + sorted
  `rglob`; a single-file `root` yields one entry).
- Per file: `prober.probe(path)` → read `format.tags.ENCODER` →
  `parse_furnace_version`; `summarize_streams(probe_json)`; build `ScanRow`.
- Probe failure (`OSError`/`RuntimeError`/`ValueError`) → `ScanRow(unreadable=True)`
  + a warning emitted by the caller to stderr.
- Filter via `row_matches`; return survivors in discovery order.

### UI/CLI

- `furnace/ui/scan_table.py` — `render_scan_table(rows, *, root, file=sys.stdout)`:
  Rich `Table` with `box=ASCII`, written through a `Console` that disables
  styling for non-TTY and is sized so no column truncates. Summary/warnings go
  to stderr (separate `Console(stderr=True)` or plain `print(..., file=sys.stderr)`).
- `cli.py` `scan` command: parse flags, build the ffprobe-backed `Prober` from
  `--config`, run `ScanService`, render. An `unreadable` row renders with
  `Status = unreadable` and `—` elsewhere.

## Error handling

- Unreadable/corrupt file → visible `unreadable` row (never silently dropped)
  + stderr warning.
- Bad `--max-version` argument → typer `BadParameter` error.
- No video files found → empty table (header only) + stderr note.
- File with no video stream → Video column `—`.

## Testing (TDD, 100% line + branch)

- `tests/core/test_scan.py` — `parse_furnace_version` (valid, foreign tag,
  malformed Furnace, `None`); `parse_version_arg` (valid + raises); 
  `summarize_streams` over crafted ffprobe JSON (multi audio/sub, missing
  language, no video stream, channel counts); `row_matches` (each predicate
  alone, OR combos, empty → all).
- `tests/services/test_scan_service.py` — mocked `Prober`: row building,
  recursion + single-file root, filtering, unreadable handling, ordering.
- `tests/ui/test_scan_table.py` — renderer emits ASCII (no Unicode box), no
  ANSI when non-TTY, multi-line audio/sub cells, long path not truncated,
  `unreadable` row formatting; summary to stderr not stdout.

## Versioning

New CLI command → **MINOR**: `1.19.4` → `1.20.0`.
