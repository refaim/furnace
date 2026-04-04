# DVD/Blu-ray Demux & Language Filtering

## Overview

Add DVD and Blu-ray disc demuxing as a first step in the `furnace plan` pipeline. When disc structures (VIDEO_TS, BDMV) are found in the source directory, Furnace demuxes them to MKV via eac3to before the normal scan/analyze/plan flow. Also: replace the single `--lang` CLI option with separate mandatory `--audio-lang` / `--sub-lang` filters.

## CLI Changes

### New arguments

```
furnace plan <source> -o <dest> -al jpn -sl rus eng [--names map.txt] [--dry-run] [--vmaf]
```

| Old | New | Short | Required |
|-----|-----|-------|----------|
| `--lang rus eng` | `--audio-lang jpn` | `-al` | Yes |
| | `--sub-lang rus eng` | `-sl` | Yes |

`--lang` is removed.

### Language filtering rules

- Audio tracks: only languages listed in `-al` are candidates.
- Subtitle tracks: only languages listed in `-sl` are candidates.
- Tracks with language `und` are always included in candidates (both audio and subs).
- Forced subtitles are discarded entirely (not offered for selection).
- These rules apply to all files: regular, demuxed, everything.

## Architecture

### New service: `DiscDemuxer` (`furnace/services/disc_demuxer.py`)

```python
class DiscDemuxer:
    def detect(self, source: Path) -> list[DiscSource]:
        """Recursively search source for VIDEO_TS/ and BDMV/ directories."""
        ...

    def demux(
        self,
        discs: list[DiscSource],
        selected_playlists: dict[DiscSource, list[DiscPlaylist]],
        demux_dir: Path,
        on_progress: Callable | None = None,
    ) -> list[Path]:
        """Demux selected playlists to MKV. Returns paths to demuxed files.

        - Skips playlists with existing .done marker (already demuxed).
        - Deletes MKV without .done marker (partial/corrupt) and re-demuxes.
        - On demux failure: fatal error, stop immediately.
        """
        ...
```

### New protocol: `DiscDemuxerPort` (`furnace/core/ports.py`)

```python
@runtime_checkable
class DiscDemuxerPort(Protocol):
    def list_playlists(self, disc_path: Path) -> list[DiscPlaylist]: ...
    def demux_to_mkv(
        self, disc_path: Path, playlist_num: int,
        output_mkv: Path, on_progress: Callable | None = None,
    ) -> int: ...
```

### New models (`furnace/core/models.py`)

```python
class DiscType(enum.Enum):
    DVD = "dvd"
    BLURAY = "bluray"

@dataclass(frozen=True)
class DiscSource:
    path: Path          # path to VIDEO_TS/ or BDMV/ directory
    disc_type: DiscType

@dataclass(frozen=True)
class DiscPlaylist:
    number: int         # playlist number in eac3to output
    duration_s: float   # duration in seconds
    raw_label: str      # original line from eac3to
```

### Eac3to adapter changes (`furnace/adapters/eac3to.py`)

Refactor: extract common `_run()` method to eliminate duplication between `denormalize` and `decode_lossless`. Add two new methods.

```python
class Eac3toAdapter:
    def _run(self, args: list[str], log_label: str,
             on_progress: Callable | None = None) -> int:
        """Common eac3to invocation with logging and progress."""
        cmd = [str(self._eac3to), *args, "-progressnumbers"]
        rc, _out = run_tool(
            cmd, on_output=on_progress or self._on_output,
            log_path=self._log_path(log_label),
        )
        return rc

    def denormalize(self, input_path: Path, output_path: Path, delay_ms: int) -> int:
        return self._run(
            [str(input_path), str(output_path), "-removeDialnorm",
             *self._delay_arg(delay_ms)], "denorm",
        )

    def decode_lossless(self, input_path: Path, output_path: Path, delay_ms: int) -> int:
        return self._run(
            [str(input_path), str(output_path), "-removeDialnorm",
             *self._delay_arg(delay_ms)], "decode",
        )

    def list_playlists(self, disc_path: Path) -> list[DiscPlaylist]:
        """Run eac3to on disc path, parse playlist listing."""
        ...

    def demux_to_mkv(self, disc_path: Path, playlist_num: int,
                     output_mkv: Path, on_progress: Callable | None = None) -> int:
        return self._run(
            [str(disc_path), f"{playlist_num})", str(output_mkv)],
            "demux", on_progress,
        )
```

Implements both `AudioDecoder` and `DiscDemuxerPort` protocols.

### Plan JSON changes

```python
@dataclass
class Plan:
    version: str            # bumped to "2"
    furnace_version: str
    created_at: str
    source: str
    destination: str
    vmaf_enabled: bool
    demux_dir: str | None   # NEW: path to .furnace_demux/ or None
    jobs: list[Job]
```

## Pipeline Flow

### `furnace plan`

```
1. DiscDemuxer.detect(source)
   -> list[DiscSource]  (may be empty)

2. If discs found:
   a. For each disc: eac3to.list_playlists()
   b. TUI: playlist selection screen (>10 min pre-selected)
   c. DiscDemuxer.demux() -> MKVs in <source>/.furnace_demux/
      - Skip already demuxed (.done marker present)
      - Delete and re-demux partial files (no .done marker)
      - Fatal error on demux failure
   d. TUI: file selection screen if >1 MKV (all pre-selected, mpv preview)
      - If exactly 1 MKV: auto-select, skip screen

3. Scanner.scan(source, dest)
   - Scans regular files only
   - Explicitly excludes .furnace_demux/ directory

4. Combine demuxed MKVs + scanned regular files

5. Analyzer -> Planner -> plan JSON
   - Audio filtered by -al, subs filtered by -sl
   - und tracks always included
   - Forced subs discarded
   - Track selection TUI as before (auto-select if 1 per language)
```

### `furnace run`

No changes to execution pipeline. After all jobs complete successfully, delete `demux_dir` if it exists.

## TUI Screens

Both screens follow existing keyboard-only pattern (no Textual Button widgets).

### Playlist selection screen

- Shown when disc structures are detected
- Lists all playlists: number, duration, raw label from eac3to
- Playlists >10 minutes pre-selected, rest unselected
- Keys: arrow keys navigate, Space toggles, Enter confirms

### Demuxed file selection screen

- Shown when demux produces >1 MKV
- Lists files: name, duration, size (MB)
- All pre-selected by default
- Keys: arrow keys navigate, Space toggles, `p` preview in mpv, Enter confirms

## Demux directory management

- Location: `<source>/.furnace_demux/`
- Each demuxed file: `<disc_folder_name>_playlist_<N>.mkv`
- Success marker: `<name>.mkv.done` (empty file, created after successful demux)
- Resumption logic:
  - `.mkv` + `.done` exists -> skip (already demuxed)
  - `.mkv` exists without `.done` -> delete .mkv, re-demux (partial/corrupt)
  - Neither exists -> demux
- Cleanup: `furnace run` deletes entire `.furnace_demux/` after all jobs succeed

## Config

No changes to `furnace.toml` — eac3to path is already configured.
