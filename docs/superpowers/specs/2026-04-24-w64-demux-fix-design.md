# Wave64 Demux Fix — Design Note

**Status:** approved, ready for implementation plan
**Date:** 2026-04-24
**Scope:** bugfix — LPCM audio tracks lost during Blu-ray demux when eac3to
emits Wave64 (`.w64`) for streams larger than 4 GB

## Problem

`DiscDemuxer` muxes the per-track files that `Eac3toAdapter.demux_title()`
produces into an intermediate MKV via `mkvmerge`. The whitelist of
muxable extensions lives in `furnace/services/disc_demuxer.py:22-37`
(`_MKV_TRACK_EXTS`) and lists `.wav` for uncompressed PCM.

eac3to **does not always** write `.wav` for PCM/LPCM tracks. The WAV
container uses 32-bit chunk sizes, so it is hard-capped at 4 GB. When the
decoded PCM of a Blu-ray LPCM track exceeds that limit (typical for a
feature-length 24-bit / 48 kHz / 5.1 or 7.1 track — roughly >3 h at 5.1
or >2 h at 7.1 24-bit), eac3to transparently switches to **Wave64**
(`.w64`), Sony's separate container with 64-bit chunk sizes and
GUID-based headers.

`.w64` is **not** in `_MKV_TRACK_EXTS`. In
`DiscDemuxer._mux_to_mkv` (line 184):

```python
for f in files:
    if f.suffix.lower() not in _MKV_TRACK_EXTS:
        continue
    ...
```

unknown extensions are silently skipped. Result: the LPCM track never
reaches the intermediate MKV, the analyzer/planner never sees it, and the
final output has no corresponding audio stream. Silent data loss.

mkvmerge cannot accept `.w64` directly: `mkvmerge --list-types` advertises
only `WAVE (uncompressed PCM audio) [wav]`. Wave64 is a different
container from WAV (not RIFF, not RF64) and is not parsed by mkvmerge's
WAV reader.

## Fix

Between `port.demux_title()` and the mux decision in
`DiscDemuxer.demux()`, transcode every `.w64` in the returned file list
to `.flac` using eac3to, then swap the path in the list and delete the
original. `.flac` is already in `_MKV_TRACK_EXTS`, so the downstream mux
path is unchanged. FLAC is lossless and fits arbitrarily large PCM (no
4 GB cap), so the re-encode is bit-perfect.

FLAC is the only conversion target that works end-to-end:

- **WAV** — same 4 GB cap, won't hold the data.
- **RF64 WAV** — ffmpeg can write it (`-rf64 auto`), but mkvmerge's
  supported-types list does not advertise RF64; treating it as supported
  would be speculative.
- **FLAC** — lossless, mkvmerge accepts it natively, and
  `Eac3toAdapter._CODEC_EXT_MAP` already produces `.flac` for native FLAC
  tracks on disc, so the code path is already exercised in the mux step.

After muxing the intermediate MKV, the main pipeline sees the track as
FLAC. `core/rules.py:16` maps `FLAC → DECODE_ENCODE`, which routes it
through `QaacAdapter` to AAC exactly like any other lossless source. The
final file is identical to what it would be if eac3to had originally
produced `.wav`.

### Why eac3to, not ffmpeg

- `DiscDemuxer` already depends on eac3to (via `DiscDemuxerPort`) and on
  `mkvmerge`. Routing w64→flac through eac3to adds no new transitive
  tool dependency.
- eac3to's progress-line parser (`_EAC3TO_PROGRESS_RE` in
  `adapters/eac3to.py:19`) is already wired; reusing `Eac3toAdapter._run`
  gets progress tracking and per-step log files for free.
- eac3to is the tool that produced the `.w64`; having it also consume
  it keeps the "PCM-from-disc" responsibility in one place.
- The ffmpeg alternative would require a new port, a new adapter method,
  a new argument on `DiscDemuxer.__init__`, and a second progress parser
  — all infrastructure for a single call site.

eac3to determines output format from the output file's extension — passing
`.flac` triggers FLAC encoding. No `-removeDialnorm` is needed (PCM has
no dialnorm metadata).

## Components

### 1. New port: `PcmTranscoder` (`furnace/core/ports.py`)

```python
@runtime_checkable
class PcmTranscoder(Protocol):
    """Transcode uncompressed PCM (Wave64) to FLAC.

    Used by DiscDemuxer to normalize eac3to's Wave64 output to a format
    mkvmerge can mux (FLAC). Lossless — the resulting stream decodes
    bit-identical to the source PCM.
    """

    def transcode_to_flac(
        self,
        input_path: Path,
        output_path: Path,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int: ...
```

A dedicated port (not added to `DiscDemuxerPort`) because the operation
is BD-specific — the DVD path (MakeMKV) never produces Wave64 and does
not need it.

### 2. `Eac3toAdapter.transcode_to_flac` (`furnace/adapters/eac3to.py`)

```python
def transcode_to_flac(
    self,
    input_path: Path,
    output_path: Path,
    on_progress: Callable[[ProgressSample], None] | None = None,
) -> int:
    rc, _ = self._run(
        [str(input_path), str(output_path)],
        "w64_to_flac",
        on_progress=on_progress,
    )
    return rc
```

Reuses the existing `_run()` helper: progress parsing, log file
(`eac3to_w64_to_flac.log`), and raw output callback are inherited. No
`-removeDialnorm` is emitted.

### 3. `DiscDemuxer` constructor (`furnace/services/disc_demuxer.py`)

```python
def __init__(
    self,
    bd_port: DiscDemuxerPort,
    dvd_port: DiscDemuxerPort,
    mkvmerge_path: Path | None = None,
    pcm_transcoder: PcmTranscoder | None = None,
) -> None:
```

`pcm_transcoder` is optional so existing tests that do not exercise BD
paths keep passing without wiring a stub. In production CLI construction
(`furnace/cli.py`), the same `Eac3toAdapter` instance that is passed as
`bd_port` is also passed as `pcm_transcoder`.

### 4. `DiscDemuxer._transcode_w64_files` (new private method)

For each `.w64` in the demux output:

1. Compute the target path: `input.with_suffix(".flac")` in the same
   title dir.
2. Call `self._pcm_transcoder.transcode_to_flac(input, output)`.
3. On non-zero rc: raise `RuntimeError` with the input path in the
   message. Leave the `.w64` on disk so the user can inspect the
   failure.
4. On success: `input.unlink()`, then substitute the `.flac` path in
   the returned file list (preserving order).

Raise `RuntimeError` when `pcm_transcoder is None` **and** the demux
output contains any `.w64` — explicit fail rather than silent regression
to the current broken behavior.

Invocation point: in `DiscDemuxer.demux()`, immediately after
`created_files = port.demux_title(...)` (line 132) and before
`self._needs_muxing(created_files)` (line 140).

## Data flow

```
eac3to -demux
  → [video.mkv, pcm.w64, chapters.txt]
      │
      ▼ _transcode_w64_files
  eac3to pcm.w64 pcm.flac
  pcm.w64 deleted
      │
      ▼
  → [video.mkv, pcm.flac, chapters.txt]
      │
      ▼ _needs_muxing (True — multiple non-mkv files)
      ▼ _mux_to_mkv
  mkvmerge -o title_N.mkv [tracks...] pcm.flac
      │
      ▼ (unchanged downstream)
  analyzer → planner: FLAC track → DECODE_ENCODE
  executor: eac3to decode_lossless → QAAC encode → AAC
```

## Testing (TDD, 100% line + branch)

### `tests/adapters/test_eac3to.py`
- `test_transcode_to_flac_builds_expected_cmd` — mock the subprocess,
  assert args are exactly `[eac3to, input.w64, output.flac,
  -progressnumbers]`. No `-removeDialnorm`.
- `test_transcode_to_flac_propagates_rc` — non-zero rc returned verbatim.
- `test_transcode_to_flac_progress_callback` — progress line triggers
  the callback with the parsed fraction.

### `tests/services/test_disc_demuxer.py`
- `test_demux_transcodes_w64_to_flac` — fake `bd_port.demux_title`
  returns `[video.mkv, audio.w64, chapters.txt]`; fake `pcm_transcoder`
  creates the `.flac` file and returns 0. Assert: `audio.w64` no longer
  exists, `audio.flac` passed to mkvmerge.
- `test_demux_multiple_w64_files` — two `.w64` in one title, each gets
  transcoded independently, both originals deleted.
- `test_demux_transcode_failure_raises` — transcoder returns non-zero;
  assert `RuntimeError` is raised and `.w64` remains on disk.
- `test_demux_w64_without_transcoder_raises` — `pcm_transcoder=None`
  and demux output contains `.w64`; assert `RuntimeError`.
- `test_demux_no_w64_skips_transcode` — demux output contains only
  `.wav` / `.dts` / `.mkv`; transcoder is not invoked. (Regression
  anchor for the existing non-Wave64 happy path.)

### Smoke check (manual, not automated)
End-to-end: one real Blu-ray known to carry a >4 GB LPCM track is
demuxed via the CLI. Verify that the resulting intermediate MKV has the
audio track (mediainfo shows FLAC), and the final output MKV has the
expected AAC track.

## Edge cases and non-goals

- `.wav` behavior is unchanged — sub-4 GB PCM still flows directly into
  mkvmerge as today.
- `_CODEC_EXT_MAP` in `adapters/eac3to.py` (`pcm → .wav`, `lpcm → .wav`)
  is **not** touched. It is only used by `_parse_track_listing`, which
  returns track metadata for UI display; actual on-disk extensions come
  from `output_dir.iterdir()` in `demux_title` (line 240).
- The intermediate `.w64` lives inside `title_dir`, which is deleted
  after successful muxing (`shutil.rmtree(title_dir, ignore_errors=True)`
  at line 148). Explicit `.w64` deletion inside `_transcode_w64_files`
  is still required so that mkvmerge does not see it if both files
  briefly coexist during the mux call.
- No attempt to pre-empt Wave64 by forcing per-track output formats on
  the initial `eac3to -demux` call — that would require a two-pass
  playlist listing and a larger refactor of `Eac3toAdapter.demux_title`.
- No attempt to cover hypothetical RF64-WAV outputs; if that ever
  appears in the field, it becomes a separate bug.

## Version

Bugfix under the SemVer rules in `CLAUDE.md`: PATCH bump.

**1.13.1 → 1.13.2**, updated in both `furnace/__init__.py` and
`pyproject.toml`.

## Out of scope

- FLAC-for-all (converting every PCM, including sub-4 GB `.wav`, to FLAC
  at demux). Discussed; rejected in favor of the minimal fix.
- Changes to the main-pipeline FLAC handling — already correct.
- ffmpeg-based transcoding path — see "Why eac3to, not ffmpeg" above.
