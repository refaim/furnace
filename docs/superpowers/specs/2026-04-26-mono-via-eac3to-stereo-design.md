# Mono downmix via eac3to stereo intermediate — design

**Status:** approved
**Version target:** 1.15.0 (MINOR — user-facing audio behaviour changes)

## Problem

`DownmixMode.MONO` is currently produced by ffmpeg's `pan` filter applied
directly to the multichannel source, using a hand-rolled ITU-R BS.775
matrix:

- 5.1 → mono: `0.707·FC + 0.5·FL + 0.5·FR + 0.354·BL + 0.354·BR`, then `alimiter=limit=0.99`
- 7.1 → mono: same idea with side + back surrounds

The matrix and limiter live in `furnace/adapters/ffmpeg.py`
(`downmix_to_mono_wav`) and bypass eac3to entirely. This is the only path
in the audio pipeline where eac3to is skipped for an `AudioAction.DECODE_ENCODE`
track.

Two issues:

1. The ITU formula is "honest" arithmetic but it ignores AC-3 / E-AC-3
   bitstream metadata (`acmod`, `cmixlev`, `surmixlev`, `dmixmod`,
   `dnorm`). For matrix-encoded sources or streams whose author specified
   custom mix levels, the result is mathematically defined but not
   author-intended.
2. For fake-surround content (where surrounds duplicate fronts or are
   silent — the very content this path was added for), summing copies
   inflates the level past unity and forces the limiter to clamp. We're
   designing for the worst case of our own input.

eac3to is the de-facto reference for this collapse: it honours bitstream
mix metadata, applies dialnorm correctly, and decodes TrueHD / DTS-HD MA
/ DTS:X via Arcsoft. Delegating the multichannel → stereo step to it
moves the hard problem to the proven tool. The remaining stereo → mono
step is `(L+R)/2`, which is mathematically trivial and cannot clip
normalised PCM.

## Solution

Replace the direct multichannel → mono path with a two-step chain that
runs inside the existing `DECODE_ENCODE` branch of `Executor._process_audio_track`:

```
multichannel source
  └── ffmpeg.extract_track          (or ffmpeg_to_wav for non-eac3to-friendly codecs)
  └── eac3to.decode_lossless(downmix=STEREO)   → stereo WAV  (delay applied here)
  └── ffmpeg.stereo_to_mono_wav                → mono WAV    (no limiter)
  └── qaac.encode_aac                          → mono m4a
```

Stereo sources flagged as fake-mono (corr_lr > 0.98) skip eac3to entirely
and go straight through `stereo_to_mono_wav` with delay applied by ffmpeg.

The planner is unchanged. It continues to emit
`AudioInstruction(action=DECODE_ENCODE, downmix=MONO, channels=...)` —
only the executor's interpretation of that instruction changes.

## Architecture

### Adapters (thin)

- **`adapters/eac3to.py`** — no code change. `decode_lossless(downmix=DownmixMode.STEREO)`
  already produces a stereo WAV with `-downStereo -removeDialnorm` and the
  delay argument. The new flow simply calls it with the existing signature.

- **`adapters/ffmpeg.py`** — `downmix_to_mono_wav` is simplified and
  renamed to `stereo_to_mono_wav`:
  - The `channels` parameter is removed from the signature.
  - The 5.1 and 7.1 branches (with `aformat=channel_layouts=...`,
    multi-term `pan` matrix, and `alimiter=limit=0.99`) are deleted.
  - The single remaining filter is `pan=mono|c0=0.5*FL+0.5*FR`.
  - Delay (`adelay` / `atrim`) is applied as before, used by the
    stereo-source-direct path.
  - The port `AudioExtractor` in `core/ports.py` updates the method name
    and signature to match.

### Service (all logic lives here)

- **`services/executor.py`** — `_process_audio_track`, branch
  `DownmixMode.MONO` (currently lines 510-540) is rewritten:

  ```
  if instr.channels == 2:
      ffmpeg.stereo_to_mono_wav(source, idx, mono.wav, delay_ms)
      qaac.encode_aac(mono.wav, m4a)
      return m4a

  if instr.channels in (6, 8):
      # extract / pre-decode (existing logic from the STEREO path)
      if codec_supported_by_eac3to(instr.codec_name):
          ffmpeg.extract_track(source, idx, raw)
      else:
          ffmpeg.ffmpeg_to_wav(source, idx, pre.wav)
          raw = pre.wav

      eac3to.decode_lossless(raw, stereo.wav, delay_ms, downmix=STEREO)
      ffmpeg.stereo_to_mono_wav(stereo.wav, 0, mono.wav, delay_ms=0)
      qaac.encode_aac(mono.wav, m4a)
      return m4a
  ```

  Delay is applied by eac3to on the multichannel path (single source of
  truth, matches STEREO/DOWN6 behaviour) and by ffmpeg on the
  stereo-direct path. The stereo→mono ffmpeg call always passes
  `delay_ms=0` after eac3to has handled it.

### Core / planner

No changes. `DownmixMode.MONO`, `AudioProfile`, `AudioMetrics`, the
classifier in `core/audio_profile.py`, and the planner all remain as-is.
`tests/core/test_audio_profile.py` and `tests/services/test_planner_downmix.py`
are not touched.

## Data flow / temp files

For one MONO track in the per-job `temp_dir`:

**Multichannel, eac3to-friendly codec (ac3/eac3/dts/truehd/flac/...):**
```
audio_{idx}_raw.{ext}      ← ffmpeg.extract_track
audio_{idx}_stereo.wav     ← eac3to.decode_lossless(downmix=STEREO)
audio_{idx}_mono.wav       ← ffmpeg.stereo_to_mono_wav
audio_{idx}.m4a            ← qaac.encode_aac
```

**Multichannel, eac3to-incompatible codec (AAC 5.1, Opus 7.1, vorbis, ...):**
```
audio_{idx}_pre.wav        ← ffmpeg.ffmpeg_to_wav
audio_{idx}_stereo.wav     ← eac3to.decode_lossless(downmix=STEREO) on the WAV
audio_{idx}_mono.wav       ← ffmpeg.stereo_to_mono_wav
audio_{idx}.m4a            ← qaac.encode_aac
```

**Stereo source flagged as fake-mono:**
```
audio_{idx}_mono.wav       ← ffmpeg.stereo_to_mono_wav (with delay)
audio_{idx}.m4a            ← qaac.encode_aac
```

Peak temp-disk for a 7.1 / 48 kHz / 24-bit / 2-hour worst case is roughly
`pre.wav` (~14 GiB) + `stereo.wav` (~3.5 GiB) + `mono.wav` (~1.7 GiB),
all live concurrently for a short window before `temp_dir` is cleared in
`Executor._execute_job`'s `finally`. This is +3.5 GiB peak vs the current
single-pass ffmpeg path. Accepted explicitly.

Progress callbacks: each of the three subprocess steps gets its own
`_make_progress_callback`, the same way `DECODE_ENCODE` already wraps
extract / decode_lossless / encode_aac. UI shows three sequential bars
instead of one.

## Error handling

Follows the existing `_process_audio_track` pattern: each step returns
an `rc`; non-zero raises `RuntimeError(...)` with a step-specific prefix
that ends up in `JobStatus.ERROR`.

New failure points:

- eac3to `-downStereo` fails (corrupt source, unsupported bit depth /
  DTS variant): `RuntimeError(f"eac3to -downStereo failed with rc={rc} for stream {idx}")`,
  log at `<job>/eac3to_decode.log`.
- `stereo_to_mono_wav` fails on the eac3to-produced WAV (disk full,
  truncated input): `RuntimeError(f"stereo_to_mono_wav failed with rc={rc} for stream {idx}")`,
  log at `<job>/ffmpeg_mono_s{idx}.log`.
- `instr.channels is None` for a MONO instruction: existing `RuntimeError`
  guard (executor.py:516-519) is preserved. The planner's Task 12
  invariant (no MONO emitted without channels) still holds.

Explicitly NOT done:

- No fallback to the old ITU path on eac3to failure. A silent fallback
  would defeat the whole "expert tool does the hard part" decision.
  Failure surfaces to the user, who can drop the track, switch to
  STEREO downmix, or fix the source.
- No `alimiter`. `(L+R)/2` for normalised PCM cannot exceed unity:
  `|0.5L + 0.5R| ≤ 0.5|L| + 0.5|R| ≤ max(|L|, |R|) ≤ 1.0`.

`graceful_shutdown` / cancellation: unchanged. The
`self._shutdown_event.is_set()` checks between steps in `_run_pipeline`
already protect this branch.

## Testing

Per CLAUDE.md: TDD strict, 100 % line + branch coverage on new and
touched code. All test runs go through `make test` / `make check`.

### Updated test files

- **`tests/test_ffmpeg_mono_downmix.py`** — shrunk:
  - 5.1 / 7.1 cases (multi-term `pan`, `aformat`, `alimiter`) deleted.
  - One remaining unit covers `stereo_to_mono_wav` building the ffmpeg
    command with `pan=mono|c0=0.5*FL+0.5*FR`, plus delay handling
    (`delay_ms > 0` → `adelay`, `delay_ms < 0` → `atrim`, `delay_ms == 0`
    → no extra filter).
  - `channels` parameter no longer exists in the signature or the test.

- **`tests/services/test_executor_downmix.py`** — rewritten:
  - 5.1 → MONO, eac3to-friendly codec: asserts the chain
    `extract_track → decode_lossless(downmix=STEREO) → stereo_to_mono_wav → encode_aac`
    via mocked Protocol adapters.
  - 7.1 → MONO, eac3to-friendly codec: same chain, channel count varies.
  - AAC 5.1 → MONO (eac3to-incompatible): asserts
    `ffmpeg_to_wav → decode_lossless(downmix=STEREO) → stereo_to_mono_wav → encode_aac`.
  - Stereo fake-mono → MONO: asserts only `stereo_to_mono_wav → encode_aac`,
    eac3to not invoked.
  - `channels is None` → `RuntimeError` (preserved).
  - eac3to step rc≠0 → `RuntimeError` with the new prefix.
  - `stereo_to_mono_wav` rc≠0 → `RuntimeError` with the new prefix.
  - Delay routing: multichannel passes `delay_ms` to eac3to and `0` to
    ffmpeg's mono step; stereo direct passes `delay_ms` to ffmpeg.

### Untouched

- `tests/adapters/test_eac3to_downmix.py` — `-downStereo` already covered.
- `tests/services/test_planner_downmix.py`, `tests/core/test_audio_profile.py`,
  `tests/test_plan.py`, `tests/test_cli.py` — planner / classifier /
  serialization behaviour is unchanged.

## Versioning

Bump to **1.15.0** in both `furnace/__init__.py` and `pyproject.toml`.
Rationale: the audio bytes a user gets out of a MONO downmix change
(different matrix on the multichannel collapse, no limiter), and the
on-disk temp-file pattern changes. That is user-facing behaviour, hence
MINOR.
