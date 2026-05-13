# Mono Downmix — Executor Stream

## Overview

Rewrite the `DownmixMode.MONO` branch inside `Executor._process_audio_track` in `furnace/services/executor.py` to follow a three-path flow: stereo source → direct `stereo_to_mono_wav`; multichannel eac3to-friendly source → `extract_track` → `decode_lossless(downmix=STEREO)` → `stereo_to_mono_wav` → `encode_aac`; multichannel eac3to-incompatible source → `ffmpeg_to_wav` → `decode_lossless(downmix=STEREO)` → `stereo_to_mono_wav` → `encode_aac`. Rewrite the `TestDecodeEncodeMonoDownmix` class in `tests/services/test_executor_downmix.py` to lock the new flow with mocked Protocol adapters. This is one of three parallel worktrees executing the design at `docs/superpowers/specs/2026-04-26-mono-via-eac3to-stereo-design.md`. The sister streams own the port and the adapter; do not modify their files here.

The project test suite (`make check`) is expected to be RED in this isolated worktree because the port still declares the old method name and the runtime-checkable Protocol check would fail against the real adapter. That cross-cutting failure is intentional and is not this stream's signal — integration verification runs on master after the three streams merge. Tests in this stream rely on mocked Protocol adapters and do not depend on the real port being renamed yet.

## Context

- Owned files (only files this worktree may modify): `furnace/services/executor.py`, `tests/services/test_executor_downmix.py`.
- Reference: design spec `docs/superpowers/specs/2026-04-26-mono-via-eac3to-stereo-design.md`.
- Architecture rule: services orchestrate adapters via Protocol interfaces. Tests here mock the Protocol; production code calls the (new) method name.
- Do not run `make test`, `make check`, `make lint`, or `make typecheck` in this worktree.
- Every subagent dispatch uses model `opus`.

## Development Approach

- TDD at the file level: rewrite the test class first, then rewrite the executor's MONO branch.
- File-level verification only — Read tool and grep, no test commands.
- Do not commit. The orchestrator commits and merges this branch.

## Testing Strategy

- Mock the Protocol adapters (`audio_extractor`, `audio_decoder`, `aac_encoder`) at the boundary; assert call ordering and argument routing.
- Cover every branch listed in Technical Details: stereo direct, multichannel via `extract_track`, multichannel via `ffmpeg_to_wav`, every rc-non-zero failure mode, the `channels is None` guard, delay routing on both paths, the regression that `DECODE_ENCODE` without `MONO` does not call `stereo_to_mono_wav`.

## Progress Tracking

- Mark `[x]` immediately when each item is done.

## Technical Details

### New executor MONO branch behaviour

Inside `_process_audio_track` (around the existing `DownmixMode.MONO` branch at roughly lines 510–540), implement three flows:

1. **`channels is None` guard.** If `instr.channels is None`, raise `RuntimeError(f"MONO downmix without channel count for stream {track_idx}")` before any subprocess work.
2. **Stereo source (`channels == 2`).** Build `mono_wav = temp_dir / f"audio_{track_idx}_mono.wav"`. Call `self._audio_extractor.stereo_to_mono_wav(input_path=source_path, stream_index=track_idx, output_wav=mono_wav, delay_ms=instr.delay_ms)`. If rc non-zero, raise `RuntimeError(f"stereo_to_mono_wav failed: rc={rc}")`. Build `m4a_path = temp_dir / f"audio_{track_idx}.m4a"`, get an `on_progress` from `self._make_progress_callback(total_s=None)`, call `self._aac_encoder.encode_aac(mono_wav, m4a_path, on_progress=on_progress)`. If rc non-zero, raise `RuntimeError(f"encode_aac failed: rc={rc}")`. Return `m4a_path`.
3. **Multichannel source (`channels != 2`, includes 6, 8, anything else non-None).** If `_codec_supported_by_eac3to(instr.codec_name)` is true: build `extracted = temp_dir / f"audio_{track_idx}_raw{ext}"`, get an `on_progress` from `self._make_progress_callback(total_s=job.duration_s or None)`, call `self._audio_extractor.extract_track(source_path, track_idx, extracted, on_progress=on_progress)`. If rc non-zero, raise `RuntimeError(f"Audio extract (MONO multichannel) failed with rc={rc} for stream {track_idx}")`. Otherwise (`_codec_supported_by_eac3to` false): build `extracted = temp_dir / f"audio_{track_idx}_pre.wav"`, get an `on_progress`, call `self._audio_extractor.ffmpeg_to_wav(source_path, track_idx, extracted, on_progress=on_progress)`. If rc non-zero, raise `RuntimeError(f"ffmpeg pre-decode (MONO multichannel) failed with rc={rc} for stream {track_idx}")`. Then build `stereo_wav = temp_dir / f"audio_{track_idx}_stereo.wav"`, get an `on_progress` from `self._make_progress_callback(total_s=None)`, call `self._audio_decoder.decode_lossless(extracted, stereo_wav, instr.delay_ms, on_progress=on_progress, downmix=DownmixMode.STEREO)` — note the positional layout `(input, output, delay_ms, ...)` for compatibility with mocked call inspection. If rc non-zero, raise `RuntimeError(f"eac3to -downStereo failed: rc={rc}")`. Then build `mono_wav = temp_dir / f"audio_{track_idx}_mono.wav"`, call `self._audio_extractor.stereo_to_mono_wav(input_path=stereo_wav, stream_index=0, output_wav=mono_wav, delay_ms=0)` — `stream_index=0` because the stereo intermediate is a single-track WAV; `delay_ms=0` because the delay was already applied by `decode_lossless`. If rc non-zero, raise `RuntimeError(f"stereo_to_mono_wav failed: rc={rc}")`. Then build `m4a_path = temp_dir / f"audio_{track_idx}.m4a"`, get an `on_progress`, call `self._aac_encoder.encode_aac(mono_wav, m4a_path, on_progress=on_progress)`. If rc non-zero, raise `RuntimeError(f"encode_aac failed: rc={rc}")`. Return `m4a_path`.

Indentation: this block lives inside `if instr.action == AudioAction.DECODE_ENCODE:`. Match the existing surrounding indentation exactly. The MONO block replaces the existing one entirely; everything from `if instr.downmix == DownmixMode.MONO:` through its matching `return m4a_path` (inclusive) is rewritten.

There is NO fallback to the old ITU matrix on any rc-non-zero — failures surface to the caller.

### Test surface

In `tests/services/test_executor_downmix.py`:

- Extend the existing `_instr` helper at the top of the file to accept a `delay_ms: int = 0` parameter and forward it via `make_audio_instruction(..., delay_ms=delay_ms)`. Preserve all other parameters of the helper.
- Replace the entire `TestDecodeEncodeMonoDownmix` class with a new class containing the test cases below. The two sibling classes in the file (`TestDecodeEncodeDownmixRouting`, `TestDecodeEncodeDownmixProgressWiring`) must NOT be touched.

Required test cases inside the new `TestDecodeEncodeMonoDownmix`:

- `test_5_1_dts_chains_extract_eac3to_stereo_mono_qaac` — 5.1 DTS source: asserts `extract_track` called once, `ffmpeg_to_wav` NOT called, `decode_lossless` called with `downmix=DownmixMode.STEREO`, `stereo_to_mono_wav` called with `stream_index=0` and `delay_ms=0` and `output_wav` ending in `.wav`, `encode_aac` called once.
- `test_7_1_truehd_chains_extract_eac3to_stereo_mono_qaac` — 7.1 TrueHD source: asserts the same chain (`extract_track` → `decode_lossless(downmix=STEREO)` → `stereo_to_mono_wav` → `encode_aac`).
- `test_aac_5_1_chains_ffmpeg_to_wav_eac3to_stereo_mono_qaac` — AAC 5.1 source: asserts `ffmpeg_to_wav` called once, `extract_track` NOT called, `decode_lossless` called with `downmix=DownmixMode.STEREO`, `stereo_to_mono_wav` called once, `encode_aac` called once.
- `test_multichannel_delay_goes_to_eac3to_not_to_mono_step` — 5.1 DTS source with `delay_ms=125`: asserts `decode_lossless`'s positional arg index 2 equals `125`, asserts `stereo_to_mono_wav`'s `delay_ms` kwarg equals `0`.
- `test_stereo_source_skips_eac3to_calls_mono_directly` — `channels=2` source with `delay_ms=-30`: asserts `extract_track` NOT called, `ffmpeg_to_wav` NOT called, `decode_lossless` NOT called, `stereo_to_mono_wav` called with `stream_index` equal to the instruction's `stream_index`, `input_path` equal to `Path(instr.source_file)`, `delay_ms=-30`; `encode_aac` called once.
- `test_decode_encode_without_mono_does_not_call_stereo_to_mono` — `DECODE_ENCODE` without `downmix=MONO` (downmix=None, channels=6): asserts `stereo_to_mono_wav` NOT called; `decode_lossless` called once.
- `test_multichannel_raises_when_eac3to_fails` — 5.1 DTS, `decode_lossless` returns `7`: asserts `RuntimeError` matches `r"eac3to -downStereo failed.*rc=7"`; asserts `stereo_to_mono_wav` and `encode_aac` NOT called.
- `test_multichannel_raises_when_extract_fails` — 5.1 DTS, `extract_track` returns `9`: asserts `RuntimeError` matches `r"Audio extract.*MONO.*rc=9"`; asserts `decode_lossless` and `stereo_to_mono_wav` NOT called.
- `test_multichannel_raises_when_ffmpeg_pre_decode_fails` — AAC 5.1, `ffmpeg_to_wav` returns `11`: asserts `RuntimeError` matches `r"ffmpeg pre-decode.*MONO.*rc=11"`; asserts `decode_lossless` NOT called.
- `test_multichannel_raises_when_stereo_to_mono_fails` — 5.1 DTS, `stereo_to_mono_wav` returns `5`: asserts `RuntimeError` matches `r"stereo_to_mono_wav failed.*rc=5"`; asserts `encode_aac` NOT called.
- `test_stereo_raises_when_stereo_to_mono_fails` — stereo source, `stereo_to_mono_wav` returns `5`: asserts `RuntimeError` matches `r"stereo_to_mono_wav failed.*rc=5"`; asserts `encode_aac` NOT called.
- `test_mono_raises_when_encode_aac_fails` — stereo source, `encode_aac` returns `3`: asserts `RuntimeError` matches `r"encode_aac failed.*rc=3"`.
- `test_mono_raises_when_channels_none` — `channels=None`: asserts `RuntimeError` matches `"MONO downmix without channel count"`; asserts none of `stereo_to_mono_wav`, `extract_track`, `ffmpeg_to_wav`, `decode_lossless` were called.

All tests use the existing `executor_with_mocks` fixture and the existing `_job()` helper. The mocked `audio_extractor` exposes `stereo_to_mono_wav` (mock auto-creates the method); calls to it must use keyword arguments (`input_path=`, `stream_index=`, `output_wav=`, `delay_ms=`) to match the production code.

## Implementation Steps

### Task 1: Extend the helper and rewrite the test class

- [x] in `tests/services/test_executor_downmix.py`, extend the `_instr` helper to accept `delay_ms: int = 0` and pass it through to `make_audio_instruction`
- [x] replace the entire existing `TestDecodeEncodeMonoDownmix` class with the new class containing every test case listed in Technical Details
- [x] confirm `TestDecodeEncodeDownmixRouting` and `TestDecodeEncodeDownmixProgressWiring` classes were not modified
- [x] verify via Read that the new test method names appear and no test references the old `downmix_to_mono_wav` method name
- [x] write tests for new functionality
- [x] run project tests - SKIP this step in this worktree (per plan Context: no make commands in this worktree)

### Task 2: Rewrite the executor MONO branch

- [x] in `furnace/services/executor.py`, locate the `DownmixMode.MONO` branch inside `_process_audio_track`
- [x] replace the entire `if instr.downmix == DownmixMode.MONO:` block (through its matching `return m4a_path`) with the new three-path implementation per Technical Details
- [x] preserve the surrounding indentation of the enclosing `AudioAction.DECODE_ENCODE` block
- [x] verify each rc-non-zero failure raises a `RuntimeError` with the exact prefix specified (`"MONO downmix without channel count for stream <idx>"`, `"Audio extract (MONO multichannel) failed with rc=<rc> for stream <idx>"`, `"ffmpeg pre-decode (MONO multichannel) failed with rc=<rc> for stream <idx>"`, `"eac3to -downStereo failed: rc=<rc>"`, `"stereo_to_mono_wav failed: rc=<rc>"`, `"encode_aac failed: rc=<rc>"`)
- [x] verify via Read that `downmix_to_mono_wav` no longer appears anywhere in `furnace/services/executor.py`
- [x] write tests for new functionality (completed in Task 1)
- [x] run project tests - SKIP this step in this worktree

### Task 3: Verify acceptance criteria

- [x] confirm `furnace/services/executor.py` MONO branch follows the three-path structure (None guard, stereo direct, multichannel)
- [x] confirm `tests/services/test_executor_downmix.py` contains every test case listed in Technical Details inside `TestDecodeEncodeMonoDownmix`
- [x] confirm `grep -rn "downmix_to_mono_wav" furnace/services/ tests/services/` returns zero hits
- [x] confirm no file outside `furnace/services/executor.py` and `tests/services/test_executor_downmix.py` was modified by this worktree

## Post-Completion

*Informational, no checkboxes*

- The orchestrator merges this branch to master after the sister streams complete.
- Integration `make check` runs on master post-merge.
