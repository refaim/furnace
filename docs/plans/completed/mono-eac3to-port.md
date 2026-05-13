# Mono Downmix — Port Stream

## Overview

Rename `AudioExtractor.downmix_to_mono_wav` to `AudioExtractor.stereo_to_mono_wav` on the Protocol in `furnace/core/ports.py`, dropping the `channels` parameter, and update the matching signature-lock tests and `_MinimalAudioExtractor` stub in `tests/core/test_ports.py`. This is one of three parallel worktrees executing the design at `docs/superpowers/specs/2026-04-26-mono-via-eac3to-stereo-design.md`. The sister streams own the adapter and the executor; do not modify their files here.

The project test suite (`make check`) is expected to be RED in this isolated worktree because the adapter and executor still reference the old method name against the renamed Protocol. That cross-cutting failure is intentional and is not this stream's signal — integration verification runs on master after the three streams merge.

## Context

- Owned files (only files this worktree may modify): `furnace/core/ports.py`, `tests/core/test_ports.py`.
- Reference: design spec `docs/superpowers/specs/2026-04-26-mono-via-eac3to-stereo-design.md`.
- Architecture rule: core defines Protocol interfaces; adapters implement, services compose. This stream stays inside core.
- Do not run `make test`, `make check`, `make lint`, or `make typecheck` in this worktree. The orchestrator runs `make check` on master after all three streams merge.
- Every subagent dispatch uses model `opus`.

## Development Approach

- TDD at the file level: rewrite the test surface first, then update the Protocol declaration.
- File-level verification only. Use Read tool and grep to confirm changes; no test commands.
- Do not commit. The orchestrator commits and merges this branch.

## Testing Strategy

- Verify changes with Read tool: presence of new method name and signature, absence of old method name.
- Confirm grep `downmix_to_mono_wav` over `furnace/core/` and `tests/core/` returns zero hits at the end.
- Integration `make check` runs on master post-merge.

## Progress Tracking

- Mark `[x]` immediately when each item is done.

## Technical Details

### New Protocol method

On `AudioExtractor` in `furnace/core/ports.py`:

- Method name: `stereo_to_mono_wav`.
- Parameter order: `self`, `input_path: Path`, `stream_index: int`, `output_wav: Path`, `delay_ms: int`.
- Return type: `int`.
- Docstring summary: averages a stereo source to a mono WAV via ffmpeg's `pan=mono|c0=0.5*FL+0.5*FR` filter; `delay_ms` applies `adelay` (positive) or `atrim` (negative); no limiter (averaging cannot exceed unity for normalised PCM, since `|0.5L + 0.5R| ≤ max(|L|, |R|) ≤ 1.0`); multichannel collapse is the caller's responsibility (typically eac3to `-downStereo`); returns the ffmpeg exit code.
- The old method `downmix_to_mono_wav(input_path, stream_index, output_wav, channels, delay_ms)` is removed entirely, including the `channels` parameter.

### Test surface updates

In `tests/core/test_ports.py`:

- The `_MinimalAudioExtractor` stub class has a method `downmix_to_mono_wav` (around lines 123–157 in the current file). Replace that stub method with `stereo_to_mono_wav` matching the new signature, returning `0`, using `# noqa: ARG002` on every unused parameter (the existing stub uses that pattern).
- Replace `test_audio_extractor_has_downmix_to_mono_wav` with `test_audio_extractor_has_stereo_to_mono_wav`. It asserts `hasattr(AudioExtractor, "stereo_to_mono_wav")` and `callable(AudioExtractor.stereo_to_mono_wav)`.
- Replace `test_audio_extractor_downmix_to_mono_wav_signature` with `test_audio_extractor_stereo_to_mono_wav_signature`. It asserts the parameter list is exactly `["self", "input_path", "stream_index", "output_wav", "delay_ms"]` and the type hints map `input_path → Path`, `stream_index → int`, `output_wav → Path`, `delay_ms → int`, return → `int`.
- Update `test_minimal_audio_extractor_satisfies_runtime_checkable_protocol` to invoke the stub as `stub.stereo_to_mono_wav(Path("/dev/null"), 1, tmp_path / "out.wav", 0)` and assert the return value equals `0`. The `isinstance(stub, AudioExtractor)` assertion stays.
- Update `test_minimal_audio_extractor_method_surface` to call `stub.stereo_to_mono_wav(Path("/dev/null"), 0, tmp_path / "o.wav", -50)` and assert it returns `0`. The two other method-surface calls (`extract_track`, `ffmpeg_to_wav`) stay as-is.

## Implementation Steps

### Task 1: Rewrite the test surface in test_ports.py

- [x] in `tests/core/test_ports.py`, replace the `_MinimalAudioExtractor.downmix_to_mono_wav` stub method with the `stereo_to_mono_wav` stub described in Technical Details
- [x] replace `test_audio_extractor_has_downmix_to_mono_wav` with `test_audio_extractor_has_stereo_to_mono_wav` per the spec above
- [x] replace `test_audio_extractor_downmix_to_mono_wav_signature` with `test_audio_extractor_stereo_to_mono_wav_signature` per the spec above (parameter order and type hint asserts)
- [x] update `test_minimal_audio_extractor_satisfies_runtime_checkable_protocol` to call the stub via the new method name and new signature, asserting return value `0`
- [x] update `test_minimal_audio_extractor_method_surface` to call the stub's new method with the new signature and a non-zero `delay_ms` argument
- [x] verify via Read that `downmix_to_mono_wav` no longer appears anywhere in `tests/core/test_ports.py`
- [x] write tests for new functionality (signature-lock + stub-surface tests above ARE the tests for the renamed Protocol method)
- [x] run project tests - SKIP this step in this worktree (the orchestrator runs the integration check on master after merge)

### Task 2: Update the AudioExtractor Protocol declaration

- [x] in `furnace/core/ports.py`, locate the `downmix_to_mono_wav` method declaration on `AudioExtractor`
- [x] replace the method declaration with `stereo_to_mono_wav(self, input_path: Path, stream_index: int, output_wav: Path, delay_ms: int) -> int`
- [x] write the docstring described in Technical Details
- [x] verify via Read that `downmix_to_mono_wav` no longer appears anywhere in `furnace/core/ports.py`
- [x] write tests for new functionality (signature-lock + stub-surface tests landed in Task 1 cover the renamed Protocol method)
- [x] run project tests - SKIP this step in this worktree

### Task 3: Verify acceptance criteria

- [x] confirm `furnace/core/ports.py` defines `AudioExtractor.stereo_to_mono_wav` with the exact parameters and return type specified
- [x] confirm `tests/core/test_ports.py` references the new method name in the stub, both signature-lock tests, and both surface tests
- [x] confirm `grep -rn "downmix_to_mono_wav" furnace/core/ tests/core/` returns zero hits
- [x] confirm no file outside `furnace/core/ports.py` and `tests/core/test_ports.py` was modified by this worktree (compare against the worktree's base ref)

## Post-Completion

*Informational, no checkboxes*

- The orchestrator merges this branch to master after the sister streams complete.
- Integration `make check` runs on master post-merge.
