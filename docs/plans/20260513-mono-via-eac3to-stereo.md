# Mono Downmix via eac3to Stereo Intermediate

## Overview

Replace the hand-rolled ITU-R BS.775 multichannel-to-mono downmix in `FFmpegAdapter` with a two-step chain: eac3to performs the multichannel-to-stereo collapse using its bitstream-aware mix matrix (honouring `acmod`, `cmixlev`, `surmixlev`, `dmixmod`, `dnorm`), then ffmpeg averages `(L+R)/2` to produce the final mono WAV. Stereo sources flagged as fake-mono bypass eac3to and run straight through ffmpeg with delay applied there. The current `pan`-based 5.1/7.1 matrix plus `alimiter` is deleted entirely — it is mathematically defined but not author-intended, and the limiter only exists to mask the matrix inflating fake-surround levels past unity.

The work is suitable for a team of agents working in parallel. The new port contract (`AudioExtractor.stereo_to_mono_wav` with parameters `input_path`, `stream_index`, `output_wav`, `delay_ms`) is the central agreement. Once the contract is locked, the three modified production files (`core/ports.py`, `adapters/ffmpeg.py`, `services/executor.py`) and the three modified test files (`tests/core/test_ports.py`, `tests/test_ffmpeg_mono_downmix.py`, `tests/services/test_executor_downmix.py`) touch disjoint paths and can be modified concurrently. Verification is the only step that must run after all parallel work is complete.

## Context

- Source plan: `docs/superpowers/plans/2026-04-26-mono-via-eac3to-stereo.md`
- Source design spec: `docs/superpowers/specs/2026-04-26-mono-via-eac3to-stereo-design.md`
- Affected port: `AudioExtractor` in `furnace/core/ports.py` — method `downmix_to_mono_wav(channels, ...)` becomes `stereo_to_mono_wav(...)` with `channels` removed.
- Affected adapter: `furnace/adapters/ffmpeg.py` — the multichannel matrix branches and `alimiter` are deleted; only the stereo-averaging filter and the delay logic remain.
- Affected service: `furnace/services/executor.py`, `_process_audio_track`, the `DownmixMode.MONO` branch (currently around lines 510–540).
- Untouched on purpose: `furnace/adapters/eac3to.py` (`decode_lossless(downmix=STEREO)` already works as needed), the planner, the classifier, plan serialization, CLI surface, and their tests (`tests/services/test_planner_downmix.py`, `tests/core/test_audio_profile.py`, `tests/test_plan.py`, `tests/test_cli.py`, `tests/adapters/test_eac3to_downmix.py`).
- Architecture rule: hexagonal split must remain — core defines the Protocol, the adapter implements it, the service composes them.
- Version target: `1.15.0` (MINOR — user-facing audio bytes change, on-disk temp pattern changes).
- No intermediate commits during execution. The user commits at the end.

## Development Approach

- TDD strict and unconditional, per `CLAUDE.md`: every code-changing task starts with the failing test, then the production change. No exceptions for "too small".
- Parallel-by-default: tasks marked as parallel batches dispatch independent agents in the same step, one per file pair. Agents within a batch must not modify each other's files.
- 100 % line **and** branch coverage on every new or touched code path. Uncovered branches block completion.
- All test, lint, and type runs go exclusively through the Makefile (`make test`, `make lint`, `make typecheck`, `make check`). Never invoke `uv run pytest`, `uv run ruff`, or `uv run mypy` directly.
- No git worktrees — work in the main checkout.
- Every spawned subagent uses model `opus`.
- Review loop to zero: after the implementation phase, dispatch a separate code-reviewer agent, address every comment, redispatch review, and only stop when the reviewer returns zero comments.

## Testing Strategy

- Failing test before any production change — Task 1 writes the full new test surface for ports, adapter, and executor. Suite is RED after Task 1 for the *expected* reason only (rename mismatch).
- Task 2 implements the new port contract, the new adapter method, and the rewritten executor MONO branch. Suite must turn GREEN after Task 2.
- Coverage gates run inside `make check`. The new code paths to cover are: the `channels is None` guard, the `channels == 2` direct-stereo branch, the multichannel branch with eac3to-friendly codec, the multichannel branch with eac3to-incompatible codec, every individual rc-non-zero failure (`extract_track`, `ffmpeg_to_wav`, `decode_lossless`, multichannel `stereo_to_mono_wav`, stereo-direct `stereo_to_mono_wav`, `encode_aac`), the three `delay_ms` regimes (zero, positive, negative) in `stereo_to_mono_wav`, and both branches of the `self._log_dir` guard.
- After production code is green, run a separate code-reviewer subagent. Re-run until the reviewer returns zero comments.

## Progress Tracking

- Mark completed items with `[x]` immediately when the underlying change is done — do not batch updates.
- If scope changes during implementation, update this plan in place before proceeding.
- The plan does not include commit steps. The user commits explicitly when the whole plan is verified green.

## Technical Details

### New port contract

`AudioExtractor.stereo_to_mono_wav(input_path: Path, stream_index: int, output_wav: Path, delay_ms: int) -> int`. Returns the ffmpeg exit code. The old method `downmix_to_mono_wav(input_path, stream_index, output_wav, channels, delay_ms)` is removed entirely, including the `channels` parameter.

### New adapter behaviour

The adapter method builds an ffmpeg command line whose `-af` value is the filter `pan=mono|c0=0.5*FL+0.5*FR`. When `delay_ms > 0`, the filter chain appends `adelay=<delay_ms>`. When `delay_ms < 0`, the chain appends `atrim=start=<abs(delay_ms)/1000.0>` formatted to three decimal places. When `delay_ms == 0`, no delay filter is appended. There is no `aformat=channel_layouts=...` and no `alimiter=limit=0.99` — averaging stereo PCM cannot exceed unity (`|0.5L + 0.5R| ≤ max(|L|, |R|) ≤ 1.0`), so a limiter is mathematically unnecessary. Output is forced to 1 audio channel (`-ac 1`), wav muxer with `-rf64 auto`. The log path is `<log_dir>/ffmpeg_mono_s<stream_index>.log` when `set_log_dir` was called, otherwise `None`.

### New executor flow

In `_process_audio_track`, the `DownmixMode.MONO` branch of `AudioAction.DECODE_ENCODE` is rewritten as three paths:

1. **`channels is None` guard.** Raise `RuntimeError` with message `"MONO downmix without channel count for stream <track_idx>"`. No subprocess work runs.
2. **Stereo source (`channels == 2`).** Call `stereo_to_mono_wav(input_path=source_path, stream_index=track_idx, output_wav=<temp>/audio_<idx>_mono.wav, delay_ms=instr.delay_ms)`. Then `encode_aac` to `<temp>/audio_<idx>.m4a`. Neither `extract_track`, `ffmpeg_to_wav`, nor `decode_lossless` is invoked.
3. **Multichannel source (`channels` in 6 or 8, or any non-2 value).** If `_codec_supported_by_eac3to(instr.codec_name)` is true: call `extract_track` to `<temp>/audio_<idx>_raw.<ext>`. Otherwise: call `ffmpeg_to_wav` to `<temp>/audio_<idx>_pre.wav`. Then call `decode_lossless` with `downmix=DownmixMode.STEREO` to produce `<temp>/audio_<idx>_stereo.wav`, passing `instr.delay_ms` so eac3to handles the delay during multichannel collapse. Then call `stereo_to_mono_wav` with `stream_index=0`, `delay_ms=0` (delay already applied) on the stereo intermediate, output `<temp>/audio_<idx>_mono.wav`. Finally `encode_aac` to `<temp>/audio_<idx>.m4a`.

Every subprocess step receives its own `_make_progress_callback`. Each non-zero rc raises a `RuntimeError` with a step-specific prefix: `"Audio extract (MONO multichannel) failed with rc=<rc> for stream <idx>"`, `"ffmpeg pre-decode (MONO multichannel) failed with rc=<rc> for stream <idx>"`, `"eac3to -downStereo failed: rc=<rc>"`, `"stereo_to_mono_wav failed: rc=<rc>"`, `"encode_aac failed: rc=<rc>"`. There is **no** fallback to the old ITU matrix — failures surface to the user.

### Temp-file shapes

Multichannel eac3to-friendly: `audio_<idx>_raw.<ext>` → `audio_<idx>_stereo.wav` → `audio_<idx>_mono.wav` → `audio_<idx>.m4a`. Multichannel eac3to-incompatible (AAC 5.1, Opus 7.1, vorbis): `audio_<idx>_pre.wav` → `audio_<idx>_stereo.wav` → `audio_<idx>_mono.wav` → `audio_<idx>.m4a`. Stereo direct: `audio_<idx>_mono.wav` → `audio_<idx>.m4a`. Peak temp-disk for a 7.1 / 48 kHz / 24-bit / 2-hour worst case is roughly the `pre.wav` plus the stereo and mono intermediates concurrent in `temp_dir` before the per-job cleanup. The `~3.5 GiB` peak overhead vs the current single-pass ffmpeg path is accepted explicitly.

### Test surface

`tests/test_ffmpeg_mono_downmix.py` covers: presence of the stereo-averaging `pan` formula, absence of `alimiter`, absence of `aformat=`, no delay filter when `delay_ms == 0`, `adelay=<ms>` when positive, `atrim=start=<seconds>` when negative, propagation of `run_tool`'s exit code as the return value, and that `set_log_dir` causes the log path to flow into `run_tool`. All tests patch `furnace.adapters.ffmpeg.run_tool` so no real ffmpeg is invoked.

`tests/services/test_executor_downmix.py` covers, inside a single test class for the MONO branch: 5.1 DTS chain (extract → eac3to → stereo_to_mono_wav → qaac), 7.1 TrueHD chain (same), AAC 5.1 chain (ffmpeg_to_wav → eac3to → stereo_to_mono_wav → qaac), stereo direct (only stereo_to_mono_wav → qaac, neither extract nor ffmpeg_to_wav nor decode_lossless called), delay routing on multichannel (eac3to sees `delay_ms`, stereo_to_mono_wav sees `0`), delay routing on stereo direct (stereo_to_mono_wav sees the original `delay_ms`), the `channels is None` guard, each of the rc-non-zero failure points, the regression that `DECODE_ENCODE` without `downmix=MONO` does not call `stereo_to_mono_wav`. The `_instr` helper at the top of the file gains a `delay_ms` parameter (default 0). Other test classes in the file (`TestDecodeEncodeDownmixRouting`, `TestDecodeEncodeDownmixProgressWiring`) are not touched.

`tests/core/test_ports.py` covers: `AudioExtractor` has attribute `stereo_to_mono_wav`, the signature is `(self, input_path: Path, stream_index: int, output_wav: Path, delay_ms: int) -> int` exactly, the `_MinimalAudioExtractor` stub satisfies the runtime-checkable Protocol, and the method-surface exercise call uses the new signature. The old `test_audio_extractor_has_downmix_to_mono_wav` and `test_audio_extractor_downmix_to_mono_wav_signature` are replaced (not added to).

## Implementation Steps

### Task 1: Lock the new test surface in parallel across all three test files

This task drives the system into the RED state required by TDD: tests describe the new API and the suite fails for the *expected* reason (rename mismatch). Dispatch three subagents in one batch; each owns one test file and must not modify the other two. No production code is touched in this task.

- [ ] dispatch subagent A to fully rewrite `tests/test_ffmpeg_mono_downmix.py` so it locks the new `stereo_to_mono_wav` adapter contract: stereo-averaging filter presence, no `alimiter`, no `aformat=`, the three delay regimes, exit-code propagation, and `set_log_dir` path flow; `run_tool` must be patched in every test
- [ ] dispatch subagent B to extend the `_instr` helper in `tests/services/test_executor_downmix.py` with a `delay_ms` parameter and then replace the existing `TestDecodeEncodeMonoDownmix` class with the full new flow-coverage suite described in Technical Details (multichannel via extract, multichannel via ffmpeg_to_wav, stereo direct, delay routing, every failure mode, the `channels is None` guard, the `DECODE_ENCODE` non-MONO regression); leave the other two test classes in the file untouched
- [ ] dispatch subagent C to replace `_MinimalAudioExtractor.downmix_to_mono_wav` and the four signature-lock tests in `tests/core/test_ports.py` (`test_audio_extractor_has_downmix_to_mono_wav`, `test_audio_extractor_downmix_to_mono_wav_signature`, `test_minimal_audio_extractor_satisfies_runtime_checkable_protocol`, `test_minimal_audio_extractor_method_surface`) with the new `stereo_to_mono_wav` versions
- [ ] verify no production file under `furnace/` has been modified in this task
- [ ] run `make test` — expected failures are limited to the three rewritten test files, all tracing back to the rename; any failure outside those files must be investigated before proceeding
- [ ] write tests for new functionality
- [ ] run project tests - must pass before next task (suite is expected RED; "pass" here means the RED is the documented expected RED)

### Task 2: Implement the new contract in parallel across port, adapter, and executor

With the tests in place, dispatch three subagents in one batch to produce the GREEN state. Each agent owns one production file. Files are disjoint, so concurrent edits do not conflict. The Protocol, the adapter, and the executor must agree on the new signature; the test surface from Task 1 enforces that agreement.

- [ ] dispatch subagent A to update `AudioExtractor` in `furnace/core/ports.py`: replace the `downmix_to_mono_wav` declaration with `stereo_to_mono_wav(input_path, stream_index, output_wav, delay_ms) -> int`, update the docstring to describe the stereo-averaging contract and that multichannel collapse is the caller's responsibility
- [ ] dispatch subagent B to replace the `downmix_to_mono_wav` method in `furnace/adapters/ffmpeg.py` with `stereo_to_mono_wav`: drop the `channels` parameter, delete the 5.1 and 7.1 matrix branches, delete the `aformat=channel_layouts=...` filter, delete the `alimiter=limit=0.99` filter, keep only the `pan=mono|c0=0.5*FL+0.5*FR` filter plus the `adelay`/`atrim` delay handling, keep the existing `set_log_dir` log-path logic, keep `-ac 1` and `-rf64 auto`, keep `run_tool` invocation and exit-code propagation
- [ ] dispatch subagent C to rewrite the `DownmixMode.MONO` branch inside `_process_audio_track` in `furnace/services/executor.py`: implement the `channels is None` guard, the `channels == 2` stereo-direct path (single `stereo_to_mono_wav` call with `delay_ms=instr.delay_ms` then `encode_aac`), the multichannel path branching on `_codec_supported_by_eac3to(instr.codec_name)` between `extract_track` and `ffmpeg_to_wav`, the eac3to step via `decode_lossless(downmix=STEREO)` carrying the delay, the final `stereo_to_mono_wav` with `stream_index=0` and `delay_ms=0`, then `encode_aac`; each subprocess step gets its own `_make_progress_callback`; every non-zero rc raises `RuntimeError` with the step-specific prefix specified in Technical Details; preserve the surrounding indentation of the enclosing `AudioAction.DECODE_ENCODE` block
- [ ] verify no references to `downmix_to_mono_wav` remain anywhere in `furnace/` or `tests/`
- [ ] run `make test` — expected GREEN; if any failure remains, the most likely causes are indentation drift in the executor block, a leftover reference to the old method name, or a coverage gap on a new branch
- [ ] write tests for new functionality (any uncovered branch reported by coverage requires a focused test added to the appropriate file from Task 1)
- [ ] run project tests - must pass before next task

### Task 3: Bump version to 1.15.0 across both manifest locations

The audio bytes produced by a MONO downmix change (different matrix on the multichannel collapse, no limiter) and the on-disk temp-file pattern changes. Per `CLAUDE.md` SemVer rules this is a MINOR bump. Both manifest files must move together. This task is fully independent of Tasks 1 and 2 and may dispatch concurrently with them — list it explicitly here so the team does not forget it.

- [ ] dispatch subagent to set `VERSION = "1.15.0"` in `furnace/__init__.py`
- [ ] dispatch subagent to set `version = "1.15.0"` under `[project]` in `pyproject.toml`
- [ ] confirm both files report `1.15.0` and no other version string elsewhere needs to move
- [ ] write tests for new functionality (no new tests required — version constants are exercised by the existing surface)
- [ ] run project tests - must pass before next task

### Task 4: Code review loop to zero comments

Per the agent rules in `CLAUDE.md`: dispatch a separate code-reviewer agent (never self-review) against the diff produced by Tasks 1–3. Address every comment. Redispatch review after fixes. Only zero-comment review closes the task.

- [ ] dispatch a separate code-reviewer subagent (model opus) with the full diff of the work so far and the design spec at `docs/superpowers/specs/2026-04-26-mono-via-eac3to-stereo-design.md` as reference
- [ ] address every reviewer comment in place; if a comment is technically questionable, verify the underlying behaviour before pushing back
- [ ] redispatch the code-reviewer subagent after each round of fixes
- [ ] repeat until the reviewer returns zero comments
- [ ] write tests for new functionality (any reviewer-driven behaviour change requires a matching test in the relevant test file)
- [ ] run project tests - must pass before next task

### Task 5: Verify acceptance criteria

- [ ] verify all requirements from Overview are implemented: ITU matrix and `alimiter` removed from `furnace/adapters/ffmpeg.py`; port renamed and `channels` parameter dropped on `AudioExtractor`; executor MONO branch uses the three-path structure (stereo direct, multichannel via extract, multichannel via ffmpeg_to_wav); eac3to handles multichannel delay and `stereo_to_mono_wav` is called with `delay_ms=0` on the multichannel path
- [ ] run full project test suite via `make test` — full GREEN
- [ ] run project linter via `make lint` — clean
- [ ] run `make typecheck` — strict mypy clean
- [ ] run `make check` — single source of truth; ruff clean, mypy strict clean, pytest with 100 % line + branch coverage on `furnace/` and `tests/`
- [ ] verify coverage hits every branch listed in Testing Strategy (the `channels is None` guard, the `channels == 2` branch, both multichannel codec branches, every rc-non-zero failure, the three `delay_ms` regimes, both `self._log_dir` branches); add focused tests for any gap and re-run until coverage is 100 % line and branch
- [ ] confirm `furnace/__init__.py` and `pyproject.toml` both report version `1.15.0`
- [ ] confirm `grep -rn "downmix_to_mono_wav" furnace/ tests/` returns zero hits

## Post-Completion

*Items requiring manual intervention - no checkboxes, informational only*

- The user commits the work explicitly when ready. No commit step is part of this plan.
- After commit, the changelog/version note for `1.15.0` should mention: bitstream-aware multichannel collapse via eac3to, removal of the hand-rolled ITU matrix and the `alimiter`, fake-mono stereo sources bypassing eac3to, and the temporary disk overhead trade-off (~3.5 GiB peak vs the previous single-pass path on a 7.1 / 48 kHz / 24-bit / 2-hour worst case).
- If a real-world track surfaces a failure mode the failure-mode tests do not cover (e.g., a specific DTS-X variant eac3to refuses), capture the source as a regression sample and add a dedicated test before changing the code.
