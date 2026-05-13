# Mono Downmix — Adapter Stream

## Overview

Replace the `downmix_to_mono_wav` method on `FFmpegAdapter` (`furnace/adapters/ffmpeg.py`) with `stereo_to_mono_wav`: drop the `channels` parameter, delete the 5.1 and 7.1 ITU matrix branches, delete the `aformat=channel_layouts=...` and `alimiter=limit=0.99` filters, keep only the stereo-averaging `pan` filter and the `adelay`/`atrim` delay handling. Rewrite the adapter tests in `tests/test_ffmpeg_mono_downmix.py` to lock the new command-line surface. This is one of three parallel worktrees executing the design at `docs/superpowers/specs/2026-04-26-mono-via-eac3to-stereo-design.md`. The sister streams own the port and the executor; do not modify their files here.

The project test suite (`make check`) is expected to be RED in this isolated worktree because the port still declares the old method name and the executor still calls it. That cross-cutting failure is intentional and is not this stream's signal — integration verification runs on master after the three streams merge.

## Context

- Owned files (only files this worktree may modify): `furnace/adapters/ffmpeg.py`, `tests/test_ffmpeg_mono_downmix.py`.
- Reference: design spec `docs/superpowers/specs/2026-04-26-mono-via-eac3to-stereo-design.md`.
- Architecture rule: adapters implement Protocol interfaces from `furnace/core/ports.py`. The port rename happens in a sister worktree; in this isolated worktree, the adapter will temporarily not satisfy the Protocol.
- Do not run `make test`, `make check`, `make lint`, or `make typecheck` in this worktree.
- Every subagent dispatch uses model `opus`.

## Development Approach

- TDD at the file level: rewrite the test surface first, then rewrite the adapter method.
- File-level verification only — Read tool and grep, no test commands.
- Do not commit. The orchestrator commits and merges this branch.

## Testing Strategy

- Verify command-line construction: the `-af` value must contain the `pan=mono|c0=0.5*FL+0.5*FR` filter; it must not contain `alimiter` or `aformat=`; `adelay=<ms>` appears when `delay_ms > 0`, `atrim=start=<seconds>` appears when `delay_ms < 0` (three decimals), neither appears when `delay_ms == 0`.
- Verify return value: the method returns whatever exit code `run_tool` returned.
- Verify log path: when `set_log_dir` is configured, `run_tool` receives `log_path=<log_dir>/ffmpeg_mono_s<stream_index>.log`; when not set, `log_path` is `None`.
- All tests patch `furnace.adapters.ffmpeg.run_tool` so no real ffmpeg is invoked.

## Progress Tracking

- Mark `[x]` immediately when each item is done.

## Technical Details

### New adapter method

In `furnace/adapters/ffmpeg.py`, replace the `downmix_to_mono_wav` method (currently around lines 586–642) with `stereo_to_mono_wav`:

- Signature: `stereo_to_mono_wav(self, input_path: Path, stream_index: int, output_wav: Path, delay_ms: int) -> int`. No `channels` parameter.
- Filter chain construction: start with the single base filter `pan=mono|c0=0.5*FL+0.5*FR`. If `delay_ms > 0`, append `adelay=<delay_ms>`. If `delay_ms < 0`, compute `seconds = abs(delay_ms) / 1000.0` and append `atrim=start=<seconds:.3f>` (three decimal places). If `delay_ms == 0`, append nothing.
- Filter chain join: comma-separated; passed as the `-af` argument.
- ffmpeg command: `<ffmpeg> -hide_banner -loglevel warning -i <input_path> -map 0:<stream_index> -af <af_value> -ac 1 -f wav -rf64 auto -y <output_wav>`.
- Log path: `<log_dir>/ffmpeg_mono_s<stream_index>.log` if `self._log_dir` is truthy, otherwise `None`. Pass via `log_path=` kwarg to `run_tool`.
- `run_tool` call: `run_tool(cmd, on_output=self._on_output, log_path=log_path)`. Returns `(rc, _out)`; return `rc`.
- Docstring summary: averages stereo to mono via the `pan` filter, optional `adelay`/`atrim` delay handling, no limiter (averaging cannot exceed unity for normalised PCM); multichannel collapse is the caller's responsibility (typically eac3to `-downStereo`).
- The old `downmix_to_mono_wav` method (with the 5.1/7.1 branches, multi-term `pan` matrix, `aformat=channel_layouts=...`, and `alimiter=limit=0.99`) is removed entirely.

### Test surface

In `tests/test_ffmpeg_mono_downmix.py`, replace the entire file with a focused suite that locks the new contract. All tests patch `furnace.adapters.ffmpeg.run_tool` (no real ffmpeg).

Required tests:

- `test_stereo_averages_fronts` — the `-af` value contains `pan=mono|c0=0.5*FL+0.5*FR`.
- `test_no_alimiter` — `-af` does not contain `alimiter`.
- `test_no_layout_normalizer` — `-af` does not contain `aformat=`.
- `test_zero_delay_has_no_delay_filter` — when `delay_ms=0`, `-af` contains neither `adelay` nor `atrim`.
- `test_positive_delay_appends_adelay` — when `delay_ms=50`, `-af` contains `adelay=50` and still contains the `pan` filter.
- `test_negative_delay_appends_atrim` — when `delay_ms=-50`, `-af` contains `atrim=start=0.050`, contains the `pan` filter, and does not contain `adelay`.
- `test_returns_run_tool_exit_code` — when `run_tool` returns `(42, "")`, the adapter method returns `42`.
- `test_log_path_uses_log_dir_when_set` — after `adapter.set_log_dir(tmp_path)`, the `log_path` kwarg passed to `run_tool` equals `tmp_path / "ffmpeg_mono_s<stream_index>.log"`.

Helper structure (use this design):

- An `adapter` fixture builds `FFmpegAdapter(ffmpeg_path=Path("ffmpeg"), ffprobe_path=Path("ffprobe"))`.
- A helper `_invoke(adapter, tmp_path, *, delay_ms=0)` patches `run_tool` (return `(0, "")`), calls `adapter.stereo_to_mono_wav(input_path=tmp_path/"a.mkv", stream_index=1, output_wav=tmp_path/"out.wav", delay_ms=delay_ms)`, and returns the `-af` value extracted from `run_tool.call_args`.
- A helper `_af_value(call_args)` extracts the `-af` argument value from the captured `run_tool` call positional `cmd` list.
- The 5.1 / 7.1 / `alimiter` / `channels`-parameter tests from the old file are deleted entirely; no `channels` parameter exists in the new signature.

## Implementation Steps

### Task 1: Rewrite the adapter test file

- [x] fully replace the contents of `tests/test_ffmpeg_mono_downmix.py` with the focused suite described in Technical Details (every test patches `furnace.adapters.ffmpeg.run_tool`, no real ffmpeg invocation)
- [x] include the eight tests listed above (`test_stereo_averages_fronts`, `test_no_alimiter`, `test_no_layout_normalizer`, `test_zero_delay_has_no_delay_filter`, `test_positive_delay_appends_adelay`, `test_negative_delay_appends_atrim`, `test_returns_run_tool_exit_code`, `test_log_path_uses_log_dir_when_set`)
- [x] include the `adapter` fixture and the `_invoke` / `_af_value` helpers
- [x] verify via Read that the new test names are present and no test references `channels=`, `alimiter`, or `aformat=` from the old surface (other than the new negative-assert tests)
- [x] write tests for new functionality
- [x] run project tests - SKIP this step in this worktree

### Task 2: Rewrite the adapter method in ffmpeg.py

- [x] in `furnace/adapters/ffmpeg.py`, locate the existing `downmix_to_mono_wav` method
- [x] replace the entire method with `stereo_to_mono_wav` per Technical Details: drop the `channels` parameter, build the filter chain as a single base `pan` filter plus optional `adelay`/`atrim`, drop `aformat=channel_layouts=...` and `alimiter=limit=0.99`, keep the `-ac 1 -f wav -rf64 auto` flags, keep the `set_log_dir`-based log path logic with file name `ffmpeg_mono_s<stream_index>.log`, keep `run_tool(cmd, on_output=self._on_output, log_path=log_path)` and return its `rc`
- [x] write the new docstring per Technical Details
- [x] verify via Read that `downmix_to_mono_wav`, `alimiter`, `aformat=channel_layouts`, and the multi-term `pan` matrices no longer appear anywhere in `furnace/adapters/ffmpeg.py` (only unrelated `aformat=channel_layouts={layout}` at line 525 in `_decode_pcm_window` audio-profiling method remains — outside the downmix scope)
- [x] write tests for new functionality (already in place from Task 1)
- [x] run project tests - SKIP this step in this worktree

### Task 3: Verify acceptance criteria

- [x] confirm `furnace/adapters/ffmpeg.py` defines `stereo_to_mono_wav` with the exact parameters and return type, the single `pan` filter as the base, and the conditional `adelay` / `atrim` appending
- [x] confirm `tests/test_ffmpeg_mono_downmix.py` contains every test listed in Technical Details
- [x] confirm `grep -n "downmix_to_mono_wav\|alimiter\|aformat=channel_layouts" furnace/adapters/ffmpeg.py tests/test_ffmpeg_mono_downmix.py` returns zero hits (the `test_no_alimiter` / `test_no_layout_normalizer` tests use those strings in negative assertions — that is the only acceptable occurrence; treat any match outside those negative-assert lines as a violation) — only acceptable hits remain: the negative-assert lines in `test_no_alimiter`, plus the unrelated `aformat=channel_layouts={layout}` at ffmpeg.py:525 in the `_decode_pcm_window` audio-profiling method (out of downmix scope, as noted in Task 2)
- [x] confirm no file outside `furnace/adapters/ffmpeg.py` and `tests/test_ffmpeg_mono_downmix.py` was modified by this worktree — `git diff --name-only master...HEAD` shows only `docs/plans/mono-eac3to-adapter.md` (the progress tracker), `furnace/adapters/ffmpeg.py`, and `tests/test_ffmpeg_mono_downmix.py`

## Post-Completion

*Informational, no checkboxes*

- The orchestrator merges this branch to master after the sister streams complete.
- Integration `make check` runs on master post-merge.
