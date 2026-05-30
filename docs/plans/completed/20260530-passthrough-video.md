# Passthrough video (`--copy-video`)

## Overview

Add a new `plan --copy-video` flag (alias `-cv`) that runs the entire Furnace
pipeline as usual but copies the video stream of "simple" progressive files
byte-for-byte instead of re-encoding it. Audio, subtitles, chapters,
attachments and container-level color/HDR metadata are handled exactly as
today. Re-encoding is the slow, lossy bottleneck; for sources already in an
acceptable codec/quality the user wants Furnace's full muxing/normalisation
machinery without touching the video.

The flag is global to the `plan` run but applied best-effort per file: streams
that cannot be safely passed through fall back to the normal encode pipeline,
so a single run can produce a mixed plan (some passthrough jobs, some encode
jobs).

## Context

- Impacted components: `furnace/cli.py` (new flag), `furnace/core/models.py`
  (`VideoParams.passthrough`), `furnace/services/planner.py` (classification +
  fallback + report summary), `furnace/services/executor.py` (video step
  branch), `furnace/core/ports.py` + `furnace/adapters/ffmpeg.py` (new
  `VideoCopier` port).
- Architecture is hexagonal (UI -> Services -> Core <- Adapters). Core stays
  pure; adapters implement `core/ports.py` Protocols; services receive adapters
  via constructor injection.
- The video-vs-passthrough decision lives entirely in the planner; the executor
  only forks the video step. Everything downstream (mux -> tag -> mkclean) is
  unchanged.
- NVEncC emits a raw HEVC elementary stream named `video.mkv`
  (`adapters/nvencc.py:105`); the muxer consumes "track 0" of `video_path`
  (`adapters/mkvmerge.py:144`) identically whether that input is a raw ES or a
  real MKV. Passthrough writes a real MKV via `ffmpeg -map 0:v:0 -c:v copy`.
- Adopted from spec
  `docs/superpowers/specs/2026-05-30-passthrough-video-design.md`.

## Development Approach

- Testing approach: TDD (mandated by project CLAUDE.md) — write the failing
  test before the implementation for every feature, including small ones.
- Subagents (if dispatched): Opus only; no git worktrees; work in the main
  checkout.
- Complete each task fully before moving to the next.
- Version bump (`furnace/__init__.py` + `pyproject.toml`) belongs to the final
  commit, not intermediate steps; do not commit until explicitly asked.
- Update this plan when scope changes during implementation.

## Testing Strategy

- Unit tests required for every code-changing Task (`tests/core`,
  `tests/services`, `tests/` integration as appropriate).
- 100% line AND branch coverage on all new or touched code.
- Linters and tests run ONLY via the Makefile - use `make check` (ruff + mypy
  strict + pytest with 100% coverage). Never invoke `uv run ruff/mypy/pytest`
  directly.
- Run `make check` after each Task before proceeding.

## Technical Details

Fallback classification when `--copy-video` is set (per source video):

| Condition | Action |
|---|---|
| Progressive, not DV P7 FEL, not HDR10+ | passthrough (copy verbatim) |
| Interlaced (`video.interlaced`) | fall back to normal encode |
| Dolby Vision Profile 7 FEL (`dv_profile == 7`) | fall back to normal encode (P7->P8.1 as today) |
| HDR10+ | rejected with `ValueError`, as today (no change) |

- `VideoParams` gains `passthrough: bool = False`. When `True`: `crop=None`,
  `deinterlace=False`, `cq`/`gop` inert; color/HDR/SAR fields still populated
  for container flags. Plan format version stays `"2"` (optional field with a
  default, old plans deserialize unchanged).
- New `VideoCopier` Protocol:
  `copy_video(input_path, output_path, on_progress=None) -> int`, implemented by
  `FFmpegAdapter` as
  `ffmpeg -hide_banner -loglevel fatal -i <in> -map 0:v:0 -c:v copy -progress pipe:1 -y <out>`,
  reusing the existing `-progress pipe:1` parsing.
- Executor: when `passthrough`, call `video_copier.copy_video(...)` instead of
  `encoder.encode(...)`, skip DV RPU extraction, skip VMAF/SSIM. ENCODER tag
  value `"Furnace vX.Y.Z"`, settings string `"video stream copied
  (passthrough)"`. mux/tag/mkclean unchanged.
- Reporting: `_format_plan_summary` (or caller) renders `"passthrough (copy
  video)"`, `"encode (interlaced)"`, or `"encode (DV P7 FEL)"` per file.
- Known limitation: passthrough preserves the source SAR as-is; the DVD 4:3 SAR
  override only applies to encoded jobs.

## Implementation Steps

### Task 1: `VideoParams.passthrough` model field + plan serialization

- [x] Add `passthrough: bool = False` to `VideoParams` in
      `furnace/core/models.py`
- [x] Ensure plan JSON round-trip preserves `passthrough` (save/load in
      `furnace/plan.py`), with old plans lacking the field defaulting to `False`
- [x] write tests: `VideoParams` default value; plan round-trip with
      `passthrough=True`, `passthrough=False`, and a legacy plan missing the
      field
- [x] run `make check` - must pass before next task

### Task 2: `VideoCopier` port + `FFmpegAdapter.copy_video`

- [x] Add `VideoCopier` Protocol to `furnace/core/ports.py` with
      `copy_video(input_path, output_path, on_progress=None) -> int`
- [x] Implement `copy_video` in `furnace/adapters/ffmpeg.py` using
      `ffmpeg -map 0:v:0 -c:v copy -progress pipe:1`, reusing the existing
      progress-line parsing
- [x] write tests: command construction is correct; progress callback path is
      exercised; return code propagates
- [x] run `make check` - must pass before next task

### Task 3: Planner classification and fallback

- [x] Thread `copy_video: bool` through `PlannerService.create_plan` and
      `_build_job`
- [x] In `_build_video_params` (or `_build_job`), classify per the fallback
      table: eligible -> `passthrough=True` and skip cropdetect; interlaced or
      DV P7 FEL -> normal encode path and record the fallback reason; HDR10+ ->
      unchanged `ValueError`
- [x] Ensure `passthrough=True` produces inert crop/deinterlace while keeping
      color/HDR/SAR populated
- [x] write tests: progressive non-DV -> passthrough and cropdetect not called;
      interlaced -> encode + reason; DV P7 FEL -> encode + P7->P8.1 preserved;
      HDR10+ -> raises; `copy_video=False` -> behaviour unchanged
- [x] run `make check` - must pass before next task

### Task 4: Executor passthrough branch

- [x] Add `video_copier: VideoCopier` constructor dependency to `Executor`
- [x] In the video step, branch on `job.video_params.passthrough`: call
      `video_copier.copy_video(...)` instead of `encoder.encode(...)`
- [x] Skip DV RPU extraction and VMAF/SSIM for passthrough jobs; set ENCODER tag
      settings string to `"video stream copied (passthrough)"`
- [x] Keep mux/tag/mkclean path unchanged
- [x] write tests: passthrough job calls `copy_video` and never `encode`; DV RPU
      skipped; mux/tag/mkclean invoked; tag settings string correct; encode path
      still works when `passthrough=False`
- [x] run `make check` - must pass before next task

### Task 5: CLI flag wiring

- [x] Add `--copy-video` / `-cv` (bool, default `False`) to the `plan` command
      in `furnace/cli.py`
- [x] Pass it into `PlannerService.create_plan(..., copy_video=...)` and wire
      `FFmpegAdapter` as the executor's `video_copier`
- [x] write tests: flag parsed and forwarded to the planner; executor wiring
      provides a `video_copier`
- [x] run `make check` - must pass before next task

### Task 6: Plan report summary for passthrough / fallback

- [x] Update `_format_plan_summary` (or its caller in `planner.py`) to render
      `"passthrough (copy video)"`, `"encode (interlaced)"`, and
      `"encode (DV P7 FEL)"` per file
- [x] Ensure the reason is visible in the `plan` TUI/report path
- [x] write tests: summary string for each of passthrough, interlaced fallback,
      DV P7 fallback, and the unchanged normal-encode case
- [x] run `make check` - must pass before next task

### Task 7: Verify acceptance criteria

- [x] Add/confirm an integration test for a mixed plan (passthrough +
      fallback-encode jobs) that serializes and reloads
- [x] Verify all requirements from Overview are implemented (flag copies video
      verbatim for eligible files; interlaced/DV P7 fall back; HDR10+ rejected;
      fallback shown in report)
- [x] run full project test suite via `make check` (ruff + mypy strict + pytest,
      100% line + branch coverage)
- [x] confirm `make check` reports zero lint/type/test failures

## Post-Completion

*Items requiring manual intervention - no checkboxes, informational only*

- Bump version in `furnace/__init__.py` and `pyproject.toml` (MINOR — new CLI
  option / new feature) as part of the final commit.
- Do not commit until the user explicitly approves; the spec doc commit is also
  pending user approval.
- After implementation, dispatch a separate code-reviewer agent (Opus, no
  self-review) and loop to zero comments before closing the task.
