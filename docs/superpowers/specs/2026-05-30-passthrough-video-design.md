# Passthrough video (`--copy-video`) — Design

## Summary

A new `plan --copy-video` flag (alias `-cv`) that runs the entire pipeline as
usual but copies the video stream of "simple" progressive files byte-for-byte
instead of re-encoding it. Audio, subtitles, chapters, attachments and
container-level color/HDR metadata are handled exactly as today.

The flag is global to the `plan` run, but applied **best-effort per file**:
streams that cannot be safely passed through fall back to the normal encode
pipeline. A single run can therefore produce a mixed plan (some passthrough
jobs, some encode jobs).

## Motivation

Re-encoding is the slow, lossy bottleneck. For sources that are already in an
acceptable codec/quality, the user wants Furnace's full muxing/normalisation
machinery (audio AAC, subtitle recode, chapter mojibake fix, container HDR
flags, mkclean) without touching the video. Copying the elementary stream
verbatim is near-instant and lossless.

## Fallback rules

When `--copy-video` is set, each file is classified:

| Condition (on the source video) | Action |
|---|---|
| Progressive, not DV P7 FEL, not HDR10+ | **passthrough** (copy verbatim) |
| Interlaced (`video.interlaced`) | fall back to normal encode |
| Dolby Vision Profile 7 FEL (`dv_profile == 7`) | fall back to normal encode (P7→P8.1 as today) |
| HDR10+ | **rejected** with `ValueError`, as today (no change) |

Rationale:
- **Interlaced / crop** cannot be applied on a verbatim copy (they require
  re-encoding), so such files are re-encoded. Interlacing is known from
  `video.interlaced` (ffprobe), so no cropdetect is needed to decide.
- **DV P7 FEL** is converted to P8.1 today for player/Plex compatibility; a raw
  P7 copy would regress that, so P7 falls back to the normal pipeline.
- **HDR10+** is already unsupported and continues to raise — passthrough does
  not change this.

Fallback reasons are surfaced explicitly in the `plan` report/TUI per file.

## Architecture (chosen: Variant A)

The video step in `executor._run_pipeline` gains a branch. The decision
("passthrough vs encode") lives entirely in the planner; the executor only
forks the video step. Everything downstream of the video step is unchanged.

### Why the intermediate format is fine

NVEncC currently emits a **raw HEVC elementary stream** (the temp file is named
`video.mkv` but contains a raw ES — see `adapters/nvencc.py:105`). The muxer
(`adapters/mkvmerge.py:144`) simply consumes "track 0" of `video_path` and
attaches container-level color/HDR flags to it. mkvmerge reads that track
identically whether `video_path` is a raw ES or a real MKV.

Passthrough cannot reliably reproduce a raw HEVC ES because the source codec
varies (HEVC / AVC / MPEG2 / VC-1) and raw-ES extraction needs fragile
bitstream filters. So passthrough writes a **real MKV** wrapping the copied
source video stream (`ffmpeg -map 0:v:0 -c:v copy video.mkv`). This:

- works for any source codec,
- preserves timing, SAR and DV RPU (carried inside the stream/container; `-c
  copy` keeps SEI NAL units),
- is muxed by mkvmerge exactly like the encoder's raw ES — the difference in
  the intermediate's wrapping does not leak past mkvmerge.

The final container, the container-level color/HDR duplication, the ENCODER tag
and mkclean are therefore a single shared code path producing a consistent
output regardless of encode vs passthrough.

## Component changes

### 1. CLI (`furnace/cli.py`)
- Add option `--copy-video` / `-cv` (bool, default `False`) to the `plan`
  command.
- Thread it into `PlannerService.create_plan(..., copy_video=<bool>)`.

### 2. Model (`furnace/core/models.py`)
- `VideoParams` gains `passthrough: bool = False`.
- When `passthrough` is `True`: `crop=None`, `deinterlace=False`, and
  `cq`/`gop` are inert. Color/HDR/SAR fields are still populated (needed for
  container-level flags).
- Plan format version stays `"2"` — the new field is optional with a default,
  so existing plans deserialize unchanged.

### 3. Planner (`furnace/services/planner.py`)
- `create_plan` / `_build_job` receive `copy_video: bool`.
- When `copy_video` is `True`, classify per the fallback table:
  - eligible → build `VideoParams` with `passthrough=True` and **skip
    cropdetect** entirely.
  - interlaced or DV P7 FEL → normal encode path (`passthrough=False`); record
    the fallback reason for the report.
  - HDR10+ → unchanged `ValueError` (already raised in `_build_video_params`).
- `_build_video_params` gains a `passthrough` path that returns inert
  crop/deinterlace while preserving color/HDR/SAR.

### 4. Executor (`furnace/services/executor.py`)
- New constructor dependency `video_copier: VideoCopier`.
- In the video step: if `job.video_params.passthrough`, call
  `video_copier.copy_video(main_source, video_output, on_progress)` instead of
  `encoder.encode(...)`.
- Skip step 3 (DV RPU extraction) when `passthrough` — not needed.
- No VMAF/SSIM for passthrough jobs.
- ENCODER tag for passthrough: value `"Furnace vX.Y.Z"`, settings string
  `"video stream copied (passthrough)"`.
- mux → tag → mkclean unchanged.

### 5. Port + adapter (`furnace/core/ports.py`, `furnace/adapters/ffmpeg.py`)
- New `VideoCopier` Protocol:
  ```python
  def copy_video(
      self,
      input_path: Path,
      output_path: Path,
      on_progress: Callable[[ProgressSample], None] | None = None,
  ) -> int: ...
  ```
- `FFmpegAdapter` implements it as
  `ffmpeg -hide_banner -loglevel fatal -i <in> -map 0:v:0 -c:v copy -progress
  pipe:1 -y <out>`, reusing the existing `-progress pipe:1` parsing used by
  `extract_track`.
- `FFmpegAdapter` is wired as the `video_copier` argument in `cli.py`'s
  `_run_executor`.

### 6. Reporting (`furnace/services/planner.py`, UI)
- `_format_plan_summary` (or its caller) renders:
  - passthrough → `"passthrough (copy video)"`,
  - interlaced fallback → `"encode (interlaced)"`,
  - DV P7 fallback → `"encode (DV P7 FEL)"`.
- The fallback reason is derivable from existing `VideoParams` fields plus the
  `copy_video` flag in planner scope (no extra report-only model state needed,
  or a minimal explicit field if cleaner during implementation).

## Testing (TDD, 100% lines + branches on touched code)

- **core**: `VideoParams.passthrough` default; plan JSON round-trip with and
  without `passthrough`.
- **services/planner**:
  - `copy_video=True` + progressive non-DV → `passthrough=True`, cropdetect not
    called.
  - interlaced → `passthrough=False`, encode path, fallback reason recorded.
  - DV P7 FEL → `passthrough=False`, P7→P8.1 path preserved.
  - HDR10+ → still raises `ValueError`.
  - `copy_video=False` → behaviour unchanged (regression guard).
- **services/executor**:
  - passthrough job calls `video_copier.copy_video`, never `encoder.encode`.
  - DV RPU extraction skipped for passthrough.
  - mux/tag/mkclean invoked as usual; ENCODER tag settings string correct.
- **adapters/ffmpeg**: `copy_video` builds the expected command; progress
  parsing path covered.
- **integration**: mixed plan (passthrough + fallback-encode jobs) serializes
  and reloads.

## Known limitations

- Passthrough preserves the source SAR as-is. The DVD 4:3 SAR override only
  applies to encoded jobs (square-pixel DVDs that would need it are typically
  interlaced and thus already fall back to encode).
- The temp video file is named `video.mkv`; for passthrough it is a genuine
  MKV (for encode it remains a raw HEVC ES, as today).
- The flag is global per `plan` run; there is no per-file opt-in/opt-out beyond
  the automatic fallback rules.
