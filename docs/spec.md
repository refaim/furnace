# Furnace — Project Specification

## 1. Overview

Furnace is a tool for transcoding and archiving a personal movie collection for home archival. It takes video files (or directories) as input, analyzes them, offers interactive track selection, and produces optimized MKV files with HEVC video, processed audio, and properly tagged subtitles.

Works in two phases:
1. **Planning** — interactive: the user walks through all files, selects tracks, confirms crop. Result — a JSON plan file.
2. **Execution** — unattended: the program reads the plan and encodes everything without questions.

This separation lets you answer all questions at once, start encoding, and walk away.

## 2. Input and Output

### 2.1 Sources

- Single video file
- Directory (recursive traversal)
- Satellite files: if `movie.rus.ac3`, `movie.eng.srt`, etc. sit next to `movie.mkv`, they are treated as additional tracks of the same movie

### 2.2 Output

- MKV container (always)
- Output directory structure mirrors the source structure
- Output filenames are cleaned of Windows-forbidden characters (`< > / : \ | ? *`, double quotes → single, trailing dots removed)
- Optional rename map (`--names map.txt`):
  ```
  film.2020.bdrip.x264-group.mkv = Movie Title (2020)
  ```
  The `.mkv` extension is added automatically.

### 2.3 Skip Logic

A file is skipped if:
- The output file already exists, OR
- The source has an MKV `ENCODER` tag starting with `Furnace/`

## 3. Command-Line Interface

```
furnace plan <source> -o <destination> [options]
furnace run <plan.json>
```

### 3.1 `furnace plan`

Scans the source, probes all files, shows a TUI for track selection and crop confirmation. Saves the result to a JSON plan file.

Options:
- `-o <path>` — output directory (required)
- `--audio-lang <code>[,<code>]` — audio language filter (e.g. `rus,eng`)
- `--sub-lang <code>[,<code>]` — subtitle language filter (e.g. `rus,eng`)
- `--names <path>` — rename map file
- `--dry-run` — show what would be planned without saving the plan file
- `--vmaf` — enable VMAF quality scoring in the plan

### 3.2 `furnace run`

Reads the plan file and executes all pending jobs. No interactive prompts. Can be interrupted with ESC (graceful shutdown of the current encode) and resumed later (rerun the same command).

## 4. Plan File (JSON)

Contains a list of jobs. Each job is one output file:

- Paths to source files (main video + satellite files)
- Path to output file
- Video encoding parameters (CQ, crop, deinterlace, color space, HDR metadata)
- Selected audio tracks with processing instructions (copy / denormalize / re-encode)
- Selected subtitle tracks with processing instructions (copy / recode to UTF-8)
- Attachments to copy (fonts)
- Chapter source
- Job status: `pending` | `done` | `error`
- Error message (if status is `error`)

After each job completes, the plan file is updated on disk — resume works after interruption.

## 5. Video

### 5.1 Codec

HEVC (H.265) via NVIDIA NVENC (`hevc_nvenc`). Always 10-bit (profile `main10`, pixel format `p010le`).

NVENC parameters:
- Preset: p5
- Tune: uhq
- Rate control: VBR with constant quality (CQ)
- Spatial AQ: enabled
- Temporal AQ: enabled
- RC Lookahead: 32 frames
- Multipass: qres
- Forced IDR: enabled
- GOP: 5-second keyframe interval (calculated from source FPS)

### 5.2 Quality (CQ)

Adaptive CQ based on source resolution (pixel area), interpolated:

| Resolution | CQ |
|---|---|
| SD (854x480) | 22 |
| 720p (1280x720) | 24 |
| 1080p (1920x1080) | 25 |
| 1440p (2560x1440) | 28 |
| 4K (3840x2160) | 31 |

In-between resolutions are linearly interpolated by pixel area.

### 5.3 Crop

Enabled by default. Uses ffmpeg `cropdetect` to detect black bars at 5 points across the timeline (10%, 30%, 50%, 70%, 90%). Values are auto-applied and aligned to a 16x8 grid (NVENC requirement).

### 5.4 Deinterlace

Auto-detected via ffprobe `field_order`. If interlaced (TFF or BFF), `bwdif_cuda` is applied automatically without prompting.

### 5.5 Color Space

Forced by resolution:
- HD (>= 720p): BT.709
- SD (< 720p): BT.601
- BT.2020 (HDR source): unchanged

### 5.6 Color Range

Always Limited (16-235).

### 5.7 HDR

If the source contains HDR10 metadata (mastering display color volume, content light level), it is passed through to the output via ffmpeg SEI side data.

### 5.8 Dolby Vision and HDR10+

If Dolby Vision or HDR10+ (dynamic metadata) is detected in the source, the file is skipped with a warning. Encoding is not performed. Static HDR10 and HLG are fully supported (see 5.7).

### 5.9 Bloat Guard

After encoding, if the output file is larger than the source — the output is deleted and the job is marked as error. Prevents pointless re-encoding of already efficiently compressed sources.

## 6. Audio

### 6.1 Track Selection

Interactive TUI (Textual) during the planning phase:

```
Audio tracks:                   [Space] toggle  [Enter] preview  [Up/Down] navigate

> [x]  rus  AC3 5.1    448 kbps   "Dubbed"
  [ ]  rus  AAC 2.0    128 kbps   "Voiceover"
  [x]  eng  TrueHD 7.1            "Original"

  [Done]
```

- Up/Down arrows — navigate
- Space — toggle track on/off
- Enter — preview the track in mpv (opens video with this audio)
- Enter on `[Done]` — confirm selection

With `--audio-lang` filter: if exactly one track per language — auto-selected. If multiple — the TUI is shown with only the matching languages.

The first selected track becomes the default track in MKV.

### 6.2 Processing Pipeline

| Source Codec | Action |
|---|---|
| AAC (any variant) | Copy as-is |
| AC3, EAC3, DTS (core) | eac3to denormalize → copy |
| DTS-HD MA, DTS-HRA, DTS-ES, TrueHD, FLAC, PCM | eac3to decode to WAV → qaac64 encode to AAC |
| Exotic (WMA, AMR, Vorbis, MP2, MP3, etc.) | ffmpeg decode to WAV → qaac64 encode to AAC |
| Unknown codec | Warning, file skipped |

qaac64 parameters: `--tvbr 91 --quality 2 --rate keep --no-delay --threading`

### 6.3 Audio Delay

Determined from ffprobe `start_pts`. If non-zero:
- For tracks going through eac3to: delay passed as `+Xms` / `-Xms` argument
- For tracks copied as-is: delay passed to mkvmerge via `--sync`

### 6.4 Codec Whitelist

Only known codecs are processed. If an audio track contains a codec not in the whitelist, the entire file is skipped with a warning. Batch processing continues with the next file.

## 7. Subtitles

### 7.1 Track Selection

Same TUI as audio. Enter opens subtitle preview in mpv overlaid on the video.

### 7.2 Processing

| Source Codec | Action |
|---|---|
| SRT | Copy, recode to UTF-8 if needed |
| ASS/SSA | Copy, recode to UTF-8 if needed |
| PGS | Copy as-is |
| VOBSUB | Copy as-is |
| Unknown codec | Warning, file skipped |

Text subtitle encoding detected via `charset-normalizer`. If the file is not valid UTF-8, it is recoded from the detected source encoding to UTF-8.

### 7.3 Forced Subtitle Detection

Three-stage algorithm:

1. **Filename keywords**: `forced`, `форсир`, `только надписи` (excluding `normal`) → mark as forced
2. **Track name keywords**: `forced`, `caption` → mark as forced. `sdh` → mark as full (not forced)
3. **Statistics**: for each language, if a subtitle track contains < 50% of the frame/caption count of the largest track in the same language → mark as forced

### 7.4 Attachments

All attachments from the source file (fonts, images) are copied to the output MKV. Required for correct rendering of styled ASS subtitles.

### 7.5 Codec Whitelist

Same as audio: unknown subtitle codecs cause the file to be skipped with a warning.

## 8. Container and Metadata

### 8.1 Assembly

Final assembly via `mkvmerge`. All processed tracks (video, audio, subtitles), chapters, and attachments are combined into a single MKV file.

### 8.2 mkclean

After assembly, `mkclean` is run to move the seek index to the beginning of the file. Required for smooth network playback (SMB) without seek delays.

### 8.3 Metadata

- `ENCODER` tag set to `Furnace/X.X.X` (tool version)
- All track names removed
- Video track language — `und` (undetermined)
- Audio and subtitle track languages preserved from source
- All other tags removed (`--no-track-tags --no-global-tags --disable-track-statistics-tags`)

### 8.4 Chapters

Copied from source when present.

### 8.5 Default Tracks

- First selected audio track marked as default
- First selected subtitle track marked as default (if forced — also marked as forced)

## 9. Batch Processing

### 9.1 Directory Scanning

Recursive traversal of the source directory. For each video file found:
1. Detect satellite files (same name, different extension)
2. Probe metadata
3. Check skip logic (already processed or output exists)
4. Show in TUI for track selection

### 9.2 Two-Phase Execution

- **Phase 1 (plan)**: all interactive decisions happen here. Result: JSON plan file.
- **Phase 2 (run)**: reads the plan, encodes sequentially. No prompts. Updates job status in JSON after each file.

### 9.3 Resumption

`furnace run plan.json` skips jobs with status `done`, retries jobs with status `error` or `pending`. Safe to restart after interruption.

### 9.4 VMAF

Optional, enabled with `--vmaf` flag during planning. If enabled, a VMAF score is calculated after each encode (libvmaf filter in ffmpeg) and saved in the plan file.

### 9.5 Summary Report

After all jobs complete, a summary is printed:
- Files processed / skipped / errored
- Total size before and after
- Total space savings
- Average VMAF score (if enabled)

### 9.6 Graceful Interruption

ESC key during encoding:
- Terminates the ffmpeg process tree (via psutil)
- Current job status remains `pending`
- Program exits
- Resume: `furnace run plan.json`

## 10. User Interface

### 10.1 Planning Phase (Textual TUI)

- Track selection widgets with keyboard navigation
- Crop confirmation dialog
- File information display (resolution, codecs, duration, size)

### 10.2 Encoding Phase (Textual TUI)

- Progress bar for each file with fps / speed / ETA
- Source and output file information
- Step tracker for the encoding pipeline
- Tool output log
- Overall batch progress (X of Y files)

### 10.3 mpv Integration

During planning, Enter on a track opens mpv for preview:
- Audio track: plays video with the selected audio
- Subtitle track: plays video with selected subtitles overlaid
- User closes mpv — returns to TUI

## 11. Architecture

### 11.1 Pattern: Ports & Adapters (Hexagonal Architecture)

The core (pure logic) has no dependencies on external tools, UI, or the file system. Communication happens through protocols (abstractions).

```
furnace/
├── __main__.py              # entry point
├── cli.py                   # typer CLI (plan, run)
│
├── core/                    # core — pure Python, no I/O
│   ├── models.py            # Movie, Track, Job, Step (dataclasses)
│   ├── rules.py             # codec whitelists, audio routing
│   ├── quality.py           # CQ interpolation, crop alignment
│   ├── detect.py            # forced subs, interlace, HDR/DV detection rules
│   └── ports.py             # protocols (Prober, Encoder, Muxer, etc.)
│
├── services/                # orchestration — connects core with adapters
│   ├── scanner.py           # directory traversal, satellite file grouping
│   ├── analyzer.py          # file probing, model population
│   ├── planner.py           # applies rules to models → list of Jobs
│   └── executor.py          # reads plan, executes Jobs sequentially
│
├── adapters/                # port implementations — tool wrappers
│   ├── ffmpeg.py            # encode, probe, vmaf, cropdetect
│   ├── eac3to.py            # denorm, decode lossless
│   ├── qaac.py              # encode AAC
│   ├── mkvmerge.py          # mux, metadata
│   ├── mkclean.py           # index optimization
│   └── mpv.py               # track preview
│
├── ui/                      # everything the user sees
│   ├── tui.py               # Textual — track selection, crop confirm
│   └── progress.py          # Rich — encoding progress
│
└── plan.py                  # JSON plan serialization/deserialization
```

### 11.2 Dependency Direction

```
UI ──→ Services ──→ Core ←── Adapters
                      ↑
                   ports.py (protocols)
```

- Core does not import from other layers
- Adapters implement protocols from Core (dependency inversion)
- Services receive adapters via constructor/arguments
- UI calls Services

## 12. Dependencies

| Dependency | Purpose |
|---|---|
| Python >= 3.13 | Runtime |
| uv | Package management |
| ffmpeg / ffprobe | Video encoding, metadata analysis, VMAF |
| mkvmerge / mkvpropedit | MKV assembly, tagging |
| mkclean | MKV index optimization |
| eac3to | Audio denormalization, lossless decoding |
| qaac64 | AAC encoding (Apple CoreAudio) |
| mpv | Track preview |
| MakeMKV | DVD demux |
| Rich | Progress display |
| Textual | TUI widgets |
| charset-normalizer | Subtitle encoding detection |
| psutil | Process tree management (graceful shutdown) |

## 13. Out of Scope

- Video scaling / downscaling
- Audio downmix
- Device profiles
- TV series renaming
- CPU encoding (libx264 / libx265)
- AV1 codec
- Deleting originals
- Forced DAR override
- Batch script generation
