# Furnace Specification

## 1. Overview

Furnace is a batch video transcoder for home archival. Two-phase workflow:
**plan** (interactive TUI for track selection) then **run** (unattended encoding).

### CLI

```
furnace plan <source> -o <output> --audio-lang <langs> --sub-lang <langs>
             [--names <map.json>] [--dry-run] [--vmaf] [--config <path>]

furnace run <plan.json> [--config <path>]
```

| Flag | Short | Description |
|------|-------|-------------|
| `--audio-lang` | `-al` | Comma-separated audio languages (e.g. `jpn` or `rus,eng`) |
| `--sub-lang` | `-sl` | Comma-separated subtitle languages |
| `--names` | | JSON rename map file `{"old.mkv": "New Name"}` |
| `--dry-run` | | Show plan without saving |
| `--vmaf` | | Enable VMAF quality metric |
| `--config` | | Path to `furnace.toml` |

Config search order: explicit `--config` path, `./furnace.toml`, project root `furnace.toml`, `%APPDATA%\furnace\furnace.toml`.

---

## 2. Input and Output

### Source

- Single video file or directory (recursive walk).
- Video extensions: `.mkv`, `.avi`, `.mp4`, `.m4v`, `.mov`, `.wmv`, `.flv`, `.ts`, `.mpg`, `.mpeg`.
- Satellite files (same directory, filename starts with video stem):
  - Audio: `.ac3`, `.dts`, `.eac3`, `.flac`, `.m4a`, `.mp3`, `.wav`
  - Subtitle: `.srt`, `.ass`, `.ssa`, `.sup`

### Output

- Always MKV.
- Mirrors source directory structure under output root.
- Filename sanitization for Windows: `<>/:"|?*` removed, `"` replaced with `'`, trailing `.` stripped.

### Skip Logic

A file is skipped when any condition is true:
1. Output file already exists at the target path.
2. Source file's ENCODER tag starts with `"Furnace"`.

### Rename Map

`--names` accepts a JSON file mapping `"original.mkv"` to `"New Stem"` (no extension). The new stem is sanitized and `.mkv` appended.

---

## 3. Disc Demuxing

### Detection

Recursive search for `VIDEO_TS/` (DVD) and `BDMV/` (Blu-ray) directories under source.

### Staging

All demuxed files go to `<source>/.furnace_demux/`. Per-title done markers (`<label>_title_<N>.done`) enable resumption. Partial MKV files (no done marker) are deleted and re-demuxed.

### Blu-ray (eac3to)

1. `eac3to <BDMV>` lists playlists.
2. TUI playlist selector (auto-select if only one).
3. `eac3to <BDMV> N) -demux` extracts raw streams to a title subdirectory.
4. `mkvmerge` assembles separate track files into a single MKV (language tags extracted from `[xxx]` in filenames, chapters included).

### DVD (MakeMKV)

1. `makemkvcon --noscan info file:<VIDEO_TS>` lists titles.
2. `makemkvcon --noscan mkv file:<VIDEO_TS> <index> <output_dir>` demuxes to MKV directly.
3. TUI file selector with SAR override option (for anamorphic DVDs, sets SAR to 64:45).
4. mpv preview with optional aspect override.

### Cleanup

`.furnace_demux/` is deleted after all jobs complete successfully.

---

## 4. Video Encoding

### Encoder: NVEncC (rigaya)

NVEncC is the sole video encoder. Output is written to a temporary MKV (video-only) -- mkvmerge handles final muxing with all tracks.

**Why NVEncC over ffmpeg hevc_nvenc:**
- Built-in VMAF and SSIM measurement in one pass (no separate analysis step).
- Native Dolby Vision RPU injection (`--dolby-vision-rpu`).
- Better NVIDIA hardware utilization and encoding performance.

### Parameters

| Parameter | Value |
|-----------|-------|
| Codec | HEVC (`-c hevc`) |
| Profile | main10 |
| Output depth | 10-bit |
| Tier | high |
| Preset | P5 |
| Tune | uhq |
| Rate control | QVBR (`--qvbr <CQ>`) |
| AQ | `--aq --aq-temporal` |
| Lookahead | 32 frames, level 3 |
| Multipass | `2pass-quarter` |
| GOP | `ceil(fps) * 5` (5-second keyframe interval) |
| Flags | `--strict-gop --repeat-headers` |

### Hardware Decode

NVDEC hardware decode (`--avhw`) for supported codecs:
`h264`, `hevc`, `mpeg2video`, `mpeg4`, `vp8`, `vp9`, `vc1`, `av1`, `mpeg1video`.

Falls back to software decode (`--avsw`) when:
- Source codec not in the NVDEC set.
- Left crop offset > 0 (NVEncC limitation).

### ENCODER_SETTINGS Tag

Format: slash-separated values. Example:
```
hevc_nvenc / NVEncC=8.12 / main10 / qvbr=24 / preset=P5 / tune=uhq / aq / aq-temporal / lookahead=32 / lookahead-level=3 / multipass=2pass-quarter
```

Optional suffixes when filters applied: `deinterlace=nnedi(...)`, `crop=T:B:L:R`, `sar=WxH`, `dolby-vision=8.1`.

---

## 5. Quality (CQ)

CQ is determined by linear interpolation over pixel area anchors:

| Resolution | Pixel Area | CQ |
|------------|------------|-----|
| SD 854x480 | 409,920 | 22 |
| 720p 1280x720 | 921,600 | 24 |
| 1080p 1920x1080 | 2,073,600 | 25 |
| 1440p 2560x1440 | 3,686,400 | 28 |
| 4K 3840x2160 | 8,294,400 | 31 |

Below SD: clamp to 22. Above 4K: clamp to 31. If crop is applied, the cropped pixel area (`crop.w * crop.h`) is used instead of the source pixel area.

---

## 6. Crop Detection

### Algorithm

1. Run `cropdetect=24:16:0` at multiple sample points across the timeline.
2. Each sample: seek to position, analyze 2 seconds, take last `crop=W:H:X:Y` line.
3. Cluster analysis: group values within tolerance=16 on all 4 coordinates, take the largest cluster.
4. Accept only if cluster contains >50% of samples.
5. Take per-coordinate median of the cluster.
6. Skip if crop equals full frame (no black bars).
7. Align to mod-8 via `align_dimensions()` (HEVC CU alignment).

### Sample Points

Standard (10 points): `0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90`

DVD (15 points): `0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 0.75, 0.85, 0.90`

DVD resolution: 720x480 (NTSC) or 720x576 (PAL).

### Pre-filter

Interlaced content: `yadif` pre-filter before cropdetect (`yadif,cropdetect=24:16:0`).

**Why yadif for cropdetect:** Quality is irrelevant for crop analysis; yadif is fast and removes combing artifacts that would confuse cropdetect. The high-quality nnedi filter is reserved for the actual encode.

**Why 15 DVD samples:** Interlaced DVD content has higher variance in crop values due to analog capture artifacts; more samples improve cluster reliability.

**Why tolerance=16:** Accounts for analog jitter in DVD sources while staying within one HEVC CU boundary (16 pixels).

---

## 7. Deinterlace

### Detection Pipeline

1. Read `field_order` from ffprobe.
2. If `field_order` not in `{tt, bb}`: progressive, no deinterlace.
3. If `field_order` in `{tt, bb}` and `fps >= 48`: always deinterlace (TV interlace).
4. If `field_order` in `{tt, bb}` and `fps < 48`: run idet analysis to confirm.

### idet Analysis

- Sample 1000 frames at 5 points: 10%, 30%, 50%, 70%, 90% of duration.
- Parse "Multi frame detection" line for TFF + BFF (interlaced) vs Progressive counts.
- Ratio = interlaced / (interlaced + progressive).
- Threshold: ratio > 5% triggers deinterlace.

Uses `r_frame_rate` (field rate) from ffprobe for FPS, not `avg_frame_rate`.

### NVEncC Filter

```
--vpp-nnedi nns=64,nsize=32x6,quality=slow
```

**Why nnedi:** Neural network-based deinterlacer produces the highest quality output. The `slow` quality preset and large neural network (`nns=64`, `nsize=32x6`) maximize detail preservation. NVEncC runs nnedi on the GPU, so the performance cost is acceptable.

---

## 8. Color Metadata

### VideoSystem

Determined from frame height:

| System | Heights |
|--------|---------|
| PAL | 576, 288 |
| NTSC | 480, 486, 240 |
| HD | >= 720 |

Other SD heights raise `ValueError`.

### 4-Step Resolution Algorithm

Given `matrix_raw`, `transfer_raw`, `primaries_raw` from ffprobe, `system`, and `has_hdr`:

**Step 1 -- Determine family:**
- `matrix_raw` in `{bt2020nc, bt2020c}` -> `bt2020`
- `matrix_raw == bt709` -> `bt709`
- `matrix_raw` in `{bt470bg, smpte170m}` -> `bt601`
- `matrix_raw is None`: infer from HDR (bt2020), HD (bt709), or SD (bt601)
- Other value: `ValueError`

**Step 2 -- Resolve matrix:**
Use `matrix_raw` if present; otherwise: bt2020 -> `bt2020nc`, bt709 -> `bt709`, bt601+PAL -> `bt470bg`, bt601+NTSC -> `smpte170m`.

**Step 3 -- Resolve primaries:**
Use `primaries_raw` if present; otherwise: bt2020 -> `bt2020`, bt709 -> `bt709`, bt601+PAL -> `bt470bg`, bt601+NTSC -> `smpte170m`.

**Step 4 -- Resolve transfer:**
Use `transfer_raw` if present; otherwise: bt2020+HDR -> `smpte2084`, bt2020+SDR -> `bt709`, bt709 -> `bt709`, bt601 -> infer from resolved primaries via `_TRANSFER_FROM_PRIMARIES` map (`bt470bg->bt470bg`, `smpte170m->smpte170m`, `bt470m->bt470m`, `bt709->bt709`).

### Output

`ResolvedColor(matrix, transfer, primaries)` passed to NVEncC as `--colormatrix`, `--transfer`, `--colorprim`.

Color range is always `"tv"`, mapped to NVEncC `"limited"` (`--colorrange limited`).

---

## 9. HDR10

### Side Data Extraction

1. Check stream `side_data_list` from ffprobe.
2. If absent, fall back to first video frame side data (`ffprobe -show_frames -read_intervals "%+#1"`).

### Mastering Display Color Volume (MDCV)

Parsed from side data entry containing "Mastering display metadata". Fraction numerators extracted (e.g. `8500/50000` -> `8500`).

Format string:
```
G(<green_x>,<green_y>)B(<blue_x>,<blue_y>)R(<red_x>,<red_y>)WP(<white_point_x>,<white_point_y>)L(<max_luminance>,<min_luminance>)
```

NVEncC flag: `--master-display <MDCV string>`

### Content Light Level (CLL)

Format string: `MaxCLL=<max_content>,MaxFALL=<max_average>`

NVEncC flag: `--max-cll <MaxCLL>,<MaxFALL>`

### HDR10+ Guard

`HDR10+` or `SMPTE ST 2094` in side data type raises `ValueError` in analyzer. HDR10+ is not supported.

---

## 10. Dolby Vision

### Detection

- `codec_name` in `{dvhe, dvh1}` or "Dolby Vision configuration" in side data.
- `dv_profile` and `dv_bl_signal_compatibility_id` extracted from side data.

### DvBlCompatibility Enum

```python
NONE = 0    # no fallback (Profile 5)
HDR10 = 1   # HDR10 fallback
SDR = 2     # SDR fallback
HLG = 4     # HLG fallback
```

### DvMode Enum

```python
COPY = 0    # extract RPU as-is (dovi_tool extract-rpu, no -m flag)
TO_8_1 = 2  # convert P7 FEL -> P8.1 (dovi_tool -m 2 extract-rpu)
```

### Mode Selection

- Profile 7: `DvMode.TO_8_1` (convert to Profile 8.1).
- All other profiles: `DvMode.COPY`.

**Why Profile 7 -> Profile 8.1:** Profile 7 uses a separate FEL (Full Enhancement Layer) that NVEncC cannot encode. Converting to Profile 8.1 preserves the RPU metadata with an HDR10-compatible base layer, maintaining DV tone mapping on supported displays while allowing HDR10 fallback on others.

### Pipeline

1. `dovi_tool extract-rpu <input> -o RPU.bin` (with `-m 2` for TO_8_1 mode).
2. NVEncC encode with `--dolby-vision-rpu RPU.bin --dolby-vision-profile 8.1`.
3. If crop is applied: `--dolby-vision-rpu-prm crop=true` adjusts the RPU active area.

---

## 11. SAR (Anamorphic)

### Detection

SAR (sample aspect ratio) from ffprobe `sample_aspect_ratio` field (e.g. `64:45`). Default: `1:1`.

### Correction

When `sar_num != sar_den`, scale to square pixels:
- `sar_num > sar_den`: scale width up: `display_w = round(width * sar_num / sar_den)`
- `sar_num < sar_den`: scale height up: `display_h = round(height * sar_den / sar_num)`

### NVEncC Flags

```
--output-res <aligned_w>x<aligned_h> --vpp-resize spline64 --sar 1:1
```

Output dimensions are aligned to mod-8 via `align_dimensions()`.

### DVD SAR Override

TUI file selector allows user to set SAR to `64:45` for anamorphic DVD files (applied to `movie.video.sar_num` / `sar_den` before planning).

---

## 12. VMAF / SSIM

### Integration

Quality metrics are built into NVEncC (not a separate pass).

- `--vmaf`: optional, enabled by `--vmaf` CLI flag.
- `--ssim`: always enabled when VMAF is enabled.

### VMAF Model Selection

| Condition | Model |
|-----------|-------|
| Pixel area >= 3,686,400 (>= 1440p) | `vmaf_4k_v0.6.1` |
| Pixel area < 3,686,400 | `vmaf_v0.6.1` |

Pixel area uses cropped dimensions if crop is applied.

### Parameters

```
--vmaf model=<model>,threads=<N>,subsample=8
```

Threads: `max(1, cpu_count - 2)`.

### Results

VMAF and SSIM scores are parsed from NVEncC stderr/stdout, stored in `EncodeResult`, propagated to `Job.vmaf_score` and `Job.ssim_score`, and written to the plan JSON.

---

## 13. Audio

### Processing Pipeline

| Codec | AudioCodecId | Action | Pipeline |
|-------|-------------|--------|----------|
| AAC LC | `AAC_LC` | COPY | Extract from container (ffmpeg) |
| AAC HE | `AAC_HE` | COPY | Extract from container |
| AAC HE v2 | `AAC_HE_V2` | COPY | Extract from container |
| AC3 | `AC3` | DENORM | Extract -> eac3to `-removeDialnorm` |
| E-AC3 | `EAC3` | DENORM | Extract -> eac3to `-removeDialnorm` |
| DTS core | `DTS` | DENORM | Extract -> eac3to `-removeDialnorm` |
| DTS-ES | `DTS_ES` | DECODE_ENCODE | Extract -> eac3to WAV -> qaac64 AAC |
| DTS-HD HRA | `DTS_HRA` | DECODE_ENCODE | Extract -> eac3to WAV -> qaac64 AAC |
| DTS-HD MA | `DTS_MA` | DECODE_ENCODE | Extract -> eac3to WAV -> qaac64 AAC |
| TrueHD | `TRUEHD` | DECODE_ENCODE | Extract -> eac3to WAV -> qaac64 AAC |
| FLAC | `FLAC` | DECODE_ENCODE | Extract -> eac3to WAV -> qaac64 AAC |
| PCM S16LE | `PCM_S16LE` | DECODE_ENCODE | Extract -> eac3to WAV -> qaac64 AAC |
| PCM S24LE | `PCM_S24LE` | DECODE_ENCODE | Extract -> eac3to WAV -> qaac64 AAC |
| PCM S16BE | `PCM_S16BE` | DECODE_ENCODE | Extract -> eac3to WAV -> qaac64 AAC |
| MP2 | `MP2` | FFMPEG_ENCODE | ffmpeg WAV -> qaac64 AAC |
| MP3 | `MP3` | FFMPEG_ENCODE | ffmpeg WAV -> qaac64 AAC |
| Vorbis | `VORBIS` | FFMPEG_ENCODE | ffmpeg WAV -> qaac64 AAC |
| Opus | `OPUS` | FFMPEG_ENCODE | ffmpeg WAV -> qaac64 AAC |
| WMA v2 | `WMA_V2` | FFMPEG_ENCODE | ffmpeg WAV -> qaac64 AAC |
| WMA Pro | `WMA_PRO` | FFMPEG_ENCODE | ffmpeg WAV -> qaac64 AAC |
| AMR | `AMR` | FFMPEG_ENCODE | ffmpeg WAV -> qaac64 AAC |

Note: eac3to always uses `-removeDialnorm` (user's eac3to.ini has `-keepDialnorm`).

### qaac64 Parameters

```
qaac64 --tvbr 91 --quality 2 --rate keep --no-delay --threading <input> -o <output>
```

### Delay Handling

Audio delay detected from ffprobe `start_pts` (MKV: already in ms) or `start_time * 1000`. Passed to eac3to as `+Nms` or `-Nms` for DENORM/DECODE_ENCODE actions, or to mkvmerge `--sync` for COPY action.

### Track Selection

1. Filter by `--audio-lang` (keep matching languages + `und`).
2. If exactly one track per language: auto-select.
3. If multiple tracks per language: TUI track selector.
4. Sort by language filter order; first track gets `is_default=true`.

### Und Language Resolution

- No `und` tracks: no action.
- Single language in filter: auto-assign to all `und` tracks.
- Multiple languages: TUI language selector per `und` track.

---

## 14. Subtitles

### Processing Pipeline

| Codec | SubtitleCodecId | Action | Pipeline |
|-------|----------------|--------|----------|
| SubRip (SRT) | `SRT` (`subrip`) | COPY_RECODE | Extract + recode to UTF-8 |
| ASS/SSA | `ASS` (`ass`) | COPY_RECODE | Extract + recode to UTF-8 |
| PGS | `PGS` (`hdmv_pgs_subtitle`) | COPY | Extract as-is |
| VOBSUB | `VOBSUB` (`dvd_subtitle`) | COPY | Extract as-is (wrapped in MKV) |

### Charset Detection

Text subtitles (SRT, ASS) are analyzed with `charset_normalizer` to detect source encoding. Recode step converts to UTF-8; if already UTF-8, file is copied as-is.

### Forced Subtitle Detection (3-Stage)

**Stage 1 -- Filename keywords (satellite files):**
- Match: `forced`, `форсир`, `только надписи`, `forsed`, `tolko nadpisi`
- Exclude: `normal`

**Stage 2 -- Track name keywords:**
- Match: `forced`, `caption`
- Exclude: `sdh`

**Stage 3 -- Statistical analysis:**
1. Exclude tracks with language `chi` and tracks with `sdh` in title.
2. Split remaining into binary (PGS, VOBSUB) and text (SRT, ASS) groups.
3. Per language within each group: find max metric (binary: `num_frames`, text: `num_captions`).
4. Track < 50% of max for its language -> mark as forced.
5. Check both `num_frames` and `num_captions` when available; either below 50% triggers forced.

Full-subtitle detection: tracks with `sdh` in title are tagged via `FULL_TRACKNAME_KEYWORDS`.

### Track Selection

1. Filter by `--sub-lang` (keep matching languages + `und`, discard forced).
2. Auto-select if one track per language; TUI if multiple.
3. Sort by language filter order; first track gets `is_default=true`.

### Attachments

Font attachments (`filename` + `mime_type`) from source MKV are carried through to output.

---

## 15. Container and Metadata

### Assembly (mkvmerge)

Execution order: audio processing -> subtitle processing -> DV RPU extraction -> video encode -> mux -> tag -> mkclean.

mkvmerge flags:
- `--no-track-tags --no-global-tags --disable-track-statistics-tags` (strip existing metadata)
- `--title ""` (clean title)
- `--normalize-language-ietf canonical` (normalize language codes: `fre->fra`, `chi->zho`)
- Video: `--track-name 0: --language 0:und --no-chapters`
- Audio: `--language 0:<lang> --default-track-flag 0:<yes|no>`, `--sync 0:<delay>` if delay != 0
- Subtitles: `--language 0:<lang> --default-track-flag --forced-display-flag --sub-charset 0:<encoding>`
- `--track-order` explicit: video first, then audio, then subtitles

### Color/HDR at Container Level

mkvmerge duplicates color/HDR metadata at the container level (for Plex/Jellyfin/TV compatibility):
- `--color-range`, `--color-primaries`, `--color-transfer-characteristics` (mapped to MKV numeric IDs)
- `--max-content-light`, `--max-frame-light` (HDR10 CLL values)

### mkclean

`mkclean <input.mkv> <output.mkv>` optimizes MKV index/seek structure. If mkclean fails, the unoptimized muxed output is used.

### Tags (mkvpropedit)

Sets global tags via XML:
- `ENCODER`: `"Furnace v<version>"`
- `ENCODER_SETTINGS`: slash-separated NVEncC parameters (see Section 4)

### Chapters

Extracted from source via ffprobe, written as OGM-format `.txt` file, passed to mkvmerge `--chapters`. Mojibake in chapter titles is auto-detected and fixed.

### Atomic Write

Plan JSON uses atomic write (write-to-temp-then-rename via `os.replace`) for crash safety. Plan is updated after each job completes.

---

## 16. Architecture

Hexagonal (Ports & Adapters). Dependency direction:

```
UI --> Services --> Core <-- Adapters
```

Core MUST NOT import from services, adapters, or UI.

### Module Tree

```
furnace/
  __init__.py              # VERSION constant
  __main__.py              # Entry point
  cli.py                   # Typer CLI: plan and run commands
  config.py                # furnace.toml loader (ToolPaths dataclass)
  plan.py                  # JSON plan serialization/deserialization, atomic write

  core/
    models.py              # All dataclasses and enums (Job, Plan, Movie, Track, VideoParams, etc.)
    ports.py               # Protocol interfaces (Prober, Encoder, Muxer, Tagger, Cleaner, etc.)
    detect.py              # Detection logic (crop clustering, interlace, color, HDR, forced subs, skip)
    quality.py             # CQ interpolation, GOP calc, SAR correction, dimension alignment
    rules.py               # Audio/subtitle codec-to-action mapping tables
    chapters.py            # Chapter mojibake detection and OGM chapter writing

  services/
    scanner.py             # Recursive file discovery, satellite matching, output path building
    analyzer.py            # ffprobe parsing, VideoInfo/Track/Attachment construction, idet dispatch
    planner.py             # Job building, track selection, CQ/crop/color/DV, und resolution
    disc_demuxer.py        # Disc detection, demux orchestration, mkvmerge assembly
    executor.py            # Job execution pipeline (audio -> subs -> DV RPU -> encode -> mux -> tag -> clean)

  adapters/
    _subprocess.py         # Shared subprocess runner with logging and output callback
    ffmpeg.py              # FFmpegAdapter: Prober + AudioExtractor (ffprobe, cropdetect, idet, HDR side data)
    nvencc.py              # NVEncCAdapter: Encoder (HEVC encoding, VMAF/SSIM)
    eac3to.py              # Eac3toAdapter: AudioDecoder + DiscDemuxerPort for Blu-ray
    makemkv.py             # MakemkvAdapter: DiscDemuxerPort for DVD
    qaac.py                # QaacAdapter: AacEncoder (WAV -> AAC)
    mkvmerge.py            # MkvmergeAdapter: Muxer (MKV assembly with color/HDR flags)
    mkvpropedit.py         # MkvpropeditAdapter: Tagger (ENCODER/ENCODER_SETTINGS via XML)
    mkclean.py             # MkcleanAdapter: Cleaner (MKV index optimization)
    dovi_tool.py           # DoviToolAdapter: DoviProcessor (RPU extraction/conversion)
    mpv.py                 # MpvAdapter: Previewer (audio/subtitle/file preview)

  ui/
    tui.py                 # Textual screens: TrackSelector, PlaylistSelector, FileSelector, LanguageSelector
    run_tui.py             # RunApp: Textual TUI for encoding progress display
    progress.py            # ReportPrinter: Rich summary after all jobs complete
```

### Port Protocols -> Adapter Mapping

| Protocol | Adapter |
|----------|---------|
| `Prober` | `FFmpegAdapter` |
| `Encoder` | `NVEncCAdapter` |
| `AudioExtractor` | `FFmpegAdapter` |
| `AudioDecoder` | `Eac3toAdapter` |
| `AacEncoder` | `QaacAdapter` |
| `Muxer` | `MkvmergeAdapter` |
| `Tagger` | `MkvpropeditAdapter` |
| `Cleaner` | `MkcleanAdapter` |
| `DoviProcessor` | `DoviToolAdapter` |
| `Previewer` | `MpvAdapter` |
| `DiscDemuxerPort` | `Eac3toAdapter` (BD), `MakemkvAdapter` (DVD) |

### Pipeline Flow

```
plan command:
  Scanner.scan() -> ScanResult[]
  Analyzer.analyze(ScanResult) -> Movie | None
  PlannerService.create_plan(Movie[]) -> Plan
  save_plan(Plan) -> furnace-plan.json

run command:
  load_plan() -> Plan
  Executor.run(Plan):
    for each Job:
      process audio tracks (extract/denorm/decode/encode)
      process subtitle tracks (extract/recode)
      extract DV RPU (if needed)
      encode video (NVEncC)
      mux (mkvmerge)
      set tags (mkvpropedit)
      optimize (mkclean)
      move to final output
      update_job_status() in plan JSON
```

---

## 17. Dependencies

### Python (from pyproject.toml)

| Package | Version | Purpose |
|---------|---------|---------|
| typer | >= 0.15 | CLI framework |
| rich | >= 13.0 | Terminal formatting, progress report |
| textual | >= 3.0 | TUI framework (track selection, encoding progress) |
| charset-normalizer | >= 3.0 | Subtitle encoding detection |
| psutil | >= 6.0 | Process tree management (graceful shutdown) |

Requires Python >= 3.13.

### External Tools

All paths configured via `furnace.toml`. Never hardcoded or resolved via PATH.

| Tool | Purpose | Link |
|------|---------|------|
| NVEncC | HEVC video encoding with NVDEC decode | https://github.com/rigaya/NVEnc |
| ffmpeg | Audio extraction, exotic codec decode to WAV | https://ffmpeg.org |
| ffprobe | Media probing (metadata, cropdetect, idet, HDR) | https://ffmpeg.org |
| eac3to | Audio denormalization, lossless decode, BD demux | https://forum.doom9.org/showthread.php?t=125966 |
| qaac64 | AAC encoding (TVBR 91, Apple encoder) | https://github.com/nu774/qaac |
| mkvmerge | MKV muxing with color/HDR metadata | https://mkvtoolnix.download |
| mkvpropedit | MKV tag editing (ENCODER, ENCODER_SETTINGS) | https://mkvtoolnix.download |
| mkclean | MKV index optimization | https://www.matroska.org/downloads/mkclean.html |
| MakeMKV (makemkvcon) | DVD demuxing | https://www.makemkv.com |
| dovi_tool | Dolby Vision RPU extraction/conversion (optional) | https://github.com/quietvoid/dovi_tool |
| mpv | Track preview during planning | https://mpv.io |

---

## 18. Out of Scope

Furnace does NOT:

- **Encode with software codecs** (x264, x265, SVT-AV1). NVEncC is the sole encoder.
- **Support HDR10+.** Raises `ValueError` at analysis time.
- **Transcode video to non-HEVC codecs.** Always outputs HEVC main10.
- **Stream or serve media.** Batch processing only.
- **Encode on non-NVIDIA GPUs.** NVEncC requires NVIDIA hardware.
- **Handle multi-angle Blu-ray discs.** Single playlist selection per disc.
- **Perform subtitle OCR.** PGS and VOBSUB are copied as binary.
- **Auto-select disc playlists by duration.** User selects via TUI.
- **Manage a media library or database.** Operates on files directly.
- **Support lossless audio passthrough.** Lossless codecs (DTS-HD MA, TrueHD, FLAC, PCM) are decoded to WAV and re-encoded as AAC.
- **Produce output in any container other than MKV.**
