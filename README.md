# Furnace

Batch video transcoder for home archival. Scans your movie collection, lets you pick tracks in a TUI, saves a JSON plan, then encodes everything with NVEncC (NVIDIA hardware encoder).

## Why Furnace

- **Plan, review, then run** — choose tracks and preview in a TUI, save a JSON plan you can inspect or edit before hours of encoding
- **Resumable** — failed jobs retry on next run, plan updated atomically after each job
- **Live progress on every long step** — ffmpeg extraction, eac3to, qaac, mkvmerge, NVEncC all stream into one unified progress bar; no more silent multi-gigabyte waits
- **Auto quality** — CQ value interpolated by pixel area, no manual tuning across SD/720p/1080p/4K
- **Disc demux** — Blu-ray (BDMV) and DVD (VIDEO_TS) fed straight into the pipeline with playlist/title selection
- **Anamorphic SAR fix** — detects and corrects wrong sample aspect ratio on DVD sources
- **Dolby Vision** — Profile 7 FEL (converted to P8.1) and Profile 8 MEL with RPU passthrough via dovi_tool
- **HDR10 passthrough** — mastering display, content light level, BT.2020/PQ preserved through encode
- **Auto deinterlace** — detects interlaced content from the video stream and applies nnedi (neural network) automatically
- **Smart crop** — black bars detected automatically across the timeline
- **mpv preview** — audition audio tracks, check subtitles, or preview video right from the TUI before committing
- **Per-track downmix** — fold 7.1 or 5.1 into stereo or 5.1 from the track selector, useful when the multichannel mix is a fake upmix or the movie is dialogue-heavy
- **Satellite files** — external audio and subtitle files next to the video are picked up as extra tracks automatically

## Workflow

```
furnace plan <source> -o <output> --audio-lang rus,eng --sub-lang rus,eng
# -> opens TUI for track selection -> saves furnace-plan.json

furnace run furnace-plan.json
# -> encodes all pending jobs with live progress TUI
```

![Run TUI](docs/screenshot.png)

## Requirements

- Python 3.13+
- [ffmpeg / ffprobe](https://ffmpeg.org)
- [MKVToolNix](https://mkvtoolnix.download) (mkvmerge, mkvpropedit)
- [eac3to](https://forum.doom9.org/showthread.php?t=125966)
- [qaac64](https://github.com/nu774/qaac)
- [mkclean](https://www.matroska.org/downloads/mkclean.html)
- [mpv](https://mpv.io) (track preview)
- [MakeMKV](https://www.makemkv.com) (DVD demux)
- [NVEncC](https://github.com/rigaya/NVEnc) (video encoder)
- [dovi_tool](https://github.com/quietvoid/dovi_tool) (Dolby Vision RPU, optional)

## Install

```bash
uv pip install .
```

## Configuration

Copy [`furnace.toml.example`](furnace.toml.example) to `furnace.toml` and set paths to your tools. Searched in order: `--config` flag, current directory, `%APPDATA%\furnace\`.

## Usage

Plan with dry run (no TUI, just print what would happen):
```bash
furnace plan D:\Movies -o E:\Encoded --audio-lang jpn --sub-lang eng --dry-run
```

Plan and encode:
```bash
furnace plan D:\Movies -o E:\Encoded --audio-lang rus,eng --sub-lang rus,eng
furnace run E:\Encoded\furnace-plan.json
```

Enable VMAF + SSIM quality scoring (single pass):
```bash
furnace plan D:\Movies -o E:\Encoded --audio-lang eng --sub-lang eng --vmaf
```
