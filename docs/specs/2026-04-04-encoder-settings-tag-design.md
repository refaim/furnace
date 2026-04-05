# ENCODER_SETTINGS MKV Tag

## Summary

Write video encoder settings into the output MKV file as an `ENCODER_SETTINGS` global tag, similar to how x264/x265 embed settings in SEI userdata. Since hevc_nvenc doesn't support SEI userdata, we use an MKV tag that MediaInfo can display.

## Format

Slash-separated string. NVENC parameters always present, filters only when applied:

```
hevc_nvenc / main10 / cq=25 / preset=p5 / tune=uhq / rc=vbr / spatial-aq=1 / temporal-aq=1 / rc-lookahead=32 / multipass=qres / deinterlace=yadif_cuda / crop=0:132:0:132 / align=1920x1072
```

Without filters:

```
hevc_nvenc / main10 / cq=25 / preset=p5 / tune=uhq / rc=vbr / spatial-aq=1 / temporal-aq=1 / rc-lookahead=32 / multipass=qres
```

### Parameters (always present)

| Key | Source |
|-----|--------|
| `hevc_nvenc` | codec name |
| `main10` | profile |
| `cq=<N>` | interpolated CQ value |
| `preset=p5` | NVENC preset |
| `tune=uhq` | NVENC tune |
| `rc=vbr` | rate control mode |
| `spatial-aq=1` | spatial AQ flag |
| `temporal-aq=1` | temporal AQ flag |
| `rc-lookahead=32` | lookahead frames |
| `multipass=qres` | multipass mode |

### Filters (only when applied)

| Key | When |
|-----|------|
| `deinterlace=yadif_cuda` or `deinterlace=yadif` | source is interlaced |
| `crop=T:B:L:R` | crop was applied |
| `align=WxH` | dimensions were adjusted for CU alignment |

## Implementation

### Data flow

1. **`FfmpegEncoder.encode()`** builds the settings string after constructing the ffmpeg command, returns it alongside the encode result.
2. **`Executor`** receives the string, passes it to `Tagger`.
3. **`Tagger.set_encoder_tag()`** writes both `ENCODER` and `ENCODER_SETTINGS` as global MKV tags in a single mkvpropedit call.

### What doesn't change

- No new files, classes, or interfaces.
- No changes to the JSON plan format — the string is a runtime artifact, not a plan parameter.
- The existing `ENCODER` tag remains unchanged.
