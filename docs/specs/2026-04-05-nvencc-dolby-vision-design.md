# NVEncC Migration + Dolby Vision Support

## Scope

Replace ffmpeg hevc_nvenc with NVEncC as the video encoder. Add Dolby Vision
support (Profile 7 FEL, Profile 8 MEL). Upgrade deinterlace from bwdif to nnedi.
Move VMAF/SSIM from a separate ffmpeg pass into NVEncC's built-in metrics.

### In scope

- NVEncC as the sole video encoder (replaces ffmpeg hevc_nvenc)
- Dolby Vision Profile 7 FEL (convert to P8.1 via dovi_tool)
- Dolby Vision Profile 8 MEL (extract RPU via dovi_tool, pass to NVEncC)
- Deinterlace: bwdif -> nnedi (quality upgrade)
- VMAF/SSIM computed inside NVEncC encode pass (one pass instead of two)
- HDR10+ content -> hard error (not silent skip)

### Out of scope

- HDR10+ encoding support (NVEncC can do it; deferred to future)
- ffmpeg replacement for probe/cropdetect/idet/audio (stays on ffmpeg)
- mkvmerge changes (mkvmerge already handles DV since v81)

### New external dependencies

- `NVEncC64` -- video encoder (mandatory, configured in furnace.toml)
- `dovi_tool` -- DV RPU extraction/conversion (required when DV content present)

## Architecture

### Adapter split

Currently `FFmpegAdapter` implements three ports: `Prober`, `Encoder`,
`AudioExtractor`. After migration:

```
FFmpegAdapter   (stays)  -> Prober + AudioExtractor
NVEncCAdapter   (new)    -> Encoder
DoviToolAdapter (new)    -> DoviProcessor
```

### Port changes

**Encoder port -- updated:**

```python
@dataclass
class EncodeResult:
    return_code: int
    encoder_settings: str
    vmaf_score: float | None = None
    ssim_score: float | None = None

class Encoder(Protocol):
    def encode(
        self,
        input_path: Path,
        output_path: Path,
        video_params: VideoParams,
        source_size: int,
        on_progress: Callable[[float, str], None] | None = None,
        vmaf_enabled: bool = False,
    ) -> EncodeResult: ...
```

`compute_quality()` removed from Encoder port. VMAF/SSIM returned in
EncodeResult.

**New port -- DoviProcessor:**

```python
class DoviProcessor(Protocol):
    def extract_rpu(
        self, input_path: Path, output_rpu: Path, mode: DvMode,
    ) -> int:
        """Extract RPU from HEVC stream.
        mode=COPY: extract as-is.
        mode=TO_8_1: convert P7 FEL -> P8.1.
        Returns exit code.
        """
        ...
```

### Encoding pipelines

```
HDR10 / SDR (no DV):
  source -> NVEncC encode -> video.hevc -> mux

DV Profile 8 (MEL):
  source -> dovi_tool extract-rpu -> RPU.bin
         -> NVEncC encode (--dolby-vision-rpu RPU.bin --dolby-vision-profile 8.1)
         -> video.hevc -> mux

DV Profile 7 (FEL):
  source -> dovi_tool -m 2 extract-rpu -> RPU.bin
         -> NVEncC encode (--dolby-vision-rpu RPU.bin --dolby-vision-profile 8.1)
         -> video.hevc -> mux
```

All DV content goes through dovi_tool. `--dolby-vision-rpu copy` is not used.

## Models

### New enums

```python
class DvBlCompatibility(enum.IntEnum):
    NONE = 0    # no fallback (Profile 5)
    HDR10 = 1   # HDR10 fallback
    SDR = 2     # SDR fallback
    HLG = 4     # HLG fallback

class DvMode(enum.IntEnum):
    COPY = 0      # extract RPU as-is
    TO_8_1 = 2    # convert P7 FEL -> P8.1
```

`DvMode.TO_8_1` value (2) maps to dovi_tool `-m 2`. `DvMode.COPY` means no `-m`
flag (dovi_tool copies RPU untouched by default when `-m` is omitted).

### HdrMetadata -- extended

```python
@dataclass(frozen=True)
class HdrMetadata:
    mastering_display: str | None
    content_light: str | None
    is_dolby_vision: bool = False
    is_hdr10_plus: bool = False
    dv_profile: int | None = None                      # NEW: 5, 7, 8, etc.
    dv_bl_compatibility: DvBlCompatibility | None = None  # NEW
```

`dv_profile` and `dv_bl_compatibility` parsed from ffprobe side_data
"Dolby Vision configuration" fields.

### VideoParams -- DV field

```python
@dataclass
class VideoParams:
    # ... existing fields ...
    dv_mode: DvMode | None = None   # None=no DV, COPY=extract as-is, TO_8_1=P7->P8.1
```

## Detection (Analyzer)

Current behavior:
- DV -> silent skip
- HDR10+ -> silent skip

New behavior:
- DV -> detect profile, proceed with encoding
- HDR10+ -> `raise ValueError("HDR10+ not supported: {filename}")`

DV profile detection: parse `dv_profile` and `dv_bl_signal_compatibility_id`
from ffprobe "Dolby Vision configuration" side_data.

## NVEncCAdapter

### Flag mapping

All NVEncC flags MUST be researched against NVEncC documentation during
implementation. Do not blindly map from ffmpeg. Key areas:

- Encode settings: preset, tune, rate control (qvbr), AQ, lookahead, multipass
- Color metadata: colorprim, transfer, colormatrix, colorrange -- investigate
  `auto` mode vs explicit values, interaction with `determine_color_space()`
- HDR metadata: max-cll, master-display -- investigate `copy` vs explicit
- DV flags: dolby-vision-rpu, dolby-vision-profile
- Filters: crop (L,T,R,B format), nnedi deinterlace, SAR/resize
- Output: raw HEVC or MKV, progress format
- Quality metrics: --ssim, --vmaf flags and output parsing

### Crop format conversion

Furnace `CropRect(w, h, x, y)` -> NVEncC `--crop left,top,right,bottom`:
```
left   = x
top    = y
right  = source_width - x - w
bottom = source_height - y - h
```

### Deinterlace

`--vpp-nnedi` replaces bwdif. Quality upgrade.

### SAR correction

`--output-res {display_w}x{h} --sar 1:1` replaces `-vf scale=W:H,setsar=1:1`.

### CU alignment

Verify whether NVEncC auto-aligns to mod 8 with `--crop`. If not, adjust crop
values via `align_dimensions()` as currently done.

### Progress parsing

NVEncC outputs progress to stderr in human-readable format (frame count, fps,
bitrate, ETA). Parser needs to extract frame number and compute percentage from
total frame count.

### encoder_settings string

Updated format:
```
hevc_nvenc / NVEncC={version} / main10 / qvbr={cq} / preset=P5 / tune=uhq / ...
```

### Removed: bloat check

Mid-encoding and post-encoding bloat checks are removed entirely.

### Removed: separate VMAF pass

VMAF/SSIM computed inside encode() via NVEncC --ssim / --vmaf. No second pass.

## Planner

`_build_video_params()` DV logic:

1. `is_hdr10_plus` -> raise ValueError
2. `is_dolby_vision` and `dv_profile == 7` -> `dv_mode = DvMode.TO_8_1`
3. `is_dolby_vision` and `dv_profile != 7` -> `dv_mode = DvMode.COPY`
4. No DV -> `dv_mode = None`

Planner does no I/O. Only sets `dv_mode` in VideoParams.

## Executor

Updated pipeline:

```
Step 1: Audio (unchanged)
Step 2: Subtitles (unchanged)
Step 3: DV RPU extraction (if dv_mode is not None)
    dovi_tool [-m 2 if TO_8_1] extract-rpu source.mkv -o temp/RPU.bin
Step 4: Video encode via NVEncC (with VMAF/SSIM if enabled)
    If dv_mode is not None: --dolby-vision-rpu temp/RPU.bin --dolby-vision-profile 8.1
Step 5: Mux via mkvmerge (unchanged; mkvmerge auto-detects DV in HEVC stream)
Step 6: ENCODER tag via mkvpropedit (unchanged)
Step 7: mkclean (unchanged)
```

Removed: bloat check, separate VMAF pass.

Executor receives `EncodeResult` from encode() and writes vmaf_score/ssim_score
to job directly.

## Config (furnace.toml)

```toml
[tools]
nvencc = "C:/tools/NVEncC64.exe"       # mandatory
dovi_tool = "C:/tools/dovi_tool.exe"   # required when DV content is present
```

## JSON plan

`dv_mode` added to video_params:

```json
{
  "video_params": {
    "dv_mode": 2
  }
}
```

Values: `null` (no DV), `0` (COPY), `2` (TO_8_1).

## Cleanup

### Remove from FFmpegAdapter
- `encode()`
- `compute_quality()`
- `_build_encode_cmd()`
- `_build_vf_chain()`
- `_build_encoder_settings()`
- `_should_use_cuda()`
- `_check_mid_encoding_bloat()`

### Remove from Executor
- `_check_bloat()`
- Separate VMAF pass block (`if self._vmaf_enabled`)

### Remove from Encoder port
- `compute_quality()`

### Update Analyzer
- DV: remove skip, proceed to planning
- HDR10+: change from silent skip to raise ValueError
