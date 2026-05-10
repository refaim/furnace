# HDR-Aware Cropdetect

## Problem

`FFmpegAdapter.detect_crop` runs `cropdetect=24:16:0` on every video
regardless of color characteristics. On HDR10 / HLG / Dolby-Vision content
encoded with the PQ or HLG transfer this consistently fails to find the
black bars: ffmpeg's cropdetect either emits no `crop=` line at all or
reports the full frame, so the planner concludes "no crop" and the encoded
output keeps the letterbox bars baked in.

Two reproducible failing samples in
`C:\Users\Roma\Desktop\furnace\recode\NEEDS_FIXES\cropdetect hdr\`:

- `Zhili.byli.2017.WEB-DL.IVI.HEVC.HDR.2160p-SOFCJ.mkv` — HDR10 WEB-DL,
  3840×2160, transfer `smpte2084`, primaries `bt2020`, MaxCLL 1000,
  x265-encoded.
- `Звездный десант.1997.UHD.Blu-Ray.Remux.2160p.mkv` — UHD Blu-ray remux,
  3840×2160, Dolby Vision Profile 7.6 (BL+EL+RPU) with HDR10 fallback,
  transfer `smpte2084`, MaxCLL 4342.

### Root cause

`cropdetect` decides "this row/column is part of the bar" by comparing
luma to a fixed `limit` (here 24). For SDR limited-range BT.709 black
sits at code 16 and the +8 margin is enough to absorb dithering noise.

For PQ-encoded HDR the EOTF is extremely steep near zero — a small
amount of compression noise or dithering in the bars produces luma
values well above the SDR-calibrated threshold. The bars are visually
black but cropdetect never marks them as such.

`cropdetect` itself does **not** rescale `limit` to the input bit depth
(see `libavfilter/vf_cropdetect.c`, `checkline()`): on 10-bit input
`limit=24` means code 24/1023, which is below limited-range black (64)
and effectively a no-op. The PQ-noise problem and the bit-depth problem
compound.

## Goal

1. `detect_crop` reliably finds black bars on PQ (`smpte2084`) and HLG
   (`arib-std-b67`) sources, including the two samples above.
2. SDR detection is byte-for-byte unchanged — same filter chain, same
   threshold, same sample points, same cluster rule.
3. No new tunable threshold, no HDR-specific magic numbers — the SDR
   `limit=24` keeps working because the input is normalised to SDR before
   cropdetect sees it.

Out of scope: rethinking the SDR sampling strategy, manual crop override,
HDR10+ dynamic-metadata-aware detection.

## Design

### New helper

In `furnace/core/detect.py`:

```python
_HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})


def hdr_transfer_for_cropdetect(color_transfer: str | None) -> str | None:
    """Return the transfer string when cropdetect needs HDR tonemapping.

    Maps PQ ('smpte2084') and HLG ('arib-std-b67') through unchanged so
    the adapter can plug them straight into ``zscale=tin=...``. Anything
    else (including None) returns None — SDR path unchanged.
    """
    return color_transfer if color_transfer in _HDR_TRANSFERS else None
```

### Adapter signature change

`Prober.detect_crop` Protocol (`furnace/core/ports.py`) and
`FFmpegAdapter.detect_crop` (`furnace/adapters/ffmpeg.py`) gain one
keyword:

```python
def detect_crop(
    self,
    path: Path,
    duration_s: float,
    *,
    interlaced: bool = False,
    is_dvd: bool = False,
    hdr_transfer: str | None = None,   # "smpte2084" | "arib-std-b67" | None
    on_progress: Callable[[ProgressSample], None] | None = None,
) -> CropRect | None:
```

`hdr_transfer=None` (default) preserves the current SDR codepath
exactly. Callers that pass `is_hdr`-style booleans don't exist outside
the planner — no other migration needed.

### Filter chain construction

`detect_crop` builds the `-vf` argument from a list of segments:

```python
parts: list[str] = []
if interlaced:
    parts.append("yadif")
if hdr_transfer is not None:
    # PQ/HLG -> linear (npl=100 normalises to SDR peak; clips highlights
    # but leaves shadows untouched, which is all cropdetect cares about).
    parts.append(
        f"zscale=tin={hdr_transfer}:min=2020_ncl:pin=2020:t=linear:npl=100"
    )
    # linear -> BT.709 SDR transfer, limited range.
    parts.append(
        "zscale=tin=linear:min=2020_ncl:pin=2020:t=bt709:m=bt709:p=bt709:r=tv"
    )
    # cropdetect does not rescale `limit` to bit depth; force 8-bit so
    # the SDR `limit=24` keeps its intended meaning.
    parts.append("format=yuv420p")
parts.append("cropdetect=24:16:0")
vf = ",".join(parts)
```

Concrete chains:

- **SDR progressive**: `cropdetect=24:16:0` *(unchanged)*
- **SDR interlaced**: `yadif,cropdetect=24:16:0` *(unchanged)*
- **HDR10 / DV BL (PQ) progressive**:
  `zscale=tin=smpte2084:min=2020_ncl:pin=2020:t=linear:npl=100,zscale=tin=linear:min=2020_ncl:pin=2020:t=bt709:m=bt709:p=bt709:r=tv,format=yuv420p,cropdetect=24:16:0`
- **HLG progressive**: same as above with `tin=arib-std-b67`.
- **HDR interlaced**: `yadif,` prepended to the HDR chain (rare in
  practice but the matrix is symmetric).

### Why these zscale parameters

- **Explicit `tin`/`min`/`pin` on both stages.** zscale auto-detects
  these from frame metadata when omitted, but on `-ss` seeks the parser
  can land before VUI propagates from the keyframe, and some sources
  (DV P5, broken muxes) lack reliable per-frame color metadata. Without
  an explicit `tin`, zscale falls back to `bt709`, which makes
  `npl` a silent no-op and produces an identity round-trip — exactly the
  failure we are trying to fix. Stating the parameters on both stages
  also makes the intermediate (linear, BT.2020 primaries) explicit so
  the second zscale doesn't have to infer it.
- **`npl=100`.** Sets nominal peak luminance for the linear output. With
  `npl=100`, PQ codes representing >100 cd/m² clip at 1.0 in linear and
  squash into the upper end of the BT.709 range. **Shadows, where the
  letterbox bars live, are unaffected** — code 64 in PQ (≈0 cd/m²) maps
  to ≈0 in linear regardless of `npl`, then through inverse-709 OETF
  back to code 16 in 8-bit limited range. Higher `npl` would *raise* the
  noise floor and make detection worse.
- **`format=yuv420p`.** Forces 8-bit output so the SDR `cropdetect=24`
  threshold has its intended meaning (see Root cause above). This is
  load-bearing — annotated with a comment in code.
- **No explicit `tonemap` filter.** Operators like `hable`/`mobius`
  shape highlights; they are identity on shadows. For cropdetect on
  letterbox bars they add CPU cost without changing the answer.

### Planner integration

Single call-site change in `furnace/services/planner.py:182`:

```python
raw_crop = self._prober.detect_crop(
    movie.main_file,
    movie.video.duration_s,
    interlaced=movie.video.interlaced,
    is_dvd=is_dvd,
    hdr_transfer=hdr_transfer_for_cropdetect(movie.video.color_transfer),
    on_progress=self._on_crop_progress,
)
```

`movie.video.color_transfer` is already populated from ffprobe in
`analyzer.py` — no changes to `VideoInfo` / `Movie`.

### DV Profile 5 note

Single-layer Dolby Vision (`dvhe.05`) marks the stream as `smpte2084`
in container metadata and HEVC VUI even though the underlying signal is
IPT-PQ-C2, not BT.2020 RGB. zscale will tonemap it as if it were HDR10,
which would corrupt colors — but cropdetect only inspects luma magnitude
near zero, and "black" maps to ~0 intensity in IPT just as it does in
YCbCr. The crop geometry comes out correct by accident. A code comment
near the zscale block records this so a future reader doesn't try to
"fix" it.

## Failure modes

- **SDR misclassified as HDR** (planner bug): zscale with
  `tin=smpte2084` on actual BT.709 input still completes — it
  inverts PQ from data that wasn't PQ-encoded, then re-applies BT.709
  OETF. Highlights warp, shadows survive intact, cropdetect still finds
  the bars. No crash, no wrong crop.
- **Missing color metadata on input**: explicit `tin=` overrides any
  missing frame metadata. The chain runs successfully on the assumption
  the planner's classification was correct. If `color_transfer` was
  `None` in `VideoInfo`, the planner passes `hdr_transfer=None` and the
  SDR path is taken — no change from today.
- **ffmpeg without libzimg**: `zscale` fails with "no such filter".
  cropdetect produces no `crop=` line, the existing
  `if not crop_values: return None` path triggers, planner logs
  "cropdetect unable to determine crop", encoding proceeds without
  crop. Same observable behaviour as a noisy SDR file today; the user's
  ffmpeg build is known to include libzimg.

## Testing

TDD order — failing tests first, then implementation.

### `tests/core/test_detect.py` — `hdr_transfer_for_cropdetect`

Parametrised cases:

| input | expected |
|------|----------|
| `"smpte2084"` | `"smpte2084"` |
| `"arib-std-b67"` | `"arib-std-b67"` |
| `"bt709"` | `None` |
| `"smpte170m"` | `None` |
| `None` | `None` |

### `tests/adapters/test_ffmpeg_cropdetect_hdr.py` (new file)

Mock `subprocess.run`, capture the command list, assert the `-vf`
argument matches the expected chain string. Coverage matrix:

| `interlaced` | `hdr_transfer` | expected `-vf` |
|--------------|----------------|----------------|
| `False` | `None` | `cropdetect=24:16:0` |
| `True`  | `None` | `yadif,cropdetect=24:16:0` |
| `False` | `"smpte2084"` | full PQ chain |
| `False` | `"arib-std-b67"` | full HLG chain |
| `True`  | `"smpte2084"` | `yadif,` + PQ chain |

### `tests/services/test_planner_crop_detect.py`

Mock `Prober`, capture `detect_crop` kwargs, assert `hdr_transfer`:

| `movie.video.color_transfer` | expected `hdr_transfer` kwarg |
|-------------------------------|-------------------------------|
| `"bt709"` | `None` |
| `"smpte2084"` | `"smpte2084"` |
| `"arib-std-b67"` | `"arib-std-b67"` |

### `tests/core/test_ports.py`

Add a check that the `Prober` Protocol accepts the new `hdr_transfer`
keyword (compile-time `runtime_checkable` test).

### Coverage

100 % lines + branches on `hdr_transfer_for_cropdetect`, on the new HDR
branch in `detect_crop`'s chain construction, and on the planner's new
kwarg expression. The two existing SDR tests for `detect_crop` continue
to pass without modification (regression guarantee for SDR).

### Manual validation (out-of-band)

Not part of the automated suite — run after implementation lands:

1. `furnace plan` on `Zhili.byli.2017.WEB-DL.IVI.HEVC.HDR.2160p-SOFCJ.mkv`
   → log shows `crop detected …` with `h < 2160`.
2. `furnace plan` on `Звездный десант.1997.UHD.Blu-Ray.Remux.2160p.mkv`
   → log shows a 2.39:1-ish crop (~3840×1606 or similar).
3. `furnace plan` on any existing SDR file in the user's library →
   identical result to before this change.

## Versioning

PATCH bump 1.14.2 → **1.14.3**. User-visible bugfix: HDR cropdetect
starts returning correct values where it previously returned no crop.
No CLI / JSON schema changes, no new flags.

- `furnace/__init__.py` → `VERSION = "1.14.3"`
- `pyproject.toml`     → `version = "1.14.3"`
