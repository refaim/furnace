# DVD Crop Detection Improvement

## Problem

Crop detection is unreliable on DVD sources:
- Interlaced content produces combing artifacts that cause cropdetect to report inconsistent values frame-to-frame
- Near-identical crop values (differing by 1-2 pixels) are treated as distinct, fragmenting the vote
- The >50% reliability threshold rejects valid results that only achieved 3-4/10 agreement due to the above

On HD/4K sources cropdetect is stable and requires no changes.

## Solution

Three targeted changes to `detect_crop`, applied only to DVD-resolution sources:

### 1. DVD Resolution Detection

Pure function in `core/detect.py`:

```python
_DVD_RESOLUTIONS = {(720, 480), (720, 576)}

def is_dvd_resolution(width: int, height: int) -> bool:
    return (width, height) in _DVD_RESOLUTIONS
```

720x480 = NTSC, 720x576 = PAL. Everything else is HD+.

### 2. Deinterlace Before Cropdetect

For interlaced content, prepend `yadif` to the `-vf` filter chain:
- Interlaced: `-vf yadif,cropdetect=24:16:0`
- Progressive: `-vf cropdetect=24:16:0` (unchanged)

`yadif` chosen over `bwdif` because visual quality is irrelevant for analysis -- we only need combing removed so cropdetect sees stable boundaries. `yadif` is faster.

This applies to all interlaced content (DVD and non-DVD), since deinterlacing before cropdetect is always correct and costs negligible time on 2-second samples.

### 3. More Samples for DVD

DVD sources get 15 sample points instead of 10:

```python
_CROP_SAMPLE_POINTS_DVD: tuple[float, ...] = (
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
    0.45, 0.50, 0.55, 0.60, 0.65, 0.75, 0.85, 0.90,
)
```

Gap at 0.38-0.42 avoids mid-film intermissions/fades. DVD decodes fast at 720x480/576, so 5 extra 2-second samples add ~5-10 seconds to analysis.

HD sources keep the existing 10-point tuple unchanged.

### 4. Crop Value Clustering

Pure function in `core/detect.py`:

```python
def cluster_crop_values(
    crops: list[CropRect],
    tolerance: int = 16,
) -> tuple[CropRect, int]:
```

**Algorithm:**
1. For each crop value, count how many other values are "close" (all 4 coordinates within +/-tolerance)
2. The crop with the largest neighborhood becomes the cluster anchor
3. Collect all values within tolerance of that anchor
4. Return the **per-coordinate median** of the cluster and the cluster size

Two CropRect values are "close" if `abs(a.w - b.w) <= tolerance` AND `abs(a.h - b.h) <= tolerance` AND `abs(a.x - b.x) <= tolerance` AND `abs(a.y - b.y) <= tolerance`.

**tolerance = 16**: aligned to HEVC CU size, wide enough to absorb DVD analog jitter, narrow enough to distinguish genuinely different crops (e.g. letterbox vs full frame).

The existing >50% threshold remains: cluster size must be > `len(samples) // 2`.

For HD sources this changes nothing -- their crop values already agree exactly, so each cluster has size 1 or all-same.

## Interface Changes

### `core/ports.py` -- Prober Protocol

```python
def detect_crop(
    self, path: Path, duration_s: float,
    interlaced: bool = False, is_dvd: bool = False,
) -> CropRect | None:
```

Both parameters default to `False` for backward compatibility.

### `adapters/ffmpeg.py` -- FFmpegProber.detect_crop

Updated to:
- Accept `interlaced` and `is_dvd` parameters
- Select sample points based on `is_dvd`
- Build `-vf` chain with `yadif` prepended when `interlaced=True`
- Collect `list[CropRect]` instead of `list[str]`
- Use `cluster_crop_values()` instead of `Counter(str)`
- Apply >50% threshold on cluster size

### `services/planner.py` -- _build_job

Compute `is_dvd` from `movie.video.width/height`, pass both flags:

```python
is_dvd = is_dvd_resolution(movie.video.width, movie.video.height)
raw_crop = self._prober.detect_crop(
    movie.main_file, movie.video.duration_s,
    interlaced=movie.video.interlaced,
    is_dvd=is_dvd,
)
```

## Files Changed

| File | Change |
|------|--------|
| `furnace/core/detect.py` | Add `is_dvd_resolution()`, `cluster_crop_values()` |
| `furnace/core/ports.py` | Update `detect_crop` signature |
| `furnace/adapters/ffmpeg.py` | Deinterlace, DVD sample points, clustering |
| `furnace/services/planner.py` | Pass `interlaced` and `is_dvd` to `detect_crop` |
| `tests/core/test_detect.py` | Unit tests for `is_dvd_resolution`, `cluster_crop_values` |
| `tests/services/test_planner_crop.py` | Planner tests with mocked crop for DVD source |

## Tests

### Unit tests (`tests/core/test_detect.py`)

**`is_dvd_resolution`:**
- `(720, 480)` -> True (NTSC)
- `(720, 576)` -> True (PAL)
- `(1920, 1080)` -> False
- `(1280, 720)` -> False
- `(3840, 2160)` -> False

**`cluster_crop_values`:**
- All identical values -> cluster = all, median = that value
- Values within tolerance -> single cluster, median correct
- Two distinct groups outside tolerance -> largest cluster wins
- Single value -> cluster size 1
- Empty list -> handled by caller (detect_crop returns None before calling cluster)
- Tolerance boundary: values exactly at tolerance included, tolerance+1 excluded

### Service tests (`tests/services/test_planner_crop.py`)

- DVD source (720x576) with interlaced=True -> `detect_crop` called with `interlaced=True, is_dvd=True`
- HD source (1920x1080) -> `detect_crop` called with `interlaced=False, is_dvd=False`

## What Does NOT Change

- HD/4K crop detection path (both flags default False)
- Crop alignment in `core/quality.py`
- NVEncC crop application
- The >50% reliability threshold
- cropdetect ffmpeg filter parameters (24:16:0)
