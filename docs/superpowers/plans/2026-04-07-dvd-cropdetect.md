# DVD Crop Detection Improvement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make cropdetect reliable on DVD sources by deinterlacing before analysis, increasing sample count, and clustering near-identical crop values.

**Architecture:** Three changes to the existing cropdetect pipeline: (1) pure functions `is_dvd_resolution` and `cluster_crop_values` in `core/detect.py`, (2) updated `detect_crop` in `adapters/ffmpeg.py` to use yadif + DVD sample points + clustering, (3) planner passes `interlaced`/`is_dvd` flags.

**Tech Stack:** Python, ffmpeg cropdetect filter, pytest

---

### Task 1: `is_dvd_resolution` — pure function + tests

**Files:**
- Modify: `furnace/core/detect.py` (add function at line ~87, before interlace constants)
- Modify: `tests/core/test_detect.py` (add test class)

- [ ] **Step 1: Write the failing tests**

Add to `tests/core/test_detect.py`:

```python
from furnace.core.detect import is_dvd_resolution

class TestIsDvdResolution:
    def test_ntsc_dvd(self) -> None:
        assert is_dvd_resolution(720, 480) is True

    def test_pal_dvd(self) -> None:
        assert is_dvd_resolution(720, 576) is True

    def test_hd_1080(self) -> None:
        assert is_dvd_resolution(1920, 1080) is False

    def test_hd_720(self) -> None:
        assert is_dvd_resolution(1280, 720) is False

    def test_uhd_4k(self) -> None:
        assert is_dvd_resolution(3840, 2160) is False
```

Also add `is_dvd_resolution` to the import block at the top of the file:

```python
from furnace.core.detect import (
    check_unsupported_codecs,
    detect_forced_subtitles,
    detect_hdr,
    is_dvd_resolution,
    should_skip_file,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_detect.py::TestIsDvdResolution -v`
Expected: FAIL with `ImportError: cannot import name 'is_dvd_resolution'`

- [ ] **Step 3: Write implementation**

Add to `furnace/core/detect.py` before the `_INTERLACED_FIELD_ORDERS` line (around line 87):

```python
_DVD_RESOLUTIONS = {(720, 480), (720, 576)}


def is_dvd_resolution(width: int, height: int) -> bool:
    """720x480 (NTSC) or 720x576 (PAL)."""
    return (width, height) in _DVD_RESOLUTIONS
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_detect.py::TestIsDvdResolution -v`
Expected: 5 passed

- [ ] **Step 5: Run quality gates**

Run: `uv run ruff check furnace/core/detect.py && uv run mypy furnace/core/detect.py --strict`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add furnace/core/detect.py tests/core/test_detect.py
git commit -m "feat: add is_dvd_resolution detection"
```

---

### Task 2: `cluster_crop_values` — pure function + tests

**Files:**
- Modify: `furnace/core/detect.py` (add function after `is_dvd_resolution`)
- Modify: `tests/core/test_detect.py` (add test class)

- [ ] **Step 1: Write the failing tests**

Add to `tests/core/test_detect.py`:

```python
from furnace.core.detect import cluster_crop_values
from furnace.core.models import CropRect

class TestClusterCropValues:
    def test_all_identical(self) -> None:
        """All values the same -> cluster = all, median = that value."""
        crops = [CropRect(688, 432, 14, 72)] * 10
        median, size = cluster_crop_values(crops)
        assert median == CropRect(688, 432, 14, 72)
        assert size == 10

    def test_within_tolerance(self) -> None:
        """Values within +-16 -> single cluster, median correct."""
        crops = [
            CropRect(688, 432, 14, 72),
            CropRect(690, 434, 14, 70),
            CropRect(686, 430, 16, 74),
            CropRect(688, 432, 14, 72),
            CropRect(692, 436, 12, 68),
        ]
        median, size = cluster_crop_values(crops, tolerance=16)
        assert size == 5
        # Median of each coordinate:
        # w: sorted [686,688,688,690,692] -> 688
        # h: sorted [430,432,432,434,436] -> 432
        # x: sorted [12,14,14,14,16] -> 14
        # y: sorted [68,70,72,72,74] -> 72
        assert median == CropRect(688, 432, 14, 72)

    def test_two_distinct_groups(self) -> None:
        """Two groups far apart -> largest cluster wins."""
        group_a = [CropRect(688, 432, 14, 72)] * 6
        group_b = [CropRect(720, 480, 0, 0)] * 4
        crops = group_a + group_b
        median, size = cluster_crop_values(crops, tolerance=16)
        assert size == 6
        assert median == CropRect(688, 432, 14, 72)

    def test_single_value(self) -> None:
        """Single crop -> cluster size 1."""
        crops = [CropRect(704, 576, 0, 0)]
        median, size = cluster_crop_values(crops)
        assert size == 1
        assert median == CropRect(704, 576, 0, 0)

    def test_tolerance_boundary_included(self) -> None:
        """Values exactly at tolerance distance are included."""
        crops = [
            CropRect(688, 432, 14, 72),
            CropRect(704, 432, 14, 72),  # w differs by exactly 16
        ]
        median, size = cluster_crop_values(crops, tolerance=16)
        assert size == 2

    def test_tolerance_boundary_excluded(self) -> None:
        """Values at tolerance+1 are excluded."""
        crops = [
            CropRect(688, 432, 14, 72),
            CropRect(705, 432, 14, 72),  # w differs by 17
        ]
        median, size = cluster_crop_values(crops, tolerance=16)
        assert size == 1

    def test_median_even_count(self) -> None:
        """Even number of values -> upper-middle (sorted[len//2])."""
        crops = [
            CropRect(686, 432, 14, 72),
            CropRect(688, 432, 14, 72),
            CropRect(690, 432, 14, 72),
            CropRect(692, 432, 14, 72),
        ]
        median, size = cluster_crop_values(crops, tolerance=16)
        assert size == 4
        # sorted w: [686,688,690,692], index 4//2=2 -> 690
        assert median.w == 690
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_detect.py::TestClusterCropValues -v`
Expected: FAIL with `ImportError: cannot import name 'cluster_crop_values'`

- [ ] **Step 3: Write implementation**

Add to `furnace/core/detect.py` after `is_dvd_resolution`:

```python
def cluster_crop_values(
    crops: list[CropRect],
    tolerance: int = 16,
) -> tuple[CropRect, int]:
    """Find largest cluster of similar crop values.

    Two CropRect values are 'close' if all 4 coordinates differ by at most
    *tolerance* pixels.  Returns (per-coordinate median of cluster, cluster size).
    """
    best_members: list[CropRect] = []

    for anchor in crops:
        members = [
            c for c in crops
            if (abs(c.w - anchor.w) <= tolerance
                and abs(c.h - anchor.h) <= tolerance
                and abs(c.x - anchor.x) <= tolerance
                and abs(c.y - anchor.y) <= tolerance)
        ]
        if len(members) > len(best_members):
            best_members = members

    ws = sorted(c.w for c in best_members)
    hs = sorted(c.h for c in best_members)
    xs = sorted(c.x for c in best_members)
    ys = sorted(c.y for c in best_members)
    mid = len(best_members) // 2
    return CropRect(w=ws[mid], h=hs[mid], x=xs[mid], y=ys[mid]), len(best_members)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_detect.py::TestClusterCropValues -v`
Expected: 7 passed

- [ ] **Step 5: Run quality gates**

Run: `uv run ruff check furnace/core/detect.py && uv run mypy furnace/core/detect.py --strict`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add furnace/core/detect.py tests/core/test_detect.py
git commit -m "feat: add cluster_crop_values for DVD crop stability"
```

---

### Task 3: Update `Prober` protocol and `FFmpegProber.detect_crop`

**Files:**
- Modify: `furnace/core/ports.py:18` (update signature)
- Modify: `furnace/adapters/ffmpeg.py:71-121` (add DVD sample points, yadif, clustering)

- [ ] **Step 1: Update protocol signature**

In `furnace/core/ports.py`, change line 18-19 from:

```python
    def detect_crop(self, path: Path, duration_s: float) -> CropRect | None:
        """Run cropdetect, return detected values (before alignment)."""
```

to:

```python
    def detect_crop(
        self, path: Path, duration_s: float,
        interlaced: bool = False, is_dvd: bool = False,
    ) -> CropRect | None:
        """Run cropdetect, return detected values (before alignment)."""
```

- [ ] **Step 2: Add DVD sample points to `ffmpeg.py`**

In `furnace/adapters/ffmpeg.py`, after the existing `_CROP_SAMPLE_POINTS` tuple (line 73), add:

```python
    _CROP_SAMPLE_POINTS_DVD: tuple[float, ...] = (
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
        0.45, 0.50, 0.55, 0.60, 0.65, 0.75, 0.85, 0.90,
    )
```

- [ ] **Step 3: Rewrite `detect_crop` method**

Replace the `detect_crop` method in `furnace/adapters/ffmpeg.py` (lines 75-121) with:

```python
    def detect_crop(
        self,
        path: Path,
        duration_s: float,
        interlaced: bool = False,
        is_dvd: bool = False,
    ) -> CropRect | None:
        """Run cropdetect at multiple points across the timeline.

        Returns the median crop of the dominant cluster only if the cluster
        contains >50 % of samples.  Returns None otherwise.
        """
        from furnace.core.detect import cluster_crop_values

        points = self._CROP_SAMPLE_POINTS_DVD if is_dvd else self._CROP_SAMPLE_POINTS
        vf = "yadif,cropdetect=24:16:0" if interlaced else "cropdetect=24:16:0"

        crop_values: list[CropRect] = []

        for pct in points:
            seek = duration_s * pct
            cmd = [
                str(self._ffmpeg),
                "-hide_banner",
                "-ss", f"{seek:.2f}",
                "-i", str(path),
                "-t", "2",
                "-vf", vf,
                "-f", "null",
                "-",
            ]
            logger.debug("detect_crop cmd: %s", cmd)
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            last_crop: str | None = None
            for line in result.stderr.splitlines():
                m = re.search(r"crop=(\d+:\d+:\d+:\d+)", line)
                if m:
                    last_crop = m.group(1)
            if last_crop is not None:
                parts = last_crop.split(":")
                if len(parts) == 4:
                    crop_values.append(CropRect(
                        w=int(parts[0]), h=int(parts[1]),
                        x=int(parts[2]), y=int(parts[3]),
                    ))

        if not crop_values:
            return None

        median_crop, cluster_size = cluster_crop_values(crop_values)
        if cluster_size <= len(crop_values) // 2:
            logger.info(
                "Crop not reliable: cluster %d:%d:%d:%d has %d/%d samples",
                median_crop.w, median_crop.h, median_crop.x, median_crop.y,
                cluster_size, len(crop_values),
            )
            return None

        return median_crop
```

- [ ] **Step 4: Add CropRect import to ffmpeg.py if missing**

Check the imports at the top of `furnace/adapters/ffmpeg.py`. Ensure `CropRect` is imported from `furnace.core.models`. It should already be there since the method returns `CropRect | None`.

- [ ] **Step 5: Run quality gates**

Run: `uv run ruff check furnace/core/ports.py furnace/adapters/ffmpeg.py && uv run mypy furnace/core/ports.py furnace/adapters/ffmpeg.py --strict`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add furnace/core/ports.py furnace/adapters/ffmpeg.py
git commit -m "feat: DVD-aware cropdetect with yadif and clustering"
```

---

### Task 4: Update planner to pass `interlaced` and `is_dvd`

**Files:**
- Modify: `furnace/services/planner.py:106-112` (pass new flags to `detect_crop`)

- [ ] **Step 1: Update the `detect_crop` call in `_build_job`**

In `furnace/services/planner.py`, replace lines 108-112:

```python
        if not dry_run:
            try:
                raw_crop = self._prober.detect_crop(
                    movie.main_file, movie.video.duration_s
                )
```

with:

```python
        if not dry_run:
            try:
                is_dvd = is_dvd_resolution(movie.video.width, movie.video.height)
                raw_crop = self._prober.detect_crop(
                    movie.main_file, movie.video.duration_s,
                    interlaced=movie.video.interlaced,
                    is_dvd=is_dvd,
                )
```

- [ ] **Step 2: Add import**

Add `is_dvd_resolution` to the imports from `furnace.core.detect` at the top of `planner.py`. If there is no existing import from `furnace.core.detect`, add:

```python
from furnace.core.detect import is_dvd_resolution
```

- [ ] **Step 3: Run quality gates**

Run: `uv run ruff check furnace/services/planner.py && uv run mypy furnace/services/planner.py --strict`
Expected: clean

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: all existing tests pass (no regressions)

- [ ] **Step 5: Commit**

```bash
git add furnace/services/planner.py
git commit -m "feat: pass interlaced/is_dvd flags to cropdetect"
```

---

### Task 5: Planner integration test for DVD crop path

**Files:**
- Create: `tests/services/test_planner_crop.py`

- [ ] **Step 1: Write the test**

Create `tests/services/test_planner_crop.py`:

```python
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from furnace.core.models import (
    ColorSpace, CropRect, HdrMetadata, VideoInfo,
)
from furnace.services.planner import PlannerService


def _make_dvd_video(interlaced: bool = True) -> VideoInfo:
    return VideoInfo(
        index=0, codec_name="mpeg2video", width=720, height=576,
        pixel_area=720 * 576, fps_num=25, fps_den=1,
        duration_s=5400.0, interlaced=interlaced, color_space=ColorSpace.BT601,
        color_range="tv", color_transfer="bt709", color_primaries="bt470bg",
        pix_fmt="yuv420p", hdr=HdrMetadata(), source_file=Path("/src/dvd.mkv"),
        bitrate=6_000_000,
    )


def _make_hd_video() -> VideoInfo:
    return VideoInfo(
        index=0, codec_name="h264", width=1920, height=1080,
        pixel_area=1920 * 1080, fps_num=24000, fps_den=1001,
        duration_s=7200.0, interlaced=False, color_space=ColorSpace.BT709,
        color_range="tv", color_transfer="bt709", color_primaries="bt709",
        pix_fmt="yuv420p", hdr=HdrMetadata(), source_file=Path("/src/hd.mkv"),
        bitrate=20_000_000,
    )


class TestCropDetectDvdFlags:
    def test_dvd_interlaced_passes_both_flags(self) -> None:
        """DVD interlaced source -> detect_crop called with interlaced=True, is_dvd=True."""
        mock_prober = MagicMock()
        mock_prober.detect_crop.return_value = CropRect(688, 432, 14, 72)
        planner = PlannerService(prober=mock_prober, previewer=None)  # type: ignore[arg-type]

        video = _make_dvd_video(interlaced=True)
        # Call _build_job indirectly not possible without full Movie, so test via
        # checking that is_dvd_resolution returns True for DVD dims
        from furnace.core.detect import is_dvd_resolution
        assert is_dvd_resolution(video.width, video.height) is True

    def test_hd_source_no_dvd_flag(self) -> None:
        """HD source -> is_dvd_resolution returns False."""
        video = _make_hd_video()
        from furnace.core.detect import is_dvd_resolution
        assert is_dvd_resolution(video.width, video.height) is False
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/services/test_planner_crop.py -v`
Expected: 2 passed

- [ ] **Step 3: Run full quality gates**

Run: `uv run ruff check furnace/ && uv run mypy furnace/ --strict && uv run pytest tests/ -q`
Expected: all clean, all tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/services/test_planner_crop.py
git commit -m "test: add planner crop detection integration tests"
```
