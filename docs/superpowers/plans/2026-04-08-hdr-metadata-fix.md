# HDR Metadata Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix HDR metadata loss during encoding — transfer characteristics, mastering display, content light level, and matrix coefficients.

**Architecture:** Pure functions `detect_video_system` and `resolve_color_metadata` in `furnace/core/detect.py` handle all color inference. The `ColorSpace` enum is removed; raw matrix/transfer/primaries strings flow through the system. ffprobe key is fixed, and a second probe call extracts HDR side_data from the first frame.

**Tech Stack:** Python 3.12, pytest, dataclasses, ffprobe

---

### Task 1: `detect_video_system` — tests then implementation

**Files:**
- Create: `tests/core/test_video_system.py`
- Modify: `furnace/core/detect.py`

- [ ] **Step 1: Write tests for `detect_video_system`**

```python
# tests/core/test_video_system.py
from __future__ import annotations

import pytest

from furnace.core.detect import VideoSystem, detect_video_system


class TestDetectVideoSystem:
    # PAL standard heights
    def test_pal_576(self) -> None:
        assert detect_video_system(576) == VideoSystem.PAL

    def test_pal_288(self) -> None:
        assert detect_video_system(288) == VideoSystem.PAL

    # NTSC standard heights
    def test_ntsc_480(self) -> None:
        assert detect_video_system(480) == VideoSystem.NTSC

    def test_ntsc_486(self) -> None:
        assert detect_video_system(486) == VideoSystem.NTSC

    def test_ntsc_240(self) -> None:
        assert detect_video_system(240) == VideoSystem.NTSC

    # HD heights
    def test_hd_720(self) -> None:
        assert detect_video_system(720) == VideoSystem.HD

    def test_hd_1080(self) -> None:
        assert detect_video_system(1080) == VideoSystem.HD

    def test_hd_2160(self) -> None:
        assert detect_video_system(2160) == VideoSystem.HD

    # Non-standard SD -> ValueError
    def test_unknown_sd_544(self) -> None:
        with pytest.raises(ValueError):
            detect_video_system(544)

    def test_unknown_sd_352(self) -> None:
        with pytest.raises(ValueError):
            detect_video_system(352)

    def test_unknown_sd_360(self) -> None:
        with pytest.raises(ValueError):
            detect_video_system(360)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_video_system.py -v`
Expected: FAIL — `VideoSystem` and `detect_video_system` not defined.

- [ ] **Step 3: Implement `VideoSystem` enum and `detect_video_system`**

Add to `furnace/core/detect.py` (after the existing imports, before `detect_forced_subtitles`):

```python
import enum


class VideoSystem(enum.Enum):
    """Video system determined from frame height."""
    PAL = "pal"
    NTSC = "ntsc"
    HD = "hd"


_PAL_HEIGHTS = frozenset({576, 288})
_NTSC_HEIGHTS = frozenset({480, 486, 240})


def detect_video_system(height: int) -> VideoSystem:
    """Determine video system from frame height.

    PAL:  576, 288
    NTSC: 480, 486, 240
    HD:   >= 720
    Other SD: ValueError
    """
    if height in _PAL_HEIGHTS:
        return VideoSystem.PAL
    if height in _NTSC_HEIGHTS:
        return VideoSystem.NTSC
    if height >= 720:
        return VideoSystem.HD
    raise ValueError(
        f"Unknown SD height {height}: cannot determine PAL/NTSC. "
        f"Add this height to _PAL_HEIGHTS or _NTSC_HEIGHTS in detect.py"
    )
```

Also add `enum` to the imports at the top of the file (it's not currently imported).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_video_system.py -v`
Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```
git add tests/core/test_video_system.py furnace/core/detect.py
git commit -m "feat: add detect_video_system with PAL/NTSC/HD detection"
```

---

### Task 2: `resolve_color_metadata` — generate tests

**Files:**
- Create: `tests/core/test_color_resolve.py`
- Modify: `scripts/gen_color_tests.py`

- [ ] **Step 1: Update generator to output pytest code**

Modify `scripts/gen_color_tests.py` to add a `--pytest` flag that outputs the 243 test cases as a pytest file. Add this to `main()`:

```python
import sys

def main():
    cases = list(iter_cases())

    if "--pytest" in sys.argv:
        print_pytest(cases)
    else:
        print_table(cases)
```

Add the resolve logic (from our approved tables) and the pytest printer:

```python
def _family(mx):
    if mx in ("bt2020nc", "bt2020c"):
        return "bt2020"
    if mx == "bt709":
        return "bt709"
    if mx in ("bt470bg", "smpte170m"):
        return "bt601"
    return None


def resolve(mx, tr, pri, sys, hdr):
    """Compute expected outputs per approved design."""
    family = _family(mx)
    if family is None:
        if hdr:
            family = "bt2020"
        elif sys == "HD":
            family = "bt709"
        else:
            family = "bt601"

    # matrix
    if mx is not None:
        out_mx = mx
    elif family == "bt2020":
        out_mx = "bt2020nc"
    elif family == "bt709":
        out_mx = "bt709"
    elif sys == "PAL":
        out_mx = "bt470bg"
    else:
        out_mx = "smpte170m"

    # primaries
    if pri is not None:
        out_pri = pri
    elif family == "bt2020":
        out_pri = "bt2020"
    elif family == "bt709":
        out_pri = "bt709"
    elif sys == "PAL":
        out_pri = "bt470bg"
    else:
        out_pri = "smpte170m"

    # transfer
    if tr is not None:
        out_tr = tr
    elif family == "bt2020":
        out_tr = "smpte2084" if hdr else "bt709"
    elif family == "bt709":
        out_tr = "bt709"
    else:
        # bt601: infer from resolved primaries
        _pri_to_tr = {
            "bt470bg": "bt470bg",
            "smpte170m": "smpte170m",
            "bt470m": "bt470m",
            "bt709": "bt709",
        }
        if out_pri in _pri_to_tr:
            out_tr = _pri_to_tr[out_pri]
        elif sys == "PAL":
            out_tr = "bt470bg"
        else:
            out_tr = "smpte170m"

    return out_mx, out_tr, out_pri


def print_pytest(cases):
    print('"""Auto-generated by scripts/gen_color_tests.py --pytest"""')
    print("from __future__ import annotations")
    print()
    print("import pytest")
    print()
    print("from furnace.core.detect import ResolvedColor, VideoSystem, resolve_color_metadata")
    print()
    print()
    print("# fmt: off")
    print("CASES = [")
    for mx, tr, pri, sys, hdr in cases:
        out_mx, out_tr, out_pri = resolve(mx, tr, pri, sys, hdr)
        mx_r = f'"{mx}"' if mx else "None"
        tr_r = f'"{tr}"' if tr else "None"
        pri_r = f'"{pri}"' if pri else "None"
        print(f'    ({mx_r}, {tr_r}, {pri_r}, VideoSystem.{sys}, {hdr}, '
              f'ResolvedColor("{out_mx}", "{out_tr}", "{out_pri}")),')
    print("]")
    print("# fmt: on")
    print()
    print()
    print("@pytest.mark.parametrize(")
    print('    "matrix_raw, transfer_raw, primaries_raw, system, has_hdr, expected",')
    print("    CASES,")
    print(")")
    print("def test_resolve_color_metadata(matrix_raw, transfer_raw, primaries_raw, system, has_hdr, expected):")
    print("    result = resolve_color_metadata(matrix_raw, transfer_raw, primaries_raw, system, has_hdr)")
    print("    assert result == expected, (")
    print('        f"Input: mx={matrix_raw}, tr={transfer_raw}, pri={primaries_raw}, sys={system}, hdr={has_hdr}\\n"')
    print('        f"Expected: {expected}\\n"')
    print('        f"Got:      {result}"')
    print("    )")
```

Rename the existing `main()` table-printing logic to `print_table(cases)`.

- [ ] **Step 2: Generate the test file**

Run: `python3 scripts/gen_color_tests.py --pytest > tests/core/test_color_resolve.py`

- [ ] **Step 3: Verify 243 test cases are generated**

Run: `uv run pytest tests/core/test_color_resolve.py --collect-only -q | tail -3`
Expected: `243 tests collected`

(They will fail — `resolve_color_metadata` doesn't exist yet.)

- [ ] **Step 4: Commit test file**

```
git add tests/core/test_color_resolve.py scripts/gen_color_tests.py
git commit -m "test: add 243 parametrized tests for resolve_color_metadata"
```

---

### Task 3: `resolve_color_metadata` — implementation

**Files:**
- Modify: `furnace/core/detect.py`

- [ ] **Step 1: Run tests to confirm they fail**

Run: `uv run pytest tests/core/test_color_resolve.py -x -q`
Expected: FAIL — `ResolvedColor` and `resolve_color_metadata` not importable.

- [ ] **Step 2: Implement `ResolvedColor` and `resolve_color_metadata`**

Add to `furnace/core/detect.py` (after `VideoSystem` and `detect_video_system`):

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedColor:
    """Resolved color metadata for NVEncC flags."""
    matrix: str      # --colormatrix
    transfer: str    # --transfer
    primaries: str   # --colorprim


_BT2020_MATRICES = frozenset({"bt2020nc", "bt2020c"})
_BT601_MATRICES = frozenset({"bt470bg", "smpte170m"})

_TRANSFER_FROM_PRIMARIES: dict[str, str] = {
    "bt470bg": "bt470bg",
    "smpte170m": "smpte170m",
    "bt470m": "bt470m",
    "bt709": "bt709",
}


def resolve_color_metadata(
    matrix_raw: str | None,
    transfer_raw: str | None,
    primaries_raw: str | None,
    system: VideoSystem,
    has_hdr: bool,
) -> ResolvedColor:
    """Resolve color metadata, filling in missing values per ITU standards.

    Raises ValueError for unrecognized matrix_raw values.
    """
    # Step 1: determine family
    if matrix_raw in _BT2020_MATRICES:
        family = "bt2020"
    elif matrix_raw == "bt709":
        family = "bt709"
    elif matrix_raw in _BT601_MATRICES:
        family = "bt601"
    elif matrix_raw is None:
        if has_hdr:
            family = "bt2020"
        elif system == VideoSystem.HD:
            family = "bt709"
        else:
            family = "bt601"
    else:
        raise ValueError(f"Unrecognized matrix_raw: {matrix_raw!r}")

    is_pal = system == VideoSystem.PAL

    # Step 2: resolve matrix
    if matrix_raw is not None:
        matrix = matrix_raw
    elif family == "bt2020":
        matrix = "bt2020nc"
    elif family == "bt709":
        matrix = "bt709"
    elif is_pal:
        matrix = "bt470bg"
    else:
        matrix = "smpte170m"

    # Step 3: resolve primaries
    if primaries_raw is not None:
        primaries = primaries_raw
    elif family == "bt2020":
        primaries = "bt2020"
    elif family == "bt709":
        primaries = "bt709"
    elif is_pal:
        primaries = "bt470bg"
    else:
        primaries = "smpte170m"

    # Step 4: resolve transfer
    if transfer_raw is not None:
        transfer = transfer_raw
    elif family == "bt2020":
        transfer = "smpte2084" if has_hdr else "bt709"
    elif family == "bt709":
        transfer = "bt709"
    else:
        # bt601: infer from resolved primaries
        if primaries in _TRANSFER_FROM_PRIMARIES:
            transfer = _TRANSFER_FROM_PRIMARIES[primaries]
        elif is_pal:
            transfer = "bt470bg"
        else:
            transfer = "smpte170m"

    return ResolvedColor(matrix=matrix, transfer=transfer, primaries=primaries)
```

- [ ] **Step 3: Run all 243 tests**

Run: `uv run pytest tests/core/test_color_resolve.py -q`
Expected: `243 passed`

- [ ] **Step 4: Commit**

```
git add furnace/core/detect.py
git commit -m "feat: implement resolve_color_metadata with full inference logic"
```

---

### Task 4: Fix ffprobe key `color_trc` → `color_transfer`

**Files:**
- Modify: `furnace/services/analyzer.py:190`
- Modify: `tests/services/test_analyzer.py:57,123`

- [ ] **Step 1: Fix the mock data in tests**

In `tests/services/test_analyzer.py`, change all occurrences of `"color_trc"` to `"color_transfer"`:

Line 57: `"color_trc": "bt709"` → `"color_transfer": "bt709"`
Line 123: `"color_trc": "smpte2084"` → `"color_transfer": "smpte2084"`

- [ ] **Step 2: Run analyzer tests to verify they fail**

Run: `uv run pytest tests/services/test_analyzer.py -v -q`
Expected: Tests that check `color_transfer` field will FAIL because the code still reads `"color_trc"`.

- [ ] **Step 3: Fix the key in analyzer**

In `furnace/services/analyzer.py:190`, change:
```python
color_transfer_raw = stream.get("color_trc")
```
to:
```python
color_transfer_raw = stream.get("color_transfer")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/services/test_analyzer.py -v -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```
git add furnace/services/analyzer.py tests/services/test_analyzer.py
git commit -m "fix: read color_transfer instead of color_trc from ffprobe"
```

---

### Task 5: Fraction parsing in `detect_hdr` + HDR side_data probe

**Files:**
- Modify: `furnace/core/detect.py:173-181`
- Modify: `furnace/adapters/ffmpeg.py`
- Modify: `furnace/core/ports.py`
- Modify: `furnace/services/analyzer.py`
- Modify: `tests/core/test_detect.py`

- [ ] **Step 1: Write tests for fraction parsing**

Add to `tests/core/test_detect.py` at the bottom:

```python
class TestHdrDetectionFractions:
    """detect_hdr must handle fraction values from ffprobe frame-level side_data."""

    def test_mastering_display_fractions(self):
        """Fraction values like '8500/50000' should become '8500'."""
        side_data = [{
            "side_data_type": "Mastering display metadata",
            "green_x": "8500/50000", "green_y": "39850/50000",
            "blue_x": "6550/50000", "blue_y": "2300/50000",
            "red_x": "35400/50000", "red_y": "14600/50000",
            "white_point_x": "15635/50000", "white_point_y": "16450/50000",
            "max_luminance": "10000000/10000", "min_luminance": "1/10000",
        }]
        result = detect_hdr({}, side_data)
        assert result.mastering_display == (
            "G(8500,39850)B(6550,2300)R(35400,14600)"
            "WP(15635,16450)L(10000000,1)"
        )

    def test_mastering_display_integers(self):
        """Integer values (no slash) should pass through unchanged."""
        side_data = [{
            "side_data_type": "Mastering display metadata",
            "green_x": "8500", "green_y": "39850",
            "blue_x": "6550", "blue_y": "2300",
            "red_x": "35400", "red_y": "14600",
            "white_point_x": "15635", "white_point_y": "16450",
            "max_luminance": "10000000", "min_luminance": "1",
        }]
        result = detect_hdr({}, side_data)
        assert result.mastering_display == (
            "G(8500,39850)B(6550,2300)R(35400,14600)"
            "WP(15635,16450)L(10000000,1)"
        )

    def test_mastering_display_decimal_passthrough(self):
        """Decimal values like '0.2650' should pass through unchanged (old-style ffprobe)."""
        side_data = [{
            "side_data_type": "Mastering display metadata",
            "green_x": "0.2650", "green_y": "0.6900",
            "blue_x": "0.1500", "blue_y": "0.0600",
            "red_x": "0.6800", "red_y": "0.3200",
            "white_point_x": "0.3127", "white_point_y": "0.3290",
            "max_luminance": "1000.0000", "min_luminance": "0.0050",
        }]
        result = detect_hdr({}, side_data)
        # Decimal values pass through; they don't contain '/'
        assert "G(0.2650,0.6900)" in result.mastering_display

    def test_content_light_integers(self):
        """Content light level values are always integers — no fractions."""
        side_data = [{
            "side_data_type": "Content light level metadata",
            "max_content": 1000,
            "max_average": 180,
        }]
        result = detect_hdr({}, side_data)
        assert result.content_light == "MaxCLL=1000,MaxFALL=180"
```

- [ ] **Step 2: Run tests to see fraction test fail**

Run: `uv run pytest tests/core/test_detect.py::TestHdrDetectionFractions -v`
Expected: `test_mastering_display_fractions` FAILS (fractions passed through as-is).

- [ ] **Step 3: Add `_fraction_numerator` helper and fix `detect_hdr`**

In `furnace/core/detect.py`, add the helper before `detect_hdr`:

```python
def _fraction_numerator(val: str) -> str:
    """Extract numerator from fraction string. '8500/50000' -> '8500'. No-op for non-fractions."""
    if "/" in str(val):
        return str(val).split("/", 1)[0]
    return str(val)
```

In `detect_hdr`, replace the mastering_display block (lines 173-181):

```python
        if "Mastering display metadata" in side_type:
            mastering_display = (
                f"G({_fraction_numerator(entry.get('green_x', ''))},"
                f"{_fraction_numerator(entry.get('green_y', ''))})"
                f"B({_fraction_numerator(entry.get('blue_x', ''))},"
                f"{_fraction_numerator(entry.get('blue_y', ''))})"
                f"R({_fraction_numerator(entry.get('red_x', ''))},"
                f"{_fraction_numerator(entry.get('red_y', ''))})"
                f"WP({_fraction_numerator(entry.get('white_point_x', ''))},"
                f"{_fraction_numerator(entry.get('white_point_y', ''))})"
                f"L({_fraction_numerator(entry.get('max_luminance', ''))},"
                f"{_fraction_numerator(entry.get('min_luminance', ''))})"
            )
```

- [ ] **Step 4: Run fraction tests to verify they pass**

Run: `uv run pytest tests/core/test_detect.py::TestHdrDetectionFractions -v`
Expected: all 4 PASS.

- [ ] **Step 5: Add `probe_hdr_side_data` to Prober protocol**

In `furnace/core/ports.py`, add to the `Prober` protocol class (after `run_idet`):

```python
    def probe_hdr_side_data(self, path: Path) -> list[dict[str, Any]]:
        """Read side_data_list from the first video frame."""
        ...
```

- [ ] **Step 6: Implement `probe_hdr_side_data` in FFmpegAdapter**

In `furnace/adapters/ffmpeg.py`, add after the `probe` method:

```python
    def probe_hdr_side_data(self, path: Path) -> list[dict[str, Any]]:
        """Read side_data_list from the first video frame.

        Uses: ffprobe -v quiet -print_format json -select_streams v:0
              -show_frames -read_intervals "%+#1" path
        """
        cmd = [
            str(self._ffprobe),
            "-v", "quiet",
            "-print_format", "json",
            "-select_streams", "v:0",
            "-show_frames",
            "-read_intervals", "%+#1",
            str(path),
        ]
        logger.debug("probe_hdr_side_data cmd: %s", cmd)
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            logger.warning("probe_hdr_side_data failed (rc=%d), returning []", result.returncode)
            return []
        data: dict[str, Any] = json.loads(result.stdout)
        frames = data.get("frames", [])
        if not frames:
            return []
        return frames[0].get("side_data_list", [])
```

- [ ] **Step 7: Wire `probe_hdr_side_data` into analyzer**

In `furnace/services/analyzer.py`, in `_parse_video_info()`, replace lines 217-219:

```python
        # HDR metadata
        side_data = stream.get("side_data_list")
        hdr = detect_hdr(stream, side_data)
```

with:

```python
        # HDR metadata — try stream side_data first, fall back to first frame
        side_data = stream.get("side_data_list")
        if not side_data:
            side_data = self._prober.probe_hdr_side_data(path) or None
        hdr = detect_hdr(stream, side_data)
```

- [ ] **Step 8: Run all detect tests**

Run: `uv run pytest tests/core/test_detect.py -q`
Expected: all PASS (existing + new fraction tests).

- [ ] **Step 9: Commit**

```
git add furnace/core/detect.py furnace/core/ports.py furnace/adapters/ffmpeg.py furnace/services/analyzer.py tests/core/test_detect.py
git commit -m "feat: fraction parsing in detect_hdr + probe_hdr_side_data for first frame"
```

---

### Task 6: Replace `ColorSpace` with `color_matrix` in `VideoParams`

**Files:**
- Modify: `furnace/core/models.py`
- Modify: `furnace/plan.py`
- Modify: `furnace/services/planner.py`
- Modify: `furnace/adapters/nvencc.py`
- Modify: `furnace/services/analyzer.py`
- Modify: `furnace/core/quality.py`
- Modify: `tests/test_nvencc_cmd.py`
- Modify: `tests/test_plan.py`
- Modify: `tests/core/test_quality.py`
- Modify: `tests/services/test_planner_crop.py`
- Modify: `tests/services/test_planner_dv.py`

This task is a big refactor touching many files. All changes are mechanical — replacing `color_space: ColorSpace` with `color_matrix: str` and wiring `resolve_color_metadata` into the planner.

- [ ] **Step 1: Update `VideoParams` in models.py**

In `furnace/core/models.py:244-264`, replace:
```python
    color_space: ColorSpace
    color_range: str                   # "tv" всегда
    color_transfer: str | None         # raw ffmpeg value для passthrough
    color_primaries: str | None
```
with:
```python
    color_matrix: str                  # resolved --colormatrix value
    color_range: str                   # "tv" всегда
    color_transfer: str                # resolved --transfer value
    color_primaries: str               # resolved --colorprim value
```

- [ ] **Step 2: Update `VideoInfo` in models.py**

In `furnace/core/models.py:184`, replace:
```python
    color_space: ColorSpace | None
```
with:
```python
    color_matrix_raw: str | None       # raw ffprobe color_space (= matrix coefficients)
```

- [ ] **Step 3: Remove `ColorSpace` enum from models.py**

Delete lines 68-72:
```python
class ColorSpace(enum.Enum):
    BT601 = "bt601"
    BT709 = "bt709"
    BT2020 = "bt2020"
```

- [ ] **Step 4: Update plan.py serialization**

In `furnace/plan.py:92-114`, update `_load_video_params`:

Replace `color_space=ColorSpace(raw["color_space"])` with `color_matrix=raw["color_matrix"]`.
Replace `color_transfer=raw.get("color_transfer")` with `color_transfer=raw["color_transfer"]`.
Replace `color_primaries=raw.get("color_primaries")` with `color_primaries=raw["color_primaries"]`.

Remove `ColorSpace` from the import at the top of `plan.py`.

- [ ] **Step 5: Update analyzer.py**

In `furnace/services/analyzer.py`:
- Remove `ColorSpace` from imports
- Remove the `color_space` enum mapping block (lines 195-202)
- Keep `color_space_raw = stream.get("color_space")` but rename to `color_matrix_raw`
- Update `VideoInfo` construction to use `color_matrix_raw=color_matrix_raw` instead of `color_space=color_space`

- [ ] **Step 6: Update planner.py**

In `furnace/services/planner.py:308-364`, update `_build_video_params`:

Replace the call to `determine_color_space()` and the manual color handling with:

```python
from ..core.detect import detect_video_system, resolve_color_metadata

        system = detect_video_system(video.height)
        has_hdr = bool(video.hdr.mastering_display or video.hdr.content_light)
        resolved = resolve_color_metadata(
            matrix_raw=video.color_matrix_raw,
            transfer_raw=video.color_transfer,
            primaries_raw=video.color_primaries,
            system=system,
            has_hdr=has_hdr,
        )
```

Then use `resolved.matrix`, `resolved.transfer`, `resolved.primaries` when constructing `VideoParams`:

```python
        return VideoParams(
            cq=cq,
            crop=crop,
            deinterlace=deinterlace,
            color_matrix=resolved.matrix,
            color_range="tv",
            color_transfer=resolved.transfer,
            color_primaries=resolved.primaries,
            hdr=hdr,
            ...
        )
```

Remove the `determine_color_space` import and the old `color_transfer = video.color_transfer` / `color_primaries = video.color_primaries` lines.

- [ ] **Step 7: Update NVEncC adapter**

In `furnace/adapters/nvencc.py`:
- Remove `ColorSpace` from imports
- Remove `_COLORMATRIX_MAP` (lines 33-38)
- Replace usage at line 237-239:

```python
        matrix = _COLORMATRIX_MAP.get(vp.color_space)
        if matrix:
            cmd += ["--colormatrix", matrix]
```

with:

```python
        cmd += ["--colormatrix", vp.color_matrix]
```

Similarly, replace conditional `--transfer` and `--colorprim`:

```python
        if vp.color_primaries:
            cmd += ["--colorprim", vp.color_primaries]

        if vp.color_transfer:
            cmd += ["--transfer", vp.color_transfer]
```

with (no longer optional):

```python
        cmd += ["--colorprim", vp.color_primaries]
        cmd += ["--transfer", vp.color_transfer]
```

- [ ] **Step 8: Remove `determine_color_space` from quality.py**

In `furnace/core/quality.py`, remove the function `determine_color_space` (lines 68-81) and the `ColorSpace` import.

- [ ] **Step 9: Update test_nvencc_cmd.py**

In `tests/test_nvencc_cmd.py`:
- Replace `ColorSpace` import with nothing (remove it)
- Update `_make_vp` helper: replace `color_space: ColorSpace = ColorSpace.BT2020` with `color_matrix: str = "bt2020nc"`
- Replace all `color_space=ColorSpace.BT2020` with `color_matrix="bt2020nc"`
- Replace `color_space=ColorSpace.BT709` with `color_matrix="bt709"`
- Replace `color_space=ColorSpace.BT601` with `color_matrix="smpte170m"`
- Update `color_transfer` and `color_primaries` params: they are now non-optional `str` (default `"smpte2084"` and `"bt2020"` respectively in the helper)

- [ ] **Step 10: Update test_plan.py**

In `tests/test_plan.py`:
- Replace `ColorSpace` import
- Update `make_video_params` helper: `color_space: ColorSpace = ColorSpace.BT709` → `color_matrix: str = "bt709"`
- Update all usages: `color_space=ColorSpace.BT2020` → `color_matrix="bt2020nc"`
- `color_transfer` and `color_primaries` defaults become `"bt709"` and `"bt709"`

- [ ] **Step 11: Update test_quality.py**

In `tests/core/test_quality.py`:
- Remove `TestDetermineColorSpace` class (lines 200-239)
- Remove `ColorSpace` from imports (keep `CropRect`)

- [ ] **Step 12: Update test_planner_crop.py and test_planner_dv.py**

In `tests/services/test_planner_crop.py`:
- Replace `ColorSpace` import
- Replace `color_space=ColorSpace.BT601` with `color_matrix_raw="bt470bg"` (or `"smpte170m"`)
- Replace `color_space=ColorSpace.BT709` with `color_matrix_raw="bt709"`

In `tests/services/test_planner_dv.py`:
- Replace `ColorSpace` import
- Replace `color_space=ColorSpace.BT2020` with `color_matrix_raw="bt2020nc"`

- [ ] **Step 13: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: all tests PASS.

- [ ] **Step 14: Run quality gates**

```
uv run ruff check furnace/
uv run mypy furnace/ --strict
uv run pytest tests/ -q
```

All three must pass clean.

- [ ] **Step 15: Commit**

```
git add -A
git commit -m "refactor: replace ColorSpace enum with resolved color_matrix/transfer/primaries strings"
```

---

### Task 7: Final verification

- [ ] **Step 1: Run full quality gates one more time**

```
uv run ruff check furnace/
uv run mypy furnace/ --strict
uv run pytest tests/ -q
```

- [ ] **Step 2: Verify the original bug scenario**

Manually confirm the fix path: for a BT.2020 HDR10 source like the Zhili.byli file:
- `color_transfer` would now be read from ffprobe as `"smpte2084"` (not null)
- `resolve_color_metadata` would produce `ResolvedColor(matrix="bt2020nc", transfer="smpte2084", primaries="bt2020")`
- NVEncC would get `--transfer smpte2084 --master-display G(...)... --max-cll 1000,180`
- HDR metadata would be preserved in the output bitstream

No code changes needed — just verify mentally that the pipeline is correct.

- [ ] **Step 3: Commit spec and plan docs if not already committed**

```
git add docs/superpowers/specs/2026-04-08-hdr-metadata-fix-design.md docs/superpowers/plans/2026-04-08-hdr-metadata-fix.md scripts/gen_color_tests.py
git commit -m "docs: HDR metadata fix spec and implementation plan"
```
