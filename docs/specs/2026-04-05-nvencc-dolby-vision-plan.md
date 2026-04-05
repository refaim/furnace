# NVEncC Migration + Dolby Vision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ffmpeg hevc_nvenc with NVEncC as the video encoder, adding Dolby Vision support (Profile 7 FEL, Profile 8 MEL), nnedi deinterlace, and built-in VMAF/SSIM.

**Architecture:** FFmpegAdapter splits: keeps Prober+AudioExtractor. New NVEncCAdapter implements Encoder. New DoviToolAdapter implements DoviProcessor. Executor pipeline gains RPU extraction step. Bloat check and separate VMAF pass removed.

**Tech Stack:** Python 3.12, NVEncC64 (rigaya), dovi_tool (quietvoid), ffmpeg/ffprobe (retained for probe/audio), mkvmerge, pytest, mypy --strict, ruff

**Spec:** `docs/specs/2026-04-05-nvencc-dolby-vision-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `furnace/core/models.py` | Modify | Add DvBlCompatibility, DvMode enums; extend HdrMetadata, VideoParams |
| `furnace/core/ports.py` | Modify | Update Encoder port (EncodeResult, vmaf_enabled); add DoviProcessor port |
| `furnace/core/detect.py` | Modify | Parse DV profile from ffprobe side_data |
| `furnace/adapters/nvencc.py` | Create | NVEncCAdapter implementing Encoder |
| `furnace/adapters/dovi_tool.py` | Create | DoviToolAdapter implementing DoviProcessor |
| `furnace/adapters/ffmpeg.py` | Modify | Remove encode/quality methods, keep probe/audio |
| `furnace/services/analyzer.py` | Modify | Remove DV skip, add HDR10+ error |
| `furnace/services/planner.py` | Modify | Add dv_mode logic to _build_video_params |
| `furnace/services/executor.py` | Modify | Add RPU extraction step, use EncodeResult, remove bloat check + VMAF pass |
| `furnace/plan.py` | Modify | Serialize/deserialize dv_mode, new HdrMetadata fields |
| `furnace/config.py` | Modify | Add nvencc, dovi_tool to ToolPaths |
| `furnace/cli.py` | Modify | Wire NVEncCAdapter + DoviToolAdapter |
| `tests/core/test_detect.py` | Modify | Add DV profile detection tests |
| `tests/core/test_models.py` | Create | Test new enums |
| `tests/test_plan.py` | Modify | Add dv_mode roundtrip tests |
| `tests/test_nvencc_cmd.py` | Create | Test NVEncC command building |
| `tests/test_dovi_tool_cmd.py` | Create | Test dovi_tool command building |
| `tests/services/test_analyzer.py` | Modify | Test DV proceeds, HDR10+ raises |
| `tests/services/test_planner_dv.py` | Create | Test dv_mode assignment in planner |
| `tests/test_ffmpeg_encode_cmd.py` | Delete | Old ffmpeg encode tests, no longer relevant |

---

### Task 1: New enums — DvBlCompatibility, DvMode

**Files:**
- Modify: `furnace/core/models.py:67-72` (after ColorSpace)
- Create: `tests/core/test_models.py`

- [ ] **Step 1: Write tests for new enums**

```python
# tests/core/test_models.py
from __future__ import annotations

from furnace.core.models import DvBlCompatibility, DvMode


class TestDvBlCompatibility:
    def test_values(self) -> None:
        assert DvBlCompatibility.NONE == 0
        assert DvBlCompatibility.HDR10 == 1
        assert DvBlCompatibility.SDR == 2
        assert DvBlCompatibility.HLG == 4

    def test_from_int(self) -> None:
        assert DvBlCompatibility(1) == DvBlCompatibility.HDR10
        assert DvBlCompatibility(4) == DvBlCompatibility.HLG


class TestDvMode:
    def test_values(self) -> None:
        assert DvMode.COPY == 0
        assert DvMode.TO_8_1 == 2

    def test_from_int(self) -> None:
        assert DvMode(0) == DvMode.COPY
        assert DvMode(2) == DvMode.TO_8_1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_models.py -v`
Expected: ImportError — DvBlCompatibility, DvMode not defined

- [ ] **Step 3: Add enums to models.py**

In `furnace/core/models.py`, after the `ColorSpace` enum (line 71), add:

```python
class DvBlCompatibility(enum.IntEnum):
    """Dolby Vision base layer compatibility."""
    NONE = 0    # no fallback (Profile 5)
    HDR10 = 1   # HDR10 fallback
    SDR = 2     # SDR fallback
    HLG = 4     # HLG fallback


class DvMode(enum.IntEnum):
    """DV RPU extraction mode. Values match dovi_tool -m flag."""
    COPY = 0      # extract RPU as-is (no -m flag)
    TO_8_1 = 2    # convert P7 FEL -> P8.1 (-m 2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_models.py -v`
Expected: PASS

- [ ] **Step 5: Extend HdrMetadata**

In `furnace/core/models.py`, update HdrMetadata (line 79-85):

```python
@dataclass(frozen=True)
class HdrMetadata:
    """HDR10 static metadata. None означает отсутствие."""
    mastering_display: str | None = None
    content_light: str | None = None
    is_dolby_vision: bool = False
    is_hdr10_plus: bool = False
    dv_profile: int | None = None
    dv_bl_compatibility: DvBlCompatibility | None = None
```

- [ ] **Step 6: Add dv_mode to VideoParams**

In `furnace/core/models.py`, add to VideoParams (after sar_den, line 238):

```python
    dv_mode: DvMode | None = None         # None=no DV, COPY=as-is, TO_8_1=P7->P8.1
```

- [ ] **Step 7: Update VideoParams comment**

Change the `deinterlace` comment from `нужен ли bwdif_cuda` to `нужен ли деинтерлейс`.

- [ ] **Step 8: Run quality gates**

Run: `uv run ruff check furnace/ && uv run mypy furnace/ --strict && uv run pytest tests/ -q`
Expected: all pass (existing tests may need minor updates for new default fields)

- [ ] **Step 9: Commit**

```bash
git add furnace/core/models.py tests/core/test_models.py
git commit -m "feat: add DvBlCompatibility, DvMode enums; extend HdrMetadata and VideoParams"
```

---

### Task 2: DV profile detection in detect_hdr

**Files:**
- Modify: `furnace/core/detect.py:119-164`
- Modify: `tests/core/test_detect.py`

- [ ] **Step 1: Write tests for DV profile parsing**

Add to `tests/core/test_detect.py`:

```python
from furnace.core.models import DvBlCompatibility


class TestDvProfileDetection:
    def test_dv_profile_from_side_data(self) -> None:
        """Parse dv_profile from Dolby Vision configuration."""
        side_data = [{
            "side_data_type": "Dolby Vision configuration record",
            "dv_profile": 8,
            "dv_bl_signal_compatibility_id": 1,
        }]
        result = detect_hdr({}, side_data)
        assert result.is_dolby_vision
        assert result.dv_profile == 8
        assert result.dv_bl_compatibility == DvBlCompatibility.HDR10

    def test_dv_profile7_fel(self) -> None:
        """Profile 7 FEL with HDR10 compatibility."""
        side_data = [{
            "side_data_type": "Dolby Vision configuration record",
            "dv_profile": 7,
            "dv_bl_signal_compatibility_id": 1,
        }]
        result = detect_hdr({}, side_data)
        assert result.dv_profile == 7
        assert result.dv_bl_compatibility == DvBlCompatibility.HDR10

    def test_dv_profile5_no_compat(self) -> None:
        """Profile 5 with no BL compatibility."""
        side_data = [{
            "side_data_type": "Dolby Vision configuration record",
            "dv_profile": 5,
            "dv_bl_signal_compatibility_id": 0,
        }]
        result = detect_hdr({}, side_data)
        assert result.dv_profile == 5
        assert result.dv_bl_compatibility == DvBlCompatibility.NONE

    def test_dv_codec_name_no_side_data_no_profile(self) -> None:
        """DV detected via codec_name but no side_data -> profile is None."""
        result = detect_hdr({"codec_name": "dvhe"}, [])
        assert result.is_dolby_vision
        assert result.dv_profile is None
        assert result.dv_bl_compatibility is None

    def test_no_dv_fields_none(self) -> None:
        """Regular HDR10 -> dv fields are None."""
        side_data = [{
            "side_data_type": "Mastering display metadata",
            "green_x": "0.265", "green_y": "0.690",
            "blue_x": "0.150", "blue_y": "0.060",
            "red_x": "0.680", "red_y": "0.320",
            "white_point_x": "0.3127", "white_point_y": "0.3290",
            "max_luminance": "1000", "min_luminance": "0.005",
        }]
        result = detect_hdr({}, side_data)
        assert result.dv_profile is None
        assert result.dv_bl_compatibility is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_detect.py::TestDvProfileDetection -v`
Expected: FAIL — dv_profile attribute not populated

- [ ] **Step 3: Update detect_hdr to parse DV profile**

In `furnace/core/detect.py`, update the `detect_hdr` function. Add import for `DvBlCompatibility` at top. Inside the function, add variables `dv_profile` and `dv_bl_compat`, parse them from "Dolby Vision configuration" side_data entry:

```python
from .models import AudioCodecId, DvBlCompatibility, HdrMetadata, SubtitleCodecId, Track


def detect_hdr(stream_data: dict[str, Any], side_data: list[dict[str, Any]] | None) -> HdrMetadata:
    mastering_display: str | None = None
    content_light: str | None = None
    is_dolby_vision: bool = False
    is_hdr10_plus: bool = False
    dv_profile: int | None = None
    dv_bl_compatibility: DvBlCompatibility | None = None

    sd = side_data or []

    for entry in sd:
        side_type = entry.get("side_data_type", "")

        if "Mastering display metadata" in side_type:
            mastering_display = (
                f"G({entry.get('green_x', '')},{entry.get('green_y', '')})"
                f"B({entry.get('blue_x', '')},{entry.get('blue_y', '')})"
                f"R({entry.get('red_x', '')},{entry.get('red_y', '')})"
                f"WP({entry.get('white_point_x', '')},{entry.get('white_point_y', '')})"
                f"L({entry.get('max_luminance', '')},{entry.get('min_luminance', '')})"
            )

        elif "Content light level metadata" in side_type:
            max_cll = entry.get("max_content", "")
            max_fall = entry.get("max_average", "")
            content_light = f"MaxCLL={max_cll},MaxFALL={max_fall}"

        elif "Dolby Vision configuration" in side_type:
            is_dolby_vision = True
            raw_profile = entry.get("dv_profile")
            if raw_profile is not None:
                dv_profile = int(raw_profile)
            raw_compat = entry.get("dv_bl_signal_compatibility_id")
            if raw_compat is not None:
                try:
                    dv_bl_compatibility = DvBlCompatibility(int(raw_compat))
                except ValueError:
                    pass

        elif "HDR10+" in side_type or "SMPTE ST 2094" in side_type:
            is_hdr10_plus = True

    codec_name = stream_data.get("codec_name", "")
    if codec_name in ("dvhe", "dvh1"):
        is_dolby_vision = True

    return HdrMetadata(
        mastering_display=mastering_display,
        content_light=content_light,
        is_dolby_vision=is_dolby_vision,
        is_hdr10_plus=is_hdr10_plus,
        dv_profile=dv_profile,
        dv_bl_compatibility=dv_bl_compatibility,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/test_detect.py -v`
Expected: all pass (new and old)

- [ ] **Step 5: Run quality gates**

Run: `uv run ruff check furnace/ && uv run mypy furnace/ --strict && uv run pytest tests/ -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add furnace/core/detect.py tests/core/test_detect.py
git commit -m "feat: parse DV profile and BL compatibility from ffprobe side_data"
```

---

### Task 3: Update Encoder port and add DoviProcessor port

**Files:**
- Modify: `furnace/core/ports.py:31-66`
- Modify: `furnace/core/models.py` (add EncodeResult)

- [ ] **Step 1: Add EncodeResult to models.py**

In `furnace/core/models.py`, after the `CropRect` class (line 94), add:

```python
@dataclass(frozen=True)
class EncodeResult:
    """Result of video encoding."""
    return_code: int
    encoder_settings: str
    vmaf_score: float | None = None
    ssim_score: float | None = None
```

- [ ] **Step 2: Update Encoder protocol in ports.py**

Replace the Encoder protocol (lines 31-66) with:

```python
@runtime_checkable
class Encoder(Protocol):
    """Video encoding via NVEncC."""

    def encode(
        self,
        input_path: Path,
        output_path: Path,
        video_params: VideoParams,
        source_size: int,
        on_progress: Callable[[float, str], None] | None = None,
        vmaf_enabled: bool = False,
        rpu_path: Path | None = None,
    ) -> EncodeResult:
        """Encode video. Returns EncodeResult with return code, settings, and optional metrics.

        source_size is passed for informational purposes.
        on_progress callback receives (progress_pct, status_line).
        vmaf_enabled: compute VMAF/SSIM during encode if True.
        rpu_path: path to extracted DV RPU file (None if no DV).
        """
        ...
```

Add EncodeResult to the import from `.models`.

- [ ] **Step 3: Add DoviProcessor protocol**

After the Encoder protocol, add:

```python
@runtime_checkable
class DoviProcessor(Protocol):
    """Extract/convert Dolby Vision RPU metadata via dovi_tool."""

    def extract_rpu(
        self,
        input_path: Path,
        output_rpu: Path,
        mode: DvMode,
    ) -> int:
        """Extract RPU from HEVC stream.

        mode=COPY: extract as-is (no -m flag).
        mode=TO_8_1: convert P7 FEL -> P8.1 (-m 2).
        Returns exit code.
        """
        ...
```

Add DvMode to the import from `.models`.

- [ ] **Step 4: Run quality gates**

Run: `uv run ruff check furnace/ && uv run mypy furnace/ --strict && uv run pytest tests/ -q`
Expected: some tests may fail due to FFmpegAdapter no longer matching Encoder protocol (encode returns tuple instead of EncodeResult). That's expected — we fix FFmpegAdapter in Task 7.

- [ ] **Step 5: Commit**

```bash
git add furnace/core/models.py furnace/core/ports.py
git commit -m "feat: update Encoder port to return EncodeResult; add DoviProcessor port"
```

---

### Task 4: Analyzer — remove DV skip, add HDR10+ error

**Files:**
- Modify: `furnace/services/analyzer.py:76-82`
- Modify: `tests/services/test_analyzer.py`

- [ ] **Step 1: Write tests for new analyzer behavior**

Add to `tests/services/test_analyzer.py` (or update existing DV/HDR10+ tests). First read the existing test file to understand the mock structure, then add:

```python
class TestDvHandling:
    def test_dv_content_not_skipped(self) -> None:
        """DV content should be analyzed, not skipped."""
        # Create a mock prober that returns DV content
        # (follow existing mock patterns in the file)
        # Assert that analyze() returns a Movie, not None

    def test_hdr10_plus_raises_error(self) -> None:
        """HDR10+ content should raise ValueError."""
        # Create a mock prober that returns HDR10+ content
        # Assert ValueError is raised
```

The exact mock setup depends on patterns in the existing test file — read it first and follow the same approach.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/services/test_analyzer.py::TestDvHandling -v`
Expected: FAIL — DV still skipped, HDR10+ doesn't raise

- [ ] **Step 3: Update analyzer.py**

Replace lines 76-82 in `furnace/services/analyzer.py`:

```python
        # Check HDR10+ — not supported, raise error
        if video_info.hdr.is_hdr10_plus:
            raise ValueError(f"HDR10+ not supported: {main_file.name}")
        # DV content proceeds to planning (no skip)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/services/test_analyzer.py -v`
Expected: all pass

- [ ] **Step 5: Run quality gates**

Run: `uv run ruff check furnace/ && uv run mypy furnace/ --strict && uv run pytest tests/ -q`

- [ ] **Step 6: Commit**

```bash
git add furnace/services/analyzer.py tests/services/test_analyzer.py
git commit -m "feat: allow DV content through analyzer; raise on HDR10+"
```

---

### Task 5: Planner — add dv_mode logic

**Files:**
- Modify: `furnace/services/planner.py:303-346`
- Create: `tests/services/test_planner_dv.py`

- [ ] **Step 1: Write tests for planner DV logic**

```python
# tests/services/test_planner_dv.py
from __future__ import annotations

from pathlib import Path

import pytest

from furnace.core.models import (
    ColorSpace,
    DvBlCompatibility,
    DvMode,
    HdrMetadata,
    VideoInfo,
)
from furnace.services.planner import PlannerService


def _make_video(
    hdr: HdrMetadata | None = None,
    width: int = 3840,
    height: int = 2160,
) -> VideoInfo:
    if hdr is None:
        hdr = HdrMetadata()
    return VideoInfo(
        index=0,
        codec_name="hevc",
        width=width,
        height=height,
        pixel_area=width * height,
        fps_num=24000,
        fps_den=1001,
        duration_s=7200.0,
        interlaced=False,
        color_space=ColorSpace.BT2020,
        color_range="tv",
        color_transfer="smpte2084",
        color_primaries="bt2020",
        pix_fmt="yuv420p10le",
        hdr=hdr,
        source_file=Path("/src/movie.mkv"),
        bitrate=80_000_000,
    )


class TestPlannerDvMode:
    def test_no_dv_mode_none(self) -> None:
        """Plain HDR10 -> dv_mode is None."""
        hdr = HdrMetadata(
            mastering_display="G(0.265,0.690)B(0.150,0.060)R(0.680,0.320)WP(0.3127,0.3290)L(1000,0.005)",
            content_light="MaxCLL=1000,MaxFALL=400",
        )
        video = _make_video(hdr=hdr)
        planner = PlannerService(prober=None, previewer=None)  # type: ignore[arg-type]
        vp = planner._build_video_params(video, crop=None)
        assert vp.dv_mode is None

    def test_dv_profile8_mode_copy(self) -> None:
        """DV Profile 8 -> dv_mode = COPY."""
        hdr = HdrMetadata(
            mastering_display="G(0.265,0.690)B(0.150,0.060)R(0.680,0.320)WP(0.3127,0.3290)L(1000,0.005)",
            content_light="MaxCLL=1000,MaxFALL=400",
            is_dolby_vision=True,
            dv_profile=8,
            dv_bl_compatibility=DvBlCompatibility.HDR10,
        )
        video = _make_video(hdr=hdr)
        planner = PlannerService(prober=None, previewer=None)  # type: ignore[arg-type]
        vp = planner._build_video_params(video, crop=None)
        assert vp.dv_mode == DvMode.COPY

    def test_dv_profile7_mode_to_8_1(self) -> None:
        """DV Profile 7 FEL -> dv_mode = TO_8_1."""
        hdr = HdrMetadata(
            mastering_display="G(0.265,0.690)B(0.150,0.060)R(0.680,0.320)WP(0.3127,0.3290)L(1000,0.005)",
            content_light="MaxCLL=1000,MaxFALL=400",
            is_dolby_vision=True,
            dv_profile=7,
            dv_bl_compatibility=DvBlCompatibility.HDR10,
        )
        video = _make_video(hdr=hdr)
        planner = PlannerService(prober=None, previewer=None)  # type: ignore[arg-type]
        vp = planner._build_video_params(video, crop=None)
        assert vp.dv_mode == DvMode.TO_8_1

    def test_dv_profile5_mode_copy(self) -> None:
        """DV Profile 5 -> dv_mode = COPY."""
        hdr = HdrMetadata(
            is_dolby_vision=True,
            dv_profile=5,
            dv_bl_compatibility=DvBlCompatibility.NONE,
        )
        video = _make_video(hdr=hdr)
        planner = PlannerService(prober=None, previewer=None)  # type: ignore[arg-type]
        vp = planner._build_video_params(video, crop=None)
        assert vp.dv_mode == DvMode.COPY

    def test_hdr10_plus_raises(self) -> None:
        """HDR10+ in planner -> ValueError (should be caught by analyzer, but guard)."""
        hdr = HdrMetadata(is_hdr10_plus=True)
        video = _make_video(hdr=hdr)
        planner = PlannerService(prober=None, previewer=None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="HDR10\\+"):
            planner._build_video_params(video, crop=None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/services/test_planner_dv.py -v`
Expected: FAIL — dv_mode not set

- [ ] **Step 3: Update _build_video_params**

In `furnace/services/planner.py`, update `_build_video_params` (line 303+). Add import for `DvMode` at the top. Replace the HDR/DV section:

```python
    def _build_video_params(self, video: VideoInfo, crop: CropRect | None) -> VideoParams:
        """CQ interpolation, GOP calc, colorspace determination, deinterlace detection."""
        # Use cropped area for CQ if crop is applied
        if crop is not None:
            pixel_area = crop.w * crop.h
        else:
            pixel_area = video.pixel_area

        cq = interpolate_cq(pixel_area)
        gop = calculate_gop(video.fps_num, video.fps_den)

        color_space = determine_color_space(
            video.width, video.height,
            video.color_space.value if video.color_space is not None else None,
        )

        deinterlace = video.interlaced

        # HDR10+ guard (should be caught by analyzer, but double-check)
        if video.hdr.is_hdr10_plus:
            raise ValueError(f"HDR10+ not supported: {video.source_file.name}")

        # DV mode
        dv_mode: DvMode | None = None
        if video.hdr.is_dolby_vision:
            if video.hdr.dv_profile == 7:
                dv_mode = DvMode.TO_8_1
            else:
                dv_mode = DvMode.COPY

        # HDR metadata passthrough
        hdr = video.hdr if (video.hdr.mastering_display or video.hdr.content_light) else None

        # Color info passthrough
        color_transfer = video.color_transfer
        color_primaries = video.color_primaries

        return VideoParams(
            cq=cq,
            crop=crop,
            deinterlace=deinterlace,
            color_space=color_space,
            color_range="tv",
            color_transfer=color_transfer,
            color_primaries=color_primaries,
            hdr=hdr,
            gop=gop,
            fps_num=video.fps_num,
            fps_den=video.fps_den,
            source_width=video.width,
            source_height=video.height,
            source_codec=video.codec_name,
            source_bitrate=video.bitrate,
            sar_num=video.sar_num,
            sar_den=video.sar_den,
            dv_mode=dv_mode,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/services/test_planner_dv.py -v`
Expected: all pass

- [ ] **Step 5: Run quality gates**

Run: `uv run ruff check furnace/ && uv run mypy furnace/ --strict && uv run pytest tests/ -q`

- [ ] **Step 6: Commit**

```bash
git add furnace/services/planner.py tests/services/test_planner_dv.py
git commit -m "feat: set dv_mode in planner based on DV profile"
```

---

### Task 6: JSON plan — serialize/deserialize dv_mode and new HdrMetadata fields

**Files:**
- Modify: `furnace/plan.py:68-98`
- Modify: `tests/test_plan.py`

- [ ] **Step 1: Write roundtrip tests for dv_mode**

Add to `tests/test_plan.py`:

```python
from furnace.core.models import DvBlCompatibility, DvMode


class TestPlanDvModeRoundtrip:
    def test_roundtrip_dv_mode_to_8_1(self, tmp_path) -> None:
        """dv_mode=TO_8_1 survives roundtrip."""
        vp = make_video_params()
        vp_dict = dataclasses.asdict(vp)
        vp_dict["dv_mode"] = DvMode.TO_8_1
        # Reconstruct with dv_mode
        from furnace.core.models import VideoParams as VP
        vp_with_dv = VP(**{**vp_dict, "crop": vp.crop, "hdr": vp.hdr, "color_space": vp.color_space})
        job = Job(
            id="dv-job", source_files=["/src/dv.mkv"], output_file="/out/dv.mkv",
            video_params=vp_with_dv, audio=[], subtitles=[], attachments=[],
            copy_chapters=False, chapters_source=None, status=JobStatus.PENDING, source_size=0,
        )
        plan = make_plan(jobs=[job])
        plan_path = tmp_path / "plan.json"
        save_plan(plan, plan_path)
        loaded = load_plan(plan_path)
        assert loaded.jobs[0].video_params.dv_mode == DvMode.TO_8_1

    def test_roundtrip_dv_mode_none(self, tmp_path) -> None:
        """dv_mode=None survives roundtrip."""
        plan = make_plan()
        plan_path = tmp_path / "plan.json"
        save_plan(plan, plan_path)
        loaded = load_plan(plan_path)
        assert loaded.jobs[0].video_params.dv_mode is None

    def test_roundtrip_hdr_with_dv_fields(self, tmp_path) -> None:
        """HdrMetadata with dv_profile and dv_bl_compatibility survives roundtrip."""
        hdr = HdrMetadata(
            mastering_display="G(0.265,0.69)B(0.15,0.06)R(0.68,0.32)WP(0.3127,0.329)L(1000,0.005)",
            content_light="MaxCLL=1000,MaxFALL=400",
            is_dolby_vision=True,
            dv_profile=8,
            dv_bl_compatibility=DvBlCompatibility.HDR10,
        )
        vp = make_video_params(color_space=ColorSpace.BT2020, hdr=hdr)
        # Need to set dv_mode too
        import dataclasses as dc
        vp = dc.replace(vp, dv_mode=DvMode.COPY)
        job = Job(
            id="dv-hdr-job", source_files=["/src/dv.mkv"], output_file="/out/dv.mkv",
            video_params=vp, audio=[], subtitles=[], attachments=[],
            copy_chapters=False, chapters_source=None, status=JobStatus.PENDING, source_size=0,
        )
        plan = make_plan(jobs=[job])
        plan_path = tmp_path / "plan.json"
        save_plan(plan, plan_path)
        loaded = load_plan(plan_path)
        loaded_hdr = loaded.jobs[0].video_params.hdr
        assert loaded_hdr is not None
        assert loaded_hdr.is_dolby_vision is True
        assert loaded_hdr.dv_profile == 8
        assert loaded_hdr.dv_bl_compatibility == DvBlCompatibility.HDR10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plan.py::TestPlanDvModeRoundtrip -v`
Expected: FAIL — _load_hdr and _load_video_params don't handle new fields

- [ ] **Step 3: Update _load_hdr in plan.py**

```python
def _load_hdr(raw: dict[str, Any] | None) -> HdrMetadata | None:
    if raw is None:
        return None
    dv_bl_raw = raw.get("dv_bl_compatibility")
    dv_bl: DvBlCompatibility | None = None
    if dv_bl_raw is not None:
        dv_bl = DvBlCompatibility(dv_bl_raw)
    return HdrMetadata(
        mastering_display=raw.get("mastering_display"),
        content_light=raw.get("content_light"),
        is_dolby_vision=raw.get("is_dolby_vision", False),
        is_hdr10_plus=raw.get("is_hdr10_plus", False),
        dv_profile=raw.get("dv_profile"),
        dv_bl_compatibility=dv_bl,
    )
```

Add `DvBlCompatibility, DvMode` to the imports from `furnace.core.models`.

- [ ] **Step 4: Update _load_video_params in plan.py**

Add at the end of `_load_video_params`:

```python
    dv_mode_raw = raw.get("dv_mode")
    dv_mode: DvMode | None = DvMode(dv_mode_raw) if dv_mode_raw is not None else None
```

And add `dv_mode=dv_mode` to the VideoParams constructor call.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_plan.py -v`
Expected: all pass (new and old)

- [ ] **Step 6: Run quality gates**

Run: `uv run ruff check furnace/ && uv run mypy furnace/ --strict && uv run pytest tests/ -q`

- [ ] **Step 7: Commit**

```bash
git add furnace/plan.py tests/test_plan.py
git commit -m "feat: serialize/deserialize dv_mode and DV HdrMetadata fields in JSON plan"
```

---

### Task 7: Config — add nvencc and dovi_tool paths

**Files:**
- Modify: `furnace/config.py`

- [ ] **Step 1: Update ToolPaths dataclass**

Add two new fields to `ToolPaths`:

```python
@dataclass(frozen=True)
class ToolPaths:
    ffmpeg: Path
    ffprobe: Path
    mkvmerge: Path
    mkvpropedit: Path
    mkclean: Path
    eac3to: Path
    qaac64: Path
    mpv: Path
    makemkvcon: Path
    nvencc: Path
    dovi_tool: Path | None  # optional: only needed for DV content
```

- [ ] **Step 2: Update load_config**

Update `tool_names` tuple and handling. `nvencc` is mandatory, `dovi_tool` is optional:

```python
    mandatory_tools = ("ffmpeg", "ffprobe", "mkvmerge", "mkvpropedit", "mkclean", "eac3to", "qaac64", "mpv", "makemkvcon", "nvencc")
    resolved: dict[str, Path] = {}

    for name in mandatory_tools:
        if name not in tools_section:
            raise KeyError(f"Missing required key [tools].{name} in config")
        tool_path = Path(tools_section[name])
        if not tool_path.exists():
            raise FileNotFoundError(
                f"Tool '{name}' not found at path: {tool_path}"
            )
        resolved[name] = tool_path

    # Optional tools
    dovi_tool_path: Path | None = None
    if "dovi_tool" in tools_section:
        dovi_tool_path = Path(tools_section["dovi_tool"])
        if not dovi_tool_path.exists():
            raise FileNotFoundError(
                f"Tool 'dovi_tool' not found at path: {dovi_tool_path}"
            )

    return ToolPaths(
        ffmpeg=resolved["ffmpeg"],
        ffprobe=resolved["ffprobe"],
        mkvmerge=resolved["mkvmerge"],
        mkvpropedit=resolved["mkvpropedit"],
        mkclean=resolved["mkclean"],
        eac3to=resolved["eac3to"],
        qaac64=resolved["qaac64"],
        mpv=resolved["mpv"],
        makemkvcon=resolved["makemkvcon"],
        nvencc=resolved["nvencc"],
        dovi_tool=dovi_tool_path,
    )
```

- [ ] **Step 3: Run quality gates**

Run: `uv run ruff check furnace/ && uv run mypy furnace/ --strict && uv run pytest tests/ -q`
Note: some tests may fail if they mock config — fix as needed.

- [ ] **Step 4: Commit**

```bash
git add furnace/config.py
git commit -m "feat: add nvencc (mandatory) and dovi_tool (optional) to config"
```

---

### Task 8: DoviToolAdapter

**Files:**
- Create: `furnace/adapters/dovi_tool.py`
- Create: `tests/test_dovi_tool_cmd.py`

- [ ] **Step 1: Write tests for dovi_tool command building**

```python
# tests/test_dovi_tool_cmd.py
from __future__ import annotations

from pathlib import Path

from furnace.adapters.dovi_tool import DoviToolAdapter
from furnace.core.models import DvMode


class TestDoviToolCommand:
    def test_extract_rpu_copy_mode(self) -> None:
        """COPY mode: no -m flag."""
        adapter = DoviToolAdapter(Path("dovi_tool.exe"))
        cmd = adapter._build_extract_cmd(
            Path("input.mkv"), Path("RPU.bin"), DvMode.COPY,
        )
        assert cmd[0] == "dovi_tool.exe"
        assert "-m" not in cmd
        assert "extract-rpu" in cmd
        assert "input.mkv" in [str(c) for c in cmd]

    def test_extract_rpu_to_8_1_mode(self) -> None:
        """TO_8_1 mode: -m 2 flag present."""
        adapter = DoviToolAdapter(Path("dovi_tool.exe"))
        cmd = adapter._build_extract_cmd(
            Path("input.mkv"), Path("RPU.bin"), DvMode.TO_8_1,
        )
        assert "-m" in [str(c) for c in cmd]
        m_idx = [str(c) for c in cmd].index("-m")
        assert cmd[m_idx + 1] == "2"

    def test_output_flag(self) -> None:
        """Output path passed via -o."""
        adapter = DoviToolAdapter(Path("dovi_tool.exe"))
        cmd = adapter._build_extract_cmd(
            Path("input.mkv"), Path("/tmp/RPU.bin"), DvMode.COPY,
        )
        str_cmd = [str(c) for c in cmd]
        o_idx = str_cmd.index("-o")
        assert str_cmd[o_idx + 1] == "/tmp/RPU.bin"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dovi_tool_cmd.py -v`
Expected: ImportError — DoviToolAdapter doesn't exist

- [ ] **Step 3: Implement DoviToolAdapter**

```python
# furnace/adapters/dovi_tool.py
from __future__ import annotations

import logging
from pathlib import Path

from ..core.models import DvMode
from ._subprocess import OutputCallback, run_tool

logger = logging.getLogger(__name__)


class DoviToolAdapter:
    """Implements DoviProcessor port via dovi_tool CLI."""

    def __init__(
        self, dovi_tool_path: Path,
        on_output: OutputCallback = None,
        log_dir: Path | None = None,
    ) -> None:
        self._dovi_tool = dovi_tool_path
        self._on_output = on_output
        self._log_dir = log_dir

    def set_log_dir(self, log_dir: Path | None) -> None:
        self._log_dir = log_dir

    def _build_extract_cmd(
        self, input_path: Path, output_rpu: Path, mode: DvMode,
    ) -> list[str | Path]:
        cmd: list[str | Path] = [self._dovi_tool]
        if mode == DvMode.TO_8_1:
            cmd += ["-m", "2"]
        cmd += ["extract-rpu", input_path, "-o", output_rpu]
        return cmd

    def extract_rpu(
        self, input_path: Path, output_rpu: Path, mode: DvMode,
    ) -> int:
        """Extract RPU from HEVC stream."""
        cmd = self._build_extract_cmd(input_path, output_rpu, mode)
        logger.debug("dovi_tool cmd: %s", " ".join(str(c) for c in cmd))
        log_path = self._log_dir / "dovi_tool_extract.log" if self._log_dir else None
        rc, _out = run_tool(cmd, on_output=self._on_output, log_path=log_path)
        return rc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dovi_tool_cmd.py -v`
Expected: all pass

- [ ] **Step 5: Run quality gates**

Run: `uv run ruff check furnace/ && uv run mypy furnace/ --strict && uv run pytest tests/ -q`

- [ ] **Step 6: Commit**

```bash
git add furnace/adapters/dovi_tool.py tests/test_dovi_tool_cmd.py
git commit -m "feat: add DoviToolAdapter for RPU extraction"
```

---

### Task 9: NVEncCAdapter — command building

This is the core task. All NVEncC flags MUST be researched against the NVEncC documentation during implementation. The code below is a starting point — the implementer must read the NVEncC docs and verify/adjust every flag.

**Files:**
- Create: `furnace/adapters/nvencc.py`
- Create: `tests/test_nvencc_cmd.py`

- [ ] **Step 1: Research NVEncC flags**

Before writing any code, read the NVEncC documentation thoroughly:
- https://github.com/rigaya/NVEnc/blob/master/NVEncC_Options.en.md

Research and document (in comments or a scratch file) the correct flags for:
- Encode settings: `--preset`, `--tune`, `--qvbr`, `--aq`, `--aq-temporal`, `--lookahead`, `--multipass`
- Color: `--colorprim`, `--transfer`, `--colormatrix`, `--colorrange` — test `auto` vs explicit
- HDR: `--max-cll`, `--master-display` — test `copy` mode
- DV: `--dolby-vision-rpu`, `--dolby-vision-profile`
- Filters: `--crop`, `--vpp-nnedi`, `--output-res`, `--sar`
- Output format and progress
- VMAF/SSIM: `--vmaf`, `--ssim`
- Profile/level: `--profile`, `--output-depth`, `--tier`

- [ ] **Step 2: Write tests for NVEncC command building**

```python
# tests/test_nvencc_cmd.py
from __future__ import annotations

from pathlib import Path

from furnace.adapters.nvencc import NVEncCAdapter
from furnace.core.models import ColorSpace, CropRect, DvMode, HdrMetadata, VideoParams


def _make_vp(
    source_codec: str = "hevc",
    crop: CropRect | None = None,
    deinterlace: bool = False,
    cq: int = 31,
    color_space: ColorSpace = ColorSpace.BT2020,
    color_transfer: str | None = "smpte2084",
    color_primaries: str | None = "bt2020",
    hdr: HdrMetadata | None = None,
    dv_mode: DvMode | None = None,
) -> VideoParams:
    return VideoParams(
        cq=cq,
        crop=crop,
        deinterlace=deinterlace,
        color_space=color_space,
        color_range="tv",
        color_transfer=color_transfer,
        color_primaries=color_primaries,
        hdr=hdr,
        gop=120,
        fps_num=24000,
        fps_den=1001,
        source_width=3840,
        source_height=2160,
        source_codec=source_codec,
        source_bitrate=80_000_000,
        dv_mode=dv_mode,
    )


class TestNVEncCBasicCommand:
    def test_hevc_encoder(self) -> None:
        adapter = NVEncCAdapter(Path("NVEncC64.exe"))
        cmd = adapter._build_encode_cmd(Path("in.mkv"), Path("out.hevc"), _make_vp())
        str_cmd = [str(c) for c in cmd]
        assert "-c" in str_cmd
        assert "hevc" in str_cmd

    def test_qvbr_value(self) -> None:
        adapter = NVEncCAdapter(Path("NVEncC64.exe"))
        cmd = adapter._build_encode_cmd(Path("in.mkv"), Path("out.hevc"), _make_vp(cq=25))
        str_cmd = [str(c) for c in cmd]
        idx = str_cmd.index("--qvbr")
        assert str_cmd[idx + 1] == "25"


class TestNVEncCCrop:
    def test_crop_format_ltrb(self) -> None:
        """Crop converted from CropRect(w,h,x,y) to --crop L,T,R,B."""
        crop = CropRect(w=3680, h=2076, x=80, y=42)
        adapter = NVEncCAdapter(Path("NVEncC64.exe"))
        cmd = adapter._build_encode_cmd(
            Path("in.mkv"), Path("out.hevc"),
            _make_vp(crop=crop),
        )
        str_cmd = [str(c) for c in cmd]
        crop_idx = str_cmd.index("--crop")
        # left=80, top=42, right=3840-80-3680=80, bottom=2160-42-2076=42
        assert str_cmd[crop_idx + 1] == "80,42,80,42"


class TestNVEncCDeinterlace:
    def test_nnedi_when_deinterlace(self) -> None:
        adapter = NVEncCAdapter(Path("NVEncC64.exe"))
        cmd = adapter._build_encode_cmd(
            Path("in.mkv"), Path("out.hevc"),
            _make_vp(deinterlace=True),
        )
        str_cmd = [str(c) for c in cmd]
        assert "--vpp-nnedi" in str_cmd

    def test_no_nnedi_when_progressive(self) -> None:
        adapter = NVEncCAdapter(Path("NVEncC64.exe"))
        cmd = adapter._build_encode_cmd(
            Path("in.mkv"), Path("out.hevc"),
            _make_vp(deinterlace=False),
        )
        str_cmd = [str(c) for c in cmd]
        assert "--vpp-nnedi" not in str_cmd


class TestNVEncCDolbyVision:
    def test_dv_rpu_path(self) -> None:
        adapter = NVEncCAdapter(Path("NVEncC64.exe"))
        vp = _make_vp(dv_mode=DvMode.COPY)
        cmd = adapter._build_encode_cmd(
            Path("in.mkv"), Path("out.hevc"), vp,
            rpu_path=Path("/tmp/RPU.bin"),
        )
        str_cmd = [str(c) for c in cmd]
        assert "--dolby-vision-rpu" in str_cmd
        rpu_idx = str_cmd.index("--dolby-vision-rpu")
        assert str_cmd[rpu_idx + 1] == "/tmp/RPU.bin"
        assert "--dolby-vision-profile" in str_cmd

    def test_no_dv_flags_when_no_dv(self) -> None:
        adapter = NVEncCAdapter(Path("NVEncC64.exe"))
        cmd = adapter._build_encode_cmd(
            Path("in.mkv"), Path("out.hevc"),
            _make_vp(dv_mode=None),
        )
        str_cmd = [str(c) for c in cmd]
        assert "--dolby-vision-rpu" not in str_cmd
        assert "--dolby-vision-profile" not in str_cmd


class TestNVEncCSar:
    def test_sar_correction(self) -> None:
        vp = _make_vp()
        # Override SAR
        vp.sar_num = 64
        vp.sar_den = 45
        adapter = NVEncCAdapter(Path("NVEncC64.exe"))
        cmd = adapter._build_encode_cmd(Path("in.mkv"), Path("out.hevc"), vp)
        str_cmd = [str(c) for c in cmd]
        assert "--output-res" in str_cmd
        assert "--sar" in str_cmd
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_nvencc_cmd.py -v`
Expected: ImportError

- [ ] **Step 4: Implement NVEncCAdapter**

Create `furnace/adapters/nvencc.py`. This is the core implementation. The implementer MUST research each flag against NVEncC docs. Skeleton:

```python
# furnace/adapters/nvencc.py
from __future__ import annotations

import logging
import re
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

from ..core.models import ColorSpace, CropRect, DvMode, EncodeResult, VideoParams
from ..core.quality import align_dimensions
from ._subprocess import OutputCallback

logger = logging.getLogger(__name__)


class NVEncCAdapter:
    """Implements Encoder port via NVEncC64."""

    def __init__(
        self, nvencc_path: Path,
        on_output: OutputCallback = None,
        log_dir: Path | None = None,
    ) -> None:
        self._nvencc = nvencc_path
        self._on_output = on_output
        self._log_dir = log_dir

    def set_log_dir(self, log_dir: Path | None) -> None:
        self._log_dir = log_dir

    def _build_encode_cmd(
        self,
        input_path: Path,
        output_path: Path,
        vp: VideoParams,
        *,
        vmaf_enabled: bool = False,
        rpu_path: Path | None = None,
    ) -> list[str | Path]:
        """Build NVEncC command line.

        ALL FLAGS MUST BE VERIFIED AGAINST NVEncC DOCUMENTATION.
        """
        cmd: list[str | Path] = [self._nvencc]

        # Input: hw decode via NVDEC
        cmd += ["--avhw", "-i", input_path]

        # Codec
        cmd += ["-c", "hevc"]
        cmd += ["--profile", "main10"]
        cmd += ["--output-depth", "10"]

        # Rate control
        # RESEARCH: --qvbr is equivalent to ffmpeg's -rc vbr -cq
        cmd += ["--preset", "P5"]
        cmd += ["--tune", "uhq"]
        cmd += ["--qvbr", str(vp.cq)]
        cmd += ["--aq", "--aq-temporal"]
        cmd += ["--lookahead", "32"]
        cmd += ["--multipass", "2pass-quarter"]

        # GOP
        cmd += ["--gop-len", str(vp.gop)]
        # RESEARCH: --strict-gop + --repeat-headers as replacement for -forced-idr 1
        cmd += ["--strict-gop", "--repeat-headers"]

        # Crop (L,T,R,B format)
        if vp.crop is not None:
            left = vp.crop.x
            top = vp.crop.y
            right = vp.source_width - vp.crop.x - vp.crop.w
            bottom = vp.source_height - vp.crop.y - vp.crop.h
            cmd += ["--crop", f"{left},{top},{right},{bottom}"]

        # Deinterlace
        if vp.deinterlace:
            cmd += ["--vpp-nnedi"]

        # SAR correction
        if vp.sar_num != vp.sar_den:
            display_w = round(vp.source_width * vp.sar_num / vp.sar_den)
            cur_h = vp.source_height
            if vp.crop is not None:
                display_w = round(vp.crop.w * vp.sar_num / vp.sar_den)
                cur_h = vp.crop.h
            cmd += ["--output-res", f"{display_w}x{cur_h}"]
            cmd += ["--sar", "1:1"]

        # Color metadata
        # RESEARCH: investigate auto mode vs explicit for each flag
        cmd += ["--colorrange", "limited"]
        if vp.color_primaries:
            cmd += ["--colorprim", vp.color_primaries]
        if vp.color_transfer:
            cmd += ["--transfer", vp.color_transfer]
        if vp.color_space == ColorSpace.BT2020:
            cmd += ["--colormatrix", "bt2020nc"]
        elif vp.color_space == ColorSpace.BT709:
            cmd += ["--colormatrix", "bt709"]
        elif vp.color_space == ColorSpace.BT601:
            cmd += ["--colormatrix", "smpte170m"]

        # HDR metadata
        # RESEARCH: investigate --max-cll copy / --master-display copy
        if vp.hdr and vp.hdr.content_light:
            # Parse MaxCLL=X,MaxFALL=Y
            for part in vp.hdr.content_light.split(","):
                if part.startswith("MaxCLL="):
                    cll = part.split("=", 1)[1]
                elif part.startswith("MaxFALL="):
                    fall = part.split("=", 1)[1]
            cmd += ["--max-cll", f"{cll},{fall}"]
        if vp.hdr and vp.hdr.mastering_display:
            cmd += ["--master-display", vp.hdr.mastering_display]

        # Dolby Vision
        if vp.dv_mode is not None and rpu_path is not None:
            cmd += ["--dolby-vision-rpu", str(rpu_path)]
            cmd += ["--dolby-vision-profile", "8.1"]

        # Quality metrics
        if vmaf_enabled:
            cmd += ["--ssim"]
            # RESEARCH: --vmaf flags and parameters
            cmd += ["--vmaf"]

        # Output
        cmd += ["-o", output_path]

        return cmd

    def _build_encoder_settings(self, vp: VideoParams) -> str:
        """Build ENCODER_SETTINGS string for MKV tag."""
        parts: list[str] = ["hevc_nvenc"]

        # RESEARCH: get NVEncC version
        version = self._get_version()
        if version:
            parts.append(f"NVEncC={version}")

        parts += [
            "main10",
            f"qvbr={vp.cq}",
            "preset=P5",
            "tune=uhq",
            "aq",
            "aq-temporal",
            "lookahead=32",
            "multipass=2pass-quarter",
        ]

        if vp.deinterlace:
            parts.append("deinterlace=nnedi")

        if vp.crop is not None:
            top = vp.crop.y
            bottom = vp.source_height - vp.crop.y - vp.crop.h
            left = vp.crop.x
            right = vp.source_width - vp.crop.x - vp.crop.w
            parts.append(f"crop={top}:{bottom}:{left}:{right}")

        if vp.dv_mode is not None:
            parts.append("dolby-vision=8.1")

        return " / ".join(parts)

    def _get_version(self) -> str:
        """Get NVEncC version string. Cached."""
        cached: str | None = getattr(self, "_version_cached", None)
        if cached is not None:
            return cached
        try:
            result = subprocess.run(
                [str(self._nvencc), "--version"],
                capture_output=True, text=True, timeout=5,
            )
            m = re.search(r"NVEncC.*?(\d+\.\d+)", result.stdout)
            self._version_cached: str = m.group(1) if m else ""
        except Exception:
            self._version_cached = ""
        return self._version_cached

    def encode(
        self,
        input_path: Path,
        output_path: Path,
        video_params: VideoParams,
        source_size: int,
        on_progress: Callable[[float, str], None] | None = None,
        vmaf_enabled: bool = False,
        rpu_path: Path | None = None,
    ) -> EncodeResult:
        """Encode video via NVEncC. Parse progress from stderr."""
        cmd = self._build_encode_cmd(
            input_path, output_path, video_params,
            vmaf_enabled=vmaf_enabled, rpu_path=rpu_path,
        )
        encoder_settings = self._build_encoder_settings(video_params)
        logger.debug("nvencc cmd: %s", " ".join(str(c) for c in cmd))

        # Open log file
        encode_log = None
        if self._log_dir:
            encode_log = (self._log_dir / "nvencc_encode.log").open("w", encoding="utf-8")
            encode_log.write(f"$ {' '.join(str(c) for c in cmd)}\n\n")
            encode_log.flush()

        process = subprocess.Popen(
            [str(c) for c in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # NVEncC outputs progress to stderr
        stderr_lines: list[str] = []
        vmaf_score: float | None = None
        ssim_score: float | None = None

        # RESEARCH: exact progress output format and VMAF/SSIM result format
        # Parse frame count, fps, percentage from stderr

        def _read_stderr() -> None:
            nonlocal vmaf_score, ssim_score
            assert process.stderr is not None
            for line in process.stderr:
                line = line.rstrip()
                if not line:
                    continue
                stderr_lines.append(line)
                if encode_log:
                    encode_log.write(line + "\n")
                    encode_log.flush()
                if self._on_output is not None:
                    self._on_output(line)

                # Parse progress: "[XX.X%] XX frames ..."
                m = re.search(r"\[(\d+\.?\d*)%\]", line)
                if m and on_progress is not None:
                    pct = float(m.group(1))
                    on_progress(pct, line)

                # Parse VMAF/SSIM results from final output
                # RESEARCH: exact format of NVEncC VMAF/SSIM output
                vm = re.search(r"VMAF\s*:\s*([\d.]+)", line, re.IGNORECASE)
                if vm:
                    vmaf_score = float(vm.group(1))
                sm = re.search(r"SSIM\s*:\s*([\d.]+)", line, re.IGNORECASE)
                if sm:
                    ssim_score = float(sm.group(1))

        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()

        # Read stdout (NVEncC may write some info here)
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if line:
                stderr_lines.append(line)
                if encode_log:
                    encode_log.write(line + "\n")
                    encode_log.flush()

        process.wait()
        stderr_thread.join(timeout=5)

        if encode_log:
            encode_log.write(f"\n--- exit code: {process.returncode} ---\n")
            encode_log.close()

        if process.returncode != 0:
            logger.error("NVEncC encode failed (rc=%d): %s", process.returncode, "\n".join(stderr_lines[-10:]))

        return EncodeResult(
            return_code=process.returncode,
            encoder_settings=encoder_settings,
            vmaf_score=vmaf_score,
            ssim_score=ssim_score,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_nvencc_cmd.py -v`
Expected: all pass

- [ ] **Step 6: Run quality gates**

Run: `uv run ruff check furnace/ && uv run mypy furnace/ --strict && uv run pytest tests/ -q`

- [ ] **Step 7: Commit**

```bash
git add furnace/adapters/nvencc.py tests/test_nvencc_cmd.py
git commit -m "feat: add NVEncCAdapter implementing Encoder port"
```

---

### Task 10: Clean up FFmpegAdapter — remove encode methods

**Files:**
- Modify: `furnace/adapters/ffmpeg.py`
- Delete: `tests/test_ffmpeg_encode_cmd.py`

- [ ] **Step 1: Remove encode-related code from FFmpegAdapter**

Delete these functions/methods from `furnace/adapters/ffmpeg.py`:
- `_build_vf_chain()` (lines 20-67)
- `_build_encoder_settings()` (lines 70-110)
- `encode()` method (lines 259-389)
- `_NVDEC_CODECS` class variable (lines 392-394)
- `_should_use_cuda()` method (lines 396-402)
- `_build_encode_cmd()` method (lines 404-464)
- `_check_mid_encoding_bloat()` method (lines 466-478)
- `compute_quality()` method (lines 484-615)

Keep: `__init__`, `set_log_dir`, `_get_ffmpeg_version`, `probe`, `detect_crop`, `get_encoder_tag`, `run_idet`, `extract_track`, `ffmpeg_to_wav`.

- [ ] **Step 2: Delete old encode tests**

```bash
rm tests/test_ffmpeg_encode_cmd.py
```

- [ ] **Step 3: Run quality gates**

Run: `uv run ruff check furnace/ && uv run mypy furnace/ --strict && uv run pytest tests/ -q`

- [ ] **Step 4: Commit**

```bash
git add furnace/adapters/ffmpeg.py
git rm tests/test_ffmpeg_encode_cmd.py
git commit -m "refactor: remove encode/quality methods from FFmpegAdapter"
```

---

### Task 11: Executor — RPU extraction, EncodeResult, remove bloat check + VMAF pass

**Files:**
- Modify: `furnace/services/executor.py`

- [ ] **Step 1: Update Executor constructor**

Add `dovi_processor` parameter:

```python
from ..core.ports import (
    AacEncoder,
    AudioDecoder,
    AudioExtractor,
    Cleaner,
    DoviProcessor,
    Encoder,
    Muxer,
    Prober,
    Tagger,
)

class Executor:
    def __init__(
        self,
        encoder: Encoder,
        audio_extractor: AudioExtractor,
        audio_decoder: AudioDecoder,
        aac_encoder: AacEncoder,
        muxer: Muxer,
        tagger: Tagger,
        cleaner: Cleaner,
        prober: Prober,
        dovi_processor: DoviProcessor | None = None,
        progress: Any | None = None,
        log_dir: Path | None = None,
    ) -> None:
        # ...
        self._dovi_processor = dovi_processor
```

- [ ] **Step 2: Add RPU extraction step to _run_pipeline**

After subtitle processing, before video encoding:

```python
        # Step 3: DV RPU extraction (if needed)
        rpu_path: Path | None = None
        if job.video_params.dv_mode is not None:
            if self._shutdown_event.is_set():
                return
            if self._dovi_processor is None:
                raise RuntimeError("DV content requires dovi_tool but it is not configured")

            rpu_path = temp_dir / "RPU.bin"
            status_msg = f"Extracting DV RPU (mode={job.video_params.dv_mode.name})"
            logger.info(status_msg)
            if self._progress is not None:
                self._progress.update_status(status_msg)
                self._progress.add_tool_line(f"[furnace] {status_msg}")

            rc = self._dovi_processor.extract_rpu(
                input_path=main_source,
                output_rpu=rpu_path,
                mode=job.video_params.dv_mode,
            )
            if rc != 0:
                raise RuntimeError(f"DV RPU extraction failed with return code {rc}")
```

- [ ] **Step 3: Update video encoding call**

Pass `vmaf_enabled` and `rpu_path` to encoder:

```python
        rc_result = self._encoder.encode(
            input_path=main_source,
            output_path=video_output,
            video_params=job.video_params,
            source_size=job.source_size,
            on_progress=_encode_progress,
            vmaf_enabled=self._vmaf_enabled,
            rpu_path=rpu_path,
        )
        if rc_result.return_code != 0:
            raise RuntimeError(f"Video encoding failed with return code {rc_result.return_code}")

        # Store metrics from encode
        if rc_result.vmaf_score is not None:
            job.vmaf_score = rc_result.vmaf_score
        if rc_result.ssim_score is not None:
            job.ssim_score = rc_result.ssim_score
        encoder_settings = rc_result.encoder_settings
```

- [ ] **Step 4: Remove bloat check**

Delete `_check_bloat` method and its call in `_run_pipeline`.

- [ ] **Step 5: Remove separate VMAF pass**

Delete the entire Step 8 block (`if self._vmaf_enabled and not self._shutdown_event.is_set():` ...).

- [ ] **Step 6: Update encoder_settings usage**

The ENCODER tag step uses `encoder_settings` from `EncodeResult` instead of the old tuple.

- [ ] **Step 7: Run quality gates**

Run: `uv run ruff check furnace/ && uv run mypy furnace/ --strict && uv run pytest tests/ -q`

- [ ] **Step 8: Commit**

```bash
git add furnace/services/executor.py
git commit -m "feat: add RPU extraction step; use EncodeResult; remove bloat check and VMAF pass"
```

---

### Task 12: CLI wiring — NVEncCAdapter + DoviToolAdapter

**Files:**
- Modify: `furnace/cli.py`

- [ ] **Step 1: Update `run` command to use new adapters**

In the `_run_executor` function inside the `run` command, replace:

```python
        # Old:
        # encoder=ffmpeg_adapter,

        # New:
        from .adapters.nvencc import NVEncCAdapter
        from .adapters.dovi_tool import DoviToolAdapter

        nvencc_adapter = NVEncCAdapter(cfg.nvencc, on_output=tool_output)

        dovi_adapter: DoviToolAdapter | None = None
        if cfg.dovi_tool is not None:
            dovi_adapter = DoviToolAdapter(cfg.dovi_tool, on_output=tool_output)

        executor = Executor(
            encoder=nvencc_adapter,
            audio_extractor=ffmpeg_adapter,
            audio_decoder=eac3to_adapter,
            aac_encoder=qaac_adapter,
            muxer=mkvmerge_adapter,
            tagger=mkvpropedit_adapter,
            cleaner=mkclean_adapter,
            prober=ffmpeg_adapter,
            dovi_processor=dovi_adapter,
            progress=progress,
            log_dir=log_dir,
        )
```

- [ ] **Step 2: Run quality gates**

Run: `uv run ruff check furnace/ && uv run mypy furnace/ --strict && uv run pytest tests/ -q`

- [ ] **Step 3: Commit**

```bash
git add furnace/cli.py
git commit -m "feat: wire NVEncCAdapter and DoviToolAdapter in CLI run command"
```

---

### Task 13: Version bump

**Files:**
- Modify: `furnace/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Bump version to 1.7.0**

This is a MINOR version bump (new feature: NVEncC + Dolby Vision support).

In `furnace/__init__.py`:
```python
VERSION = "1.7.0"
```

In `pyproject.toml`:
```toml
version = "1.7.0"
```

- [ ] **Step 2: Run quality gates**

Run: `uv run ruff check furnace/ && uv run mypy furnace/ --strict && uv run pytest tests/ -q`

- [ ] **Step 3: Commit**

```bash
git add furnace/__init__.py pyproject.toml
git commit -m "Bump to 1.7.0: NVEncC migration, Dolby Vision support, nnedi deinterlace"
```
