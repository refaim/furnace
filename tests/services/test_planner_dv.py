from __future__ import annotations
from pathlib import Path
import pytest
from furnace.core.models import (
    DvBlCompatibility, DvMode, HdrMetadata, VideoInfo,
)
from furnace.services.planner import PlannerService


def _make_video(hdr: HdrMetadata | None = None) -> VideoInfo:
    if hdr is None:
        hdr = HdrMetadata()
    return VideoInfo(
        index=0, codec_name="hevc", width=3840, height=2160,
        pixel_area=3840 * 2160, fps_num=24000, fps_den=1001,
        duration_s=7200.0, interlaced=False, color_matrix_raw="bt2020nc",
        color_range="tv", color_transfer="smpte2084", color_primaries="bt2020",
        pix_fmt="yuv420p10le", hdr=hdr, source_file=Path("/src/movie.mkv"),
        bitrate=80_000_000,
    )


class TestPlannerDvMode:
    def test_no_dv_mode_none(self) -> None:
        hdr = HdrMetadata(mastering_display="G(0.265,0.690)B(0.150,0.060)R(0.680,0.320)WP(0.3127,0.3290)L(1000,0.005)", content_light="MaxCLL=1000,MaxFALL=400")
        video = _make_video(hdr=hdr)
        planner = PlannerService(prober=None, previewer=None)  # type: ignore[arg-type]
        vp = planner._build_video_params(video, crop=None, source_file=video.source_file, sar_overrides=set())
        assert vp.dv_mode is None

    def test_dv_profile8_mode_copy(self) -> None:
        hdr = HdrMetadata(mastering_display="G(0.265,0.690)B(0.150,0.060)R(0.680,0.320)WP(0.3127,0.3290)L(1000,0.005)", content_light="MaxCLL=1000,MaxFALL=400", is_dolby_vision=True, dv_profile=8, dv_bl_compatibility=DvBlCompatibility.HDR10)
        video = _make_video(hdr=hdr)
        planner = PlannerService(prober=None, previewer=None)  # type: ignore[arg-type]
        vp = planner._build_video_params(video, crop=None, source_file=video.source_file, sar_overrides=set())
        assert vp.dv_mode == DvMode.COPY

    def test_dv_profile7_mode_to_8_1(self) -> None:
        hdr = HdrMetadata(mastering_display="G(0.265,0.690)B(0.150,0.060)R(0.680,0.320)WP(0.3127,0.3290)L(1000,0.005)", content_light="MaxCLL=1000,MaxFALL=400", is_dolby_vision=True, dv_profile=7, dv_bl_compatibility=DvBlCompatibility.HDR10)
        video = _make_video(hdr=hdr)
        planner = PlannerService(prober=None, previewer=None)  # type: ignore[arg-type]
        vp = planner._build_video_params(video, crop=None, source_file=video.source_file, sar_overrides=set())
        assert vp.dv_mode == DvMode.TO_8_1

    def test_dv_profile5_mode_copy(self) -> None:
        hdr = HdrMetadata(is_dolby_vision=True, dv_profile=5, dv_bl_compatibility=DvBlCompatibility.NONE)
        video = _make_video(hdr=hdr)
        planner = PlannerService(prober=None, previewer=None)  # type: ignore[arg-type]
        vp = planner._build_video_params(video, crop=None, source_file=video.source_file, sar_overrides=set())
        assert vp.dv_mode == DvMode.COPY

    def test_hdr10_plus_raises(self) -> None:
        hdr = HdrMetadata(is_hdr10_plus=True)
        video = _make_video(hdr=hdr)
        planner = PlannerService(prober=None, previewer=None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="HDR10\\+"):
            planner._build_video_params(video, crop=None, source_file=video.source_file, sar_overrides=set())
