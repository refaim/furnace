from __future__ import annotations

from pathlib import Path

import pytest

from furnace.core.models import (
    DvBlCompatibility,
    DvMode,
    HdrMetadata,
    VideoInfo,
    VideoParams,
)
from furnace.core.target_quality import grain_uses_svt
from furnace.services.planner import PlannerService
from tests.conftest import make_video_info

_MASTERING_DISPLAY = "G(0.265,0.690)B(0.150,0.060)R(0.680,0.320)WP(0.3127,0.3290)L(1000,0.005)"


def _make_video(hdr: HdrMetadata) -> VideoInfo:
    return make_video_info(
        codec_name="hevc",
        width=3840,
        height=2160,
        fps_num=24000,
        fps_den=1001,
        duration_s=7200.0,
        color_matrix_raw="bt2020nc",
        color_transfer="smpte2084",
        color_primaries="bt2020",
        pix_fmt="yuv420p10le",
        hdr=hdr,
        bitrate=80_000_000,
    )


def _build(video: VideoInfo, *, grain_overrides: dict[Path, bool] | None = None) -> VideoParams:
    planner = PlannerService(previewer=None)
    return planner._build_video_params(
        video,
        crop=None,
        source_file=video.source_file,
        sar_overrides=set(),
        grain_overrides=grain_overrides if grain_overrides is not None else {},
    )


class TestPlannerHdr10Plus:
    def test_hdr10_plus_plans_without_raising(self) -> None:
        hdr = HdrMetadata(
            mastering_display=_MASTERING_DISPLAY,
            content_light="MaxCLL=1000,MaxFALL=400",
            is_hdr10_plus=True,
        )
        vp = _build(_make_video(hdr))
        assert vp.hdr is not None
        assert vp.hdr.is_hdr10_plus

    def test_hdr10_plus_with_dolby_vision_keeps_both(self) -> None:
        hdr = HdrMetadata(
            mastering_display=_MASTERING_DISPLAY,
            content_light="MaxCLL=1000,MaxFALL=400",
            is_hdr10_plus=True,
            is_dolby_vision=True,
            dv_profile=8,
            dv_bl_compatibility=DvBlCompatibility.HDR10,
        )
        vp = _build(_make_video(hdr))
        assert vp.dv_mode == DvMode.COPY
        assert vp.hdr is not None
        assert vp.hdr.is_hdr10_plus

    def test_hdr10_plus_without_static_metadata_survives(self) -> None:
        vp = _build(_make_video(HdrMetadata(is_hdr10_plus=True)))
        assert vp.hdr is not None
        assert vp.hdr.is_hdr10_plus
        assert vp.hdr.mastering_display is None
        assert vp.hdr.content_light is None

    def test_untagged_hdr10_plus_resolves_to_pq(self) -> None:
        video = make_video_info(
            codec_name="hevc",
            width=3840,
            height=2160,
            color_matrix_raw=None,
            color_transfer=None,
            color_primaries=None,
            pix_fmt="yuv420p10le",
            hdr=HdrMetadata(is_hdr10_plus=True),
        )
        vp = _build(video)
        assert vp.color_transfer == "smpte2084"
        assert vp.color_primaries == "bt2020"
        assert vp.color_matrix == "bt2020nc"

    def test_plain_sdr_still_carries_no_hdr(self) -> None:
        video = make_video_info(hdr=HdrMetadata())
        vp = _build(video)
        assert vp.hdr is None

    def test_grainy_sd_hdr10_plus_is_rerouted_off_svt(self) -> None:
        video = make_video_info(
            codec_name="hevc",
            width=1440,
            height=576,
            fps_num=25,
            fps_den=1,
            color_matrix_raw="bt2020nc",
            color_transfer="smpte2084",
            color_primaries="bt2020",
            pix_fmt="yuv420p10le",
            hdr=HdrMetadata(mastering_display=_MASTERING_DISPLAY, is_hdr10_plus=True),
            grainy=True,
        )
        vp = _build(video)
        assert not vp.grain
        assert not grain_uses_svt(vp)
        assert vp.hdr is not None
        assert vp.hdr.is_hdr10_plus

    def test_defense_in_depth_hdr10_plus_landing_on_svt_raises(self) -> None:
        video = make_video_info(
            codec_name="mpeg2video",
            width=720,
            height=576,
            fps_num=25,
            fps_den=1,
            color_matrix_raw="bt470bg",
            color_transfer="bt709",
            color_primaries="bt470bg",
            hdr=HdrMetadata(is_hdr10_plus=True),
            grainy=True,
        )
        with pytest.raises(ValueError, match="HDR10\\+"):
            _build(video)

    def test_sd_grain_without_hdr10_plus_still_uses_svt(self) -> None:
        video = make_video_info(
            codec_name="mpeg2video",
            width=720,
            height=576,
            fps_num=25,
            fps_den=1,
            color_matrix_raw="bt470bg",
            color_transfer="bt709",
            color_primaries="bt470bg",
            hdr=HdrMetadata(),
            grainy=True,
        )
        vp = _build(video)
        assert grain_uses_svt(vp)

    def test_hd_hdr10_plus_grain_stays_on_nvencc(self) -> None:
        hdr = HdrMetadata(mastering_display=_MASTERING_DISPLAY, is_hdr10_plus=True)
        video = _make_video(hdr)
        video.grainy = True
        vp = _build(video)
        assert vp.grain
        assert not grain_uses_svt(vp)
