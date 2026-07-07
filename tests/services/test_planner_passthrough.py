"""Tests for planner passthrough behaviour and fallback.

The ``--copy-video`` flow asks the planner to copy a source video stream
verbatim instead of re-encoding it. Eligibility is decided per source video by
``furnace.core.detect.classify_passthrough`` (unit-tested in core); these tests
cover how the planner *acts* on that verdict end-to-end:

- progressive, non-DV-P7-FEL, non-HDR10+ -> passthrough (crop forced off)
- interlaced -> fall back to encode (deinterlace on)
- Dolby Vision Profile 7 FEL -> fall back to encode (dv_mode TO_8_1)
- HDR10+ -> rejected with ValueError (unchanged)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from furnace.core.models import (
    AudioCodecId,
    CropRect,
    DvBlCompatibility,
    DvMode,
    HdrMetadata,
    Movie,
    TrackType,
    VideoInfo,
)
from furnace.services.planner import PlannerService
from tests.conftest import make_movie, make_track, make_video_info

_MD = "G(0.265,0.690)B(0.150,0.060)R(0.680,0.320)WP(0.3127,0.3290)L(1000,0.005)"


def _make_video(
    *,
    interlaced: bool = False,
    hdr: HdrMetadata | None = None,
    source_file: Path | None = None,
) -> VideoInfo:
    return make_video_info(
        codec_name="hevc",
        width=3840,
        height=2160,
        fps_num=24000,
        fps_den=1001,
        duration_s=7200.0,
        interlaced=interlaced,
        color_matrix_raw="bt2020nc",
        color_transfer="smpte2084",
        color_primaries="bt2020",
        pix_fmt="yuv420p10le",
        hdr=hdr if hdr is not None else HdrMetadata(),
        bitrate=80_000_000,
        sar_num=4,
        sar_den=3,
        source_file=source_file,
    )


def _make_movie(tmp_path: Path, video: VideoInfo) -> Movie:
    main = tmp_path / "movie.mkv"
    main.write_bytes(b"")
    # rebind the video's source_file to the on-disk main file
    video.source_file = main
    return make_movie(
        main_file=main,
        video=video,
        audio_tracks=[
            make_track(
                index=1,
                track_type=TrackType.AUDIO,
                codec_name="aac",
                codec_id=AudioCodecId.AAC_LC,
                language="eng",
                is_default=True,
                source_file=main,
                channels=2,
                bitrate=192_000,
            ),
        ],
    )


def _dv_hdr(profile: int) -> HdrMetadata:
    return HdrMetadata(
        mastering_display=_MD,
        content_light="MaxCLL=1000,MaxFALL=400",
        is_dolby_vision=True,
        dv_profile=profile,
        dv_bl_compatibility=DvBlCompatibility.HDR10,
    )


class TestBuildVideoParamsPassthrough:
    """`passthrough=True` makes crop/deinterlace inert, keeps color/HDR/SAR."""

    def test_passthrough_sets_flag_and_inert_fields(self) -> None:
        planner = PlannerService(previewer=None)
        video = _make_video(hdr=_dv_hdr(8))
        vp = planner._build_video_params(
            video,
            crop=None,
            source_file=video.source_file,
            sar_overrides=set(),
            grain_overrides={},
            passthrough=True,
        )
        assert vp.passthrough is True
        assert vp.crop is None
        assert vp.deinterlace is False
        # color/HDR/SAR still populated for container flags
        assert vp.color_transfer == "smpte2084"
        assert vp.hdr is not None
        assert vp.sar_num == 4
        assert vp.sar_den == 3
        # DV P8 still maps to COPY
        assert vp.dv_mode == DvMode.COPY

    def test_passthrough_forces_deinterlace_false_even_if_interlaced(self) -> None:
        planner = PlannerService(previewer=None)
        video = _make_video(interlaced=True)
        vp = planner._build_video_params(
            video,
            crop=None,
            source_file=video.source_file,
            sar_overrides=set(),
            grain_overrides={},
            passthrough=True,
        )
        assert vp.deinterlace is False

    def test_non_passthrough_default_unchanged(self) -> None:
        planner = PlannerService(previewer=None)
        video = _make_video(interlaced=True)
        vp = planner._build_video_params(
            video,
            crop=None,
            source_file=video.source_file,
            sar_overrides=set(),
            grain_overrides={},
        )
        assert vp.passthrough is False
        assert vp.deinterlace is True


class TestCreatePlanCopyVideo:
    """End-to-end through create_plan with the copy_video flag."""

    def test_progressive_passthrough(self, tmp_path: Path) -> None:
        movie = _make_movie(tmp_path, _make_video())
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            copy_video=True,
        )

        vp = plan.jobs[0].video_params
        assert vp.passthrough is True
        assert vp.crop is None

    def test_passthrough_forces_crop_none_even_with_precomputed_crop(self, tmp_path: Path) -> None:
        """A passthrough job drops any crop the map supplies for its file."""
        movie = _make_movie(tmp_path, _make_video())
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            precomputed_crops={movie.main_file: CropRect(w=3840, h=1600, x=0, y=280)},
            copy_video=True,
        )

        vp = plan.jobs[0].video_params
        assert vp.passthrough is True
        assert vp.crop is None

    def test_interlaced_falls_back_to_encode(self, tmp_path: Path) -> None:
        movie = _make_movie(tmp_path, _make_video(interlaced=True))
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            # interlaced is an encode, so a precomputed crop IS honoured
            precomputed_crops={movie.main_file: CropRect(w=3840, h=1600, x=0, y=280)},
            copy_video=True,
        )

        vp = plan.jobs[0].video_params
        assert vp.passthrough is False
        assert vp.deinterlace is True
        assert vp.crop is not None
        assert (vp.crop.w, vp.crop.h) == (3840, 1600)

    def test_dv_p7_fel_falls_back_preserving_to_8_1(self, tmp_path: Path) -> None:
        movie = _make_movie(tmp_path, _make_video(hdr=_dv_hdr(7)))
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            copy_video=True,
        )

        vp = plan.jobs[0].video_params
        assert vp.passthrough is False
        assert vp.dv_mode == DvMode.TO_8_1

    def test_hdr10_plus_still_raises(self, tmp_path: Path) -> None:
        movie = _make_movie(tmp_path, _make_video(hdr=HdrMetadata(is_hdr10_plus=True)))
        planner = PlannerService(previewer=None)

        with pytest.raises(ValueError, match="HDR10\\+"):
            planner.create_plan(
                [(movie, tmp_path / "out.mkv")],
                audio_lang_filter=["eng"],
                sub_lang_filter=["eng"],
                vmaf_enabled=False,
                copy_video=True,
            )

    def test_copy_video_default_false_unchanged(self, tmp_path: Path) -> None:
        movie = _make_movie(tmp_path, _make_video())
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
        )

        assert plan.jobs[0].video_params.passthrough is False
