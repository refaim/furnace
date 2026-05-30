"""Tests for Task 3: planner passthrough classification and fallback.

The ``--copy-video`` flow asks the planner to copy a source video stream
verbatim instead of re-encoding it. Eligibility is decided per source video:

- progressive, non-DV-P7-FEL, non-HDR10+ -> passthrough
- interlaced -> fall back to encode (reason "interlaced")
- Dolby Vision Profile 7 FEL -> fall back to encode (reason "DV P7 FEL")
- HDR10+ -> rejected with ValueError (unchanged)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from furnace.core.models import (
    AudioCodecId,
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


class TestClassifyPassthrough:
    """Unit tests for the per-video classification helper."""

    def test_copy_video_disabled_no_passthrough(self) -> None:
        planner = PlannerService(prober=MagicMock(), previewer=None)
        video = _make_video()
        assert planner._classify_passthrough(video, copy_video=False) == (False, None)

    def test_progressive_non_dv_eligible(self) -> None:
        planner = PlannerService(prober=MagicMock(), previewer=None)
        video = _make_video()
        assert planner._classify_passthrough(video, copy_video=True) == (True, None)

    def test_interlaced_falls_back(self) -> None:
        planner = PlannerService(prober=MagicMock(), previewer=None)
        video = _make_video(interlaced=True)
        assert planner._classify_passthrough(video, copy_video=True) == (False, "interlaced")

    def test_dv_p7_fel_falls_back(self) -> None:
        planner = PlannerService(prober=MagicMock(), previewer=None)
        video = _make_video(hdr=_dv_hdr(7))
        assert planner._classify_passthrough(video, copy_video=True) == (False, "DV P7 FEL")

    def test_dv_p8_eligible(self) -> None:
        planner = PlannerService(prober=MagicMock(), previewer=None)
        video = _make_video(hdr=_dv_hdr(8))
        assert planner._classify_passthrough(video, copy_video=True) == (True, None)


class TestBuildVideoParamsPassthrough:
    """`passthrough=True` makes crop/deinterlace inert, keeps color/HDR/SAR."""

    def test_passthrough_sets_flag_and_inert_fields(self) -> None:
        planner = PlannerService(prober=MagicMock(), previewer=None)
        video = _make_video(hdr=_dv_hdr(8))
        vp = planner._build_video_params(
            video,
            crop=None,
            source_file=video.source_file,
            sar_overrides=set(),
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
        planner = PlannerService(prober=MagicMock(), previewer=None)
        video = _make_video(interlaced=True)
        vp = planner._build_video_params(
            video,
            crop=None,
            source_file=video.source_file,
            sar_overrides=set(),
            passthrough=True,
        )
        assert vp.deinterlace is False

    def test_non_passthrough_default_unchanged(self) -> None:
        planner = PlannerService(prober=MagicMock(), previewer=None)
        video = _make_video(interlaced=True)
        vp = planner._build_video_params(
            video,
            crop=None,
            source_file=video.source_file,
            sar_overrides=set(),
        )
        assert vp.passthrough is False
        assert vp.deinterlace is True


class TestCreatePlanCopyVideo:
    """End-to-end through create_plan with the copy_video flag."""

    def test_progressive_passthrough_skips_cropdetect(self, tmp_path: Path) -> None:
        movie = _make_movie(tmp_path, _make_video())
        prober = MagicMock()
        planner = PlannerService(prober=prober, previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
            copy_video=True,
        )

        vp = plan.jobs[0].video_params
        assert vp.passthrough is True
        assert vp.crop is None
        prober.detect_crop.assert_not_called()

    def test_interlaced_falls_back_to_encode(self, tmp_path: Path) -> None:
        movie = _make_movie(tmp_path, _make_video(interlaced=True))
        prober = MagicMock()
        prober.detect_crop.return_value = None
        planner = PlannerService(prober=prober, previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
            copy_video=True,
        )

        vp = plan.jobs[0].video_params
        assert vp.passthrough is False
        assert vp.deinterlace is True
        # interlaced still runs cropdetect (encode path)
        prober.detect_crop.assert_called_once()

    def test_dv_p7_fel_falls_back_preserving_to_8_1(self, tmp_path: Path) -> None:
        movie = _make_movie(tmp_path, _make_video(hdr=_dv_hdr(7)))
        prober = MagicMock()
        prober.detect_crop.return_value = None
        planner = PlannerService(prober=prober, previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
            copy_video=True,
        )

        vp = plan.jobs[0].video_params
        assert vp.passthrough is False
        assert vp.dv_mode == DvMode.TO_8_1

    def test_hdr10_plus_still_raises(self, tmp_path: Path) -> None:
        movie = _make_movie(tmp_path, _make_video(hdr=HdrMetadata(is_hdr10_plus=True)))
        prober = MagicMock()
        planner = PlannerService(prober=prober, previewer=None)

        with pytest.raises(ValueError, match="HDR10\\+"):
            planner.create_plan(
                [(movie, tmp_path / "out.mkv")],
                audio_lang_filter=["eng"],
                sub_lang_filter=["eng"],
                vmaf_enabled=False,
                dry_run=False,
                copy_video=True,
            )

    def test_copy_video_default_false_unchanged(self, tmp_path: Path) -> None:
        movie = _make_movie(tmp_path, _make_video())
        prober = MagicMock()
        prober.detect_crop.return_value = None
        planner = PlannerService(prober=prober, previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
        )

        assert plan.jobs[0].video_params.passthrough is False
        prober.detect_crop.assert_called_once()
