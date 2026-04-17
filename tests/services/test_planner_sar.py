"""Tests for the SAR override refactor: sar_overrides as explicit parameter."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from furnace.core.models import (
    AudioCodecId,
    Movie,
    TrackType,
)
from furnace.services.planner import PlannerService
from tests.conftest import make_movie, make_track, make_video_info


def _make_movie_sar(tmp_path: Path, sar_num: int = 1, sar_den: int = 1) -> Movie:
    main = tmp_path / "movie.mkv"
    main.write_bytes(b"")
    return make_movie(
        main_file=main,
        video=make_video_info(
            codec_name="mpeg2video",
            width=720, height=480,
            fps_num=30000, fps_den=1001,
            color_matrix_raw="smpte170m",
            color_transfer="smpte170m",
            color_primaries="smpte170m",
            pix_fmt="yuv420p",
            source_file=main,
            bitrate=8_000_000,
            sar_num=sar_num, sar_den=sar_den,
        ),
        audio_tracks=[make_track(
            index=1, track_type=TrackType.AUDIO, codec_name="ac3",
            codec_id=AudioCodecId.AC3, language="eng",
            is_default=True, source_file=main,
            channels=2, bitrate=192_000,
        )],
    )


class TestSarOverrides:
    def test_sar_override_applied(self, tmp_path: Path) -> None:
        movie = _make_movie_sar(tmp_path, sar_num=1, sar_den=1)
        prober = MagicMock()
        prober.detect_crop.return_value = None
        planner = PlannerService(prober=prober, previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=True,
            sar_overrides={movie.main_file},
        )

        vp = plan.jobs[0].video_params
        assert vp.sar_num == 64
        assert vp.sar_den == 45

    def test_sar_not_overridden_when_path_not_in_set(self, tmp_path: Path) -> None:
        movie = _make_movie_sar(tmp_path, sar_num=1, sar_den=1)
        prober = MagicMock()
        prober.detect_crop.return_value = None
        planner = PlannerService(prober=prober, previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=True,
            sar_overrides=set(),
        )

        vp = plan.jobs[0].video_params
        assert vp.sar_num == 1
        assert vp.sar_den == 1

    def test_sar_overrides_none_behaves_as_empty(self, tmp_path: Path) -> None:
        """Omitting sar_overrides (None default) must leave SAR at source."""
        movie = _make_movie_sar(tmp_path, sar_num=1, sar_den=1)
        prober = MagicMock()
        prober.detect_crop.return_value = None
        planner = PlannerService(prober=prober, previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=True,
        )

        vp = plan.jobs[0].video_params
        assert vp.sar_num == 1
        assert vp.sar_den == 1

    def test_movie_video_sar_not_mutated_by_planner(self, tmp_path: Path) -> None:
        """Regression guard: the planner must NOT mutate movie.video.sar_num/den."""
        movie = _make_movie_sar(tmp_path, sar_num=1, sar_den=1)
        original_num = movie.video.sar_num
        original_den = movie.video.sar_den
        prober = MagicMock()
        prober.detect_crop.return_value = None
        planner = PlannerService(prober=prober, previewer=None)

        planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=True,
            sar_overrides={movie.main_file},
        )

        assert movie.video.sar_num == original_num
        assert movie.video.sar_den == original_den
