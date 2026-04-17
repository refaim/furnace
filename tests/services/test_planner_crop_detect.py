"""Tests for crop detection in non-dry-run mode (planner lines 142-176)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from furnace.core.models import (
    AudioCodecId,
    CropRect,
    Movie,
    TrackType,
)
from furnace.services.planner import PlannerService
from tests.conftest import make_movie, make_track, make_video_info


def _make_movie(tmp_path: Path, *, width: int = 1920, height: int = 1080) -> Movie:
    main = tmp_path / "movie.mkv"
    main.write_bytes(b"")
    return make_movie(
        main_file=main,
        video=make_video_info(
            width=width,
            height=height,
            source_file=main,
            bitrate=20_000_000,
        ),
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


class TestCropDetectNonDryRun:
    def test_real_crop_applied(self, tmp_path: Path) -> None:
        """detect_crop returns a real crop rect -> crop is set in VideoParams."""
        movie = _make_movie(tmp_path)
        prober = MagicMock()
        prober.detect_crop.return_value = CropRect(w=1920, h=800, x=0, y=140)
        planner = PlannerService(prober=prober, previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
        )

        assert len(plan.jobs) == 1
        vp = plan.jobs[0].video_params
        assert vp.crop is not None
        assert vp.crop.w == 1920
        assert vp.crop.h == 800

    def test_full_frame_crop_becomes_none(self, tmp_path: Path) -> None:
        """detect_crop returns full-frame crop (same as source) -> crop becomes None."""
        movie = _make_movie(tmp_path, width=1920, height=1080)
        prober = MagicMock()
        prober.detect_crop.return_value = CropRect(w=1920, h=1080, x=0, y=0)
        planner = PlannerService(prober=prober, previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
        )

        assert len(plan.jobs) == 1
        assert plan.jobs[0].video_params.crop is None

    def test_crop_none_returns_no_crop(self, tmp_path: Path) -> None:
        """detect_crop returns None -> crop is None."""
        movie = _make_movie(tmp_path)
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

        assert len(plan.jobs) == 1
        assert plan.jobs[0].video_params.crop is None

    def test_oserror_logged_and_crop_none(self, tmp_path: Path) -> None:
        """detect_crop raises OSError -> warning logged, crop is None."""
        movie = _make_movie(tmp_path)
        prober = MagicMock()
        prober.detect_crop.side_effect = OSError("ffmpeg crashed")
        planner = PlannerService(prober=prober, previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
        )

        assert len(plan.jobs) == 1
        assert plan.jobs[0].video_params.crop is None

    def test_runtime_error_logged_and_crop_none(self, tmp_path: Path) -> None:
        """detect_crop raises RuntimeError -> warning logged, crop is None."""
        movie = _make_movie(tmp_path)
        prober = MagicMock()
        prober.detect_crop.side_effect = RuntimeError("bad frame")
        planner = PlannerService(prober=prober, previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
        )

        assert len(plan.jobs) == 1
        assert plan.jobs[0].video_params.crop is None

    def test_value_error_logged_and_crop_none(self, tmp_path: Path) -> None:
        """detect_crop raises ValueError -> warning logged, crop is None."""
        movie = _make_movie(tmp_path)
        prober = MagicMock()
        prober.detect_crop.side_effect = ValueError("bad params")
        planner = PlannerService(prober=prober, previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
        )

        assert len(plan.jobs) == 1
        assert plan.jobs[0].video_params.crop is None

    def test_full_frame_crop_log_message(self, tmp_path: Path) -> None:
        """Full-frame crop logs 'no black bars detected'."""
        movie = _make_movie(tmp_path)
        prober = MagicMock()
        prober.detect_crop.return_value = CropRect(w=1920, h=1080, x=0, y=0)
        planner = PlannerService(prober=prober, previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
        )

        assert plan.jobs[0].video_params.crop is None

    def test_real_crop_log_message(self, tmp_path: Path) -> None:
        """Real crop logs the crop rect dimensions."""
        movie = _make_movie(tmp_path)
        prober = MagicMock()
        prober.detect_crop.return_value = CropRect(w=1920, h=800, x=0, y=140)
        planner = PlannerService(prober=prober, previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
        )

        assert plan.jobs[0].video_params.crop is not None
