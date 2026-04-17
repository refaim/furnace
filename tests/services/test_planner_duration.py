from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from furnace.core.models import Movie
from furnace.services.planner import PlannerService
from tests.conftest import make_movie, make_track, make_video_info


def _make_movie_dur(duration_s: float = 1234.5) -> Movie:
    return make_movie(
        video=make_video_info(
            fps_num=24000, fps_den=1001,
            duration_s=duration_s,
            bitrate=20_000_000,
        ),
        audio_tracks=[make_track(is_default=True, bitrate=192000)],
        file_size=1_000_000_000,
    )


class TestJobDurationS:
    def test_duration_s_populated_from_movie_video(self) -> None:
        """Job.duration_s must be set from movie.video.duration_s."""
        movie = _make_movie_dur(duration_s=1234.5)
        output_path = Path("/out/movie.mkv")

        prober = MagicMock()
        prober.detect_crop.return_value = None

        planner = PlannerService(prober=prober, previewer=None)
        plan = planner.create_plan(
            movies=[(movie, output_path)],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=True,
        )

        assert len(plan.jobs) == 1
        assert plan.jobs[0].duration_s == 1234.5

    def test_duration_s_zero_when_video_duration_zero(self) -> None:
        """Job.duration_s == 0.0 when source video duration is unknown."""
        movie = _make_movie_dur(duration_s=0.0)
        output_path = Path("/out/movie.mkv")

        prober = MagicMock()
        prober.detect_crop.return_value = None

        planner = PlannerService(prober=prober, previewer=None)
        plan = planner.create_plan(
            movies=[(movie, output_path)],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=True,
        )

        assert len(plan.jobs) == 1
        assert plan.jobs[0].duration_s == 0.0
