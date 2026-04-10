from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from furnace.core.models import (
    AudioCodecId,
    HdrMetadata,
    Movie,
    Track,
    TrackType,
    VideoInfo,
)
from furnace.services.planner import PlannerService


def _make_video(duration_s: float = 1234.5) -> VideoInfo:
    return VideoInfo(
        index=0, codec_name="h264", width=1920, height=1080,
        pixel_area=1920 * 1080, fps_num=24000, fps_den=1001,
        duration_s=duration_s, interlaced=False, color_matrix_raw="bt709",
        color_range="tv", color_transfer="bt709", color_primaries="bt709",
        pix_fmt="yuv420p", hdr=HdrMetadata(), source_file=Path("/src/movie.mkv"),
        bitrate=20_000_000,
    )


def _make_audio_track() -> Track:
    return Track(
        index=0,
        track_type=TrackType.AUDIO,
        codec_name="aac",
        codec_id=AudioCodecId.AAC_LC,
        language="eng",
        title="",
        is_default=True,
        is_forced=False,
        source_file=Path("/src/movie.mkv"),
        channels=2,
        bitrate=192000,
    )


def _make_movie(duration_s: float = 1234.5) -> Movie:
    return Movie(
        main_file=Path("/src/movie.mkv"),
        satellite_files=[],
        video=_make_video(duration_s=duration_s),
        audio_tracks=[_make_audio_track()],
        subtitle_tracks=[],
        attachments=[],
        has_chapters=False,
        file_size=1_000_000_000,
    )


class TestJobDurationS:
    def test_duration_s_populated_from_movie_video(self) -> None:
        """Job.duration_s must be set from movie.video.duration_s."""
        movie = _make_movie(duration_s=1234.5)
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
        movie = _make_movie(duration_s=0.0)
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
