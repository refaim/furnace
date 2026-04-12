"""Tests for the X-A trigger: force TUI when an audio track has >2 channels."""
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


def _make_movie_with_audio(tmp_path: Path, audio: list[Track]) -> Movie:
    main = tmp_path / "movie.mkv"
    main.write_bytes(b"")
    video = VideoInfo(
        index=0,
        codec_name="hevc",
        width=1920, height=1080,
        pixel_area=1920 * 1080,
        fps_num=24, fps_den=1,
        duration_s=5400.0,
        interlaced=False,
        color_matrix_raw="bt709",
        color_range="tv",
        color_transfer="bt709",
        color_primaries="bt709",
        pix_fmt="yuv420p10le",
        hdr=HdrMetadata(
            mastering_display=None, content_light=None,
            is_dolby_vision=False, is_hdr10_plus=False,
        ),
        source_file=main,
        bitrate=10_000_000,
        sar_num=1, sar_den=1,
    )
    return Movie(
        main_file=main, satellite_files=[], file_size=0, video=video,
        audio_tracks=audio, subtitle_tracks=[], attachments=[],
        has_chapters=False,
    )


def _audio(index: int, language: str, channels: int | None, codec: str = "truehd") -> Track:
    codec_id_map = {
        "truehd": AudioCodecId.TRUEHD,
        "aac": AudioCodecId.AAC_LC,
        "ac3": AudioCodecId.AC3,
    }
    return Track(
        index=index, track_type=TrackType.AUDIO, codec_name=codec,
        codec_id=codec_id_map[codec], language=language, title="",
        is_default=False, is_forced=False, source_file=Path("/src/movie.mkv"),
        channels=channels, bitrate=4_500_000,
    )


class TestXATrigger:
    def test_single_multichannel_track_invokes_track_selector(self, tmp_path: Path) -> None:
        """A 7.1 track with no language ambiguity must still show the TUI."""
        movie = _make_movie_with_audio(tmp_path, [_audio(1, "eng", 8)])
        prober = MagicMock()
        prober.detect_crop.return_value = None
        selector = MagicMock(return_value=[_audio(1, "eng", 8)])
        planner = PlannerService(prober=prober, previewer=None, track_selector=selector)

        planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
        )

        # track_selector called at least once for AUDIO
        audio_calls = [
            call for call in selector.call_args_list
            if call[0][2] == TrackType.AUDIO
        ]
        assert len(audio_calls) == 1

    def test_single_stereo_track_does_not_invoke_selector(self, tmp_path: Path) -> None:
        """A 2.0 track auto-selects as before (no TUI)."""
        movie = _make_movie_with_audio(tmp_path, [_audio(1, "eng", 2, codec="aac")])
        prober = MagicMock()
        prober.detect_crop.return_value = None
        selector = MagicMock(return_value=[])
        planner = PlannerService(prober=prober, previewer=None, track_selector=selector)

        planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
        )

        audio_calls = [c for c in selector.call_args_list if c[0][2] == TrackType.AUDIO]
        assert audio_calls == []

    def test_multichannel_with_channels_none_does_not_trigger(self, tmp_path: Path) -> None:
        """A track with unknown channels should not force TUI via X-A."""
        movie = _make_movie_with_audio(tmp_path, [_audio(1, "eng", None, codec="aac")])
        prober = MagicMock()
        prober.detect_crop.return_value = None
        selector = MagicMock(return_value=[])
        planner = PlannerService(prober=prober, previewer=None, track_selector=selector)

        planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
        )

        audio_calls = [c for c in selector.call_args_list if c[0][2] == TrackType.AUDIO]
        assert audio_calls == []

    def test_headless_mode_not_affected(self, tmp_path: Path) -> None:
        """Without a track_selector callback (headless), X-A must not crash."""
        movie = _make_movie_with_audio(tmp_path, [_audio(1, "eng", 8)])
        prober = MagicMock()
        prober.detect_crop.return_value = None
        planner = PlannerService(prober=prober, previewer=None)  # no track_selector

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=True,
        )

        # Still produces a job with the 7.1 track included, no downmix
        assert len(plan.jobs) == 1
        assert len(plan.jobs[0].audio) == 1
        assert plan.jobs[0].audio[0].downmix is None
