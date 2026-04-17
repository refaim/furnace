"""Tests for the X-A trigger: force TUI when an audio track has >2 channels."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from furnace.core.models import (
    AudioCodecId,
    Movie,
    Track,
    TrackType,
)
from furnace.services.planner import PlannerService
from tests.conftest import make_movie, make_track, make_video_info


def _make_movie_with_audio(tmp_path: Path, audio: list[Track]) -> Movie:
    main = tmp_path / "movie.mkv"
    main.write_bytes(b"")
    return make_movie(
        main_file=main,
        video=make_video_info(
            codec_name="hevc", pix_fmt="yuv420p10le",
            source_file=main, bitrate=10_000_000,
        ),
        audio_tracks=audio,
    )


def _audio(index: int, language: str, channels: int | None, codec: str = "truehd") -> Track:
    codec_id_map: dict[str, AudioCodecId] = {
        "truehd": AudioCodecId.TRUEHD,
        "aac": AudioCodecId.AAC_LC,
        "ac3": AudioCodecId.AC3,
    }
    return make_track(
        index=index, track_type=TrackType.AUDIO, codec_name=codec,
        codec_id=codec_id_map[codec], language=language,
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

    def test_all_stereo_tracks_auto_select_no_tui(self, tmp_path: Path) -> None:
        """When ALL audio tracks are stereo across different langs, the for-loop
        exits without returning None (line 359->364), so no TUI is invoked."""
        movie = _make_movie_with_audio(
            tmp_path,
            [
                _audio(1, "eng", 2, codec="aac"),
                _audio(2, "rus", 2, codec="aac"),
            ],
        )
        prober = MagicMock()
        prober.detect_crop.return_value = None
        selector = MagicMock(return_value=[])
        planner = PlannerService(prober=prober, previewer=None, track_selector=selector)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng", "rus"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
        )

        audio_calls = [c for c in selector.call_args_list if c[0][2] == TrackType.AUDIO]
        assert audio_calls == []
        assert len(plan.jobs) == 1
        assert len(plan.jobs[0].audio) == 2
