"""Tests for the audio-selector trigger: force the TUI when the fake-surround
detector flags a candidate as fake or possibly fake (verdict != REAL)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from furnace.core.audio_profile import AudioMetrics, AudioProfile, Verdict
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


def _profile(verdict: Verdict) -> AudioProfile:
    """Minimal AudioProfile carrying only the verdict the planner reads."""
    metrics = AudioMetrics(
        channels=6,
        rms_l=-20.0, rms_r=-20.0, rms_c=None, rms_lfe=None,
        rms_ls=None, rms_rs=None, rms_lb=None, rms_rb=None,
        corr_lr=0.5, corr_ls_l=None, corr_rs_r=None, corr_ls_rs=None,
        corr_lb_ls=None, corr_rb_rs=None,
    )
    return AudioProfile(verdict=verdict, score=0, suggested=None, reasons=(), metrics=metrics)


def _audio(
    index: int,
    language: str,
    channels: int | None,
    codec: str = "truehd",
    *,
    verdict: Verdict | None = None,
) -> Track:
    codec_id_map: dict[str, AudioCodecId] = {
        "truehd": AudioCodecId.TRUEHD,
        "aac": AudioCodecId.AAC_LC,
        "ac3": AudioCodecId.AC3,
    }
    track = make_track(
        index=index, track_type=TrackType.AUDIO, codec_name=codec,
        codec_id=codec_id_map[codec], language=language,
        channels=channels, bitrate=4_500_000,
    )
    if verdict is not None:
        track.audio_profile = _profile(verdict)
    return track


def _make_planner(prober: MagicMock, selector: MagicMock) -> PlannerService:
    prober.detect_crop.return_value = None
    return PlannerService(prober=prober, previewer=None, track_selector=selector)


def _audio_calls(selector: MagicMock) -> list[object]:
    return [c for c in selector.call_args_list if c[0][2] == TrackType.AUDIO]


class TestVerdictTrigger:
    def test_fake_track_invokes_track_selector(self, tmp_path: Path) -> None:
        """A track the detector marks FAKE forces the TUI, even with no language ambiguity."""
        track = _audio(1, "eng", 6, verdict=Verdict.FAKE)
        movie = _make_movie_with_audio(tmp_path, [track])
        selector = MagicMock(return_value=[track])
        planner = _make_planner(MagicMock(), selector)

        planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
        )

        assert len(_audio_calls(selector)) == 1

    def test_suspicious_track_invokes_track_selector(self, tmp_path: Path) -> None:
        """A SUSPICIOUS ('might be fake') verdict also forces the TUI."""
        track = _audio(1, "eng", 6, verdict=Verdict.SUSPICIOUS)
        movie = _make_movie_with_audio(tmp_path, [track])
        selector = MagicMock(return_value=[track])
        planner = _make_planner(MagicMock(), selector)

        planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
        )

        assert len(_audio_calls(selector)) == 1

    def test_fake_stereo_track_invokes_track_selector(self, tmp_path: Path) -> None:
        """A 2.0 track flagged FAKE (e.g. dual-mono) now forces the TUI too."""
        track = _audio(1, "eng", 2, codec="aac", verdict=Verdict.FAKE)
        movie = _make_movie_with_audio(tmp_path, [track])
        selector = MagicMock(return_value=[track])
        planner = _make_planner(MagicMock(), selector)

        planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
        )

        assert len(_audio_calls(selector)) == 1

    def test_real_multichannel_track_does_not_invoke_selector(self, tmp_path: Path) -> None:
        """A 7.1 track the detector judges REAL auto-selects silently (no TUI)."""
        track = _audio(1, "eng", 8, verdict=Verdict.REAL)
        movie = _make_movie_with_audio(tmp_path, [track])
        selector = MagicMock(return_value=[])
        planner = _make_planner(MagicMock(), selector)

        planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
        )

        assert _audio_calls(selector) == []

    def test_unprofiled_track_does_not_invoke_selector(self, tmp_path: Path) -> None:
        """A track with no detector verdict (audio_profile=None) does not force the TUI."""
        track = _audio(1, "eng", 6)  # no verdict -> audio_profile stays None
        movie = _make_movie_with_audio(tmp_path, [track])
        selector = MagicMock(return_value=[])
        planner = _make_planner(MagicMock(), selector)

        planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
        )

        assert _audio_calls(selector) == []

    def test_headless_mode_not_affected(self, tmp_path: Path) -> None:
        """Without a track_selector callback (headless), a FAKE track must not crash."""
        track = _audio(1, "eng", 6, verdict=Verdict.FAKE)
        movie = _make_movie_with_audio(tmp_path, [track])
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

        assert len(plan.jobs) == 1
        assert len(plan.jobs[0].audio) == 1
        assert plan.jobs[0].audio[0].downmix is None

    def test_all_real_tracks_auto_select_no_tui(self, tmp_path: Path) -> None:
        """When every candidate is REAL across different langs, the loop exits
        without returning None, so no TUI is invoked."""
        movie = _make_movie_with_audio(
            tmp_path,
            [
                _audio(1, "eng", 2, codec="aac", verdict=Verdict.REAL),
                _audio(2, "rus", 2, codec="aac", verdict=Verdict.REAL),
            ],
        )
        selector = MagicMock(return_value=[])
        planner = _make_planner(MagicMock(), selector)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng", "rus"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
        )

        assert _audio_calls(selector) == []
        assert len(plan.jobs) == 1
        assert len(plan.jobs[0].audio) == 2
