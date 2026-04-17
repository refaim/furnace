"""Tests for the downmix feature in PlannerService."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from furnace.core.models import (
    AudioAction,
    AudioCodecId,
    DownmixMode,
    Movie,
    Track,
    TrackType,
)
from furnace.services.planner import PlannerService
from tests.conftest import make_movie, make_track, make_video_info


def _audio_track(
    index: int = 0,
    codec_name: str = "truehd",
    codec_id: AudioCodecId = AudioCodecId.TRUEHD,
    language: str = "eng",
    channels: int | None = 8,
) -> Track:
    return make_track(
        index=index,
        track_type=TrackType.AUDIO,
        codec_name=codec_name,
        codec_id=codec_id,
        language=language,
        channels=channels,
        bitrate=4_500_000,
    )


class TestBuildAudioInstructionValidation:
    def test_downmix_on_stereo_track_raises(self) -> None:
        track = _audio_track(codec_name="aac", codec_id=AudioCodecId.AAC_LC, channels=2)
        planner = PlannerService(prober=MagicMock(), previewer=None)
        with pytest.raises(ValueError, match="Downmix not applicable"):
            planner._build_audio_instruction(track, is_default=True, downmix=DownmixMode.STEREO)

    def test_downmix_on_mono_track_raises(self) -> None:
        track = _audio_track(codec_name="aac", codec_id=AudioCodecId.AAC_LC, channels=1)
        planner = PlannerService(prober=MagicMock(), previewer=None)
        with pytest.raises(ValueError, match="Downmix not applicable"):
            planner._build_audio_instruction(track, is_default=True, downmix=DownmixMode.STEREO)

    def test_downmix_on_none_channels_raises(self) -> None:
        track = _audio_track(channels=None)
        planner = PlannerService(prober=MagicMock(), previewer=None)
        with pytest.raises(ValueError, match="Downmix not applicable"):
            planner._build_audio_instruction(track, is_default=True, downmix=DownmixMode.STEREO)

    def test_down6_on_6ch_raises(self) -> None:
        track = _audio_track(channels=6)
        planner = PlannerService(prober=MagicMock(), previewer=None)
        with pytest.raises(ValueError, match="DOWN6 not applicable"):
            planner._build_audio_instruction(track, is_default=True, downmix=DownmixMode.DOWN6)

    def test_down6_on_5ch_raises(self) -> None:
        track = _audio_track(channels=5)
        planner = PlannerService(prober=MagicMock(), previewer=None)
        with pytest.raises(ValueError, match="DOWN6 not applicable"):
            planner._build_audio_instruction(track, is_default=True, downmix=DownmixMode.DOWN6)

    def test_down6_on_7ch_ok(self) -> None:
        track = _audio_track(channels=7)
        planner = PlannerService(prober=MagicMock(), previewer=None)
        instr = planner._build_audio_instruction(track, is_default=True, downmix=DownmixMode.DOWN6)
        assert instr.downmix == DownmixMode.DOWN6

    def test_down6_on_8ch_ok(self) -> None:
        track = _audio_track(channels=8)
        planner = PlannerService(prober=MagicMock(), previewer=None)
        instr = planner._build_audio_instruction(track, is_default=True, downmix=DownmixMode.DOWN6)
        assert instr.downmix == DownmixMode.DOWN6

    def test_stereo_on_6ch_ok(self) -> None:
        track = _audio_track(channels=6)
        planner = PlannerService(prober=MagicMock(), previewer=None)
        instr = planner._build_audio_instruction(track, is_default=True, downmix=DownmixMode.STEREO)
        assert instr.downmix == DownmixMode.STEREO

    def test_no_downmix_on_2ch_ok(self) -> None:
        track = _audio_track(codec_name="aac", codec_id=AudioCodecId.AAC_LC, channels=2)
        planner = PlannerService(prober=MagicMock(), previewer=None)
        instr = planner._build_audio_instruction(track, is_default=True, downmix=None)
        assert instr.downmix is None


class TestBuildAudioInstructionForcing:
    """Downmix must force AudioAction.DECODE_ENCODE regardless of source codec."""

    def test_force_on_ac3_track_overrides_denorm(self) -> None:
        track = _audio_track(codec_name="ac3", codec_id=AudioCodecId.AC3, channels=6)
        planner = PlannerService(prober=MagicMock(), previewer=None)
        instr = planner._build_audio_instruction(track, is_default=True, downmix=DownmixMode.STEREO)
        assert instr.action == AudioAction.DECODE_ENCODE
        assert instr.downmix == DownmixMode.STEREO

    def test_force_on_truehd_track_is_decode_encode(self) -> None:
        track = _audio_track(codec_name="truehd", codec_id=AudioCodecId.TRUEHD, channels=8)
        planner = PlannerService(prober=MagicMock(), previewer=None)
        instr = planner._build_audio_instruction(track, is_default=True, downmix=DownmixMode.DOWN6)
        assert instr.action == AudioAction.DECODE_ENCODE

    def test_force_on_opus_track_overrides_ffmpeg_encode(self) -> None:
        track = _audio_track(codec_name="opus", codec_id=AudioCodecId.OPUS, channels=6)
        planner = PlannerService(prober=MagicMock(), previewer=None)
        instr = planner._build_audio_instruction(track, is_default=True, downmix=DownmixMode.STEREO)
        assert instr.action == AudioAction.DECODE_ENCODE

    def test_no_downmix_preserves_default_action(self) -> None:
        """Without downmix, AC3 stays on DENORM (baseline)."""
        track = _audio_track(codec_name="ac3", codec_id=AudioCodecId.AC3, channels=6)
        planner = PlannerService(prober=MagicMock(), previewer=None)
        instr = planner._build_audio_instruction(track, is_default=True, downmix=None)
        assert instr.action == AudioAction.DENORM


class TestCreatePlanDownmixOverrides:
    """create_plan must accept downmix_overrides and thread them into AudioInstruction."""

    def test_downmix_overrides_applied_to_matching_track(self, tmp_path: Path) -> None:
        main = tmp_path / "movie.mkv"
        main.write_bytes(b"")

        track = _audio_track(index=1, channels=8, language="eng")
        track.source_file = main
        movie = make_movie(
            main_file=main,
            video=make_video_info(
                codec_name="hevc", pix_fmt="yuv420p10le",
                source_file=main, bitrate=10_000_000,
            ),
            audio_tracks=[track],
            file_size=1_000_000,
        )
        output_path = tmp_path / "out.mkv"

        prober = MagicMock()
        prober.detect_crop.return_value = None
        planner = PlannerService(prober=prober, previewer=None)

        downmix_overrides = {(Path(str(main)), 1): DownmixMode.STEREO}
        plan = planner.create_plan(
            [(movie, output_path)],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=True,
            downmix_overrides=downmix_overrides,
        )

        assert len(plan.jobs) == 1
        audio = plan.jobs[0].audio
        assert len(audio) == 1
        assert audio[0].downmix == DownmixMode.STEREO
        assert audio[0].action == AudioAction.DECODE_ENCODE

    def test_closure_mutation_after_call_is_observed(self, tmp_path: Path) -> None:
        """Regression: the dict passed to create_plan must be the same object
        the planner reads from, so a track_selector callback that mutates an
        outer dict before/during planner execution sees its updates honored.

        The bug this guards against: `effective_overrides = downmix_overrides or {}`
        in create_plan replaces the empty dict with a fresh literal, breaking
        the reference link to the caller's dict and silently dropping any
        downmix overrides that the closure adds during track selection.
        """
        main = tmp_path / "movie.mkv"
        main.write_bytes(b"")

        # Two audio tracks on the same language so the planner is forced
        # to invoke the track_selector callback (lang ambiguity path).
        t1 = _audio_track(index=1, channels=8, language="eng")
        t1.source_file = main
        t2 = _audio_track(index=2, channels=6, language="eng")
        t2.source_file = main
        movie = make_movie(
            main_file=main,
            video=make_video_info(
                codec_name="hevc", pix_fmt="yuv420p10le",
                source_file=main, bitrate=10_000_000,
            ),
            audio_tracks=[t1, t2],
        )

        # The caller's outer dict — initially empty, mutated by the callback.
        downmix_overrides: dict[tuple[Path, int], DownmixMode] = {}

        def selector(_movie: Movie, candidates: list[Track], track_type: TrackType) -> list[Track]:
            if track_type == TrackType.AUDIO:
                # Pick the first multichannel candidate and tag it for downmix.
                picked = candidates[0]
                downmix_overrides[(Path(str(picked.source_file)), picked.index)] = (
                    DownmixMode.STEREO
                )
                return [picked]
            return list(candidates)

        prober = MagicMock()
        prober.detect_crop.return_value = None
        planner = PlannerService(
            prober=prober, previewer=None, track_selector=selector,
        )

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=True,
            downmix_overrides=downmix_overrides,
        )

        audio = plan.jobs[0].audio
        assert len(audio) == 1
        assert audio[0].downmix == DownmixMode.STEREO
        assert audio[0].action == AudioAction.DECODE_ENCODE

    def test_empty_downmix_overrides_leaves_tracks_untouched(self, tmp_path: Path) -> None:
        main = tmp_path / "movie.mkv"
        main.write_bytes(b"")

        track = _audio_track(index=1, channels=2, codec_name="aac", codec_id=AudioCodecId.AAC_LC)
        track.source_file = main
        movie = make_movie(
            main_file=main,
            video=make_video_info(
                codec_name="hevc", pix_fmt="yuv420p10le",
                source_file=main, bitrate=10_000_000,
            ),
            audio_tracks=[track],
        )

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

        assert plan.jobs[0].audio[0].downmix is None
        assert plan.jobs[0].audio[0].action == AudioAction.COPY
