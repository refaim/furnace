"""Tests for guard branches in _build_audio_instruction and _build_subtitle_instruction."""
from __future__ import annotations

from unittest.mock import MagicMock

from furnace.core.models import (
    AudioAction,
    AudioCodecId,
    SubtitleAction,
    SubtitleCodecId,
    TrackType,
)
from furnace.services.planner import PlannerService
from tests.conftest import make_track


class TestBuildAudioInstructionGuards:
    def test_codec_id_none_uses_ffmpeg_encode(self) -> None:
        """Audio track with codec_id=None -> FFMPEG_ENCODE."""
        track = make_track(
            codec_name="unknown_codec",
            codec_id=None,
            channels=2,
            bitrate=128_000,
        )
        planner = PlannerService(prober=MagicMock(), previewer=None)
        instr = planner._build_audio_instruction(track, is_default=True)
        assert instr.action == AudioAction.FFMPEG_ENCODE

    def test_non_audio_codec_id_uses_ffmpeg_encode(self) -> None:
        """Audio track with a SubtitleCodecId (wrong type) -> FFMPEG_ENCODE."""
        track = make_track(
            codec_name="subrip",
            codec_id=SubtitleCodecId.SRT,
            channels=2,
            bitrate=0,
        )
        planner = PlannerService(prober=MagicMock(), previewer=None)
        instr = planner._build_audio_instruction(track, is_default=False)
        assert instr.action == AudioAction.FFMPEG_ENCODE

    def test_unknown_audio_codec_id_uses_ffmpeg_encode(self) -> None:
        """Audio track with UNKNOWN AudioCodecId -> FFMPEG_ENCODE (not in whitelist)."""
        track = make_track(
            codec_name="something",
            codec_id=AudioCodecId.UNKNOWN,
            channels=2,
            bitrate=0,
        )
        planner = PlannerService(prober=MagicMock(), previewer=None)
        instr = planner._build_audio_instruction(track, is_default=True)
        assert instr.action == AudioAction.FFMPEG_ENCODE

    def test_known_audio_codec_id_uses_whitelist(self) -> None:
        """Audio track with known AudioCodecId -> correct action from rules."""
        track = make_track(
            codec_name="ac3",
            codec_id=AudioCodecId.AC3,
            channels=6,
            bitrate=640_000,
        )
        planner = PlannerService(prober=MagicMock(), previewer=None)
        instr = planner._build_audio_instruction(track, is_default=True)
        assert instr.action == AudioAction.DENORM


class TestBuildSubtitleInstructionGuards:
    def test_codec_id_none_uses_copy(self) -> None:
        """Subtitle track with codec_id=None -> COPY."""
        track = make_track(
            track_type=TrackType.SUBTITLE,
            codec_name="unknown_sub",
            codec_id=None,
            channels=None,
        )
        planner = PlannerService(prober=MagicMock(), previewer=None)
        instr = planner._build_subtitle_instruction(track, is_default=False)
        assert instr.action == SubtitleAction.COPY

    def test_non_subtitle_codec_id_uses_copy(self) -> None:
        """Subtitle track with an AudioCodecId (wrong type) -> COPY."""
        track = make_track(
            track_type=TrackType.SUBTITLE,
            codec_name="aac",
            codec_id=AudioCodecId.AAC_LC,
            channels=None,
        )
        planner = PlannerService(prober=MagicMock(), previewer=None)
        instr = planner._build_subtitle_instruction(track, is_default=False)
        assert instr.action == SubtitleAction.COPY

    def test_unknown_subtitle_codec_id_uses_copy(self) -> None:
        """Subtitle track with UNKNOWN SubtitleCodecId -> COPY (not in whitelist)."""
        track = make_track(
            track_type=TrackType.SUBTITLE,
            codec_name="something",
            codec_id=SubtitleCodecId.UNKNOWN,
            channels=None,
        )
        planner = PlannerService(prober=MagicMock(), previewer=None)
        instr = planner._build_subtitle_instruction(track, is_default=True)
        assert instr.action == SubtitleAction.COPY

    def test_known_subtitle_codec_id_uses_whitelist(self) -> None:
        """Subtitle track with known SubtitleCodecId -> correct action from rules."""
        track = make_track(
            track_type=TrackType.SUBTITLE,
            codec_name="subrip",
            codec_id=SubtitleCodecId.SRT,
            channels=None,
        )
        planner = PlannerService(prober=MagicMock(), previewer=None)
        instr = planner._build_subtitle_instruction(track, is_default=False)
        assert instr.action == SubtitleAction.COPY_RECODE

    def test_pgs_subtitle_codec_id_uses_copy(self) -> None:
        """PGS subtitle -> COPY from whitelist."""
        track = make_track(
            track_type=TrackType.SUBTITLE,
            codec_name="hdmv_pgs_subtitle",
            codec_id=SubtitleCodecId.PGS,
            channels=None,
        )
        planner = PlannerService(prober=MagicMock(), previewer=None)
        instr = planner._build_subtitle_instruction(track, is_default=True)
        assert instr.action == SubtitleAction.COPY
