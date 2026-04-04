from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from furnace.core.models import (
    AudioCodecId,
    SubtitleCodecId,
    Track,
    TrackType,
)
from furnace.services.planner import PlannerService


def _audio_track(language: str, index: int = 0) -> Track:
    return Track(
        index=index,
        track_type=TrackType.AUDIO,
        codec_name="aac",
        codec_id=AudioCodecId.AAC_LC,
        language=language,
        title="",
        is_default=False,
        is_forced=False,
        source_file=Path("/src/movie.mkv"),
        channels=2,
        bitrate=192000,
    )


def _sub_track(language: str, index: int = 0, is_forced: bool = False) -> Track:
    return Track(
        index=index,
        track_type=TrackType.SUBTITLE,
        codec_name="subrip",
        codec_id=SubtitleCodecId.SRT,
        language=language,
        title="",
        is_default=False,
        is_forced=is_forced,
        source_file=Path("/src/movie.mkv"),
    )


class TestAudioLangFilter:
    def test_filters_by_audio_lang(self):
        tracks = [_audio_track("jpn"), _audio_track("eng"), _audio_track("rus")]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_audio_tracks_by_lang(tracks, ["jpn"])
        assert [t.language for t in result] == ["jpn"]

    def test_und_always_included(self):
        tracks = [_audio_track("jpn"), _audio_track("und")]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_audio_tracks_by_lang(tracks, ["jpn"])
        assert [t.language for t in result] == ["jpn", "und"]

    def test_multiple_langs(self):
        tracks = [_audio_track("jpn"), _audio_track("eng"), _audio_track("rus")]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_audio_tracks_by_lang(tracks, ["jpn", "eng"])
        assert [t.language for t in result] == ["jpn", "eng"]

    def test_order_follows_lang_filter(self):
        """Tracks are sorted by lang_filter order, not source order."""
        tracks = [_audio_track("eng"), _audio_track("jpn"), _audio_track("rus")]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_audio_tracks_by_lang(tracks, ["rus", "jpn", "eng"])
        assert [t.language for t in result] == ["rus", "jpn", "eng"]


class TestSubLangFilter:
    def test_filters_by_sub_lang(self):
        tracks = [_sub_track("rus"), _sub_track("eng"), _sub_track("jpn")]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_sub_tracks_by_lang(tracks, ["rus", "eng"])
        assert [t.language for t in result] == ["rus", "eng"]

    def test_forced_subs_discarded(self):
        tracks = [_sub_track("rus"), _sub_track("eng", is_forced=True)]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_sub_tracks_by_lang(tracks, ["rus", "eng"])
        assert [t.language for t in result] == ["rus"]

    def test_und_always_included(self):
        tracks = [_sub_track("rus"), _sub_track("und")]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_sub_tracks_by_lang(tracks, ["rus"])
        assert [t.language for t in result] == ["rus", "und"]

    def test_forced_und_discarded(self):
        tracks = [_sub_track("rus"), _sub_track("und", is_forced=True)]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_sub_tracks_by_lang(tracks, ["rus"])
        assert [t.language for t in result] == ["rus"]

    def test_order_follows_lang_filter(self):
        """Subs sorted by lang_filter order, not source order."""
        tracks = [_sub_track("eng"), _sub_track("rus"), _sub_track("jpn")]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_sub_tracks_by_lang(tracks, ["jpn", "rus", "eng"])
        assert [t.language for t in result] == ["jpn", "rus", "eng"]
