from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from furnace.core.models import (
    AudioCodecId,
    Movie,
    SubtitleCodecId,
    Track,
    TrackType,
)
from furnace.services.planner import PlannerService
from tests.conftest import make_movie, make_track, make_video_info


def _audio_track(language: str, index: int = 0) -> Track:
    return make_track(
        index=index,
        language=language,
        bitrate=192000,
    )


def _sub_track(language: str, index: int = 0, is_forced: bool = False) -> Track:
    return make_track(
        index=index,
        track_type=TrackType.SUBTITLE,
        codec_name="subrip",
        codec_id=SubtitleCodecId.SRT,
        language=language,
        is_forced=is_forced,
        channels=None,
    )


class TestSortAndSetDefault:
    def test_sorts_by_lang_filter_order(self) -> None:
        tracks = [_audio_track("eng", index=0), _audio_track("rus", index=1), _audio_track("jpn", index=2)]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._sort_and_set_default(tracks, ["jpn", "rus", "eng"])
        assert [t.language for t in result] == ["jpn", "rus", "eng"]

    def test_first_track_is_default(self) -> None:
        tracks = [_audio_track("eng", index=0), _audio_track("rus", index=1)]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._sort_and_set_default(tracks, ["rus", "eng"])
        assert result[0].is_default is True
        assert result[1].is_default is False

    def test_empty_list(self) -> None:
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._sort_and_set_default([], ["rus"])
        assert result == []


class TestAudioLangFilter:
    def test_filters_by_audio_lang(self) -> None:
        tracks = [_audio_track("jpn"), _audio_track("eng"), _audio_track("rus")]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_audio_tracks_by_lang(tracks, ["jpn"])
        assert [t.language for t in result] == ["jpn"]

    def test_und_always_included(self) -> None:
        tracks = [_audio_track("jpn"), _audio_track("und")]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_audio_tracks_by_lang(tracks, ["jpn"])
        assert [t.language for t in result] == ["jpn", "und"]

    def test_multiple_langs(self) -> None:
        tracks = [_audio_track("jpn"), _audio_track("eng"), _audio_track("rus")]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_audio_tracks_by_lang(tracks, ["jpn", "eng"])
        assert [t.language for t in result] == ["jpn", "eng"]

    def test_order_follows_lang_filter(self) -> None:
        """Tracks are sorted by lang_filter order, not source order."""
        tracks = [_audio_track("eng"), _audio_track("jpn"), _audio_track("rus")]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_audio_tracks_by_lang(tracks, ["rus", "jpn", "eng"])
        assert [t.language for t in result] == ["rus", "jpn", "eng"]


class TestSubLangFilter:
    def test_filters_by_sub_lang(self) -> None:
        tracks = [_sub_track("rus"), _sub_track("eng"), _sub_track("jpn")]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_sub_tracks_by_lang(tracks, ["rus", "eng"])
        assert [t.language for t in result] == ["rus", "eng"]

    def test_forced_subs_discarded(self) -> None:
        tracks = [_sub_track("rus"), _sub_track("eng", is_forced=True)]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_sub_tracks_by_lang(tracks, ["rus", "eng"])
        assert [t.language for t in result] == ["rus"]

    def test_und_always_included(self) -> None:
        tracks = [_sub_track("rus"), _sub_track("und")]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_sub_tracks_by_lang(tracks, ["rus"])
        assert [t.language for t in result] == ["rus", "und"]

    def test_forced_und_discarded(self) -> None:
        tracks = [_sub_track("rus"), _sub_track("und", is_forced=True)]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_sub_tracks_by_lang(tracks, ["rus"])
        assert [t.language for t in result] == ["rus"]

    def test_order_follows_lang_filter(self) -> None:
        """Subs sorted by lang_filter order, not source order."""
        tracks = [_sub_track("eng"), _sub_track("rus"), _sub_track("jpn")]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_sub_tracks_by_lang(tracks, ["jpn", "rus", "eng"])
        assert [t.language for t in result] == ["jpn", "rus", "eng"]


class TestResolveUndLanguages:
    def dummy_movie(self) -> MagicMock:
        return MagicMock()

    def test_no_und_tracks_unchanged(self) -> None:
        tracks = [_audio_track("jpn", index=0), _audio_track("eng", index=1)]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        cb = MagicMock()
        movie = self.dummy_movie()
        result = planner._resolve_und_languages(movie, tracks, ["jpn", "eng"], cb)
        cb.assert_not_called()
        assert [t.language for t in result] == ["jpn", "eng"]

    def test_single_lang_auto_assigns(self) -> None:
        tracks = [_audio_track("jpn", index=0), _audio_track("und", index=1)]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        cb = MagicMock()
        movie = self.dummy_movie()
        result = planner._resolve_und_languages(movie, tracks, ["jpn"], cb)
        cb.assert_not_called()
        assert [t.language for t in result] == ["jpn", "jpn"]

    def test_multiple_langs_calls_callback(self) -> None:
        tracks = [_audio_track("jpn", index=0), _audio_track("und", index=1)]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        cb = MagicMock(return_value="eng")
        movie = self.dummy_movie()
        result = planner._resolve_und_languages(movie, tracks, ["jpn", "eng"], cb)
        cb.assert_called_once_with(movie, tracks[1], ["jpn", "eng"])
        assert [t.language for t in result] == ["jpn", "eng"]

    def test_multiple_und_tracks_each_gets_callback(self) -> None:
        tracks = [_audio_track("und", index=0), _audio_track("und", index=1)]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        cb = MagicMock(side_effect=["rus", "eng"])
        movie = self.dummy_movie()
        result = planner._resolve_und_languages(movie, tracks, ["rus", "eng"], cb)
        assert cb.call_count == 2
        assert [t.language for t in result] == ["rus", "eng"]


def _make_movie_with_subs(
    tmp_path: Path,
    audio: list[Track] | None = None,
    subs: list[Track] | None = None,
) -> Movie:
    main = tmp_path / "movie.mkv"
    main.write_bytes(b"")
    default_audio = [
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
    ]
    return make_movie(
        main_file=main,
        video=make_video_info(
            codec_name="hevc",
            pix_fmt="yuv420p10le",
            source_file=main,
            bitrate=10_000_000,
        ),
        audio_tracks=audio if audio is not None else default_audio,
        subtitle_tracks=subs if subs is not None else [],
    )


class TestSubtitleAutoSelectFallback:
    """When _auto_select_from_candidates returns None for subs."""

    def test_track_selector_called_for_ambiguous_subs(self, tmp_path: Path) -> None:
        """Multiple subs per language -> track_selector callback is called for SUBTITLE."""
        main = tmp_path / "movie.mkv"
        main.write_bytes(b"")
        # Two subs with same language -> ambiguity
        subs = [
            _sub_track("eng", index=3),
            _sub_track("eng", index=4),
        ]
        for s in subs:
            s.source_file = main

        movie = _make_movie_with_subs(tmp_path, subs=subs)
        prober = MagicMock()
        prober.detect_crop.return_value = None

        selector = MagicMock(return_value=subs[:1])
        planner = PlannerService(prober=prober, previewer=None, track_selector=selector)

        planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
        )

        sub_calls = [c for c in selector.call_args_list if c[0][2] == TrackType.SUBTITLE]
        assert len(sub_calls) == 1

    def test_headless_includes_all_subs_when_ambiguous(self, tmp_path: Path) -> None:
        """Without track_selector (headless), all ambiguous subs are included."""
        main = tmp_path / "movie.mkv"
        main.write_bytes(b"")
        subs = [
            _sub_track("eng", index=3),
            _sub_track("eng", index=4),
        ]
        for s in subs:
            s.source_file = main

        movie = _make_movie_with_subs(tmp_path, subs=subs)
        prober = MagicMock()
        prober.detect_crop.return_value = None

        # No track_selector: headless mode
        planner = PlannerService(prober=prober, previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
        )

        assert len(plan.jobs) == 1
        assert len(plan.jobs[0].subtitles) == 2


class TestUndResolverIntegration:
    """und_resolver integration: called for audio and sub tracks in _build_job."""

    def test_und_resolver_called_for_audio_tracks(self, tmp_path: Path) -> None:
        """und_resolver is invoked for audio tracks with language 'und'."""
        main = tmp_path / "movie.mkv"
        main.write_bytes(b"")
        audio = [
            make_track(
                index=1,
                track_type=TrackType.AUDIO,
                codec_name="aac",
                codec_id=AudioCodecId.AAC_LC,
                language="und",
                source_file=main,
                channels=2,
                bitrate=192_000,
            ),
        ]
        movie = _make_movie_with_subs(tmp_path, audio=audio)
        prober = MagicMock()
        prober.detect_crop.return_value = None

        und_resolver = MagicMock(return_value="eng")
        planner = PlannerService(
            prober=prober,
            previewer=None,
            und_resolver=und_resolver,
        )

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=True,
        )

        assert len(plan.jobs) == 1
        assert plan.jobs[0].audio[0].language == "eng"

    def test_und_resolver_called_for_sub_tracks(self, tmp_path: Path) -> None:
        """und_resolver is invoked for subtitle tracks with language 'und'."""
        main = tmp_path / "movie.mkv"
        main.write_bytes(b"")
        audio = [
            make_track(
                index=1,
                track_type=TrackType.AUDIO,
                codec_name="aac",
                codec_id=AudioCodecId.AAC_LC,
                language="eng",
                source_file=main,
                channels=2,
                bitrate=192_000,
            ),
        ]
        subs = [
            _sub_track("und", index=3),
        ]
        subs[0].source_file = main
        movie = _make_movie_with_subs(tmp_path, audio=audio, subs=subs)
        prober = MagicMock()
        prober.detect_crop.return_value = None

        und_resolver = MagicMock(return_value="eng")
        planner = PlannerService(
            prober=prober,
            previewer=None,
            und_resolver=und_resolver,
        )

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=True,
        )

        assert len(plan.jobs) == 1
        assert plan.jobs[0].subtitles[0].language == "eng"

    def test_und_resolver_single_lang_auto_assigns_in_build_job(self, tmp_path: Path) -> None:
        """Single lang in filter -> und auto-assigned without calling resolver callback."""
        main = tmp_path / "movie.mkv"
        main.write_bytes(b"")
        audio = [
            make_track(
                index=1,
                track_type=TrackType.AUDIO,
                codec_name="aac",
                codec_id=AudioCodecId.AAC_LC,
                language="und",
                source_file=main,
                channels=2,
                bitrate=192_000,
            ),
        ]
        movie = _make_movie_with_subs(tmp_path, audio=audio)
        prober = MagicMock()
        prober.detect_crop.return_value = None

        und_resolver = MagicMock(return_value="eng")
        planner = PlannerService(
            prober=prober,
            previewer=None,
            und_resolver=und_resolver,
        )

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=True,
        )

        # Single lang auto-assigns, so callback should NOT be called
        und_resolver.assert_not_called()
        assert plan.jobs[0].audio[0].language == "eng"
