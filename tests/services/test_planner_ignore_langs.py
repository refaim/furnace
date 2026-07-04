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

# ---------------------------------------------------------------------------
# Local factories (mirror the pattern in test_planner_lang.py)
# ---------------------------------------------------------------------------

def _audio(language: str, index: int, source: Path) -> Track:
    return make_track(
        index=index,
        track_type=TrackType.AUDIO,
        codec_name="aac",
        codec_id=AudioCodecId.AAC_LC,
        language=language,
        source_file=source,
        channels=2,
        bitrate=192_000,
    )


def _sub(language: str, index: int, source: Path, *, is_forced: bool = False) -> Track:
    return make_track(
        index=index,
        track_type=TrackType.SUBTITLE,
        codec_name="subrip",
        codec_id=SubtitleCodecId.SRT,
        language=language,
        is_forced=is_forced,
        source_file=source,
        channels=None,
    )


def _make_movie(
    main: Path,
    *,
    audio: list[Track],
    subs: list[Track] | None = None,
) -> Movie:
    return make_movie(
        main_file=main,
        video=make_video_info(
            codec_name="hevc",
            pix_fmt="yuv420p10le",
            source_file=main,
            bitrate=10_000_000,
        ),
        audio_tracks=audio,
        subtitle_tracks=subs if subs is not None else [],
    )


# ---------------------------------------------------------------------------
# _eff_lang
# ---------------------------------------------------------------------------

class TestEffLang:
    def test_ignore_true_returns_und(self) -> None:
        planner = PlannerService(previewer=None, ignore_langs=True)
        track = _audio("fre", 1, Path("/src/movie.mkv"))
        assert planner._eff_lang(track) == "und"

    def test_ignore_false_returns_track_language(self) -> None:
        planner = PlannerService(previewer=None, ignore_langs=False)
        track = _audio("fre", 1, Path("/src/movie.mkv"))
        assert planner._eff_lang(track) == "fre"


# ---------------------------------------------------------------------------
# _sort_and_set_default keyword behaviour
# ---------------------------------------------------------------------------

class TestSortAndSetDefaultIgnoreLangs:
    def test_ignore_true_preserves_source_order(self) -> None:
        src = Path("/src/movie.mkv")
        tracks = [_audio("eng", 0, src), _audio("rus", 1, src), _audio("jpn", 2, src)]
        planner = PlannerService(previewer=None)
        result = planner._sort_and_set_default(tracks, ["jpn", "rus", "eng"], ignore_langs=True)
        assert [t.language for t in result] == ["eng", "rus", "jpn"]
        assert result[0].is_default is True
        assert result[1].is_default is False
        assert result[2].is_default is False

    def test_ignore_false_sorts_by_filter(self) -> None:
        src = Path("/src/movie.mkv")
        tracks = [_audio("eng", 0, src), _audio("rus", 1, src), _audio("jpn", 2, src)]
        planner = PlannerService(previewer=None)
        result = planner._sort_and_set_default(tracks, ["jpn", "rus", "eng"], ignore_langs=False)
        assert [t.language for t in result] == ["jpn", "rus", "eng"]
        assert result[0].is_default is True


# ---------------------------------------------------------------------------
# Filtering keeps mislabelled tracks under --ignore-langs
# ---------------------------------------------------------------------------

class TestFilterKeepsMislabelled:
    def test_audio_mislabelled_kept_under_ignore(self) -> None:
        src = Path("/src/movie.mkv")
        tracks = [_audio("fre", 1, src)]
        planner = PlannerService(previewer=None, ignore_langs=True)
        result = planner._filter_audio_tracks_by_lang(tracks, ["jpn"])
        assert [t.language for t in result] == ["fre"]

    def test_audio_mislabelled_dropped_without_ignore(self) -> None:
        src = Path("/src/movie.mkv")
        tracks = [_audio("fre", 1, src)]
        planner = PlannerService(previewer=None, ignore_langs=False)
        result = planner._filter_audio_tracks_by_lang(tracks, ["jpn"])
        assert result == []

    def test_subs_mislabelled_kept_under_ignore(self) -> None:
        src = Path("/src/movie.mkv")
        tracks = [_sub("fre", 3, src)]
        planner = PlannerService(previewer=None, ignore_langs=True)
        result = planner._filter_sub_tracks_by_lang(tracks, ["jpn"])
        assert [t.language for t in result] == ["fre"]


# ---------------------------------------------------------------------------
# Forced subtitles are still discarded under --ignore-langs
# ---------------------------------------------------------------------------

class TestForcedSubDiscardedUnderIgnore:
    def test_forced_sub_discarded(self) -> None:
        src = Path("/src/movie.mkv")
        tracks = [_sub("fre", 3, src), _sub("ger", 4, src, is_forced=True)]
        planner = PlannerService(previewer=None, ignore_langs=True)
        result = planner._filter_sub_tracks_by_lang(tracks, ["jpn"])
        assert [t.index for t in result] == [3]


# ---------------------------------------------------------------------------
# _assign_languages_relabel
# ---------------------------------------------------------------------------

class TestAssignLanguagesRelabel:
    def test_default_to_first_filter_lang(self) -> None:
        src = Path("/src/movie.mkv")
        tracks = [_audio("fre", 1, src), _audio("ger", 2, src)]
        planner = PlannerService(previewer=None, ignore_langs=True)
        result = planner._assign_languages_relabel(tracks, ["jpn", "eng"], {})
        assert [t.language for t in result] == ["jpn", "jpn"]

    def test_override_takes_precedence(self) -> None:
        src = Path("/src/movie.mkv")
        tracks = [_audio("fre", 1, src), _audio("ger", 2, src)]
        overrides = {(src, 2): "rus"}
        planner = PlannerService(previewer=None, ignore_langs=True)
        result = planner._assign_languages_relabel(tracks, ["jpn"], overrides)
        assert [t.language for t in result] == ["jpn", "rus"]

    def test_empty_filter_defaults_und(self) -> None:
        src = Path("/src/movie.mkv")
        tracks = [_audio("fre", 1, src)]
        planner = PlannerService(previewer=None, ignore_langs=True)
        result = planner._assign_languages_relabel(tracks, [], {})
        assert [t.language for t in result] == ["und"]


# ---------------------------------------------------------------------------
# create_plan integration under --ignore-langs
# ---------------------------------------------------------------------------

class TestIgnoreLangsPlan:
    def test_single_track_auto_selected_and_relabelled(self, tmp_path: Path) -> None:
        """One audio + one sub track: each auto-selected (no selector) and relabelled."""
        main = tmp_path / "movie.mkv"
        main.write_bytes(b"")
        audio = [_audio("fre", 1, main)]
        subs = [_sub("ger", 3, main)]
        movie = _make_movie(main, audio=audio, subs=subs)

        selector_calls: list[TrackType] = []

        def _selector(m: Movie, cands: list[Track], tt: TrackType) -> list[Track]:
            selector_calls.append(tt)
            return list(cands)

        planner = PlannerService(previewer=None, track_selector=_selector, ignore_langs=True)
        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["jpn"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
        )

        assert selector_calls == []
        assert plan.jobs[0].audio[0].language == "jpn"
        assert plan.jobs[0].subtitles[0].language == "eng"

    def test_ambiguous_candidates_keep_original_tags_at_selection(self, tmp_path: Path) -> None:
        """Two mislabelled audio tracks force the selector; candidates still carry
        their ORIGINAL tags (not yet relabelled to 'und') at selection time."""
        main = tmp_path / "movie.mkv"
        main.write_bytes(b"")
        audio = [_audio("fre", 1, main), _audio("ger", 2, main)]
        movie = _make_movie(main, audio=audio)

        captured: list[list[str]] = []

        def _selector(m: Movie, cands: list[Track], tt: TrackType) -> list[Track]:
            if tt == TrackType.AUDIO:
                captured.append([t.language for t in cands])
            return list(cands)

        planner = PlannerService(previewer=None, track_selector=_selector, ignore_langs=True)
        planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["jpn"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
        )

        assert captured == [["fre", "ger"]]

    def test_explicit_override_and_default(self, tmp_path: Path) -> None:
        """An explicit lang override wins; un-overridden selected tracks get lang_filter[0]."""
        main = tmp_path / "movie.mkv"
        main.write_bytes(b"")
        audio = [_audio("fre", 1, main), _audio("ger", 2, main)]
        movie = _make_movie(main, audio=audio)

        def _selector(m: Movie, cands: list[Track], tt: TrackType) -> list[Track]:
            return list(cands)

        planner = PlannerService(previewer=None, track_selector=_selector, ignore_langs=True)
        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["jpn"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            lang_overrides={(main, 2): "rus"},
        )

        langs = {a.stream_index: a.language for a in plan.jobs[0].audio}
        assert langs[1] == "jpn"
        assert langs[2] == "rus"

    def test_final_ordering_and_default(self, tmp_path: Path) -> None:
        """After relabel the tracks are sorted by target priority and is_default set."""
        main = tmp_path / "movie.mkv"
        main.write_bytes(b"")
        audio = [_audio("fre", 1, main), _audio("ger", 2, main)]
        movie = _make_movie(main, audio=audio)

        def _selector(m: Movie, cands: list[Track], tt: TrackType) -> list[Track]:
            return list(cands)

        planner = PlannerService(previewer=None, track_selector=_selector, ignore_langs=True)
        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["jpn", "eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            lang_overrides={(main, 1): "eng", (main, 2): "jpn"},
        )

        assert [a.language for a in plan.jobs[0].audio] == ["jpn", "eng"]
        assert plan.jobs[0].audio[0].is_default is True
        assert plan.jobs[0].audio[1].is_default is False

    def test_empty_filter_relabels_to_und(self, tmp_path: Path) -> None:
        """Empty audio filter under --ignore-langs relabels the selected track to 'und'."""
        main = tmp_path / "movie.mkv"
        main.write_bytes(b"")
        audio = [_audio("fre", 1, main)]
        movie = _make_movie(main, audio=audio)

        planner = PlannerService(previewer=None, ignore_langs=True)
        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=[],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
        )

        assert plan.jobs[0].audio[0].language == "und"


# ---------------------------------------------------------------------------
# Regression: default behaviour (ignore_langs=False) unchanged
# ---------------------------------------------------------------------------

class TestRegressionDefault:
    def test_mislabelled_dropped_and_und_resolution_still_works(self, tmp_path: Path) -> None:
        """Without --ignore-langs a mislabelled track is dropped while a genuine
        'und' track is still resolved to the single target lang."""
        main = tmp_path / "movie.mkv"
        main.write_bytes(b"")
        audio = [_audio("fre", 1, main), _audio("und", 2, main)]
        movie = _make_movie(main, audio=audio)

        und_resolver = MagicMock(return_value="eng")
        planner = PlannerService(previewer=None, und_resolver=und_resolver, ignore_langs=False)
        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
        )

        # The 'fre' track was dropped (not in filter, not 'und'); only the
        # resolved 'und' track survives, auto-assigned to the single target lang.
        assert [a.language for a in plan.jobs[0].audio] == ["eng"]
        und_resolver.assert_not_called()
