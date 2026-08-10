from __future__ import annotations

from pathlib import Path

import pytest
from textual.content import Content
from textual.widgets import Input

from furnace.core.models import CropRect, DiscTitle, DownmixMode, Track, TrackType
from furnace.ui.tui import (
    FileSelectorScreen,
    PlaylistSelectorScreen,
    TrackSelectorScreen,
    _fmt_audio_track,
    _fmt_subtitle_track,
    build_downmix_map,
    parse_crop_value,
)
from tests.conftest import make_movie, make_track


def _plain(line: str) -> str:
    return Content.from_markup(line).plain


class TestParseCropValue:
    def test_valid_crop(self) -> None:
        result = parse_crop_value("1920:800:0:140", 1920, 1080)
        assert result == CropRect(w=1920, h=800, x=0, y=140)

    def test_wrong_field_count(self) -> None:
        with pytest.raises(ValueError, match="w:h:x:y"):
            parse_crop_value("1920:800:0", 1920, 1080)

    def test_non_integer(self) -> None:
        with pytest.raises(ValueError, match="integers"):
            parse_crop_value("1920:abc:0:0", 1920, 1080)

    def test_zero_width(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            parse_crop_value("0:800:0:0", 1920, 1080)

    def test_negative_offset(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            parse_crop_value("1920:800:-1:0", 1920, 1080)

    def test_exceeds_source(self) -> None:
        with pytest.raises(ValueError, match="exceeds"):
            parse_crop_value("1920:800:0:500", 1920, 1080)

    def test_exact_fit(self) -> None:
        result = parse_crop_value("1920:1080:0:0", 1920, 1080)
        assert result == CropRect(w=1920, h=1080, x=0, y=0)


class TestBuildDownmixMap:
    def test_selected_with_downmix_in_map(self) -> None:
        track = make_track(index=1, source_file=Path("/src/movie.mkv"))
        result = build_downmix_map(
            [track],
            [True],
            [DownmixMode.STEREO],
        )
        assert result == {(Path("/src/movie.mkv"), 1): DownmixMode.STEREO}

    def test_unselected_with_downmix_not_in_map(self) -> None:
        track = make_track(index=1, source_file=Path("/src/movie.mkv"))
        result = build_downmix_map(
            [track],
            [False],
            [DownmixMode.STEREO],
        )
        assert result == {}

    def test_selected_without_downmix_not_in_map(self) -> None:
        track = make_track(index=1, source_file=Path("/src/movie.mkv"))
        result = build_downmix_map(
            [track],
            [True],
            [None],
        )
        assert result == {}

    def test_multiple_tracks_mixed(self) -> None:
        tracks = [
            make_track(index=1, source_file=Path("/src/a.mkv")),
            make_track(index=2, source_file=Path("/src/b.mkv")),
            make_track(index=3, source_file=Path("/src/c.mkv")),
        ]
        selected = [True, False, True]
        downmix_list: list[DownmixMode | None] = [
            DownmixMode.STEREO,
            DownmixMode.DOWN6,
            None,
        ]
        result = build_downmix_map(tracks, selected, downmix_list)
        assert result == {(Path("/src/a.mkv"), 1): DownmixMode.STEREO}


class TestTrackComment:
    def _screen(self) -> tuple[TrackSelectorScreen, list[Track]]:
        tracks = [make_track(index=1, track_type=TrackType.AUDIO, source_file=Path("/src/movie.mkv"))]
        movie = make_movie(audio_tracks=tracks)
        screen = TrackSelectorScreen(movie, tracks, TrackType.AUDIO)
        return screen, tracks

    def test_line_without_comment_is_untouched(self) -> None:
        screen, _ = self._screen()
        assert "#" not in screen._render_line(0)

    def test_comment_appears_in_the_line(self) -> None:
        screen, _ = self._screen()
        screen._comments[0] = "Гаврилов"
        assert "Гаврилов" in screen._render_line(0)

    def test_submit_from_another_input_is_ignored(self) -> None:
        screen, _ = self._screen()
        screen.on_input_submitted(Input.Submitted(Input(id="crop-input"), "1920:800:0:140"))
        assert screen._comments[0] == ""

    def test_comment_brackets_survive_rendering(self) -> None:
        screen, _ = self._screen()
        screen._comments[0] = "[dim]red[/dim]"
        assert "# [dim]red[/dim]" in _plain(screen._render_line(0))

    def test_trailing_backslash_does_not_leak_the_closing_tag(self) -> None:
        screen, _ = self._screen()
        screen._comments[0] = "C:\\"
        rendered = _plain(screen._render_line(0))
        assert "# C:\\" in rendered
        assert "[/dim]" not in rendered


class TestPlaylistLine:
    def test_shows_video_format(self) -> None:
        playlists = [
            DiscTitle(
                number=3,
                duration_s=6062.0,
                raw_label="3) 00025.mpls, 00076.m2ts, 1:41:02",
                video="h264/AVC, 1080p24/1.001 (16:9)",
            )
        ]
        screen = PlaylistSelectorScreen(disc_label="Disc", playlists=playlists)
        line = screen._render_line(0)
        assert "h264/AVC, 1080p24/1.001 (16:9)" in line
        assert "00025.mpls" in line

    def test_markup_in_the_disc_labels_is_escaped(self) -> None:
        playlists = [
            DiscTitle(
                number=1,
                duration_s=600.0,
                raw_label="1) [eng] 00800.mpls, 0:10:00",
                video="h264/AVC, 1080p24 [dim]",
            )
        ]
        screen = PlaylistSelectorScreen(disc_label="Disc", playlists=playlists)
        rendered = _plain(screen._render_line(0))
        assert "1) [eng] 00800.mpls" in rendered
        assert "1080p24 [dim]" in rendered

    def test_line_without_video_has_no_trailing_separator(self) -> None:
        playlists = [DiscTitle(number=1, duration_s=600.0, raw_label="1) 00800.mpls, 0:10:00")]
        screen = PlaylistSelectorScreen(disc_label="Disc", playlists=playlists)
        line = screen._render_line(0)
        assert line == line.rstrip()
        assert line.endswith("(10:00)")


class TestTrackTitleMarkup:
    def test_audio_title_markup_is_escaped(self) -> None:
        track = make_track(
            index=1,
            track_type=TrackType.AUDIO,
            title="Dubbing [dim] studio",
            source_file=Path("/src/movie.mkv"),
        )
        assert "Dubbing [dim] studio" in _plain(_fmt_audio_track(track, selected=False))

    def test_subtitle_title_markup_is_escaped(self) -> None:
        track = make_track(
            index=2,
            track_type=TrackType.SUBTITLE,
            codec_name="subrip",
            title="Forced [dim] only",
            source_file=Path("/src/movie.mkv"),
        )
        assert "Forced [dim] only" in _plain(_fmt_subtitle_track(track, selected=False))


class TestFileSelectorLine:
    def test_release_brackets_survive_rendering(self) -> None:
        screen = FileSelectorScreen(files=[(Path("/src/Кино [rus] [1979].mkv"), 600.0, 1024)])
        assert "Кино [rus] [1979].mkv" in _plain(screen._render_line(0))
