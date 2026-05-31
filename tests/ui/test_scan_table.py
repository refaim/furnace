"""Renderer tests for ``furnace scan``'s redirect-safe inventory table.

The table goes to stdout and must survive redirection to a file: ASCII box
only (no Unicode box-drawing), no ANSI when stdout is not a TTY, and columns
sized to content so long paths never truncate. Summary, warnings, and the
"no video files found" note go to stderr, keeping the redirected file pure
table.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from furnace.core.scan import AudioTrackSummary, ScanRow, SubtitleTrackSummary
from furnace.ui.scan_table import render_scan_table

# Unicode box-drawing characters that must never appear in redirect-safe output.
_UNICODE_BOX = "─│┌┐└┘├┤┬┴┼━┃┏┓┗┛╔╗╚╝═║"


def _row(
    path: str,
    *,
    version: tuple[int, int, int] | None = None,
    video: str | None = "hevc",
    audio: tuple[AudioTrackSummary, ...] = (),
    subtitles: tuple[SubtitleTrackSummary, ...] = (),
    unreadable: bool = False,
) -> ScanRow:
    return ScanRow(
        path=Path(path),
        furnace_version=version,
        video_codec=video,
        audio=audio,
        subtitles=subtitles,
        unreadable=unreadable,
    )


def _render(rows: list[ScanRow], *, root: Path, total: int | None = None) -> tuple[str, str]:
    out = io.StringIO()
    err = io.StringIO()
    render_scan_table(
        rows,
        root=root,
        total=len(rows) if total is None else total,
        file=out,
        err=err,
    )
    return out.getvalue(), err.getvalue()


class TestRedirectSafe:
    def test_no_unicode_box_drawing(self) -> None:
        root = Path("/movies")
        rows = [_row("/movies/a.mkv", version=(1, 19, 3))]
        out, _ = _render(rows, root=root)
        for ch in _UNICODE_BOX:
            assert ch not in out

    def test_uses_ascii_box(self) -> None:
        root = Path("/movies")
        rows = [_row("/movies/a.mkv", version=(1, 19, 3))]
        out, _ = _render(rows, root=root)
        assert "+" in out
        assert "-" in out
        assert "|" in out

    def test_no_ansi_escape_codes_for_non_tty(self) -> None:
        root = Path("/movies")
        rows = [_row("/movies/a.mkv", version=(1, 19, 3))]
        out, err = _render(rows, root=root)
        assert "\x1b[" not in out
        assert "\x1b[" not in err

    def test_long_path_not_truncated(self) -> None:
        root = Path("/movies")
        long_name = "a-really-very-extremely-long-movie-file-name-" * 6 + ".mkv"
        rows = [_row(f"/movies/{long_name}", version=(1, 0, 0))]
        out, _ = _render(rows, root=root)
        assert long_name in out
        assert "…" not in out  # no ellipsis from truncation


class TestColumns:
    def test_path_relative_to_root(self) -> None:
        root = Path("/movies")
        rows = [_row("/movies/sub/dir/film.mkv", version=(1, 19, 3))]
        out, _ = _render(rows, root=root)
        assert "sub/dir/film.mkv" in out
        assert "/movies/sub" not in out

    def test_single_file_root_shows_filename(self) -> None:
        root = Path("/movies/film.mkv")
        rows = [_row("/movies/film.mkv", version=(1, 19, 3))]
        out, _ = _render(rows, root=root)
        assert "film.mkv" in out

    def test_path_outside_root_shown_absolute(self) -> None:
        root = Path("/movies")
        rows = [_row("/other/place/film.mkv", version=(1, 0, 0))]
        out, _ = _render(rows, root=root)
        assert "/other/place/film.mkv" in out

    def test_furnace_version_status(self) -> None:
        root = Path("/movies")
        rows = [_row("/movies/a.mkv", version=(1, 19, 3))]
        out, _ = _render(rows, root=root)
        assert "Furnace v1.19.3" in out

    def test_not_encoded_status(self) -> None:
        root = Path("/movies")
        rows = [_row("/movies/a.mkv", version=None)]
        out, _ = _render(rows, root=root)
        assert "not encoded" in out

    def test_video_codec_shown(self) -> None:
        root = Path("/movies")
        rows = [_row("/movies/a.mkv", version=None, video="h264")]
        out, _ = _render(rows, root=root)
        assert "h264" in out

    def test_no_video_stream_shows_dash(self) -> None:
        root = Path("/movies")
        # Only the video column is empty, so the lone dash must come from it.
        rows = [_row(
            "/movies/a.mka",
            version=None,
            video=None,
            audio=(AudioTrackSummary(language="eng", codec="aac", channels=2),),
            subtitles=(SubtitleTrackSummary(language="eng", codec="subrip"),),
        )]
        out, _ = _render(rows, root=root)
        assert out.count("—") == 1  # exactly the video column

    def test_audio_line_with_channels(self) -> None:
        root = Path("/movies")
        rows = [_row(
            "/movies/a.mkv",
            version=None,
            audio=(AudioTrackSummary(language="rus", codec="ac3", channels=2),),
        )]
        out, _ = _render(rows, root=root)
        assert "rus ac3 2ch" in out

    def test_audio_missing_language_is_und(self) -> None:
        root = Path("/movies")
        rows = [_row(
            "/movies/a.mkv",
            version=None,
            audio=(AudioTrackSummary(language=None, codec="aac", channels=6),),
        )]
        out, _ = _render(rows, root=root)
        assert "und aac 6ch" in out

    def test_audio_missing_channels_omits_ch(self) -> None:
        root = Path("/movies")
        rows = [_row(
            "/movies/a.mkv",
            version=None,
            audio=(AudioTrackSummary(language="eng", codec="aac", channels=None),),
        )]
        out, _ = _render(rows, root=root)
        assert "eng aac" in out
        assert "ch" not in out.split("eng aac")[1].split("\n")[0]

    def test_no_audio_shows_dash(self) -> None:
        root = Path("/movies")
        # Video and subs are populated, so the lone dash must be the audio column.
        rows = [_row(
            "/movies/a.mkv",
            version=None,
            audio=(),
            subtitles=(SubtitleTrackSummary(language="eng", codec="subrip"),),
        )]
        out, _ = _render(rows, root=root)
        assert out.count("—") == 1  # exactly the audio column

    def test_subtitle_line(self) -> None:
        root = Path("/movies")
        rows = [_row(
            "/movies/a.mkv",
            version=None,
            subtitles=(SubtitleTrackSummary(language="eng", codec="subrip"),),
        )]
        out, _ = _render(rows, root=root)
        assert "eng subrip" in out

    def test_subtitle_missing_language_is_und(self) -> None:
        root = Path("/movies")
        rows = [_row(
            "/movies/a.mkv",
            version=None,
            subtitles=(SubtitleTrackSummary(language=None, codec="ass"),),
        )]
        out, _ = _render(rows, root=root)
        assert "und ass" in out

    def test_no_subtitles_shows_dash(self) -> None:
        root = Path("/movies")
        # Video and audio are populated, so the lone dash must be the subs column.
        rows = [_row(
            "/movies/a.mkv",
            version=None,
            audio=(AudioTrackSummary(language="eng", codec="aac", channels=2),),
            subtitles=(),
        )]
        out, _ = _render(rows, root=root)
        assert out.count("—") == 1  # exactly the subs column


class TestMultiLineCells:
    def test_multiple_audio_tracks_each_on_own_line(self) -> None:
        root = Path("/movies")
        rows = [_row(
            "/movies/a.mkv",
            version=None,
            audio=(
                AudioTrackSummary(language="rus", codec="ac3", channels=2),
                AudioTrackSummary(language="eng", codec="dts", channels=6),
            ),
        )]
        out, _ = _render(rows, root=root)
        lines = out.splitlines()
        rus_line = next(line for line in lines if "rus ac3 2ch" in line)
        eng_line = next(line for line in lines if "eng dts 6ch" in line)
        assert rus_line != eng_line

    def test_multiple_subtitle_tracks_each_on_own_line(self) -> None:
        root = Path("/movies")
        rows = [_row(
            "/movies/a.mkv",
            version=None,
            subtitles=(
                SubtitleTrackSummary(language="eng", codec="subrip"),
                SubtitleTrackSummary(language="rus", codec="ass"),
            ),
        )]
        out, _ = _render(rows, root=root)
        lines = out.splitlines()
        eng_line = next(line for line in lines if "eng subrip" in line)
        rus_line = next(line for line in lines if "rus ass" in line)
        assert eng_line != rus_line


class TestUnreadableRow:
    def test_status_is_unreadable(self) -> None:
        root = Path("/movies")
        rows = [_row("/movies/broken.mkv", unreadable=True, video=None)]
        out, _ = _render(rows, root=root)
        assert "unreadable" in out
        assert "broken.mkv" in out

    def test_stream_columns_are_dashes(self) -> None:
        root = Path("/movies")
        rows = [_row(
            "/movies/broken.mkv",
            unreadable=True,
            video="hevc",  # ignored for unreadable rows
            audio=(AudioTrackSummary(language="eng", codec="aac", channels=2),),
            subtitles=(SubtitleTrackSummary(language="eng", codec="subrip"),),
        )]
        out, _ = _render(rows, root=root)
        # Even though stream fields are populated, an unreadable row hides them.
        assert "hevc" not in out
        assert "eng aac 2ch" not in out
        assert "eng subrip" not in out
        assert "—" in out


class TestStderr:
    def test_summary_goes_to_stderr_not_stdout(self) -> None:
        root = Path("/movies")
        rows = [_row("/movies/a.mkv", version=(1, 19, 3))]
        out, err = _render(rows, root=root, total=80)
        assert "1 of 80 shown" in err
        assert "shown" not in out

    def test_no_video_files_found_note_on_stderr(self) -> None:
        root = Path("/movies")
        out, err = _render([], root=root, total=0)
        assert "no video files found" in err
        assert "no video files found" not in out

    def test_empty_table_header_still_on_stdout(self) -> None:
        root = Path("/movies")
        out, _ = _render([], root=root, total=0)
        assert "File" in out
        assert "Status" in out

    def test_warnings_go_to_stderr(self) -> None:
        root = Path("/movies")
        rows = [_row("/movies/a.mkv", version=None)]
        out = io.StringIO()
        err = io.StringIO()
        render_scan_table(
            rows,
            root=root,
            total=1,
            warnings=("could not read broken.mkv",),
            file=out,
            err=err,
        )
        assert "could not read broken.mkv" in err.getvalue()
        assert "could not read broken.mkv" not in out.getvalue()


class TestDefaults:
    def test_defaults_to_stdout_and_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        root = Path("/movies")
        rows = [_row("/movies/a.mkv", version=(1, 0, 0))]
        render_scan_table(rows, root=root, total=1)
        captured = capsys.readouterr()
        assert "a.mkv" in captured.out
        assert "1 of 1 shown" in captured.err
