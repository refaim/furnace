from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from furnace.core.models import DiscSource, DiscTitle, DiscType
from furnace.services.disc_demuxer import DiscDemuxer


def _make_demuxer(
    bd_port: MagicMock | None = None,
    dvd_port: MagicMock | None = None,
    mkvmerge_path: Path | None = None,
    pcm_transcoder: MagicMock | None = None,
    prober: MagicMock | None = None,
) -> DiscDemuxer:
    return DiscDemuxer(
        bd_port=bd_port or MagicMock(),
        dvd_port=dvd_port or MagicMock(),
        mkvmerge_path=mkvmerge_path,
        pcm_transcoder=pcm_transcoder,
        prober=prober,
    )


class TestDiscDetection:
    def test_detect_bluray(self, tmp_path: Path) -> None:
        bdmv = tmp_path / "movie" / "BDMV"
        bdmv.mkdir(parents=True)
        demuxer = _make_demuxer()
        discs = demuxer.detect(tmp_path)
        assert len(discs) == 1
        assert discs[0].disc_type == DiscType.BLURAY
        assert discs[0].path == bdmv

    def test_detect_dvd(self, tmp_path: Path) -> None:
        video_ts = tmp_path / "movie" / "VIDEO_TS"
        video_ts.mkdir(parents=True)
        demuxer = _make_demuxer()
        discs = demuxer.detect(tmp_path)
        assert len(discs) == 1
        assert discs[0].disc_type == DiscType.DVD
        assert discs[0].path == video_ts

    def test_detect_multiple_discs(self, tmp_path: Path) -> None:
        (tmp_path / "bd" / "BDMV").mkdir(parents=True)
        (tmp_path / "dvd" / "VIDEO_TS").mkdir(parents=True)
        demuxer = _make_demuxer()
        discs = demuxer.detect(tmp_path)
        assert len(discs) == 2
        types = {d.disc_type for d in discs}
        assert types == {DiscType.DVD, DiscType.BLURAY}

    def test_detect_no_discs(self, tmp_path: Path) -> None:
        (tmp_path / "movie.mkv").touch()
        demuxer = _make_demuxer()
        discs = demuxer.detect(tmp_path)
        assert discs == []

    def test_detect_recursive(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c" / "BDMV"
        deep.mkdir(parents=True)
        demuxer = _make_demuxer()
        discs = demuxer.detect(tmp_path)
        assert len(discs) == 1
        assert discs[0].path == deep

    def test_detect_ignores_furnace_demux_dir(self, tmp_path: Path) -> None:
        demux_dir = tmp_path / ".furnace_demux"
        demux_dir.mkdir()
        (demux_dir / "BDMV").mkdir()
        (tmp_path / "real" / "BDMV").mkdir(parents=True)
        demuxer = _make_demuxer()
        discs = demuxer.detect(tmp_path)
        assert len(discs) == 1
        assert ".furnace_demux" not in str(discs[0].path)


class TestDemux:
    def test_skip_already_demuxed(self, tmp_path: Path) -> None:
        """Title with .done marker and existing MKV is not re-demuxed."""
        demux_dir = tmp_path / ".furnace_demux"
        demux_dir.mkdir()
        mkv = demux_dir / "movie_title_1.mkv"
        mkv.write_bytes(b"x" * 1000)
        (demux_dir / "movie_title_1.done").touch()

        port = MagicMock()
        demuxer = _make_demuxer(bd_port=port)

        disc = DiscSource(path=tmp_path / "movie" / "BDMV", disc_type=DiscType.BLURAY)
        title = DiscTitle(number=1, duration_s=6000.0, raw_label="1) test, 1:40:00")

        result = demuxer.demux(
            discs=[disc],
            selected_titles={disc: [title]},
            demux_dir=demux_dir,
        )
        port.demux_title.assert_not_called()
        assert len(result) == 1
        assert result[0] == mkv

    def test_demux_creates_files(self, tmp_path: Path) -> None:
        """Successful demux creates MKV and .done marker."""
        demux_dir = tmp_path / ".furnace_demux"

        port = MagicMock()

        def fake_demux(disc_path: Path, title_num: int, output_dir: Path, on_progress: object = None) -> list[Path]:
            mkv = output_dir / "title_t00.mkv"
            mkv.write_bytes(b"video data")
            return [mkv]

        port.demux_title.side_effect = fake_demux
        demuxer = _make_demuxer(bd_port=port)

        disc = DiscSource(path=tmp_path / "movie" / "BDMV", disc_type=DiscType.BLURAY)
        title = DiscTitle(number=1, duration_s=6000.0, raw_label="1) test, 1:40:00")

        result = demuxer.demux(
            discs=[disc],
            selected_titles={disc: [title]},
            demux_dir=demux_dir,
        )
        port.demux_title.assert_called_once()
        assert len(result) == 1
        assert result[0].name == "movie_title_1.mkv"
        assert (demux_dir / "movie_title_1.done").exists()

    def test_routes_dvd_to_dvd_port(self, tmp_path: Path) -> None:
        """DVD discs are routed to the dvd_port."""
        demux_dir = tmp_path / ".furnace_demux"

        bd_port = MagicMock()
        dvd_port = MagicMock()

        def fake_demux(disc_path: Path, title_num: int, output_dir: Path, on_progress: object = None) -> list[Path]:
            mkv = output_dir / "title_t00.mkv"
            mkv.write_bytes(b"video data")
            return [mkv]

        dvd_port.demux_title.side_effect = fake_demux
        demuxer = _make_demuxer(bd_port=bd_port, dvd_port=dvd_port)

        disc = DiscSource(path=tmp_path / "movie" / "VIDEO_TS", disc_type=DiscType.DVD)
        title = DiscTitle(number=4, duration_s=4352.0, raw_label="Title #4 was added")

        demuxer.demux(
            discs=[disc],
            selected_titles={disc: [title]},
            demux_dir=demux_dir,
        )
        dvd_port.demux_title.assert_called_once()
        bd_port.demux_title.assert_not_called()

    def test_demux_failure_raises(self, tmp_path: Path) -> None:
        """RuntimeError from adapter propagates."""
        demux_dir = tmp_path / ".furnace_demux"

        port = MagicMock()
        port.demux_title.side_effect = RuntimeError("demux failed")

        demuxer = _make_demuxer(bd_port=port)

        disc = DiscSource(path=tmp_path / "movie" / "BDMV", disc_type=DiscType.BLURAY)
        title = DiscTitle(number=1, duration_s=6000.0, raw_label="1) test, 1:40:00")

        with pytest.raises(RuntimeError, match="demux failed"):
            demuxer.demux(
                discs=[disc],
                selected_titles={disc: [title]},
                demux_dir=demux_dir,
            )

    def test_done_marker_no_mkv_files_redemuxes(self, tmp_path: Path) -> None:
        """Done marker exists but no MKV files -> re-demux (not skip)."""
        demux_dir = tmp_path / ".furnace_demux"
        demux_dir.mkdir()
        # Create done marker but NO .mkv file
        (demux_dir / "movie_title_1.done").touch()

        port = MagicMock()

        def fake_demux(disc_path: Path, title_num: int, output_dir: Path, on_progress: object = None) -> list[Path]:
            mkv = output_dir / "title_t00.mkv"
            mkv.write_bytes(b"video data")
            return [mkv]

        port.demux_title.side_effect = fake_demux
        demuxer = _make_demuxer(bd_port=port)

        disc = DiscSource(path=tmp_path / "movie" / "BDMV", disc_type=DiscType.BLURAY)
        title = DiscTitle(number=1, duration_s=6000.0, raw_label="1) test")

        result = demuxer.demux(
            discs=[disc],
            selected_titles={disc: [title]},
            demux_dir=demux_dir,
        )

        # Should have re-demuxed since no MKV was found
        port.demux_title.assert_called_once()
        assert len(result) == 1
        assert result[0].name == "movie_title_1.mkv"

    def test_title_dir_exists_cleaned_before_demux(self, tmp_path: Path) -> None:
        """title_dir already exists -> rmtree before demux."""
        demux_dir = tmp_path / ".furnace_demux"
        demux_dir.mkdir()

        # Pre-create the title dir with some leftover file
        title_dir = demux_dir / "movie_title_1"
        title_dir.mkdir()
        leftover = title_dir / "leftover.h264"
        leftover.write_bytes(b"old data")

        port = MagicMock()

        def fake_demux(disc_path: Path, title_num: int, output_dir: Path, on_progress: object = None) -> list[Path]:
            # The leftover file should NOT be here after rmtree
            assert not (output_dir / "leftover.h264").exists()
            mkv = output_dir / "title_t00.mkv"
            mkv.write_bytes(b"video data")
            return [mkv]

        port.demux_title.side_effect = fake_demux
        demuxer = _make_demuxer(bd_port=port)

        disc = DiscSource(path=tmp_path / "movie" / "BDMV", disc_type=DiscType.BLURAY)
        title = DiscTitle(number=1, duration_s=6000.0, raw_label="1) test")

        result = demuxer.demux(
            discs=[disc],
            selected_titles={disc: [title]},
            demux_dir=demux_dir,
        )

        port.demux_title.assert_called_once()
        assert len(result) == 1


class TestListTitles:
    def test_list_titles_delegates_to_correct_port(self) -> None:
        """list_titles() delegates to the correct port based on disc type."""
        bd_port = MagicMock()
        dvd_port = MagicMock()

        bd_titles = [DiscTitle(number=1, duration_s=6000.0, raw_label="main")]
        bd_port.list_titles.return_value = bd_titles
        dvd_titles = [DiscTitle(number=4, duration_s=3600.0, raw_label="dvd title")]
        dvd_port.list_titles.return_value = dvd_titles

        demuxer = _make_demuxer(bd_port=bd_port, dvd_port=dvd_port)

        bd_disc = DiscSource(path=Path("/bd/BDMV"), disc_type=DiscType.BLURAY)
        result_bd = demuxer.list_titles(bd_disc)
        bd_port.list_titles.assert_called_once_with(bd_disc.path)
        assert result_bd == bd_titles

        dvd_disc = DiscSource(path=Path("/dvd/VIDEO_TS"), disc_type=DiscType.DVD)
        result_dvd = demuxer.list_titles(dvd_disc)
        dvd_port.list_titles.assert_called_once_with(dvd_disc.path)
        assert result_dvd == dvd_titles


class TestNeedsMuxing:
    def test_single_mkv_no_muxing(self) -> None:
        """Single .mkv file -> no muxing needed."""
        assert DiscDemuxer._needs_muxing([Path("title.mkv")]) is False

    def test_multiple_files_needs_muxing(self) -> None:
        """Multiple track files -> muxing needed."""
        files = [Path("video.h264"), Path("audio.ac3"), Path("subs.sup")]
        assert DiscDemuxer._needs_muxing(files) is True

    def test_mkv_plus_extra_needs_muxing(self) -> None:
        """MKV plus additional files -> muxing needed."""
        files = [Path("title.mkv"), Path("extra.ac3")]
        assert DiscDemuxer._needs_muxing(files) is True

    def test_multiple_mkv_needs_muxing(self) -> None:
        """Multiple MKV files -> muxing needed."""
        files = [Path("a.mkv"), Path("b.mkv")]
        assert DiscDemuxer._needs_muxing(files) is True

    def test_empty_needs_muxing(self) -> None:
        """Empty file list -> needs muxing (no single MKV)."""
        assert DiscDemuxer._needs_muxing([]) is True


class TestMuxToMkv:
    def test_mux_to_mkv_calls_run_tool(self, tmp_path: Path) -> None:
        """_mux_to_mkv calls run_tool with correct mkvmerge command."""
        mkvmerge_path = Path("/usr/bin/mkvmerge")
        demuxer = _make_demuxer(mkvmerge_path=mkvmerge_path)

        video = tmp_path / "video.h264"
        video.write_bytes(b"video")
        audio = tmp_path / "audio [eng].ac3"
        audio.write_bytes(b"audio")
        output_mkv = tmp_path / "output.mkv"

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv([video, audio], output_mkv)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == str(mkvmerge_path)
        assert "-o" in cmd
        assert str(output_mkv) in cmd
        assert str(video) in cmd
        assert str(audio) in cmd

    def test_mux_to_mkv_with_chapters(self, tmp_path: Path) -> None:
        """_mux_to_mkv includes --chapters when a .txt chapters file is present."""
        mkvmerge_path = Path("/usr/bin/mkvmerge")
        demuxer = _make_demuxer(mkvmerge_path=mkvmerge_path)

        video = tmp_path / "video.h264"
        video.write_bytes(b"video")
        chapters = tmp_path / "chapters.txt"
        # Write valid OGM chapters so fix_chapters_file doesn't crash
        chapters.write_text(
            "CHAPTER01=00:00:00.000\nCHAPTER01NAME=Chapter 1\n",
            encoding="utf-8",
        )
        output_mkv = tmp_path / "output.mkv"

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv([video, chapters], output_mkv)

        cmd = mock_run.call_args[0][0]
        assert "--chapters" in cmd
        assert str(chapters) in cmd

    def test_mux_to_mkv_fixes_mojibake_chapters(self, tmp_path: Path) -> None:
        """_mux_to_mkv fixes mojibake in chapters file when detected."""
        mkvmerge_path = Path("/usr/bin/mkvmerge")
        demuxer = _make_demuxer(mkvmerge_path=mkvmerge_path)

        video = tmp_path / "video.h264"
        video.write_bytes(b"video")
        chapters = tmp_path / "chapters.txt"
        # Write OGM chapters with mojibake: UTF-8 "Глава" encoded as Latin-1
        mojibake_name = "Глава".encode().decode("latin-1")
        chapters.write_text(
            f"CHAPTER01=00:00:00.000\nCHAPTER01NAME={mojibake_name}\n",
            encoding="utf-8",
        )
        output_mkv = tmp_path / "output.mkv"

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")):
            demuxer._mux_to_mkv([video, chapters], output_mkv)

        # Verify chapters were fixed in-place
        fixed_text = chapters.read_text(encoding="utf-8")
        assert "Глава" in fixed_text

    def test_mux_to_mkv_language_from_filename(self, tmp_path: Path) -> None:
        """_mux_to_mkv extracts language from [xxx] pattern in filename."""
        mkvmerge_path = Path("/usr/bin/mkvmerge")
        demuxer = _make_demuxer(mkvmerge_path=mkvmerge_path)

        audio = tmp_path / "audio [rus].ac3"
        audio.write_bytes(b"audio")
        output_mkv = tmp_path / "output.mkv"

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv([audio], output_mkv)

        cmd = mock_run.call_args[0][0]
        lang_idx = cmd.index("--language")
        assert cmd[lang_idx + 1] == "0:rus"

    def test_mux_to_mkv_no_mkvmerge_raises(self) -> None:
        """_mux_to_mkv raises RuntimeError if mkvmerge_path is None."""
        demuxer = _make_demuxer(mkvmerge_path=None)
        with pytest.raises(RuntimeError, match="mkvmerge path not configured"):
            demuxer._mux_to_mkv([], Path("/out.mkv"))

    def test_mux_to_mkv_failure_raises(self, tmp_path: Path) -> None:
        """_mux_to_mkv raises RuntimeError if mkvmerge returns error code >= 2."""
        mkvmerge_path = Path("/usr/bin/mkvmerge")
        demuxer = _make_demuxer(mkvmerge_path=mkvmerge_path)

        video = tmp_path / "video.h264"
        video.write_bytes(b"video")
        output_mkv = tmp_path / "output.mkv"

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(2, "error msg")):
            with pytest.raises(RuntimeError, match="mkvmerge failed"):
                demuxer._mux_to_mkv([video], output_mkv)

    def test_mux_to_mkv_warning_rc1_ok(self, tmp_path: Path) -> None:
        """mkvmerge returns 1 for warnings -> no error raised."""
        mkvmerge_path = Path("/usr/bin/mkvmerge")
        demuxer = _make_demuxer(mkvmerge_path=mkvmerge_path)

        video = tmp_path / "video.h264"
        video.write_bytes(b"video")
        output_mkv = tmp_path / "output.mkv"

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(1, "warning")):
            # Should not raise
            demuxer._mux_to_mkv([video], output_mkv)

    def test_mux_to_mkv_on_output_callback(self, tmp_path: Path) -> None:
        """on_output callback is passed to run_tool."""
        mkvmerge_path = Path("/usr/bin/mkvmerge")
        demuxer = _make_demuxer(mkvmerge_path=mkvmerge_path)

        video = tmp_path / "video.h264"
        video.write_bytes(b"video")
        output_mkv = tmp_path / "output.mkv"
        on_output = MagicMock()

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv([video], output_mkv, on_output=on_output)

        assert mock_run.call_args[1]["on_output"] is on_output


class TestDemuxMuxingPath:
    def test_needs_muxing_triggers_mux_to_mkv(self, tmp_path: Path) -> None:
        """When demux output has multiple files, _mux_to_mkv is called."""
        demux_dir = tmp_path / ".furnace_demux"
        mkvmerge_path = Path("/usr/bin/mkvmerge")

        port = MagicMock()

        def fake_demux(disc_path: Path, title_num: int, output_dir: Path, on_progress: object = None) -> list[Path]:
            video = output_dir / "video.h264"
            video.write_bytes(b"video data")
            audio = output_dir / "audio [eng].ac3"
            audio.write_bytes(b"audio data")
            return [video, audio]

        port.demux_title.side_effect = fake_demux
        demuxer = _make_demuxer(bd_port=port, mkvmerge_path=mkvmerge_path)

        disc = DiscSource(path=tmp_path / "movie" / "BDMV", disc_type=DiscType.BLURAY)
        title = DiscTitle(number=1, duration_s=6000.0, raw_label="1) test")

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            result = demuxer.demux(
                discs=[disc],
                selected_titles={disc: [title]},
                demux_dir=demux_dir,
            )

        mock_run.assert_called_once()
        assert len(result) == 1
        assert result[0].name == "movie_title_1.mkv"


class TestCleanPartial:
    def test_removes_mkv_files(self, tmp_path: Path) -> None:
        """_clean_partial removes MKV files matching the title prefix."""
        demux_dir = tmp_path
        mkv = demux_dir / "movie_title_1.mkv"
        mkv.write_bytes(b"partial")
        other_mkv = demux_dir / "other_title_2.mkv"
        other_mkv.write_bytes(b"unrelated")

        DiscDemuxer._clean_partial(demux_dir, "movie", 1)

        assert not mkv.exists()
        assert other_mkv.exists()

    def test_removes_done_marker(self, tmp_path: Path) -> None:
        """_clean_partial removes stale done marker."""
        demux_dir = tmp_path
        done = demux_dir / "movie_title_1.done"
        done.touch()

        DiscDemuxer._clean_partial(demux_dir, "movie", 1)

        assert not done.exists()

    def test_no_files_to_clean(self, tmp_path: Path) -> None:
        """_clean_partial does nothing when no matching files exist."""
        demux_dir = tmp_path
        # Just make sure it doesn't crash
        DiscDemuxer._clean_partial(demux_dir, "movie", 1)


def _source_prober(
    *,
    video_start: str | None,
    audio_starts: list[str | None],
    include_video: bool = True,
) -> MagicMock:
    """Prober mock whose `probe()` returns source-m2ts streams with
    ffprobe-style string ``start_time`` values (as ffprobe emits them).
    ``None`` in a position omits the ``start_time`` field for that stream;
    ``include_video=False`` emits no video stream at all.
    """
    streams: list[dict[str, object]] = []
    if include_video:
        v: dict[str, object] = {"codec_type": "video"}
        if video_start is not None:
            v["start_time"] = video_start
        streams.append(v)
    for s in audio_starts:
        a: dict[str, object] = {"codec_type": "audio"}
        if s is not None:
            a["start_time"] = s
        streams.append(a)
    p = MagicMock()
    p.probe.return_value = {"streams": streams}
    return p


class TestSourceTruthSyncOffsets:
    """Source-truth A/V offset (2.1.3+): when a BD title maps to a single
    source ``.m2ts``, ``--sync`` is decided solely by the REAL audio-vs-video
    start offset read from that m2ts. There is no duration/RMS heuristic:
    titles whose audio is merely shorter than the video (trailing/outro gap)
    are left untouched — the fix for the ``О чём говорят мужчины`` whole-film
    desync. When the offset cannot be determined, no ``--sync`` is applied.
    """  # noqa: RUF002 — Cyrillic film title is intentional

    def _build(
        self,
        tmp_path: Path,
        *,
        prober: MagicMock,
        audio_filenames: list[str] | None = None,
    ) -> tuple[DiscDemuxer, list[Path], Path, Path]:
        demuxer = _make_demuxer(mkvmerge_path=Path("/usr/bin/mkvmerge"), prober=prober)
        video = tmp_path / "video.h264"
        video.write_bytes(b"v")
        names = audio_filenames or ["audio [rus].dtsma"]
        audio_files = []
        for n in names:
            p = tmp_path / n
            p.write_bytes(b"a")
            audio_files.append(p)
        m2ts = tmp_path / "00001.m2ts"
        output_mkv = tmp_path / "out.mkv"
        return demuxer, [video, *audio_files], output_mkv, m2ts

    def test_aligned_source_no_sync(self, tmp_path: Path) -> None:
        """Audio & video start at the same source PTS → offset 0 → no --sync,
        even though the audio may be shorter than the video downstream."""
        prober = _source_prober(video_start="5.866667", audio_starts=["5.866667"])
        demuxer, files, out, m2ts = self._build(tmp_path, prober=prober)
        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, out, source_segments=[m2ts])
        assert mock_run.call_count == 1
        prober.probe.assert_called_once_with(m2ts)

    def test_subtitle_stream_in_source_is_ignored(self, tmp_path: Path) -> None:
        """Non-video/non-audio source streams (e.g. PGS subtitles) are skipped
        when reading start times. A subtitle must not be miscounted as audio:
        with one genuinely-delayed audio track plus a subtitle, --sync is still
        applied at the audio's real offset (a miscount would trip the
        count-mismatch guard and silently skip the fix)."""
        prober = MagicMock()
        prober.probe.return_value = {
            "streams": [
                {"codec_type": "video", "start_time": "5.0"},
                {"codec_type": "audio", "start_time": "8.0"},
                {"codec_type": "subtitle", "start_time": "5.0"},
            ],
        }
        demuxer, files, out, m2ts = self._build(tmp_path, prober=prober)
        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, out, source_segments=[m2ts])
        assert mock_run.call_count == 2
        cmd = mock_run.call_args_list[1][0][0]
        assert cmd[cmd.index("--sync") + 1] == "0:3000"

    def test_fractional_ms_offset_is_rounded(self, tmp_path: Path) -> None:
        """A sub-millisecond source offset is rounded to the nearest ms."""
        # 8.6007 - 5.0 = 3.6007 s → 3600.7 ms → round → 3601
        prober = _source_prober(video_start="5.0", audio_starts=["8.6007"])
        demuxer, files, out, m2ts = self._build(tmp_path, prober=prober)
        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, out, source_segments=[m2ts])
        assert mock_run.call_count == 2
        cmd = mock_run.call_args_list[1][0][0]
        assert cmd[cmd.index("--sync") + 1] == "0:3601"

    def test_real_intro_gap_applies_source_offset(self, tmp_path: Path) -> None:
        """Audio starts 3s after video in the source → --sync 0:3000."""
        prober = _source_prober(video_start="5.0", audio_starts=["8.0"])
        demuxer, files, out, m2ts = self._build(tmp_path, prober=prober)
        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, out, source_segments=[m2ts])
        assert mock_run.call_count == 2
        cmd = mock_run.call_args_list[1][0][0]
        sync_idx = cmd.index("--sync")
        assert cmd[sync_idx + 1] == "0:3000"
        audio_path = next(f for f in files if f.suffix == ".dtsma")
        assert cmd.index(str(audio_path)) > sync_idx

    def test_offset_below_threshold_no_sync(self, tmp_path: Path) -> None:
        """200ms source offset is below the 500ms threshold → no --sync."""
        prober = _source_prober(video_start="5.0", audio_starts=["5.2"])
        demuxer, files, out, m2ts = self._build(tmp_path, prober=prober)
        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, out, source_segments=[m2ts])
        assert mock_run.call_count == 1

    def test_negative_offset_no_sync(self, tmp_path: Path) -> None:
        """Audio starting before video (negative offset) is never --sync'd."""
        prober = _source_prober(video_start="8.0", audio_starts=["5.0"])
        demuxer, files, out, m2ts = self._build(tmp_path, prober=prober)
        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, out, source_segments=[m2ts])
        assert mock_run.call_count == 1

    def test_per_track_offsets_only_delayed_track_synced(
        self, tmp_path: Path,
    ) -> None:
        """One audio track delayed in source, the other aligned → only the
        delayed one receives --sync."""
        prober = _source_prober(video_start="5.0", audio_starts=["8.0", "5.0"])
        demuxer, files, out, m2ts = self._build(
            tmp_path,
            prober=prober,
            audio_filenames=["a1 [rus].dtsma", "a2 [eng].ac3"],
        )
        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, out, source_segments=[m2ts])
        assert mock_run.call_count == 2
        cmd = mock_run.call_args_list[1][0][0]
        sync_indices = [i for i, tok in enumerate(cmd) if tok == "--sync"]
        assert len(sync_indices) == 1
        a1 = tmp_path / "a1 [rus].dtsma"
        a2 = tmp_path / "a2 [eng].ac3"
        assert sync_indices[0] < cmd.index(str(a1))
        between = cmd[sync_indices[0] + 2 : cmd.index(str(a2))]
        assert "--sync" not in between

    def test_no_source_segments_no_sync(self, tmp_path: Path) -> None:
        """No source segments at all (e.g. label without an m2ts) → no probe,
        no --sync."""
        prober = _source_prober(video_start="5.0", audio_starts=["8.0"])
        demuxer, files, out, _m2ts = self._build(tmp_path, prober=prober)
        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, out, source_segments=None)
        assert mock_run.call_count == 1
        prober.probe.assert_not_called()

    def test_multi_segment_no_sync(self, tmp_path: Path) -> None:
        """More than one source segment → source-truth not attempted (we don't
        probe), no --sync. The audio stays at PTS 0 where eac3to aligned it."""
        prober = _source_prober(video_start="5.0", audio_starts=["8.0"])
        seg1 = tmp_path / "00001.m2ts"
        seg2 = tmp_path / "00002.m2ts"
        demuxer, files, out, _m2ts = self._build(tmp_path, prober=prober)
        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, out, source_segments=[seg1, seg2])
        assert mock_run.call_count == 1
        prober.probe.assert_not_called()

    def test_probe_error_no_sync(self, tmp_path: Path) -> None:
        """Source probe raising → cannot determine offset → no --sync."""
        prober = MagicMock()
        prober.probe.side_effect = RuntimeError("ffprobe blew up")
        demuxer, files, out, m2ts = self._build(tmp_path, prober=prober)
        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, out, source_segments=[m2ts])
        assert mock_run.call_count == 1
        prober.probe.assert_called_once_with(m2ts)

    def test_no_video_stream_no_sync(self, tmp_path: Path) -> None:
        """Source m2ts with no video stream → offset undeterminable → no --sync."""
        prober = _source_prober(
            video_start=None, audio_starts=["5.0"], include_video=False,
        )
        demuxer, files, out, m2ts = self._build(tmp_path, prober=prober)
        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, out, source_segments=[m2ts])
        assert mock_run.call_count == 1

    def test_video_without_start_time_no_sync(self, tmp_path: Path) -> None:
        """Video stream present but lacking start_time → no --sync."""
        prober = _source_prober(video_start=None, audio_starts=["5.0"])
        demuxer, files, out, m2ts = self._build(tmp_path, prober=prober)
        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, out, source_segments=[m2ts])
        assert mock_run.call_count == 1

    def test_audio_without_start_time_no_sync(self, tmp_path: Path) -> None:
        """An audio stream lacking start_time → mapping untrustworthy → no --sync."""
        prober = _source_prober(video_start="5.0", audio_starts=[None])
        demuxer, files, out, m2ts = self._build(tmp_path, prober=prober)
        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, out, source_segments=[m2ts])
        assert mock_run.call_count == 1

    def test_invalid_start_time_string_no_sync(self, tmp_path: Path) -> None:
        """A non-numeric start_time ('N/A') parses to None → no --sync."""
        prober = _source_prober(video_start="N/A", audio_starts=["5.0"])
        demuxer, files, out, m2ts = self._build(tmp_path, prober=prober)
        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, out, source_segments=[m2ts])
        assert mock_run.call_count == 1

    def test_audio_stream_count_mismatch_no_sync(self, tmp_path: Path) -> None:
        """Source audio-stream count != demuxed audio-file count → cannot map
        → no --sync."""
        # Source reports one audio stream, but two demuxed audio files exist.
        prober = _source_prober(video_start="5.0", audio_starts=["5.0"])
        demuxer, files, out, m2ts = self._build(
            tmp_path,
            prober=prober,
            audio_filenames=["a1 [rus].dtsma", "a2 [eng].ac3"],
        )
        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, out, source_segments=[m2ts])
        assert mock_run.call_count == 1


class TestBdSourceSegments:
    """`_bd_source_segments` maps a BD title's raw_label to its source m2ts
    paths under ``<BDMV>/STREAM``."""

    def test_single_segment(self, tmp_path: Path) -> None:
        demuxer = _make_demuxer()
        disc = DiscSource(path=tmp_path / "BDMV", disc_type=DiscType.BLURAY)
        title = DiscTitle(
            number=1, duration_s=5792.0, raw_label="1) 00001.mpls, 00001.m2ts, 1:36:32",
        )
        segments = demuxer._bd_source_segments(disc, title)
        assert segments == [tmp_path / "BDMV" / "STREAM" / "00001.m2ts"]

    def test_multi_segment(self, tmp_path: Path) -> None:
        demuxer = _make_demuxer()
        disc = DiscSource(path=tmp_path / "BDMV", disc_type=DiscType.BLURAY)
        title = DiscTitle(
            number=1,
            duration_s=5792.0,
            raw_label="1) 00800.mpls, 00800.m2ts+00801.m2ts, 1:36:32",
        )
        segments = demuxer._bd_source_segments(disc, title)
        assert segments == [
            tmp_path / "BDMV" / "STREAM" / "00800.m2ts",
            tmp_path / "BDMV" / "STREAM" / "00801.m2ts",
        ]

    def test_dvd_returns_none(self, tmp_path: Path) -> None:
        demuxer = _make_demuxer()
        disc = DiscSource(path=tmp_path / "VIDEO_TS", disc_type=DiscType.DVD)
        title = DiscTitle(number=1, duration_s=100.0, raw_label="1) title")
        assert demuxer._bd_source_segments(disc, title) is None

    def test_no_m2ts_in_label_returns_none(self, tmp_path: Path) -> None:
        demuxer = _make_demuxer()
        disc = DiscSource(path=tmp_path / "BDMV", disc_type=DiscType.BLURAY)
        title = DiscTitle(number=1, duration_s=100.0, raw_label="1) test")
        assert demuxer._bd_source_segments(disc, title) is None


class TestParseStartTime:
    """`_parse_start_time` parses ffprobe's string start_time to float."""

    def test_valid(self) -> None:
        from furnace.services.disc_demuxer import _parse_start_time
        assert _parse_start_time("5.866667") == pytest.approx(5.866667)

    def test_none(self) -> None:
        from furnace.services.disc_demuxer import _parse_start_time
        assert _parse_start_time(None) is None

    def test_invalid(self) -> None:
        from furnace.services.disc_demuxer import _parse_start_time
        assert _parse_start_time("N/A") is None

    def test_non_finite_rejected(self) -> None:
        """``inf``/``-inf``/``nan`` parse via float() but are rejected so the
        downstream offset arithmetic never sees a non-finite value."""
        from furnace.services.disc_demuxer import _parse_start_time
        assert _parse_start_time("inf") is None
        assert _parse_start_time("-inf") is None
        assert _parse_start_time("nan") is None


class TestDemuxThreadsSourceSegments:
    """End-to-end: demux() derives the source m2ts from the title and an
    aligned single-segment source produces NO --sync (the regression fix)."""

    def test_aligned_single_segment_no_resync(self, tmp_path: Path) -> None:
        demux_dir = tmp_path / ".furnace_demux"
        port = MagicMock()

        def fake_demux(
            disc_path: Path, title_num: int, output_dir: Path, on_progress: object = None,
        ) -> list[Path]:
            video = output_dir / "video.h264"
            video.write_bytes(b"video data")
            audio = output_dir / "audio [rus].dtsma"
            audio.write_bytes(b"audio data")
            return [video, audio]

        port.demux_title.side_effect = fake_demux
        prober = _source_prober(video_start="5.866667", audio_starts=["5.866667"])
        demuxer = _make_demuxer(
            bd_port=port,
            mkvmerge_path=Path("/usr/bin/mkvmerge"),
            prober=prober,
        )
        disc = DiscSource(path=tmp_path / "movie" / "BDMV", disc_type=DiscType.BLURAY)
        title = DiscTitle(
            number=1, duration_s=5792.0, raw_label="1) 00001.mpls, 00001.m2ts, 1:36:32",
        )
        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer.demux(
                discs=[disc],
                selected_titles={disc: [title]},
                demux_dir=demux_dir,
            )
        # Only the single mux ran — no --sync re-mux.
        mock_run.assert_called_once()
        prober.probe.assert_called_once_with(
            tmp_path / "movie" / "BDMV" / "STREAM" / "00001.m2ts",
        )


class TestW64Transcode:
    def _make_fake_transcoder(self, rc: int = 0) -> MagicMock:
        """Build a fake pcm_transcoder whose transcode_to_flac creates the
        output file (mimicking eac3to) and returns the given rc.
        """
        transcoder = MagicMock()

        def fake_transcode(input_path: Path, output_path: Path, on_progress: object = None) -> int:
            if rc == 0:
                output_path.write_bytes(b"fake flac data")
            return rc

        transcoder.transcode_to_flac.side_effect = fake_transcode
        return transcoder

    def test_demux_transcodes_w64_to_flac(self, tmp_path: Path) -> None:
        """A .w64 in demux output is transcoded to .flac; .w64 is deleted;
        mkvmerge receives the .flac.
        """
        demux_dir = tmp_path / ".furnace_demux"
        mkvmerge_path = Path("/usr/bin/mkvmerge")

        port = MagicMock()

        def fake_demux(disc_path: Path, title_num: int, output_dir: Path, on_progress: object = None) -> list[Path]:
            video = output_dir / "video.h264"
            video.write_bytes(b"video data")
            w64 = output_dir / "audio [eng].w64"
            w64.write_bytes(b"huge pcm data")
            return [video, w64]

        port.demux_title.side_effect = fake_demux
        transcoder = self._make_fake_transcoder(rc=0)
        demuxer = _make_demuxer(
            bd_port=port,
            mkvmerge_path=mkvmerge_path,
            pcm_transcoder=transcoder,
        )

        disc = DiscSource(path=tmp_path / "movie" / "BDMV", disc_type=DiscType.BLURAY)
        title = DiscTitle(number=1, duration_s=6000.0, raw_label="1) test")

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer.demux(
                discs=[disc],
                selected_titles={disc: [title]},
                demux_dir=demux_dir,
            )

        # Transcoder was invoked exactly once with the w64 as input
        transcoder.transcode_to_flac.assert_called_once()
        args = transcoder.transcode_to_flac.call_args
        input_path = args[0][0]
        output_path = args[0][1]
        assert input_path.suffix == ".w64"
        assert output_path.suffix == ".flac"
        assert output_path.stem == input_path.stem

        # mkvmerge was called with the .flac path, not the .w64
        cmd = mock_run.call_args[0][0]
        flac_args = [a for a in cmd if a.endswith(".flac")]
        w64_args = [a for a in cmd if a.endswith(".w64")]
        assert len(flac_args) == 1
        assert w64_args == []

    def test_demux_multiple_w64_files(self, tmp_path: Path) -> None:
        """Multiple .w64 files in one title are each transcoded independently."""
        demux_dir = tmp_path / ".furnace_demux"
        mkvmerge_path = Path("/usr/bin/mkvmerge")

        port = MagicMock()

        def fake_demux(disc_path: Path, title_num: int, output_dir: Path, on_progress: object = None) -> list[Path]:
            video = output_dir / "video.h264"
            video.write_bytes(b"v")
            w64_a = output_dir / "audio1 [eng].w64"
            w64_b = output_dir / "audio2 [rus].w64"
            w64_a.write_bytes(b"a")
            w64_b.write_bytes(b"b")
            return [video, w64_a, w64_b]

        port.demux_title.side_effect = fake_demux
        transcoder = self._make_fake_transcoder(rc=0)
        demuxer = _make_demuxer(
            bd_port=port,
            mkvmerge_path=mkvmerge_path,
            pcm_transcoder=transcoder,
        )

        disc = DiscSource(path=tmp_path / "movie" / "BDMV", disc_type=DiscType.BLURAY)
        title = DiscTitle(number=1, duration_s=6000.0, raw_label="1) test")

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer.demux(
                discs=[disc],
                selected_titles={disc: [title]},
                demux_dir=demux_dir,
            )

        assert transcoder.transcode_to_flac.call_count == 2
        cmd = mock_run.call_args[0][0]
        flac_args = [a for a in cmd if a.endswith(".flac")]
        assert len(flac_args) == 2

    def test_demux_transcode_failure_raises_and_keeps_w64(self, tmp_path: Path) -> None:
        """Non-zero rc from transcoder -> RuntimeError; .w64 stays on disk for
        post-mortem inspection (title_dir is NOT cleaned on failure).
        """
        demux_dir = tmp_path / ".furnace_demux"
        mkvmerge_path = Path("/usr/bin/mkvmerge")

        port = MagicMock()
        w64_holder: dict[str, Path] = {}

        def fake_demux(disc_path: Path, title_num: int, output_dir: Path, on_progress: object = None) -> list[Path]:
            video = output_dir / "video.h264"
            video.write_bytes(b"v")
            w64 = output_dir / "audio.w64"
            w64.write_bytes(b"a")
            w64_holder["path"] = w64
            return [video, w64]

        port.demux_title.side_effect = fake_demux
        transcoder = self._make_fake_transcoder(rc=2)
        demuxer = _make_demuxer(
            bd_port=port,
            mkvmerge_path=mkvmerge_path,
            pcm_transcoder=transcoder,
        )

        disc = DiscSource(path=tmp_path / "movie" / "BDMV", disc_type=DiscType.BLURAY)
        title = DiscTitle(number=1, duration_s=6000.0, raw_label="1) test")

        with pytest.raises(RuntimeError, match="transcode"):
            demuxer.demux(
                discs=[disc],
                selected_titles={disc: [title]},
                demux_dir=demux_dir,
            )

        # The .w64 must still be on disk (title_dir not cleaned on failure)
        assert w64_holder["path"].exists()

    def test_demux_w64_without_transcoder_raises(self, tmp_path: Path) -> None:
        """pcm_transcoder=None + .w64 in demux output -> RuntimeError (fail fast
        rather than silently dropping the track).
        """
        demux_dir = tmp_path / ".furnace_demux"

        port = MagicMock()

        def fake_demux(disc_path: Path, title_num: int, output_dir: Path, on_progress: object = None) -> list[Path]:
            video = output_dir / "video.h264"
            video.write_bytes(b"v")
            w64 = output_dir / "audio.w64"
            w64.write_bytes(b"a")
            return [video, w64]

        port.demux_title.side_effect = fake_demux
        demuxer = _make_demuxer(bd_port=port, pcm_transcoder=None)

        disc = DiscSource(path=tmp_path / "movie" / "BDMV", disc_type=DiscType.BLURAY)
        title = DiscTitle(number=1, duration_s=6000.0, raw_label="1) test")

        with pytest.raises(RuntimeError, match="pcm_transcoder"):
            demuxer.demux(
                discs=[disc],
                selected_titles={disc: [title]},
                demux_dir=demux_dir,
            )

    def test_demux_no_w64_skips_transcode(self, tmp_path: Path) -> None:
        """Regression: demux output without any .w64 does NOT invoke the
        transcoder, even when one is configured.
        """
        demux_dir = tmp_path / ".furnace_demux"
        mkvmerge_path = Path("/usr/bin/mkvmerge")

        port = MagicMock()

        def fake_demux(disc_path: Path, title_num: int, output_dir: Path, on_progress: object = None) -> list[Path]:
            video = output_dir / "video.h264"
            video.write_bytes(b"v")
            audio = output_dir / "audio [eng].ac3"
            audio.write_bytes(b"a")
            return [video, audio]

        port.demux_title.side_effect = fake_demux
        transcoder = MagicMock()
        demuxer = _make_demuxer(
            bd_port=port,
            mkvmerge_path=mkvmerge_path,
            pcm_transcoder=transcoder,
        )

        disc = DiscSource(path=tmp_path / "movie" / "BDMV", disc_type=DiscType.BLURAY)
        title = DiscTitle(number=1, duration_s=6000.0, raw_label="1) test")

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")):
            demuxer.demux(
                discs=[disc],
                selected_titles={disc: [title]},
                demux_dir=demux_dir,
            )

        transcoder.transcode_to_flac.assert_not_called()
