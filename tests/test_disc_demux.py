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
    audio_analyzer: MagicMock | None = None,
) -> DiscDemuxer:
    return DiscDemuxer(
        bd_port=bd_port or MagicMock(),
        dvd_port=dvd_port or MagicMock(),
        mkvmerge_path=mkvmerge_path,
        pcm_transcoder=pcm_transcoder,
        prober=prober,
        audio_analyzer=audio_analyzer,
    )


def _silence_analyzer() -> MagicMock:
    """AudioAnalyzer mock reporting any audio as silent first second (intro gap)."""
    a = MagicMock()
    a.first_second_rms_db.return_value = -100.0
    return a


def _loud_analyzer() -> MagicMock:
    """AudioAnalyzer mock reporting loud first second (outro gap, sync skipped)."""
    a = MagicMock()
    a.first_second_rms_db.return_value = -20.0
    return a


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


def _probe_with_durations(
    *,
    video_dur: str | None,
    audio_durs: list[str | None],
) -> MagicMock:
    """Return a MagicMock prober whose `probe()` yields the given track
    DURATION tag strings (e.g. "01:35:48.750000000"). None means "tag
    absent for that track".
    """
    streams: list[dict[str, object]] = []
    if video_dur is not None:
        streams.append({"codec_type": "video", "tags": {"DURATION": video_dur}})
    else:
        streams.append({"codec_type": "video", "tags": {}})
    for d in audio_durs:
        if d is not None:
            streams.append({"codec_type": "audio", "tags": {"DURATION": d}})
        else:
            streams.append({"codec_type": "audio", "tags": {}})
    prober = MagicMock()
    prober.probe.return_value = {"streams": streams}
    return prober


class TestMuxToMkvSyncCorrection:
    """When the muxed MKV shows audio shorter than video by more than a
    threshold (typical of multi-segment BD titles whose intro segment lacks
    PCM audio), `_mux_to_mkv` must re-run mkvmerge with `--sync` to push
    each audio track later in the timeline by the missing-prefix duration.
    """

    def _build(
        self,
        tmp_path: Path,
        *,
        video_dur: str | None,
        audio_durs: list[str | None],
        audio_filenames: list[str] | None = None,
        audio_analyzer: MagicMock | None = None,
    ) -> tuple[DiscDemuxer, list[Path], Path]:
        prober = _probe_with_durations(video_dur=video_dur, audio_durs=audio_durs)
        demuxer = _make_demuxer(
            mkvmerge_path=Path("/usr/bin/mkvmerge"),
            prober=prober,
            audio_analyzer=audio_analyzer,
        )
        video = tmp_path / "video.h264"
        video.write_bytes(b"video")
        names = audio_filenames or [f"audio{i}.flac" for i in range(len(audio_durs))]
        audio_files = []
        for name in names:
            p = tmp_path / name
            p.write_bytes(b"audio")
            audio_files.append(p)
        output_mkv = tmp_path / "out.mkv"
        return demuxer, [video, *audio_files], output_mkv

    def test_remux_when_audio_significantly_shorter(self, tmp_path: Path) -> None:
        """Intro-gap case: video 5748750ms, audio 5744755ms, first second of
        audio is silence → analyzer signals intro gap → resync 0:3995."""
        demuxer, files, output_mkv = self._build(
            tmp_path,
            video_dur="01:35:48.750000000",
            audio_durs=["01:35:44.755000000"],
            audio_analyzer=_silence_analyzer(),
        )

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, output_mkv)

        assert mock_run.call_count == 2
        second_cmd = mock_run.call_args_list[1][0][0]
        sync_idx = second_cmd.index("--sync")
        assert second_cmd[sync_idx + 1] == "0:3995"
        # --sync must precede the audio file it applies to
        audio_path = next(f for f in files if f.suffix == ".flac")
        assert second_cmd.index(str(audio_path)) > sync_idx

    def test_no_remux_when_durations_match(self, tmp_path: Path) -> None:
        demuxer, files, output_mkv = self._build(
            tmp_path,
            video_dur="01:35:48.750000000",
            audio_durs=["01:35:48.750000000"],
        )

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, output_mkv)

        assert mock_run.call_count == 1

    def test_no_remux_when_gap_below_threshold(self, tmp_path: Path) -> None:
        """100ms gap is within natural BD audio end-trim — no correction."""
        demuxer, files, output_mkv = self._build(
            tmp_path,
            video_dur="00:30:00.100000000",
            audio_durs=["00:30:00.000000000"],
        )

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, output_mkv)

        assert mock_run.call_count == 1

    def test_no_remux_when_audio_longer_than_video(self, tmp_path: Path) -> None:
        """Negative delta — never resync (would shift audio backwards)."""
        demuxer, files, output_mkv = self._build(
            tmp_path,
            video_dur="00:30:00.000000000",
            audio_durs=["00:30:00.500000000"],
        )

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, output_mkv)

        assert mock_run.call_count == 1

    def test_no_remux_when_prober_not_configured(self, tmp_path: Path) -> None:
        """Without a prober, `_mux_to_mkv` keeps legacy behaviour (single mux)."""
        demuxer = _make_demuxer(mkvmerge_path=Path("/usr/bin/mkvmerge"), prober=None)
        video = tmp_path / "video.h264"
        video.write_bytes(b"video")
        audio = tmp_path / "audio.flac"
        audio.write_bytes(b"audio")
        output_mkv = tmp_path / "out.mkv"

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv([video, audio], output_mkv)

        assert mock_run.call_count == 1

    def test_no_remux_when_probe_raises(self, tmp_path: Path) -> None:
        """Probe failure degrades gracefully — no second mux, no exception."""
        prober = MagicMock()
        prober.probe.side_effect = RuntimeError("ffprobe blew up")
        demuxer = _make_demuxer(mkvmerge_path=Path("/usr/bin/mkvmerge"), prober=prober)
        video = tmp_path / "video.h264"
        video.write_bytes(b"v")
        audio = tmp_path / "audio.flac"
        audio.write_bytes(b"a")
        output_mkv = tmp_path / "out.mkv"

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv([video, audio], output_mkv)

        assert mock_run.call_count == 1

    def test_no_remux_when_duration_tags_missing(self, tmp_path: Path) -> None:
        """No DURATION tags at all — cannot detect, skip correction."""
        demuxer, files, output_mkv = self._build(
            tmp_path,
            video_dur=None,
            audio_durs=[None],
        )

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, output_mkv)

        assert mock_run.call_count == 1

    def test_resync_applies_per_audio_track(self, tmp_path: Path) -> None:
        """Multiple intro-gap audio tracks all get --sync correction; both
        --sync and --language precede each respective audio file in the
        command line.
        """
        demuxer, files, output_mkv = self._build(
            tmp_path,
            video_dur="01:35:48.750000000",
            audio_durs=["01:35:44.755000000", "01:35:44.755000000"],
            audio_filenames=["audio1 [rus].flac", "audio2 [eng].ac3"],
            audio_analyzer=_silence_analyzer(),
        )

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, output_mkv)

        assert mock_run.call_count == 2
        cmd = mock_run.call_args_list[1][0][0]
        sync_indices = [i for i, tok in enumerate(cmd) if tok == "--sync"]
        lang_indices = [i for i, tok in enumerate(cmd) if tok == "--language"]
        assert len(sync_indices) == 2
        assert len(lang_indices) == 2
        for i in sync_indices:
            assert cmd[i + 1] == "0:3995"
        # For each audio file, both --sync and --language must precede it
        for name in ("audio1 [rus].flac", "audio2 [eng].ac3"):
            file_idx = cmd.index(str(tmp_path / name))
            preceding_syncs = [i for i in sync_indices if i < file_idx]
            preceding_langs = [i for i in lang_indices if i < file_idx]
            assert preceding_syncs
            assert preceding_langs

    def test_resync_only_audio_below_threshold_left_alone(self, tmp_path: Path) -> None:
        """When one audio track is short (intro gap) and another matches
        video, only the short one receives --sync."""
        demuxer, files, output_mkv = self._build(
            tmp_path,
            video_dur="01:35:48.750000000",
            audio_durs=["01:35:44.755000000", "01:35:48.750000000"],
            audio_filenames=["short [rus].flac", "ok [eng].ac3"],
            audio_analyzer=_silence_analyzer(),
        )

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, output_mkv)

        cmd = mock_run.call_args_list[1][0][0]
        sync_indices = [i for i, tok in enumerate(cmd) if tok == "--sync"]
        assert len(sync_indices) == 1
        short_path = tmp_path / "short [rus].flac"
        ok_path = tmp_path / "ok [eng].ac3"
        # --sync precedes the short file but does NOT precede the ok one
        assert sync_indices[0] < cmd.index(str(short_path))
        assert sync_indices[0] < cmd.index(str(ok_path))  # appears in cmd
        # Verify no --sync between sync_indices[0]+2 and the ok file
        between = cmd[sync_indices[0] + 2 : cmd.index(str(ok_path))]
        assert "--sync" not in between

    def test_subtitle_stream_does_not_break_detection(self, tmp_path: Path) -> None:
        """Streams with codec_type other than video/audio (e.g. subtitle) are
        ignored when computing offsets."""
        prober = MagicMock()
        prober.probe.return_value = {
            "streams": [
                {"codec_type": "video", "tags": {"DURATION": "01:35:48.750000000"}},
                {"codec_type": "audio", "tags": {"DURATION": "01:35:44.755000000"}},
                {"codec_type": "subtitle", "tags": {"DURATION": "00:30:00.000000000"}},
            ],
        }
        demuxer = _make_demuxer(
            mkvmerge_path=Path("/usr/bin/mkvmerge"),
            prober=prober,
            audio_analyzer=_silence_analyzer(),
        )
        video = tmp_path / "video.h264"
        video.write_bytes(b"v")
        audio = tmp_path / "audio.flac"
        audio.write_bytes(b"a")
        subs = tmp_path / "subs.sup"
        subs.write_bytes(b"s")
        output_mkv = tmp_path / "out.mkv"

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv([video, audio, subs], output_mkv)

        assert mock_run.call_count == 2
        cmd = mock_run.call_args_list[1][0][0]
        sync_idx = cmd.index("--sync")
        assert cmd[sync_idx + 1] == "0:3995"

    def test_audio_with_missing_duration_tag_is_skipped(self, tmp_path: Path) -> None:
        """When one audio stream has no DURATION tag, only the well-formed
        one is considered. The unparseable track gets no --sync."""
        demuxer, files, output_mkv = self._build(
            tmp_path,
            video_dur="01:35:48.750000000",
            audio_durs=["01:35:44.755000000", None],
            audio_filenames=["short [rus].flac", "untagged [eng].ac3"],
            audio_analyzer=_silence_analyzer(),
        )

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, output_mkv)

        assert mock_run.call_count == 2
        cmd = mock_run.call_args_list[1][0][0]
        sync_indices = [i for i, tok in enumerate(cmd) if tok == "--sync"]
        # Only the first audio file gets --sync; the second (no tag) is left as-is
        assert len(sync_indices) == 1
        short_path = tmp_path / "short [rus].flac"
        untagged_path = tmp_path / "untagged [eng].ac3"
        assert sync_indices[0] < cmd.index(str(short_path))
        between = cmd[sync_indices[0] + 2 : cmd.index(str(untagged_path))]
        assert "--sync" not in between

    def test_resync_skipped_when_file_stream_count_mismatch(self, tmp_path: Path) -> None:
        """If mkvmerge produced fewer/more audio streams than source audio
        files, we cannot map durations safely — skip the fix."""
        # Two source audio files but probe shows only one audio stream
        demuxer, files, output_mkv = self._build(
            tmp_path,
            video_dur="01:35:48.750000000",
            audio_durs=["01:35:44.755000000"],
            audio_filenames=["a.flac", "b.ac3"],
            audio_analyzer=_silence_analyzer(),
        )

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, output_mkv)

        assert mock_run.call_count == 1

    # --- audio analyzer gap-direction classification (1.16+) ---

    def test_no_remux_outro_gap_detected_via_loud_first_second(
        self, tmp_path: Path,
    ) -> None:
        """Audio short by 3s but first second is LOUD → outro-gap (end-credits
        without dub). Applying --sync would shift dub 3s late on whole film,
        which is the ``О чём говорят мужчины`` regression. Skip --sync.
        """  # noqa: RUF002 — Cyrillic film title is intentional
        demuxer, files, output_mkv = self._build(
            tmp_path,
            video_dur="01:35:48.750000000",
            audio_durs=["01:35:44.755000000"],
            audio_analyzer=_loud_analyzer(),
        )

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, output_mkv)

        assert mock_run.call_count == 1

    def test_no_remux_when_audio_analyzer_not_configured(
        self, tmp_path: Path,
    ) -> None:
        """Without an audio_analyzer we cannot classify gap direction.
        Safe default: do NOT apply --sync (avoids the outro-gap regression).
        """
        demuxer, files, output_mkv = self._build(
            tmp_path,
            video_dur="01:35:48.750000000",
            audio_durs=["01:35:44.755000000"],
            audio_analyzer=None,
        )

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, output_mkv)

        assert mock_run.call_count == 1

    def test_no_remux_when_analyzer_returns_none(self, tmp_path: Path) -> None:
        """analyzer.first_second_rms_db returning None (decode failure) →
        cannot classify → safe default = skip --sync."""
        analyzer = MagicMock()
        analyzer.first_second_rms_db.return_value = None
        demuxer, files, output_mkv = self._build(
            tmp_path,
            video_dur="01:35:48.750000000",
            audio_durs=["01:35:44.755000000"],
            audio_analyzer=analyzer,
        )

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, output_mkv)

        assert mock_run.call_count == 1
        analyzer.first_second_rms_db.assert_called_once()

    def test_no_remux_when_analyzer_raises(self, tmp_path: Path) -> None:
        """If analyzer raises (e.g. ffmpeg crashed), gap-direction is unknown.
        Safe default: skip --sync rather than corrupt the timeline."""
        analyzer = MagicMock()
        analyzer.first_second_rms_db.side_effect = RuntimeError("ffmpeg blew up")
        demuxer, files, output_mkv = self._build(
            tmp_path,
            video_dur="01:35:48.750000000",
            audio_durs=["01:35:44.755000000"],
            audio_analyzer=analyzer,
        )

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, output_mkv)

        assert mock_run.call_count == 1

    def test_mixed_gap_direction_only_intro_track_resyncs(
        self, tmp_path: Path,
    ) -> None:
        """Two short audio tracks: one is intro-gap (silent first sec), the
        other is outro-gap (loud first sec). Only intro-gap track gets --sync.
        """
        analyzer = MagicMock()
        # First analyzer call (audio1 short [rus]) → silence (intro gap)
        # Second call (audio2 [eng]) → loud (outro gap)
        analyzer.first_second_rms_db.side_effect = [-100.0, -20.0]
        demuxer, files, output_mkv = self._build(
            tmp_path,
            video_dur="01:35:48.750000000",
            audio_durs=["01:35:44.755000000", "01:35:44.755000000"],
            audio_filenames=["intro_gap [rus].flac", "outro_gap [eng].ac3"],
            audio_analyzer=analyzer,
        )

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, output_mkv)

        assert mock_run.call_count == 2
        cmd = mock_run.call_args_list[1][0][0]
        sync_indices = [i for i, tok in enumerate(cmd) if tok == "--sync"]
        assert len(sync_indices) == 1

        intro_path = tmp_path / "intro_gap [rus].flac"
        outro_path = tmp_path / "outro_gap [eng].ac3"
        # --sync precedes intro file, but no --sync between sync_indices[0]+2
        # and the outro file (which comes later in the command)
        intro_idx_in_cmd = cmd.index(str(intro_path))
        outro_idx_in_cmd = cmd.index(str(outro_path))
        assert sync_indices[0] < intro_idx_in_cmd
        between = cmd[sync_indices[0] + 2 : outro_idx_in_cmd]
        assert "--sync" not in between

    def test_analyzer_threshold_boundary_exactly_minus_50_is_outro_gap(
        self, tmp_path: Path,
    ) -> None:
        """RMS exactly at the threshold is treated as ``loud enough`` →
        outro gap → skip --sync. (Conservative: tie goes to skipping, since
        the cost of a wrong --sync is whole-film desync.)"""
        analyzer = MagicMock()
        analyzer.first_second_rms_db.return_value = -50.0
        demuxer, files, output_mkv = self._build(
            tmp_path,
            video_dur="01:35:48.750000000",
            audio_durs=["01:35:44.755000000"],
            audio_analyzer=analyzer,
        )

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, output_mkv)

        assert mock_run.call_count == 1

    def test_analyzer_threshold_boundary_minus_50_001_is_intro_gap(
        self, tmp_path: Path,
    ) -> None:
        """RMS just below the threshold (-50.001 dB) → silence → intro gap →
        apply --sync."""
        analyzer = MagicMock()
        analyzer.first_second_rms_db.return_value = -50.001
        demuxer, files, output_mkv = self._build(
            tmp_path,
            video_dur="01:35:48.750000000",
            audio_durs=["01:35:44.755000000"],
            audio_analyzer=analyzer,
        )

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")) as mock_run:
            demuxer._mux_to_mkv(files, output_mkv)

        assert mock_run.call_count == 2
        cmd = mock_run.call_args_list[1][0][0]
        assert "--sync" in cmd

    def test_analyzer_only_invoked_for_short_audio_tracks(
        self, tmp_path: Path,
    ) -> None:
        """If an audio track matches video duration, analyzer is NOT called
        for it (no gap to classify). Only short tracks trigger analysis."""
        analyzer = MagicMock()
        analyzer.first_second_rms_db.return_value = -100.0
        demuxer, files, output_mkv = self._build(
            tmp_path,
            video_dur="01:35:48.750000000",
            audio_durs=["01:35:48.750000000", "01:35:44.755000000"],
            audio_filenames=["full [rus].flac", "short [eng].ac3"],
            audio_analyzer=analyzer,
        )

        with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")):
            demuxer._mux_to_mkv(files, output_mkv)

        # Analyzer called once — only for the short track
        assert analyzer.first_second_rms_db.call_count == 1
        called_path = analyzer.first_second_rms_db.call_args[0][0]
        assert called_path.name == "short [eng].ac3"


class TestParseMkvDurationTag:
    """`_parse_mkv_duration_tag` converts mkvmerge's DURATION tag string
    (HH:MM:SS.nnnnnnnnn) to integer milliseconds.
    """

    def test_typical_value(self) -> None:
        from furnace.services.disc_demuxer import _parse_mkv_duration_tag
        assert _parse_mkv_duration_tag("01:35:48.750000000") == 5748750

    def test_subsecond_precision_rounds_to_ms(self) -> None:
        from furnace.services.disc_demuxer import _parse_mkv_duration_tag
        # 5744.755000000 → 5744755 ms (exact)
        assert _parse_mkv_duration_tag("01:35:44.755000000") == 5744755

    def test_short_fractional_part_is_left_aligned(self) -> None:
        from furnace.services.disc_demuxer import _parse_mkv_duration_tag
        # "00:00:01.5" should be 1500 ms, not 1000 + 5
        assert _parse_mkv_duration_tag("00:00:01.5") == 1500

    def test_no_fractional_part(self) -> None:
        from furnace.services.disc_demuxer import _parse_mkv_duration_tag
        # If the regex requires a dot, this may return None — accept either,
        # but the typical mkvmerge tag always carries fractions.
        result = _parse_mkv_duration_tag("00:00:01")
        assert result in (None, 1000)

    def test_none_input(self) -> None:
        from furnace.services.disc_demuxer import _parse_mkv_duration_tag
        assert _parse_mkv_duration_tag(None) is None

    def test_empty_string(self) -> None:
        from furnace.services.disc_demuxer import _parse_mkv_duration_tag
        assert _parse_mkv_duration_tag("") is None

    def test_garbage_input(self) -> None:
        from furnace.services.disc_demuxer import _parse_mkv_duration_tag
        assert _parse_mkv_duration_tag("not a duration") is None

    def test_multi_hour(self) -> None:
        from furnace.services.disc_demuxer import _parse_mkv_duration_tag
        assert _parse_mkv_duration_tag("10:00:00.000000000") == 36_000_000


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
