from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from furnace.core.models import DiscSource, DiscTitle, DiscType
from furnace.services.disc_demuxer import DiscDemuxer


def _make_demuxer(
    bd_port: MagicMock | None = None,
    dvd_port: MagicMock | None = None,
) -> DiscDemuxer:
    return DiscDemuxer(
        bd_port=bd_port or MagicMock(),
        dvd_port=dvd_port or MagicMock(),
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
