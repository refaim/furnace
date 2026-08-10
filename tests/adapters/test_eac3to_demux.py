from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from furnace.adapters.eac3to import Eac3toAdapter


class TestDemuxTitlePathHandling:
    def test_demux_title_passes_absolute_disc_path(self, tmp_path: Path) -> None:
        disc_root = tmp_path / "src2" / "MOVIE_BDCLUB"
        (disc_root / "BDMV").mkdir(parents=True)
        output_dir = tmp_path / "out" / "title_1"
        output_dir.mkdir(parents=True)

        original_cwd = Path.cwd()
        os.chdir(tmp_path)
        try:
            relative_disc_path = Path("src2") / "MOVIE_BDCLUB" / "BDMV"
            assert not relative_disc_path.is_absolute()

            captured: dict[str, object] = {}

            def fake_run_tool(
                cmd: list[str | Path],
                on_output: object = None,
                on_progress_line: object = None,
                log_path: object = None,
                cwd: object = None,
            ) -> tuple[int, str]:
                captured["cmd"] = [str(c) for c in cmd]
                captured["cwd"] = cwd
                return (0, "")

            adapter = Eac3toAdapter(Path("C:/Tools/eac3to.exe"))

            with patch("furnace.adapters.eac3to.run_tool", side_effect=fake_run_tool):
                adapter.demux_title(
                    relative_disc_path,
                    title_num=1,
                    output_dir=output_dir,
                )

            cmd = captured["cmd"]
            assert isinstance(cmd, list)
            disc_path_arg = Path(cmd[1])
            assert disc_path_arg.is_absolute(), (
                f"disc_path passed to eac3to must be absolute so that cwd=output_dir "
                f"does not break resolution; got {disc_path_arg}"
            )
            assert disc_path_arg.resolve() == relative_disc_path.resolve()
        finally:
            os.chdir(original_cwd)


def _demux_into(output_dir: Path, names: list[str]) -> list[Path]:
    for name in names:
        (output_dir / name).write_bytes(b"x")

    adapter = Eac3toAdapter(Path("C:/Tools/eac3to.exe"))
    with patch("furnace.adapters.eac3to.run_tool", return_value=(0, "")):
        return adapter.demux_title(Path("C:/BD/BDMV"), title_num=1, output_dir=output_dir)


class TestDemuxTitleOrdering:
    def test_orders_by_track_number_not_by_name(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        names = [
            "00047 - 1 - Chapters.txt",
            "00047 - 2 - h264, 1080p24 (16-9).h264",
            "00047 - 3 - DTS Master Audio, [rus], 5.1 channels, 48kHz.dtsma",
            "00047 - 10 - Subtitle (PGS), [eng].sup",
            "00047 - 11 - Subtitle (PGS), [spa].sup",
        ]

        result = _demux_into(output_dir, names)

        assert [p.name for p in result] == names

    def test_video_stays_ahead_of_audio_with_ten_plus_tracks(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        names = [f"BD - {n} - AC3, [eng], 5.1 channels, 448kbps, 48kHz.ac3" for n in range(3, 13)]
        names.append("BD - 2 - MPEG2, 1080p24 (16-9).mkv")

        result = _demux_into(output_dir, names)

        assert result[0].name == "BD - 2 - MPEG2, 1080p24 (16-9).mkv"
        assert [p.name for p in result[1:]] == [
            f"BD - {n} - AC3, [eng], 5.1 channels, 448kbps, 48kHz.ac3" for n in range(3, 13)
        ]

    def test_keeps_original_title_in_name(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        names = [
            "src - 2 - AC3, [rus], 5.1 channels, 448kbps, 48kHz, 'Дубляж Мосфильм'.ac3",
            "src - 1 - h264, 320x240 24p.h264",
        ]

        result = _demux_into(output_dir, names)

        assert [p.name for p in result] == [names[1], names[0]]

    def test_unnumbered_chapters_file_does_not_break_the_order(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        names = [
            "00000 - Chapters.txt",
            "00000 - 2 - h264, 1080p24.h264",
            "00000 - 3 - AC3, [rus], 3-1 channels, 640kbps, 48kHz.ac3",
            "00000 - 10 - DTS-HD Master Audio, [rus], 2.0 (L R) channels, 24 bits, 48kHz.dtsma",
            "00000 - 16 - Subtitle (PGS), [eng].sup",
        ]

        result = _demux_into(output_dir, names)

        assert [p.name for p in result] == [
            "00000 - 2 - h264, 1080p24.h264",
            "00000 - 3 - AC3, [rus], 3-1 channels, 640kbps, 48kHz.ac3",
            "00000 - 10 - DTS-HD Master Audio, [rus], 2.0 (L R) channels, 24 bits, 48kHz.dtsma",
            "00000 - 16 - Subtitle (PGS), [eng].sup",
            "00000 - Chapters.txt",
        ]

    def test_unnumbered_files_keep_name_order_among_themselves(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        names = ["00000 - Chapters.txt", "00000 - 2 - h264, 1080p24.h264", "00000 - Attachment.bin"]

        result = _demux_into(output_dir, names)

        assert [p.name for p in result] == [
            "00000 - 2 - h264, 1080p24.h264",
            "00000 - Attachment.bin",
            "00000 - Chapters.txt",
        ]

    def test_unnumbered_file_alone_does_not_warn(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        with caplog.at_level(logging.WARNING, logger="furnace.adapters.eac3to"):
            _demux_into(output_dir, ["00000 - Chapters.txt", "00000 - 2 - h264, 1080p24.h264"])

        assert caplog.records == []

    def test_falls_back_to_name_order_without_track_numbers(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        names = ["b.ac3", "a.mkv"]

        result = _demux_into(output_dir, names)

        assert [p.name for p in result] == ["a.mkv", "b.ac3"]

    def test_falls_back_when_numbers_are_not_unique(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        names = [
            "Rocky - 3 - Special - 2 - AC3, [eng], 5.1 channels.ac3",
            "Rocky - 3 - Special - 1 - h264, 1080p24.mkv",
        ]

        result = _demux_into(output_dir, names)

        assert [p.name for p in result] == sorted(names)

    def test_warns_when_nothing_carries_a_track_number(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        with caplog.at_level(logging.WARNING, logger="furnace.adapters.eac3to"):
            _demux_into(output_dir, ["b.ac3", "a.mkv"])

        assert any("track number" in r.message for r in caplog.records)

    def test_warns_when_numbers_are_ambiguous(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        names = [
            "Rocky - 3 - Special - 2 - AC3, [eng], 5.1 channels.ac3",
            "Rocky - 3 - Special - 1 - h264, 1080p24.mkv",
        ]

        with caplog.at_level(logging.WARNING, logger="furnace.adapters.eac3to"):
            _demux_into(output_dir, names)

        assert any("track number" in r.message for r in caplog.records)

    def test_ignores_subdirectories(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        (output_dir / "BD - 3 - nested").mkdir()

        result = _demux_into(output_dir, ["BD - 2 - h264, 1080p24.mkv"])

        assert [p.name for p in result] == ["BD - 2 - h264, 1080p24.mkv"]
