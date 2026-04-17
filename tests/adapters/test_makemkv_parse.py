from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from furnace.adapters.makemkv import MakemkvAdapter, _parse_duration


class TestParseMakemkvInfo:
    def test_parse_dvd_titles(self) -> None:
        """Parse typical makemkvcon info output."""
        output = (
            "MakeMKV v1.18.3 win(x64-release) started\n"
            "Title #1 has length of 30 seconds which is less than minimum title length\n"
            "Title #2 was added (1 cell(s), 0:07:49)\n"
            "Title #3 has length of 34 seconds which is less than minimum title length\n"
            "Title #4 was added (13 cell(s), 1:12:32)\n"
            "Operation successfully completed\n"
        )
        result = MakemkvAdapter._parse_info_output(output)
        assert len(result) == 2
        assert result[0].number == 2
        assert result[0].duration_s == pytest.approx(469.0)
        assert result[1].number == 4
        assert result[1].duration_s == pytest.approx(4352.0)

    def test_parse_empty_output(self) -> None:
        result = MakemkvAdapter._parse_info_output("")
        assert result == []

    def test_skipped_titles_not_included(self) -> None:
        """Titles that were skipped by makemkv are not in the result."""
        output = (
            "Title #1 has length of 30 seconds which is less than minimum title length and was therefore skipped\n"
        )
        result = MakemkvAdapter._parse_info_output(output)
        assert result == []

    def test_raw_label_contains_original_line(self) -> None:
        output = "Title #4 was added (13 cell(s), 1:12:32)\n"
        result = MakemkvAdapter._parse_info_output(output)
        assert "Title #4 was added" in result[0].raw_label


class TestParseDurationMakemkv:
    def test_m_ss_format(self) -> None:
        assert _parse_duration("1:23") == pytest.approx(83.0)

    def test_bad_input(self) -> None:
        assert _parse_duration("bad") == 0.0

    def test_h_mm_ss_format(self) -> None:
        assert _parse_duration("1:02:03") == pytest.approx(3723.0)


class TestListTitlesMakemkv:
    def test_list_titles_rc_nonzero_raises(self) -> None:
        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            return 1, "error"

        adapter = MakemkvAdapter(Path("makemkvcon.exe"))
        with patch("furnace.adapters.makemkv.run_tool", side_effect=fake_run_tool):
            with pytest.raises(RuntimeError, match="makemkvcon info failed"):
                adapter.list_titles(Path("/disc"))

    def test_list_titles_success(self) -> None:
        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            return 0, "Title #2 was added (1 cell(s), 0:07:49)\n"

        adapter = MakemkvAdapter(Path("makemkvcon.exe"))
        with patch("furnace.adapters.makemkv.run_tool", side_effect=fake_run_tool):
            titles = adapter.list_titles(Path("/disc"))
        assert len(titles) == 1
        assert titles[0].number == 2


class TestDemuxTitleMakemkv:
    def test_demux_happy_path(self, tmp_path: Path) -> None:
        """Happy path: list_titles returns the right title, demux creates an MKV file."""
        call_count = 0

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            nonlocal call_count
            str_cmd = [str(c) for c in cmd]
            call_count += 1
            if "info" in str_cmd:
                return 0, "Title #5 was added (3 cell(s), 1:00:00)\n"
            if "mkv" in str_cmd:
                # Simulate makemkvcon creating an MKV
                mkv_file = tmp_path / "out" / "title_t05.mkv"
                mkv_file.parent.mkdir(parents=True, exist_ok=True)
                mkv_file.write_text("fake mkv")
                return 0, ""
            return 1, "unexpected"

        adapter = MakemkvAdapter(Path("makemkvcon.exe"))
        output_dir = tmp_path / "out"
        with patch("furnace.adapters.makemkv.run_tool", side_effect=fake_run_tool):
            files = adapter.demux_title(Path("/disc"), title_num=5, output_dir=output_dir)
        assert len(files) == 1
        assert files[0].suffix == ".mkv"

    def test_demux_title_not_found(self, tmp_path: Path) -> None:
        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            return 0, "Title #2 was added (1 cell(s), 0:07:49)\n"

        adapter = MakemkvAdapter(Path("makemkvcon.exe"))
        output_dir = tmp_path / "out"
        with patch("furnace.adapters.makemkv.run_tool", side_effect=fake_run_tool):
            with pytest.raises(RuntimeError, match="Title 99 not found"):
                adapter.demux_title(Path("/disc"), title_num=99, output_dir=output_dir)

    def test_demux_no_mkv_files(self, tmp_path: Path) -> None:
        call_count = 0

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            nonlocal call_count
            call_count += 1
            str_cmd = [str(c) for c in cmd]
            if "info" in str_cmd:
                return 0, "Title #3 was added (1 cell(s), 0:05:00)\n"
            # Demux succeeds but creates no MKV files
            return 0, ""

        adapter = MakemkvAdapter(Path("makemkvcon.exe"))
        output_dir = tmp_path / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        with patch("furnace.adapters.makemkv.run_tool", side_effect=fake_run_tool):
            with pytest.raises(RuntimeError, match="no MKV files"):
                adapter.demux_title(Path("/disc"), title_num=3, output_dir=output_dir)

    def test_demux_rc_nonzero_raises(self, tmp_path: Path) -> None:
        call_count = 0

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            nonlocal call_count
            call_count += 1
            str_cmd = [str(c) for c in cmd]
            if "info" in str_cmd:
                return 0, "Title #3 was added (1 cell(s), 0:05:00)\n"
            # Demux fails
            return 1, "demux error"

        adapter = MakemkvAdapter(Path("makemkvcon.exe"))
        output_dir = tmp_path / "out"
        with patch("furnace.adapters.makemkv.run_tool", side_effect=fake_run_tool):
            with pytest.raises(RuntimeError, match="demux failed"):
                adapter.demux_title(Path("/disc"), title_num=3, output_dir=output_dir)


class TestMakemkvSetLogDir:
    def test_set_log_dir(self, tmp_path: Path) -> None:
        adapter = MakemkvAdapter(Path("makemkvcon.exe"))
        adapter.set_log_dir(tmp_path)
        assert adapter._log_dir == tmp_path

    def test_log_path_helper(self, tmp_path: Path) -> None:
        adapter = MakemkvAdapter(Path("makemkvcon.exe"), log_dir=tmp_path)
        assert adapter._log_path("test") == tmp_path / "makemkv_test.log"

    def test_log_path_none(self) -> None:
        adapter = MakemkvAdapter(Path("makemkvcon.exe"))
        assert adapter._log_path("test") is None
