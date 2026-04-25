from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from furnace.adapters.makemkv import MakemkvAdapter, _parse_makemkv_progress_line
from furnace.core.progress import ProgressSample


def test_prgv_basic() -> None:
    sample = _parse_makemkv_progress_line("PRGV:5,10,100")
    assert sample is not None
    assert sample.fraction == 0.05


def test_prgv_complete() -> None:
    sample = _parse_makemkv_progress_line("PRGV:100,100,100")
    assert sample is not None
    assert sample.fraction == 1.0


def test_prgv_zero_max_returns_none() -> None:
    assert _parse_makemkv_progress_line("PRGV:5,10,0") is None


def test_prgt_returns_none() -> None:
    assert _parse_makemkv_progress_line('PRGT:0,0,"Saving to MKV"') is None


def test_prgc_returns_none() -> None:
    assert _parse_makemkv_progress_line('PRGC:0,0,"Backing up disc"') is None


def test_msg_returns_none() -> None:
    assert _parse_makemkv_progress_line('MSG:1004,0,1,"Some message"') is None


def test_garbage_returns_none() -> None:
    assert _parse_makemkv_progress_line("not a progress line") is None
    assert _parse_makemkv_progress_line("PRGV:abc,def,ghi") is None
    assert _parse_makemkv_progress_line("") is None


class TestDemuxProgressWiring:
    """Cover the on_progress_line closure inside MakemkvAdapter.demux_title.

    The closure has three branches: sample is None (return False), sample
    present + callback None, sample present + callback present. All three
    must be exercised for 100% branch coverage.
    """

    @staticmethod
    def _make_fake_run_tool(
        tmp_path: Path,
        progress_lines: list[str],
        progress_returns: list[bool],
    ) -> Any:
        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            str_cmd = [str(c) for c in cmd]
            if "info" in str_cmd:
                return 0, "Title #5 was added (3 cell(s), 1:00:00)\n"
            # demux step: feed progress_lines through the closure
            if on_progress_line is not None:
                progress_returns.extend(on_progress_line(line) for line in progress_lines)
            mkv_file = tmp_path / "out" / "title_t05.mkv"
            mkv_file.parent.mkdir(parents=True, exist_ok=True)
            mkv_file.write_text("fake mkv")
            return 0, ""

        return fake_run_tool

    def test_prgv_after_saving_prgc_emits_sample(self, tmp_path: Path) -> None:
        """PRGV samples are forwarded only after the 'Saving to MKV file' PRGC."""
        samples: list[ProgressSample] = []
        progress_returns: list[bool] = []
        lines = [
            'PRGC:5017,0,"Saving to MKV file"',  # opens the gate (returns False, flows to log)
            "PRGV:50,100,100",                    # forwarded (gate is open)
        ]
        fake = self._make_fake_run_tool(tmp_path, lines, progress_returns)
        adapter = MakemkvAdapter(Path("makemkvcon.exe"))
        with patch("furnace.adapters.makemkv.run_tool", side_effect=fake):
            adapter.demux_title(
                Path("/disc"),
                title_num=5,
                output_dir=tmp_path / "out",
                on_progress=samples.append,
            )
        assert progress_returns == [False, True]
        assert samples == [ProgressSample(fraction=0.5)]

    def test_prgv_before_saving_prgc_is_dropped(self, tmp_path: Path) -> None:
        """PRGV before the 'Saving to MKV file' PRGC is consumed but not forwarded."""
        samples: list[ProgressSample] = []
        progress_returns: list[bool] = []
        lines = [
            'PRGC:3104,0,"Decrypting data"',  # not the saving label, gate stays closed
            "PRGV:50,100,100",                 # consumed (returns True) but no sample
        ]
        fake = self._make_fake_run_tool(tmp_path, lines, progress_returns)
        adapter = MakemkvAdapter(Path("makemkvcon.exe"))
        with patch("furnace.adapters.makemkv.run_tool", side_effect=fake):
            adapter.demux_title(
                Path("/disc"),
                title_num=5,
                output_dir=tmp_path / "out",
                on_progress=samples.append,
            )
        assert progress_returns == [False, True]
        assert samples == []

    def test_prgv_line_after_saving_without_callback_consumes_line(self, tmp_path: Path) -> None:
        progress_returns: list[bool] = []
        lines = [
            'PRGC:5017,0,"Saving to MKV file"',
            "PRGV:25,50,100",
        ]
        fake = self._make_fake_run_tool(tmp_path, lines, progress_returns)
        adapter = MakemkvAdapter(Path("makemkvcon.exe"))
        with patch("furnace.adapters.makemkv.run_tool", side_effect=fake):
            adapter.demux_title(
                Path("/disc"),
                title_num=5,
                output_dir=tmp_path / "out",
            )
        assert progress_returns == [False, True]

    def test_non_progress_line_not_consumed(self, tmp_path: Path) -> None:
        progress_returns: list[bool] = []
        fake = self._make_fake_run_tool(
            tmp_path,
            ['MSG:1004,0,1,"some message"'],
            progress_returns,
        )
        adapter = MakemkvAdapter(Path("makemkvcon.exe"))
        with patch("furnace.adapters.makemkv.run_tool", side_effect=fake):
            adapter.demux_title(
                Path("/disc"),
                title_num=5,
                output_dir=tmp_path / "out",
            )
        assert progress_returns == [False]

    def test_demux_uses_robot_mode_flag(self, tmp_path: Path) -> None:
        captured_cmd: list[str] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            str_cmd = [str(c) for c in cmd]
            if "info" in str_cmd:
                return 0, "Title #5 was added (3 cell(s), 1:00:00)\n"
            captured_cmd.extend(str_cmd)
            mkv_file = tmp_path / "out" / "title_t05.mkv"
            mkv_file.parent.mkdir(parents=True, exist_ok=True)
            mkv_file.write_text("fake mkv")
            return 0, ""

        adapter = MakemkvAdapter(Path("makemkvcon.exe"))
        with patch("furnace.adapters.makemkv.run_tool", side_effect=fake_run_tool):
            adapter.demux_title(Path("/disc"), title_num=5, output_dir=tmp_path / "out")
        assert "-r" in captured_cmd
