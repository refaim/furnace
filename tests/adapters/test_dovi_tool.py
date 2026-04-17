from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from furnace.adapters.dovi_tool import DoviToolAdapter
from furnace.core.models import DvMode


class TestDoviToolCommand:
    def test_extract_rpu_copy_mode(self) -> None:
        adapter = DoviToolAdapter(Path("dovi_tool.exe"))
        cmd = adapter._build_extract_cmd(Path("input.mkv"), Path("RPU.bin"), DvMode.COPY)
        str_cmd = [str(c) for c in cmd]
        assert str_cmd[0] == "dovi_tool.exe"
        assert "-m" not in str_cmd
        assert "extract-rpu" in str_cmd
        assert "input.mkv" in str_cmd

    def test_extract_rpu_to_8_1_mode(self) -> None:
        adapter = DoviToolAdapter(Path("dovi_tool.exe"))
        cmd = adapter._build_extract_cmd(Path("input.mkv"), Path("RPU.bin"), DvMode.TO_8_1)
        str_cmd = [str(c) for c in cmd]
        m_idx = str_cmd.index("-m")
        assert str_cmd[m_idx + 1] == "2"
        assert "extract-rpu" in str_cmd

    def test_output_flag(self) -> None:
        adapter = DoviToolAdapter(Path("dovi_tool.exe"))
        rpu_path = Path(tempfile.gettempdir()) / "RPU.bin"
        cmd = adapter._build_extract_cmd(Path("input.mkv"), rpu_path, DvMode.COPY)
        str_cmd = [str(c) for c in cmd]
        o_idx = str_cmd.index("-o")
        assert str_cmd[o_idx + 1] == str(rpu_path)


class TestDoviToolExtractRpu:
    """Test extract_rpu() execution with mocked run_tool."""

    def test_extract_rpu_returns_rc(self) -> None:
        captured: list[str] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            captured.extend(str(c) for c in cmd)
            return 0, ""

        adapter = DoviToolAdapter(Path("dovi_tool.exe"))
        with patch("furnace.adapters.dovi_tool.run_tool", side_effect=fake_run_tool):
            rc = adapter.extract_rpu(Path("input.hevc"), Path("rpu.bin"), DvMode.COPY)
        assert rc == 0
        assert "extract-rpu" in captured
        assert "input.hevc" in captured

    def test_extract_rpu_log_path(self, tmp_path: Path) -> None:
        captured_kwargs: dict[str, Any] = {}

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            captured_kwargs["log_path"] = log_path
            return 0, ""

        adapter = DoviToolAdapter(Path("dovi_tool.exe"), log_dir=tmp_path)
        with patch("furnace.adapters.dovi_tool.run_tool", side_effect=fake_run_tool):
            adapter.extract_rpu(Path("input.hevc"), Path("rpu.bin"), DvMode.TO_8_1)
        assert captured_kwargs["log_path"] == tmp_path / "dovi_tool_extract.log"

    def test_extract_rpu_nonzero_rc(self) -> None:
        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            return 1, "error"

        adapter = DoviToolAdapter(Path("dovi_tool.exe"))
        with patch("furnace.adapters.dovi_tool.run_tool", side_effect=fake_run_tool):
            rc = adapter.extract_rpu(Path("input.hevc"), Path("rpu.bin"), DvMode.COPY)
        assert rc == 1


class TestDoviToolSetLogDir:
    def test_set_log_dir(self, tmp_path: Path) -> None:
        adapter = DoviToolAdapter(Path("dovi_tool.exe"))
        adapter.set_log_dir(tmp_path)
        assert adapter._log_dir == tmp_path

    def test_set_log_dir_none(self, tmp_path: Path) -> None:
        adapter = DoviToolAdapter(Path("dovi_tool.exe"), log_dir=tmp_path)
        adapter.set_log_dir(None)
        assert adapter._log_dir is None
