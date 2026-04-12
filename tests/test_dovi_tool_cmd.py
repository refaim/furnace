from __future__ import annotations

import tempfile
from pathlib import Path

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
