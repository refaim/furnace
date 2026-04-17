"""Tests for mkvpropedit _build_tags_xml helper and set_encoder_tag execution."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from furnace.adapters.mkvpropedit import MkvpropeditAdapter, _build_tags_xml


class TestBuildTagsXml:
    def test_encoder_only(self) -> None:
        xml = _build_tags_xml("Furnace v1.4.0")
        assert "<Name>ENCODER</Name>" in xml
        assert "<String>Furnace v1.4.0</String>" in xml
        assert "ENCODER_SETTINGS" not in xml

    def test_with_encoder_settings(self) -> None:
        xml = _build_tags_xml("Furnace v1.4.0", "hevc_nvenc / main10 / cq=25")
        assert "<Name>ENCODER</Name>" in xml
        assert "<Name>ENCODER_SETTINGS</Name>" in xml
        assert "<String>hevc_nvenc / main10 / cq=25</String>" in xml


class TestSetEncoderTag:
    """Test set_encoder_tag() execution with mocked run_tool."""

    def test_set_encoder_tag_cmd(self, tmp_path: Path) -> None:
        mkv_path = tmp_path / "movie.mkv"
        mkv_path.write_text("fake mkv")
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

        adapter = MkvpropeditAdapter(Path("mkvpropedit.exe"))
        with patch("furnace.adapters.mkvpropedit.run_tool", side_effect=fake_run_tool):
            rc = adapter.set_encoder_tag(mkv_path, "Furnace v1.4.0")
        assert rc == 0
        assert "mkvpropedit.exe" in captured
        assert str(mkv_path) in captured
        assert "--tags" in captured
        # global: prefix
        tags_arg = [c for c in captured if c.startswith("global:")]
        assert len(tags_arg) == 1

    def test_temp_xml_cleaned_up(self, tmp_path: Path) -> None:
        mkv_path = tmp_path / "movie.mkv"
        mkv_path.write_text("fake mkv")

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            return 0, ""

        adapter = MkvpropeditAdapter(Path("mkvpropedit.exe"))
        with patch("furnace.adapters.mkvpropedit.run_tool", side_effect=fake_run_tool):
            adapter.set_encoder_tag(mkv_path, "Furnace v1.4.0", "hevc_nvenc / main10")
        # After execution, no .xml temp files should remain
        xml_files = list(tmp_path.glob("furnace_tags_*.xml"))
        assert len(xml_files) == 0

    def test_temp_xml_cleaned_on_error(self, tmp_path: Path) -> None:
        mkv_path = tmp_path / "movie.mkv"
        mkv_path.write_text("fake mkv")

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            return 1, "error"

        adapter = MkvpropeditAdapter(Path("mkvpropedit.exe"))
        with patch("furnace.adapters.mkvpropedit.run_tool", side_effect=fake_run_tool):
            rc = adapter.set_encoder_tag(mkv_path, "Furnace v1.4.0")
        assert rc == 1
        # Temp file cleaned up even on error
        xml_files = list(tmp_path.glob("furnace_tags_*.xml"))
        assert len(xml_files) == 0

    def test_log_path(self, tmp_path: Path) -> None:
        mkv_path = tmp_path / "movie.mkv"
        mkv_path.write_text("fake mkv")
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

        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        adapter = MkvpropeditAdapter(Path("mkvpropedit.exe"), log_dir=log_dir)
        with patch("furnace.adapters.mkvpropedit.run_tool", side_effect=fake_run_tool):
            adapter.set_encoder_tag(mkv_path, "Furnace v1.4.0")
        assert captured_kwargs["log_path"] == log_dir / "mkvpropedit.log"


class TestMkvpropeditSetLogDir:
    def test_set_log_dir(self, tmp_path: Path) -> None:
        adapter = MkvpropeditAdapter(Path("mkvpropedit.exe"))
        adapter.set_log_dir(tmp_path)
        assert adapter._log_dir == tmp_path
