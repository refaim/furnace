from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from furnace.adapters.qaac import QaacAdapter, _parse_qaac_progress_line
from furnace.core.progress import ProgressSample


class TestParseQaacProgressLine:
    def test_full_line(self) -> None:
        line = "[42.5%] 0:30/1:43:01.600 (30.5x), ETA 3:20"
        sample = _parse_qaac_progress_line(line)
        assert sample == ProgressSample(fraction=0.425, speed=30.5)

    def test_no_speed(self) -> None:
        sample = _parse_qaac_progress_line("[100.0%] 0:01:23.456/0:01:23.456")
        assert sample == ProgressSample(fraction=1.0, speed=None)

    def test_zero_percent(self) -> None:
        assert _parse_qaac_progress_line("[0.0%]") == ProgressSample(fraction=0.0)

    def test_plain_text(self) -> None:
        assert _parse_qaac_progress_line("Encoding with qaac64...") is None

    def test_integer_percent(self) -> None:
        assert _parse_qaac_progress_line("[50%] ...") == ProgressSample(fraction=0.5)


class TestQaacSetLogDir:
    def test_set_log_dir(self, tmp_path: Path) -> None:
        adapter = QaacAdapter(Path("qaac64.exe"))
        adapter.set_log_dir(tmp_path)
        assert adapter._log_dir == tmp_path


class TestQaacEncodeAac:
    """Test encode_aac() execution by mocking run_tool."""

    def test_cmd_flags(self) -> None:
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

        adapter = QaacAdapter(Path("qaac64.exe"))
        with patch("furnace.adapters.qaac.run_tool", side_effect=fake_run_tool):
            rc = adapter.encode_aac(Path("/src/audio.wav"), Path("/out/audio.m4a"))
        assert rc == 0
        assert "--tvbr" in captured
        assert "91" in captured
        assert "--quality" in captured
        assert "2" in captured
        assert "--rate" in captured
        assert "keep" in captured
        assert "--no-delay" in captured
        assert "--threading" in captured
        assert "/src/audio.wav" in captured
        assert "-o" in captured
        assert "/out/audio.m4a" in captured

    def test_log_path(self, tmp_path: Path) -> None:
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

        adapter = QaacAdapter(Path("qaac64.exe"), log_dir=tmp_path)
        with patch("furnace.adapters.qaac.run_tool", side_effect=fake_run_tool):
            adapter.encode_aac(Path("/src/audio.wav"), Path("/out/audio.m4a"))
        assert captured_kwargs["log_path"] == tmp_path / "qaac.log"

    def test_progress_callback(self) -> None:
        samples: list[ProgressSample] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                on_progress_line("[50.0%] 0:30/1:00.000 (30.0x)")
            return 0, ""

        adapter = QaacAdapter(Path("qaac64.exe"))
        with patch("furnace.adapters.qaac.run_tool", side_effect=fake_run_tool):
            adapter.encode_aac(Path("/src/audio.wav"), Path("/out/audio.m4a"), on_progress=samples.append)
        assert len(samples) == 1
        assert abs(samples[0].fraction - 0.5) < 0.01  # type: ignore[operator]

    def test_progress_non_progress_line(self) -> None:
        """Non-progress lines return False from the closure."""
        results: list[bool] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                results.append(on_progress_line("not progress"))
            return 0, ""

        adapter = QaacAdapter(Path("qaac64.exe"))
        with patch("furnace.adapters.qaac.run_tool", side_effect=fake_run_tool):
            adapter.encode_aac(Path("/src/audio.wav"), Path("/out/audio.m4a"))
        assert results == [False]

    def test_progress_without_callback(self) -> None:
        """Progress line consumed even when on_progress is None."""
        results: list[bool] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                results.append(on_progress_line("[50%] 0:30/1:00"))
            return 0, ""

        adapter = QaacAdapter(Path("qaac64.exe"))
        with patch("furnace.adapters.qaac.run_tool", side_effect=fake_run_tool):
            adapter.encode_aac(Path("/src/audio.wav"), Path("/out/audio.m4a"), on_progress=None)
        assert results == [True]

    def test_no_log_dir(self) -> None:
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

        adapter = QaacAdapter(Path("qaac64.exe"), log_dir=None)
        with patch("furnace.adapters.qaac.run_tool", side_effect=fake_run_tool):
            adapter.encode_aac(Path("/src/audio.wav"), Path("/out/audio.m4a"))
        assert captured_kwargs["log_path"] is None
