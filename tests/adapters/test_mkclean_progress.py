from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from furnace.adapters.mkclean import MkcleanAdapter, _parse_mkclean_progress_line
from furnace.core.progress import ProgressSample


class TestParseMkcleanProgressLine:
    def test_stage_1_zero(self) -> None:
        sample = _parse_mkclean_progress_line("Progress 1/3:   0%")
        assert sample is not None
        assert sample.fraction == pytest.approx(0.0)

    def test_stage_1_fortytwo(self) -> None:
        sample = _parse_mkclean_progress_line("Progress 1/3:  42%")
        assert sample is not None
        # (0 + 0.42) / 3 = 0.14
        assert sample.fraction == pytest.approx(0.14)

    def test_stage_2_fifty(self) -> None:
        sample = _parse_mkclean_progress_line("Progress 2/3:  50%")
        assert sample is not None
        # (1 + 0.5) / 3 = 0.5
        assert sample.fraction == pytest.approx(0.5)

    def test_stage_3_hundred(self) -> None:
        sample = _parse_mkclean_progress_line("Progress 3/3: 100%")
        assert sample is not None
        assert sample.fraction == pytest.approx(1.0)

    def test_stage_out_of_range(self) -> None:
        assert _parse_mkclean_progress_line("Progress 4/3:  10%") is None

    def test_plain_text(self) -> None:
        assert _parse_mkclean_progress_line("mkclean v0.8.7") is None


class TestMkcleanClean:
    """Test clean() execution with mocked run_tool."""

    def test_clean_cmd(self) -> None:
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

        adapter = MkcleanAdapter(Path("mkclean.exe"))
        with patch("furnace.adapters.mkclean.run_tool", side_effect=fake_run_tool):
            rc = adapter.clean(Path("input.mkv"), Path("output.mkv"))
        assert rc == 0
        assert "mkclean.exe" in captured
        assert "input.mkv" in captured
        assert "output.mkv" in captured

    def test_clean_progress_callback(self) -> None:
        samples: list[ProgressSample] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                on_progress_line("Progress 2/3:  50%")
            return 0, ""

        adapter = MkcleanAdapter(Path("mkclean.exe"))
        with patch("furnace.adapters.mkclean.run_tool", side_effect=fake_run_tool):
            adapter.clean(Path("input.mkv"), Path("output.mkv"), on_progress=samples.append)
        assert len(samples) == 1
        assert samples[0].fraction == pytest.approx(0.5)

    def test_clean_log_path(self, tmp_path: Path) -> None:
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

        adapter = MkcleanAdapter(Path("mkclean.exe"), log_dir=tmp_path)
        with patch("furnace.adapters.mkclean.run_tool", side_effect=fake_run_tool):
            adapter.clean(Path("input.mkv"), Path("output.mkv"))
        assert captured_kwargs["log_path"] == tmp_path / "mkclean.log"

    def test_clean_non_progress_line(self) -> None:
        results: list[bool] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                results.append(on_progress_line("mkclean v0.8.7"))
            return 0, ""

        adapter = MkcleanAdapter(Path("mkclean.exe"))
        with patch("furnace.adapters.mkclean.run_tool", side_effect=fake_run_tool):
            adapter.clean(Path("input.mkv"), Path("output.mkv"))
        assert results == [False]


class TestMkcleanCleanWithoutProgressCallback:
    """Test that progress lines are consumed even without on_progress."""

    def test_progress_consumed_without_callback(self) -> None:
        results: list[bool] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                results.append(on_progress_line("Progress 1/3:  50%"))
            return 0, ""

        adapter = MkcleanAdapter(Path("mkclean.exe"))
        with patch("furnace.adapters.mkclean.run_tool", side_effect=fake_run_tool):
            adapter.clean(Path("input.mkv"), Path("output.mkv"), on_progress=None)
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

        adapter = MkcleanAdapter(Path("mkclean.exe"), log_dir=None)
        with patch("furnace.adapters.mkclean.run_tool", side_effect=fake_run_tool):
            adapter.clean(Path("input.mkv"), Path("output.mkv"))
        assert captured_kwargs["log_path"] is None


class TestMkcleanSetLogDir:
    def test_set_log_dir(self, tmp_path: Path) -> None:
        adapter = MkcleanAdapter(Path("mkclean.exe"))
        adapter.set_log_dir(tmp_path)
        assert adapter._log_dir == tmp_path
