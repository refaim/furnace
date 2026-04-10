from __future__ import annotations

import pytest

from furnace.adapters.nvencc import _parse_nvencc_progress_line


class TestParseNvenccProgressLine:
    def test_basic_percent(self) -> None:
        line = "[45.3%] 12345 frames: 85.08 fps, 8547 kb/s, remain 0:02:15"
        sample = _parse_nvencc_progress_line(line)
        assert sample is not None
        assert sample.fraction == pytest.approx(0.453)
        assert sample.speed is None  # no src_fps passed

    def test_percent_with_src_fps(self) -> None:
        line = "[45.3%] 12345 frames: 48.0 fps, 8547 kb/s"
        sample = _parse_nvencc_progress_line(line, src_fps=24.0)
        assert sample is not None
        assert sample.fraction == pytest.approx(0.453)
        assert sample.speed == pytest.approx(2.0)

    def test_no_percent_marker(self) -> None:
        assert _parse_nvencc_progress_line("encoding frame 1234") is None

    def test_100_percent(self) -> None:
        sample = _parse_nvencc_progress_line("[100.0%]")
        assert sample is not None
        assert sample.fraction == 1.0

    def test_src_fps_zero_skipped(self) -> None:
        line = "[50.0%] 12345 frames: 48.0 fps"
        sample = _parse_nvencc_progress_line(line, src_fps=0.0)
        assert sample is not None
        assert sample.speed is None

    def test_malformed_percent(self) -> None:
        # Regex won't match non-numeric percent, result is None
        assert _parse_nvencc_progress_line("[not-a-number%]") is None
