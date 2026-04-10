from __future__ import annotations

import pytest

from furnace.adapters.mkclean import _parse_mkclean_progress_line


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
