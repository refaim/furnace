from __future__ import annotations

from furnace.adapters.qaac import _parse_qaac_progress_line
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
