from __future__ import annotations

from furnace.adapters.eac3to import _parse_eac3to_progress_line
from furnace.core.progress import ProgressSample


class TestParseEac3toProgressLine:
    def test_process_zero(self) -> None:
        assert _parse_eac3to_progress_line("process: 0%") == ProgressSample(fraction=0.0)

    def test_process_fifty(self) -> None:
        assert _parse_eac3to_progress_line("process: 50%") == ProgressSample(fraction=0.5)

    def test_process_hundred(self) -> None:
        assert _parse_eac3to_progress_line("process: 100%") == ProgressSample(fraction=1.0)

    def test_missing_percent(self) -> None:
        assert _parse_eac3to_progress_line("process: 50") is None

    def test_analyze_not_captured(self) -> None:
        # Only `process:` is captured; executor handles phase transitions explicitly
        assert _parse_eac3to_progress_line("analyze: 25%") is None

    def test_plain_text(self) -> None:
        assert _parse_eac3to_progress_line("Reading file...") is None

    def test_trailing_whitespace(self) -> None:
        assert _parse_eac3to_progress_line("process: 42%  ") == ProgressSample(fraction=0.42)
