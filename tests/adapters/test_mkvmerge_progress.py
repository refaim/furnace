from __future__ import annotations

from furnace.adapters.mkvmerge import _parse_mkvmerge_progress_line
from furnace.core.progress import ProgressSample


class TestParseMkvmergeProgressLine:
    def test_basic_progress(self) -> None:
        assert _parse_mkvmerge_progress_line("Progress: 45%") == ProgressSample(fraction=0.45)

    def test_zero(self) -> None:
        assert _parse_mkvmerge_progress_line("Progress: 0%") == ProgressSample(fraction=0.0)

    def test_hundred(self) -> None:
        assert _parse_mkvmerge_progress_line("Progress: 100%") == ProgressSample(fraction=1.0)

    def test_missing_colon(self) -> None:
        assert _parse_mkvmerge_progress_line("Progress 45%") is None

    def test_plain_text(self) -> None:
        assert _parse_mkvmerge_progress_line("Merging files...") is None

    def test_trailing_whitespace(self) -> None:
        assert _parse_mkvmerge_progress_line("Progress: 45%  ") == ProgressSample(fraction=0.45)
