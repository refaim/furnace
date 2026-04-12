from __future__ import annotations

import pytest

from furnace.adapters.makemkv import MakemkvAdapter


class TestParseMakemkvInfo:
    def test_parse_dvd_titles(self) -> None:
        """Parse typical makemkvcon info output."""
        output = (
            "MakeMKV v1.18.3 win(x64-release) started\n"
            "Title #1 has length of 30 seconds which is less than minimum title length\n"
            "Title #2 was added (1 cell(s), 0:07:49)\n"
            "Title #3 has length of 34 seconds which is less than minimum title length\n"
            "Title #4 was added (13 cell(s), 1:12:32)\n"
            "Operation successfully completed\n"
        )
        result = MakemkvAdapter._parse_info_output(output)
        assert len(result) == 2
        assert result[0].number == 2
        assert result[0].duration_s == pytest.approx(469.0)
        assert result[1].number == 4
        assert result[1].duration_s == pytest.approx(4352.0)

    def test_parse_empty_output(self) -> None:
        result = MakemkvAdapter._parse_info_output("")
        assert result == []

    def test_skipped_titles_not_included(self) -> None:
        """Titles that were skipped by makemkv are not in the result."""
        output = (
            "Title #1 has length of 30 seconds which is less than minimum title length and was therefore skipped\n"
        )
        result = MakemkvAdapter._parse_info_output(output)
        assert result == []

    def test_raw_label_contains_original_line(self) -> None:
        output = "Title #4 was added (13 cell(s), 1:12:32)\n"
        result = MakemkvAdapter._parse_info_output(output)
        assert "Title #4 was added" in result[0].raw_label
