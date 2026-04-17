from __future__ import annotations

import pytest

from furnace.adapters.eac3to import Eac3toAdapter


class TestParsePlaylistOutput:
    def test_parse_bluray_playlists(self) -> None:
        output = (
            "M2TS, 1 video track, 3 audio tracks, 2 subtitle tracks, 1:45:23\n"
            "\n"
            "1) 00800.mpls, 1:45:23\n"
            "2) 00801.mpls, 0:02:15\n"
            "3) 00802.mpls, 0:31:10\n"
        )
        result = Eac3toAdapter._parse_playlist_output(output)
        assert len(result) == 3
        assert result[0].number == 1
        assert result[0].duration_s == pytest.approx(6323.0)
        assert "00800.mpls" in result[0].raw_label
        assert result[1].number == 2
        assert result[1].duration_s == pytest.approx(135.0)
        assert result[2].number == 3
        assert result[2].duration_s == pytest.approx(1870.0)

    def test_parse_dvd_playlists(self) -> None:
        output = (
            "1) 01 - Title 1, 1:32:05\n"
            "2) 02 - Title 2, 0:05:30\n"
        )
        result = Eac3toAdapter._parse_playlist_output(output)
        assert len(result) == 2
        assert result[0].number == 1
        assert result[0].duration_s == pytest.approx(5525.0)
        assert result[1].number == 2
        assert result[1].duration_s == pytest.approx(330.0)

    def test_parse_empty_output(self) -> None:
        result = Eac3toAdapter._parse_playlist_output("")
        assert result == []

    def test_parse_lines_without_playlist_numbers(self) -> None:
        output = (
            "M2TS, 1 video track\n"
            "\n"
            "1) 00800.mpls, 1:00:00\n"
        )
        result = Eac3toAdapter._parse_playlist_output(output)
        assert len(result) == 1

    def test_parse_duration_hours_minutes_seconds(self) -> None:
        output = "1) test, 2:03:45\n"
        result = Eac3toAdapter._parse_playlist_output(output)
        assert result[0].duration_s == pytest.approx(2 * 3600 + 3 * 60 + 45)

    def test_parse_duration_minutes_seconds(self) -> None:
        output = "1) test, 5:30\n"
        result = Eac3toAdapter._parse_playlist_output(output)
        assert result[0].duration_s == pytest.approx(330.0)
