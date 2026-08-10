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
        output = "1) 01 - Title 1, 1:32:05\n2) 02 - Title 2, 0:05:30\n"
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
        output = "M2TS, 1 video track\n\n1) 00800.mpls, 1:00:00\n"
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


class TestParsePlaylistVideo:
    def test_video_line_per_playlist(self) -> None:
        output = (
            "1) 00047.mpls, 15:03:42\n"
            "   [64+64+64].m2ts\n"
            "   - Chapters, 801 chapters\n"
            "   - MPEG2, 1080p24/1.001 (16:9)\n"
            "   - AC3, [eng], multi-channel, 48kHz\n"
            "   - Subtitle (PGS), [eng]\n"
            "\n"
            "3) 00025.mpls, 00076.m2ts+00115.m2ts, 1:41:02\n"
            "   - Chapters, 21 chapters\n"
            "   - h264/AVC, 1080p24/1.001 (16:9)\n"
            "   - RAW/PCM, [eng], multi-channel, 48kHz\n"
        )
        result = Eac3toAdapter._parse_playlist_output(output)
        assert result[0].video == "MPEG2, 1080p24/1.001 (16:9)"
        assert result[1].video == "h264/AVC, 1080p24/1.001 (16:9)"

    def test_interlaced_standard_definition(self) -> None:
        output = "4) 00026.mpls, 00043.m2ts, 0:30:31\n   - MPEG2, 480i60/1.001 (4:3)\n"
        result = Eac3toAdapter._parse_playlist_output(output)
        assert result[0].video == "MPEG2, 480i60/1.001 (4:3)"

    def test_ultra_high_definition(self) -> None:
        output = "1) 00800.mpls, 2:10:00\n   - h265/HEVC, 2160p24/1.001 (16:9), HDR10\n"
        result = Eac3toAdapter._parse_playlist_output(output)
        assert result[0].video == "h265/HEVC, 2160p24/1.001 (16:9), HDR10"

    def test_audio_lines_are_not_mistaken_for_video(self) -> None:
        output = (
            "1) 00800.mpls, 1:00:00\n"
            "   - Chapters, 12 chapters\n"
            "   - DTS Master Audio, [eng], 5.1 channels, 1536kbps, 48kHz\n"
            "   - AC3, [rus], multi-channel, 48kHz\n"
            "   - Subtitle (PGS), [eng]\n"
        )
        result = Eac3toAdapter._parse_playlist_output(output)
        assert result[0].video is None

    def test_first_video_line_wins(self) -> None:
        output = "1) 00800.mpls, 2:00:00\n   - h264/AVC, 1080p24/1.001 (16:9)\n   - h264/MVC, 1080p24/1.001 (16:9)\n"
        result = Eac3toAdapter._parse_playlist_output(output)
        assert result[0].video == "h264/AVC, 1080p24/1.001 (16:9)"

    def test_video_line_before_any_playlist_is_ignored(self) -> None:
        output = "   - MPEG2, 1080p24/1.001 (16:9)\n1) 00800.mpls, 1:00:00\n"
        result = Eac3toAdapter._parse_playlist_output(output)
        assert result[0].video is None

    def test_playlist_without_track_listing(self) -> None:
        output = "1) 00800.mpls, 1:45:23\n2) 00801.mpls, 0:02:15\n"
        result = Eac3toAdapter._parse_playlist_output(output)
        assert result[0].video is None
        assert result[1].video is None
