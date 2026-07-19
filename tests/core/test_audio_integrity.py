from __future__ import annotations

import pytest

from furnace.core.audio_integrity import audio_is_truncated, probe_audio_duration


class TestProbeAudioDuration:
    def test_stream_numeric_duration_wins(self) -> None:
        probe = {
            "streams": [
                {"index": 1, "duration": "7345.152000", "tags": {"DURATION": "00:10:00.000000000"}},
            ],
            "format": {"duration": "9999.0"},
        }
        assert probe_audio_duration(probe, 1) == pytest.approx(7345.152)

    def test_falls_back_to_duration_tag_when_stream_duration_absent(self) -> None:
        probe = {
            "streams": [
                {"index": 1, "tags": {"DURATION": "02:02:25.152000000"}},
            ],
            "format": {"duration": "9999.0"},
        }
        assert probe_audio_duration(probe, 1) == pytest.approx(7345.152)

    def test_duration_na_falls_through_to_tag(self) -> None:
        probe = {"streams": [{"index": 1, "duration": "N/A", "tags": {"DURATION": "00:30:00.000000000"}}]}
        assert probe_audio_duration(probe, 1) == pytest.approx(1800.0)

    def test_stream_without_duration_or_tag_uses_format(self) -> None:
        probe = {"streams": [{"index": 1}], "format": {"duration": "5400.0"}}
        assert probe_audio_duration(probe, 1) == pytest.approx(5400.0)

    def test_stream_tags_none_uses_format(self) -> None:
        probe = {"streams": [{"index": 1, "tags": None}], "format": {"duration": "5400.0"}}
        assert probe_audio_duration(probe, 1) == pytest.approx(5400.0)

    def test_stream_index_not_found_uses_format(self) -> None:
        probe = {"streams": [{"index": 0}], "format": {"duration": "5400.0"}}
        assert probe_audio_duration(probe, 7) == pytest.approx(5400.0)

    def test_no_streams_no_format_is_none(self) -> None:
        assert probe_audio_duration({"chapters": []}, 1) is None

    def test_format_duration_na_is_none(self) -> None:
        assert probe_audio_duration({"format": {"duration": "N/A"}}, 1) is None

    def test_zero_duration_is_none(self) -> None:
        assert probe_audio_duration({"format": {"duration": "0.0"}}, 1) is None

    def test_malformed_tag_wrong_parts_falls_through(self) -> None:
        probe = {"streams": [{"index": 1, "tags": {"DURATION": "12:30"}}], "format": {"duration": "60.0"}}
        assert probe_audio_duration(probe, 1) == pytest.approx(60.0)

    def test_malformed_tag_nonnumeric_falls_through(self) -> None:
        probe = {"streams": [{"index": 1, "tags": {"DURATION": "aa:bb:cc"}}], "format": {"duration": "60.0"}}
        assert probe_audio_duration(probe, 1) == pytest.approx(60.0)

    def test_zero_tag_falls_through(self) -> None:
        probe = {"streams": [{"index": 1, "tags": {"DURATION": "00:00:00.000000000"}}], "format": {"duration": "60.0"}}
        assert probe_audio_duration(probe, 1) == pytest.approx(60.0)

    def test_non_string_tag_falls_through(self) -> None:
        probe = {"streams": [{"index": 1, "tags": {"DURATION": 123}}], "format": {"duration": "60.0"}}
        assert probe_audio_duration(probe, 1) == pytest.approx(60.0)

    def test_no_container_fallback_returns_none_when_stream_lacks_duration(self) -> None:
        probe = {"streams": [{"index": 1}], "format": {"duration": "7375.0"}}
        assert probe_audio_duration(probe, 1, allow_container_fallback=False) is None

    def test_no_container_fallback_still_uses_stream_tag(self) -> None:
        probe = {
            "streams": [{"index": 1, "tags": {"DURATION": "02:02:25.152000000"}}],
            "format": {"duration": "7375.0"},
        }
        assert probe_audio_duration(probe, 1, allow_container_fallback=False) == pytest.approx(7345.152)

    def test_no_container_fallback_returns_none_when_stream_missing(self) -> None:
        probe = {"streams": [{"index": 0}], "format": {"duration": "7375.0"}}
        assert probe_audio_duration(probe, 5, allow_container_fallback=False) is None


class TestAudioIsTruncated:
    def test_gross_shortfall_is_truncated(self) -> None:
        assert audio_is_truncated(7345.0, 1794.0) is True

    def test_full_length_is_not_truncated(self) -> None:
        assert audio_is_truncated(7345.0, 7345.0) is False

    def test_small_legit_tail_is_not_truncated(self) -> None:
        assert audio_is_truncated(7375.0, 7345.0) is False

    def test_boundary_at_ratio_is_not_truncated(self) -> None:
        assert audio_is_truncated(100.0, 97.0, ratio=0.97) is False

    def test_just_below_ratio_is_truncated(self) -> None:
        assert audio_is_truncated(100.0, 96.9, ratio=0.97) is True

    def test_custom_ratio(self) -> None:
        assert audio_is_truncated(100.0, 92.0, ratio=0.95) is True
        assert audio_is_truncated(100.0, 96.0, ratio=0.95) is False
