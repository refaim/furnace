from __future__ import annotations

from furnace.adapters.ffmpeg import _parse_ffmpeg_progress_block
from furnace.core.progress import ProgressSample


class TestParseFfmpegProgressBlock:
    def test_typical_block(self) -> None:
        kv = {
            "frame": "42",
            "fps": "23.97",
            "out_time_us": "60000000",
            "out_time_ms": "60000",
            "speed": "2.5x",
            "progress": "continue",
        }
        sample = _parse_ffmpeg_progress_block(kv)
        assert sample == ProgressSample(processed_s=60.0, speed=2.5)

    def test_missing_out_time(self) -> None:
        kv = {"frame": "42", "progress": "continue"}
        assert _parse_ffmpeg_progress_block(kv) is None

    def test_out_time_na(self) -> None:
        kv = {"out_time_us": "N/A", "progress": "continue"}
        assert _parse_ffmpeg_progress_block(kv) is None

    def test_malformed_out_time(self) -> None:
        kv = {"out_time_us": "not-a-number", "progress": "continue"}
        assert _parse_ffmpeg_progress_block(kv) is None

    def test_speed_na(self) -> None:
        kv = {"out_time_us": "30000000", "speed": "N/A", "progress": "continue"}
        sample = _parse_ffmpeg_progress_block(kv)
        assert sample == ProgressSample(processed_s=30.0, speed=None)

    def test_speed_without_x_suffix(self) -> None:
        kv = {"out_time_us": "30000000", "speed": "2.5", "progress": "continue"}
        sample = _parse_ffmpeg_progress_block(kv)
        assert sample == ProgressSample(processed_s=30.0, speed=None)

    def test_end_of_stream(self) -> None:
        kv = {"out_time_us": "120000000", "speed": "3.0x", "progress": "end"}
        sample = _parse_ffmpeg_progress_block(kv)
        assert sample == ProgressSample(processed_s=120.0, speed=3.0)
