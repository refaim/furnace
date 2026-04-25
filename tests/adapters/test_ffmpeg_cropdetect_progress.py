"""Tests for the per-sample-point progress callback of ``FFmpegAdapter.detect_crop``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from furnace.adapters.ffmpeg import FFmpegAdapter
from furnace.core.progress import ProgressSample


def test_detect_crop_calls_on_progress_per_point_hd() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    samples: list[ProgressSample] = []
    fake_result = MagicMock()
    fake_result.stderr = "[Parsed_cropdetect_0 @ 0x0] crop=3840:1600:0:280\n"
    fake_result.returncode = 0
    with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=fake_result):
        adapter.detect_crop(
            Path("x.mkv"),
            duration_s=1000.0,
            interlaced=False,
            is_dvd=False,
            on_progress=samples.append,
        )
    # HD has 10 sample points -> 10 progress events ending at 1.0
    assert len(samples) == 10
    assert samples[-1].fraction == 1.0


def test_detect_crop_calls_on_progress_per_point_dvd() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    samples: list[ProgressSample] = []
    fake_result = MagicMock()
    fake_result.stderr = "[Parsed_cropdetect_0 @ 0x0] crop=720:480:0:0\n"
    fake_result.returncode = 0
    with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=fake_result):
        adapter.detect_crop(
            Path("x.mkv"),
            duration_s=1000.0,
            interlaced=False,
            is_dvd=True,
            on_progress=samples.append,
        )
    # DVD has 15 sample points -> 15 progress events
    assert len(samples) == 15
    assert samples[-1].fraction == 1.0
