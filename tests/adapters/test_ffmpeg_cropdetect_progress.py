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
    # Constant crop converges after 2 HD batches (2 x 10 = 20 points), plus a
    # final terminal event so the bar reaches 1.0 even on early convergence.
    assert len(samples) == 21
    assert samples[-1].fraction == 1.0
    # Per-point fractions are reported against the cap (40), so convergence
    # at point 20 lands on 0.5 before the terminal 1.0.
    assert samples[19].fraction == 0.5


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
    # DVD batches are 15 -> converges after 2 x 15 = 30 points + terminal 1.0.
    assert len(samples) == 31
    assert samples[-1].fraction == 1.0
