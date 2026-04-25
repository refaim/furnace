"""Tests for the per-sample-point progress callback of ``FFmpegAdapter.run_idet``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from furnace.adapters.ffmpeg import FFmpegAdapter
from furnace.core.progress import ProgressSample


def test_run_idet_calls_on_progress_after_each_sample_point() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    samples: list[ProgressSample] = []

    fake_result = MagicMock()
    fake_result.stderr = "Multi frame detection: TFF: 0 BFF: 0 Progressive: 1000\n"
    fake_result.returncode = 0

    with patch(
        "furnace.adapters.ffmpeg.subprocess.run", return_value=fake_result,
    ) as mock_run:
        adapter.run_idet(Path("v.mkv"), duration_s=1000.0, on_progress=samples.append)

    # 5 sample points: 10/30/50/70/90 — fractions 0.2, 0.4, 0.6, 0.8, 1.0
    assert mock_run.call_count == 5
    for s in samples:
        assert s.fraction is not None
    assert [round(s.fraction, 1) for s in samples if s.fraction is not None] == [
        0.2, 0.4, 0.6, 0.8, 1.0,
    ]


def test_run_idet_works_without_on_progress() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    fake_result = MagicMock()
    fake_result.stderr = "Multi frame detection: TFF: 0 BFF: 0 Progressive: 1000\n"
    fake_result.returncode = 0
    with patch(
        "furnace.adapters.ffmpeg.subprocess.run", return_value=fake_result,
    ) as mock_run:
        ratio = adapter.run_idet(Path("v.mkv"), duration_s=1000.0)
    assert mock_run.call_count == 5
    assert ratio == 0.0


def test_run_idet_returns_zero_when_no_frames_detected() -> None:
    """Branch: stderr without idet line -> total == 0 -> early-return 0.0."""
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    fake_result = MagicMock()
    fake_result.stderr = ""
    fake_result.returncode = 0
    with patch(
        "furnace.adapters.ffmpeg.subprocess.run", return_value=fake_result,
    ) as mock_run:
        ratio = adapter.run_idet(Path("v.mkv"), duration_s=1000.0)
    assert mock_run.call_count == 5
    assert ratio == 0.0


def test_run_idet_computes_interlaced_ratio() -> None:
    """Branch: total > 0 with non-zero interlaced -> non-zero ratio."""
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    fake_result = MagicMock()
    fake_result.stderr = "Multi frame detection: TFF: 600 BFF: 0 Progressive: 400\n"
    fake_result.returncode = 0
    with patch(
        "furnace.adapters.ffmpeg.subprocess.run", return_value=fake_result,
    ) as mock_run:
        ratio = adapter.run_idet(Path("v.mkv"), duration_s=1000.0)
    # Each of 5 calls contributes 600 interlaced + 400 prog -> 3000/5000 = 0.6
    assert mock_run.call_count == 5
    assert ratio == 0.6
