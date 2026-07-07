"""Tests for ``FFmpegAdapter.sample_grain`` (film-grain amplitude probing).

The adapter pipes five short luma-only rawvideo windows out of ffmpeg and
measures static-block temporal flicker with numpy; the boolean GRAINY verdict
lives in ``core.detect.classify_grain``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from furnace.adapters.ffmpeg import FFmpegAdapter


def _raw_frames(n: int, noise_amp: float, seed: int = 7) -> bytes:
    rng = np.random.default_rng(seed)
    base = np.full((270, 480), 128.0, dtype=np.float32)
    frames = [
        np.clip(base + rng.normal(0.0, noise_amp, base.shape), 0, 255).astype(np.uint8)
        for _ in range(n)
    ]
    return b"".join(f.tobytes() for f in frames)


def _black_frames(n: int) -> bytes:
    frame = np.zeros((270, 480), dtype=np.uint8)
    return frame.tobytes() * n


def _fake_result(stdout: bytes, returncode: int = 0) -> MagicMock:
    result = MagicMock()
    result.stdout = stdout
    result.returncode = returncode
    return result


def test_sample_grain_probes_five_windows_with_expected_seeks() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))

    with patch(
        "furnace.adapters.ffmpeg.subprocess.run",
        return_value=_fake_result(_raw_frames(24, 2.0)),
    ) as mock_run:
        values = adapter.sample_grain(Path("v.mkv"), duration_s=1000.0)

    assert mock_run.call_count == 5
    seeks = []
    for call in mock_run.call_args_list:
        cmd = call.args[0]
        seeks.append(cmd[cmd.index("-ss") + 1])
    # Windows start at 10/30/50/70/90% of the duration.
    assert seeks == ["100.00", "300.00", "500.00", "700.00", "900.00"]
    assert len(values) == 5


def test_sample_grain_command_shape() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))

    with patch(
        "furnace.adapters.ffmpeg.subprocess.run",
        return_value=_fake_result(_raw_frames(24, 1.0)),
    ) as mock_run:
        adapter.sample_grain(Path("v.mkv"), duration_s=1000.0)

    cmd = mock_run.call_args_list[0].args[0]
    assert cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-f") + 1] == "rawvideo"
    assert cmd[cmd.index("-pix_fmt") + 1] == "gray"
    assert cmd[cmd.index("-frames:v") + 1] == "24"
    assert "scale=480:270" in cmd[cmd.index("-vf") + 1]
    assert cmd[cmd.index("-i") + 1] == "v.mkv"
    assert cmd[-1] == "-"


def test_sample_grain_amplitude_tracks_noise() -> None:
    """Louder synthetic grain yields a strictly larger per-window flicker."""
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))

    with patch(
        "furnace.adapters.ffmpeg.subprocess.run",
        return_value=_fake_result(_raw_frames(24, 2.0)),
    ):
        high = adapter.sample_grain(Path("v.mkv"), duration_s=1000.0)

    with patch(
        "furnace.adapters.ffmpeg.subprocess.run",
        return_value=_fake_result(_raw_frames(24, 0.2)),
    ):
        low = adapter.sample_grain(Path("v.mkv"), duration_s=1000.0)

    assert len(high) == 5
    assert len(low) == 5
    assert min(high) > 0.0
    assert min(low) > 0.0
    assert high[0] > low[0]


def test_sample_grain_skips_failed_window() -> None:
    """A window where ffmpeg exits non-zero contributes nothing; others survive."""
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    good = _raw_frames(24, 2.0)
    results = [
        _fake_result(good),
        _fake_result(b"", returncode=1),
        _fake_result(good),
        _fake_result(good),
        _fake_result(good),
    ]

    with patch("furnace.adapters.ffmpeg.subprocess.run", side_effect=results):
        values = adapter.sample_grain(Path("v.mkv"), duration_s=1000.0)

    assert len(values) == 4


def test_sample_grain_skips_short_read_window() -> None:
    """A window that decodes fewer than 8 frames is skipped."""
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    good = _raw_frames(24, 2.0)
    results = [
        _fake_result(good),
        _fake_result(_raw_frames(5, 2.0)),  # only 5 frames < 8 → skipped
        _fake_result(good),
        _fake_result(good),
        _fake_result(good),
    ]

    with patch("furnace.adapters.ffmpeg.subprocess.run", side_effect=results):
        values = adapter.sample_grain(Path("v.mkv"), duration_s=1000.0)

    assert len(values) == 4


def test_sample_grain_all_black_windows_return_empty() -> None:
    """All-black windows (luma < 30) have zero valid blocks → skipped everywhere."""
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))

    with patch(
        "furnace.adapters.ffmpeg.subprocess.run",
        return_value=_fake_result(_black_frames(24)),
    ):
        values = adapter.sample_grain(Path("v.mkv"), duration_s=1000.0)

    assert values == []
