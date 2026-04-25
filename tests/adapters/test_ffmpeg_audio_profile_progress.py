"""Tests for the per-sample-point progress callback of
``FFmpegAdapter.profile_audio_track``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from furnace.adapters.ffmpeg import FFmpegAdapter
from furnace.core.progress import ProgressSample


def _fake_window(channels: int) -> np.ndarray:
    return np.zeros((48000, channels), dtype=np.float32)


def test_profile_audio_track_stereo_emits_2_progress_events() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    samples: list[ProgressSample] = []
    with patch.object(
        FFmpegAdapter,
        "_decode_pcm_window",
        side_effect=lambda *a, **k: _fake_window(2),
    ):
        adapter.profile_audio_track(
            Path("x.mkv"),
            stream_index=1,
            channels=2,
            duration_s=1000.0,
            on_progress=samples.append,
        )
    assert len(samples) == 2
    assert samples[-1].fraction == 1.0


def test_profile_audio_track_5_1_emits_4_progress_events() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    samples: list[ProgressSample] = []
    with patch.object(
        FFmpegAdapter,
        "_decode_pcm_window",
        side_effect=lambda *a, **k: _fake_window(6),
    ):
        adapter.profile_audio_track(
            Path("x.mkv"),
            stream_index=1,
            channels=6,
            duration_s=1000.0,
            on_progress=samples.append,
        )
    assert len(samples) == 4
    assert samples[-1].fraction == 1.0
