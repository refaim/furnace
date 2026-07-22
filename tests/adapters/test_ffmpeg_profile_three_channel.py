from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from furnace.adapters.ffmpeg import FFmpegAdapter
from furnace.core.audio_profile import AudioMetrics


def _adapter() -> FFmpegAdapter:
    return FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))


def _window(values: tuple[float, float, float]) -> np.ndarray:
    n = 4800
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.stack([v * np.sin(2 * np.pi * (i + 1) * 10 * t) for i, v in enumerate(values)], axis=1)


def _profile(
    channels: int,
    channel_layout: str | None,
    values: tuple[float, float, float],
) -> tuple[AudioMetrics, list[str]]:
    adapter = _adapter()
    captured: list[str] = []

    def fake_decode(
        _self: FFmpegAdapter,
        _path: Path,
        _stream_index: int,
        _channels: int,
        layout: str,
        _start: float,
        _dur: float,
    ) -> np.ndarray:
        captured.append(layout)
        return _window(values)

    with patch.object(FFmpegAdapter, "_decode_pcm_window", fake_decode):
        metrics = adapter.profile_audio_track(
            Path("x.mkv"),
            stream_index=1,
            channels=channels,
            duration_s=1000.0,
            channel_layout=channel_layout,
        )
    return metrics, captured


class TestProfileTwoOne:
    def test_decodes_with_the_2_1_layout(self) -> None:
        _metrics, captured = _profile(3, "2.1", (0.5, 0.4, 0.3))
        assert set(captured) == {"2.1"}

    def test_maps_columns_to_left_right_lfe(self) -> None:
        metrics, _ = _profile(3, "2.1", (0.5, 0.25, 0.0))
        assert metrics.channels == 3
        assert metrics.rms_c is None
        assert metrics.rms_lfe == -120.0
        assert metrics.rms_l > metrics.rms_r
        assert metrics.rms_ls is None
        assert metrics.rms_rs is None
        assert metrics.corr_ls_l is None

    def test_emits_four_progress_events(self) -> None:
        adapter = _adapter()
        samples: list[Any] = []
        with patch.object(
            FFmpegAdapter,
            "_decode_pcm_window",
            side_effect=lambda *a, **k: _window((0.5, 0.5, 0.5)),
        ):
            adapter.profile_audio_track(
                Path("x.mkv"),
                stream_index=1,
                channels=3,
                duration_s=1000.0,
                channel_layout="2.1",
                on_progress=samples.append,
            )
        assert len(samples) == 4
        assert samples[-1].fraction == 1.0


class TestProfileThreeZero:
    def test_decodes_with_the_3_0_layout(self) -> None:
        _metrics, captured = _profile(3, "3.0", (0.5, 0.4, 0.3))
        assert set(captured) == {"3.0"}

    def test_maps_columns_to_left_right_center(self) -> None:
        metrics, _ = _profile(3, "3.0", (0.5, 0.25, 0.0))
        assert metrics.channels == 3
        assert metrics.rms_lfe is None
        assert metrics.rms_c == -120.0
        assert metrics.rms_l > metrics.rms_r

    def test_a_center_that_is_the_front_mix_correlates_to_one(self) -> None:
        adapter = _adapter()
        n = 4800
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        left = np.sin(2 * np.pi * 10 * t)
        right = np.sin(2 * np.pi * 20 * t)
        window = np.stack([left, right, (left + right) / 2], axis=1)

        with patch.object(FFmpegAdapter, "_decode_pcm_window", side_effect=lambda *a, **k: window):
            metrics = adapter.profile_audio_track(
                Path("x.mkv"), stream_index=1, channels=3, duration_s=1000.0, channel_layout="3.0"
            )

        assert metrics.corr_c_lr is not None
        assert metrics.corr_c_lr > 0.999

    def test_an_independent_center_does_not_correlate(self) -> None:
        metrics, _ = _profile(3, "3.0", (0.5, 0.4, 0.3))
        assert metrics.corr_c_lr is not None
        assert abs(metrics.corr_c_lr) < 0.5

    def test_two_one_leaves_the_center_correlation_unset(self) -> None:
        metrics, _ = _profile(3, "2.1", (0.5, 0.4, 0.3))
        assert metrics.corr_c_lr is None


class TestProfileThreeChannelRejects:
    @pytest.mark.parametrize("layout", [None, "3.0(back)", "stereo"])
    def test_unknown_three_channel_layout_raises(self, layout: str | None) -> None:
        with pytest.raises(ValueError, match="unsupported 3-channel layout"):
            _profile(3, layout, (0.5, 0.5, 0.5))
