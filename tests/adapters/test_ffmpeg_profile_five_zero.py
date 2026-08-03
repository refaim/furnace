from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from furnace.adapters.ffmpeg import FFmpegAdapter
from furnace.core.audio_profile import AudioMetrics

_FIVE = (0.50, 0.25, 0.40, 0.10, 0.05)


def _adapter() -> FFmpegAdapter:
    return FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))


def _window(values: tuple[float, ...]) -> np.ndarray:
    n = 4800
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.stack([v * np.sin(2 * np.pi * (i + 1) * 10 * t) for i, v in enumerate(values)], axis=1)


def _profile(
    channel_layout: str | None,
    values: tuple[float, ...] = _FIVE,
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
            channels=5,
            duration_s=1000.0,
            channel_layout=channel_layout,
        )
    return metrics, captured


class TestProfileFiveZero:
    @pytest.mark.parametrize("layout", ["5.0", "5.0(side)"])
    def test_decodes_with_the_declared_layout(self, layout: str) -> None:
        _metrics, captured = _profile(layout)
        assert set(captured) == {layout}

    def test_maps_columns_to_left_right_center_and_surrounds(self) -> None:
        metrics, _ = _profile("5.0(side)")
        assert metrics.channels == 5
        assert metrics.rms_c is not None
        assert metrics.rms_ls is not None
        assert metrics.rms_rs is not None
        assert metrics.rms_l > metrics.rms_c > metrics.rms_r > metrics.rms_ls > metrics.rms_rs
        assert metrics.rms_lb is None
        assert metrics.rms_rb is None

    def test_leaves_the_lfe_unset(self) -> None:
        metrics, _ = _profile("5.0(side)")
        assert metrics.rms_lfe is None

    def test_wires_the_surround_correlations(self) -> None:
        metrics, _ = _profile("5.0(side)")
        assert metrics.corr_ls_l is not None
        assert metrics.corr_rs_r is not None
        assert metrics.corr_ls_rs is not None
        assert metrics.corr_lb_ls is None
        assert metrics.corr_rb_rs is None

    def test_an_independent_center_does_not_correlate_with_the_fronts(self) -> None:
        metrics, _ = _profile("5.0(side)")
        assert metrics.corr_c_lr is not None
        assert abs(metrics.corr_c_lr) < 0.5

    def test_a_center_that_is_the_front_mix_correlates_to_one(self) -> None:
        adapter = _adapter()
        n = 4800
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        left = np.sin(2 * np.pi * 10 * t)
        right = np.sin(2 * np.pi * 20 * t)
        surround = np.sin(2 * np.pi * 40 * t)
        window = np.stack([left, right, (left + right) / 2, surround, surround], axis=1)

        with patch.object(FFmpegAdapter, "_decode_pcm_window", side_effect=lambda *a, **k: window):
            metrics = adapter.profile_audio_track(
                Path("x.mkv"),
                stream_index=1,
                channels=5,
                duration_s=1000.0,
                channel_layout="5.0(side)",
            )

        assert metrics.corr_c_lr is not None
        assert metrics.corr_c_lr > 0.999

    def test_a_surround_pair_copied_from_the_fronts_correlates_to_one(self) -> None:
        adapter = _adapter()
        n = 4800
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        left = np.sin(2 * np.pi * 10 * t)
        right = np.sin(2 * np.pi * 20 * t)
        center = np.sin(2 * np.pi * 30 * t)
        window = np.stack([left, right, center, left, right], axis=1)

        with patch.object(FFmpegAdapter, "_decode_pcm_window", side_effect=lambda *a, **k: window):
            metrics = adapter.profile_audio_track(
                Path("x.mkv"),
                stream_index=1,
                channels=5,
                duration_s=1000.0,
                channel_layout="5.0",
            )

        assert metrics.corr_ls_l is not None
        assert metrics.corr_ls_l > 0.999
        assert metrics.corr_rs_r is not None
        assert metrics.corr_rs_r > 0.999

    def test_emits_twelve_progress_events(self) -> None:
        adapter = _adapter()
        samples: list[object] = []
        with patch.object(
            FFmpegAdapter,
            "_decode_pcm_window",
            side_effect=lambda *a, **k: _window(_FIVE),
        ):
            adapter.profile_audio_track(
                Path("x.mkv"),
                stream_index=1,
                channels=5,
                duration_s=1000.0,
                channel_layout="5.0(side)",
                on_progress=samples.append,
            )
        assert len(samples) == 12


class TestProfileFiveChannelRejects:
    @pytest.mark.parametrize("layout", [None, "4.1", "5.1", "quad"])
    def test_unknown_five_channel_layout_raises(self, layout: str | None) -> None:
        with pytest.raises(ValueError, match="unsupported 5-channel layout"):
            _profile(layout)
