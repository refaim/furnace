from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from furnace.adapters.ffmpeg import (
    FFmpegAdapter,
    _pearson,
    _rms_db,
)
from furnace.core.audio_profile import AudioMetrics


def _adapter() -> FFmpegAdapter:
    return FFmpegAdapter(Path("ffmpeg.exe"), Path("ffprobe.exe"))


class TestRmsDb:
    def test_empty_returns_floor(self) -> None:
        assert _rms_db(np.empty(0, dtype=np.float32)) == -120.0

    def test_digital_silence_returns_floor(self) -> None:
        silent = np.zeros(1024, dtype=np.float32)
        assert _rms_db(silent) == -120.0

    def test_sine_has_expected_rms(self) -> None:
        sr = 48000
        t = np.arange(sr, dtype=np.float64) / sr
        x = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        db = _rms_db(x)
        assert abs(db - (-3.0103)) < 0.05


class TestPearson:
    def test_empty_left_returns_zero(self) -> None:
        a = np.empty(0, dtype=np.float32)
        b = np.ones(16, dtype=np.float32)
        assert _pearson(a, b) == 0.0

    def test_empty_right_returns_zero(self) -> None:
        a = np.ones(16, dtype=np.float32)
        b = np.empty(0, dtype=np.float32)
        assert _pearson(a, b) == 0.0

    def test_zero_norm_left_returns_zero(self) -> None:
        a = np.full(128, 1.0, dtype=np.float32)
        b = np.linspace(-1.0, 1.0, 128, dtype=np.float32)
        assert _pearson(a, b) == 0.0

    def test_zero_norm_right_returns_zero(self) -> None:
        a = np.linspace(-1.0, 1.0, 128, dtype=np.float32)
        b = np.full(128, -3.0, dtype=np.float32)
        assert _pearson(a, b) == 0.0

    def test_identical_signals_return_one(self) -> None:
        x = np.linspace(-1.0, 1.0, 512, dtype=np.float32)
        assert abs(_pearson(x, x) - 1.0) < 1e-6

    def test_anti_correlated_return_minus_one(self) -> None:
        x = np.linspace(-1.0, 1.0, 512, dtype=np.float32)
        assert abs(_pearson(x, -x) - (-1.0)) < 1e-6


class TestDecodePcmWindow:
    def test_rc_nonzero_returns_empty(self) -> None:
        adapter = _adapter()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = b"ffmpeg error: stream not found"
        mock_result.stdout = b""
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            out = adapter._decode_pcm_window(
                Path("video.mkv"),
                stream_index=1,
                channels=6,
                layout="5.1",
                start_s=10.0,
                dur_s=20.0,
            )
        assert out.shape == (0, 6)
        assert out.dtype == np.float32

    def test_empty_stdout_returns_empty(self) -> None:
        adapter = _adapter()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b""
        mock_result.stderr = b""
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            out = adapter._decode_pcm_window(
                Path("video.mkv"),
                stream_index=1,
                channels=2,
                layout="stereo",
                start_s=0.0,
                dur_s=20.0,
            )
        assert out.shape == (0, 2)

    def test_valid_pcm_reshaped(self) -> None:
        adapter = _adapter()
        pcm = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = pcm
        mock_result.stderr = b""
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            out = adapter._decode_pcm_window(
                Path("v.mkv"),
                stream_index=0,
                channels=2,
                layout="stereo",
                start_s=0.0,
                dur_s=20.0,
            )
        assert out.shape == (2, 2)
        assert abs(out[0, 0] - 0.1) < 1e-6
        assert abs(out[1, 1] - 0.4) < 1e-6

    def test_truncated_pcm_is_trimmed(self) -> None:
        adapter = _adapter()
        pcm = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32).tobytes()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = pcm
        mock_result.stderr = b""
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            out = adapter._decode_pcm_window(
                Path("v.mkv"),
                stream_index=0,
                channels=2,
                layout="stereo",
                start_s=0.0,
                dur_s=20.0,
            )
        assert out.shape == (2, 2)

    def test_cmd_contains_expected_filter(self) -> None:
        captured_cmd: list[list[str]] = []

        def fake_run(cmd: Any, **_: Any) -> MagicMock:
            captured_cmd.append(list(cmd))
            m = MagicMock()
            m.returncode = 0
            m.stdout = b""
            m.stderr = b""
            return m

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.subprocess.run", side_effect=fake_run):
            adapter._decode_pcm_window(
                Path("v.mkv"),
                stream_index=3,
                channels=6,
                layout="5.1",
                start_s=12.5,
                dur_s=20.0,
            )
        cmd = captured_cmd[0]
        af_idx = cmd.index("-af")
        assert cmd[af_idx + 1] == "aformat=channel_layouts=5.1:sample_rates=48000"
        assert "0:3" in cmd
        assert "f32le" in cmd


class TestProfileAudioTrack:
    @pytest.mark.parametrize("channels", [1, 4, 7])
    def test_unsupported_channels_raises(self, channels: int) -> None:
        adapter = _adapter()
        with pytest.raises(ValueError, match=f"unsupported channels={channels}"):
            adapter.profile_audio_track(Path("v.mkv"), 1, channels, 60.0)

    def test_no_windows_decoded_raises(self) -> None:
        adapter = _adapter()
        mock_result = MagicMock()
        mock_result.returncode = 2
        mock_result.stdout = b""
        mock_result.stderr = b"decode fail"
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="no windows decoded"):
                adapter.profile_audio_track(Path("v.mkv"), 0, 6, 60.0)

    def test_stereo_dispatch(self) -> None:
        adapter = _adapter()
        n = 480
        silent = np.zeros(n * 2, dtype=np.float32).tobytes()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = silent
        mock_result.stderr = b""
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            metrics = adapter.profile_audio_track(Path("v.mkv"), 0, 2, 60.0)
        assert metrics.channels == 2
        assert metrics.rms_l == -120.0
        assert metrics.rms_r == -120.0
        assert metrics.rms_c is None
        assert metrics.rms_lfe is None
        assert metrics.rms_ls is None
        assert metrics.rms_rs is None
        assert metrics.rms_lb is None
        assert metrics.rms_rb is None
        assert metrics.corr_lr == 0.0
        assert metrics.corr_ls_l is None

    def test_5_1_dispatch(self) -> None:
        adapter = _adapter()
        n = 480
        silent = np.zeros(n * 6, dtype=np.float32).tobytes()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = silent
        mock_result.stderr = b""
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            metrics = adapter.profile_audio_track(Path("v.mkv"), 0, 6, 60.0)
        assert metrics.channels == 6
        assert metrics.rms_c == -120.0
        assert metrics.rms_lfe == -120.0
        assert metrics.rms_ls == -120.0
        assert metrics.rms_rs == -120.0
        assert metrics.rms_lb is None
        assert metrics.rms_rb is None
        assert metrics.corr_ls_rs == 0.0
        assert metrics.corr_lb_ls is None
        assert metrics.corr_rb_rs is None

    def test_7_1_dispatch(self) -> None:
        adapter = _adapter()
        n = 480
        silent = np.zeros(n * 8, dtype=np.float32).tobytes()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = silent
        mock_result.stderr = b""
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            metrics = adapter.profile_audio_track(Path("v.mkv"), 0, 8, 60.0)
        assert metrics.channels == 8
        assert metrics.rms_lb == -120.0
        assert metrics.rms_rb == -120.0
        assert metrics.corr_lb_ls == 0.0
        assert metrics.corr_rb_rs == 0.0

    @pytest.mark.parametrize("channels", [2, 6])
    def test_samples_twelve_spread_windows(self, channels: int) -> None:
        adapter = _adapter()
        starts: list[float] = []

        def rec(
            path: Path,
            stream_index: int,
            channels: int,
            layout: str,
            start_s: float,
            dur_s: float,
        ) -> np.ndarray:
            starts.append(start_s)
            return np.zeros((480, channels), dtype=np.float32)

        with patch.object(adapter, "_decode_pcm_window", side_effect=rec):
            adapter.profile_audio_track(Path("v.mkv"), 0, channels, duration_s=600.0)

        assert len(starts) == 12
        assert len(set(starts)) == 12
        assert starts == sorted(starts)
        assert max(starts) - min(starts) > 0.7 * 600.0

    def test_partial_windows_drop_empty_chunks(self) -> None:
        adapter = _adapter()
        n = 480
        good_pcm = np.zeros(n * 6, dtype=np.float32).tobytes()

        call_results = iter(
            [
                MagicMock(returncode=0, stdout=good_pcm, stderr=b""),
                MagicMock(returncode=1, stdout=b"", stderr=b"err"),
            ]
            * 6
        )

        def fake_run(*_: Any, **__: Any) -> MagicMock:
            return next(call_results)

        with patch("furnace.adapters.ffmpeg.subprocess.run", side_effect=fake_run):
            metrics = adapter.profile_audio_track(Path("v.mkv"), 0, 6, 60.0)
        assert metrics.rms_l == -120.0


_DESCENDING_LEVELS = (0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05)


def _center_window(channels: int, *, derived: bool) -> np.ndarray:
    n = 4800
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    cols = [
        level * np.sin(2 * np.pi * (i + 1) * 10 * t) for i, level in enumerate(_DESCENDING_LEVELS[:channels])
    ]
    if derived:
        cols[2] = (cols[0] + cols[1]) / 2
    return np.stack(cols, axis=1)


def _profile_window(channels: int, *, derived: bool) -> AudioMetrics:
    adapter = _adapter()
    window = _center_window(channels, derived=derived)
    with patch.object(adapter, "_decode_pcm_window", side_effect=lambda *a, **k: window):
        return adapter.profile_audio_track(Path("v.mkv"), 1, channels, 1000.0)


class TestCenterDerivedFromTheFronts:
    @pytest.mark.parametrize("channels", [6, 8])
    def test_a_center_that_is_the_front_mix_correlates_to_one(self, channels: int) -> None:
        metrics = _profile_window(channels, derived=True)
        assert metrics.corr_c_lr is not None
        assert metrics.corr_c_lr > 0.999

    @pytest.mark.parametrize("channels", [6, 8])
    def test_an_independent_center_does_not_correlate_with_the_fronts(self, channels: int) -> None:
        metrics = _profile_window(channels, derived=False)
        assert metrics.corr_c_lr is not None
        assert abs(metrics.corr_c_lr) < 0.01

    def test_5_1_columns_map_to_the_declared_channel_order(self) -> None:
        m = _profile_window(6, derived=False)
        assert m.rms_c is not None
        assert m.rms_lfe is not None
        assert m.rms_ls is not None
        assert m.rms_rs is not None
        assert m.rms_l > m.rms_r > m.rms_c > m.rms_lfe > m.rms_ls > m.rms_rs

    def test_7_1_columns_map_to_the_declared_channel_order(self) -> None:
        m = _profile_window(8, derived=False)
        assert m.rms_c is not None
        assert m.rms_lfe is not None
        assert m.rms_lb is not None
        assert m.rms_rb is not None
        assert m.rms_ls is not None
        assert m.rms_rs is not None
        assert m.rms_l > m.rms_r > m.rms_c > m.rms_lfe
        assert m.rms_lfe > m.rms_lb > m.rms_rb > m.rms_ls > m.rms_rs


def _sparse_lfe_windows(channels: int, loud_window: int) -> list[np.ndarray]:
    n = 4800
    t = np.arange(n, dtype=np.float64) / 48000.0
    windows: list[np.ndarray] = []
    for i in range(12):
        w = np.zeros((n, channels), dtype=np.float32)
        w[:, 0] = (0.1 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        w[:, 1] = (0.1 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)
        if i == loud_window:
            w[:, 3] = (math.sqrt(2.0) * 1e-3 * np.sin(2 * np.pi * 40 * t)).astype(np.float32)
        windows.append(w)
    return windows


class TestSparseLfe:
    def test_5_1_lfe_alive_in_one_window_profiles_as_the_loudest_window(self) -> None:
        adapter = _adapter()
        with patch.object(adapter, "_decode_pcm_window", side_effect=_sparse_lfe_windows(6, 7)):
            metrics = adapter.profile_audio_track(Path("v.mkv"), 0, 6, 6000.0)
        assert metrics.rms_lfe == pytest.approx(-60.0, abs=0.5)

    def test_7_1_lfe_alive_in_one_window_profiles_as_the_loudest_window(self) -> None:
        adapter = _adapter()
        with patch.object(adapter, "_decode_pcm_window", side_effect=_sparse_lfe_windows(8, 2)):
            metrics = adapter.profile_audio_track(Path("v.mkv"), 0, 8, 6000.0)
        assert metrics.rms_lfe == pytest.approx(-60.0, abs=0.5)

    def test_lfe_dead_in_every_window_stays_at_the_floor(self) -> None:
        adapter = _adapter()
        with patch.object(adapter, "_decode_pcm_window", side_effect=_sparse_lfe_windows(6, -1)):
            metrics = adapter.profile_audio_track(Path("v.mkv"), 0, 6, 6000.0)
        assert metrics.rms_lfe == -120.0

    def test_a_loud_fragment_window_does_not_rescue_a_dead_lfe(self) -> None:
        adapter = _adapter()
        windows = _sparse_lfe_windows(6, -1)
        fragment = np.zeros((600, 6), dtype=np.float32)
        t = np.arange(600, dtype=np.float64) / 48000.0
        fragment[:, 3] = (0.5 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
        windows[11] = fragment
        with patch.object(adapter, "_decode_pcm_window", side_effect=windows):
            metrics = adapter.profile_audio_track(Path("v.mkv"), 0, 6, 6000.0)
        assert metrics.rms_lfe == -120.0

    def test_non_lfe_channels_stay_pooled_across_windows(self) -> None:
        adapter = _adapter()
        windows = _sparse_lfe_windows(6, -1)
        t = np.arange(windows[3].shape[0], dtype=np.float64) / 48000.0
        loud = windows[3].copy()
        loud[:, 4] = (math.sqrt(2.0) * 1e-2 * np.sin(2 * np.pi * 40 * t)).astype(np.float32)
        windows[3] = loud
        with patch.object(adapter, "_decode_pcm_window", side_effect=windows):
            metrics = adapter.profile_audio_track(Path("v.mkv"), 0, 6, 6000.0)
        assert metrics.rms_ls == pytest.approx(-40.0 - 10 * math.log10(12), abs=0.5)
