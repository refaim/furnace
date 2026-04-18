"""Unit tests for FFmpegAdapter.profile_audio_track, _decode_pcm_window, and
the module-level RMS/correlation helpers.

These tests never invoke the real ffmpeg binary — ``subprocess.run`` is
patched so every edge branch (empty input, zero-norm, non-zero return code,
truncated PCM buffer, unsupported channel count) is exercised under control.
"""

from __future__ import annotations

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


def _adapter() -> FFmpegAdapter:
    return FFmpegAdapter(Path("ffmpeg.exe"), Path("ffprobe.exe"))


class TestRmsDb:
    def test_empty_returns_floor(self) -> None:
        assert _rms_db(np.empty(0, dtype=np.float32)) == -120.0

    def test_digital_silence_returns_floor(self) -> None:
        # Exact zeros → rms below 1e-9 → clamped at -120 dB floor.
        silent = np.zeros(1024, dtype=np.float32)
        assert _rms_db(silent) == -120.0

    def test_sine_has_expected_rms(self) -> None:
        # Full-scale 1 kHz sine at 48 kHz: RMS = 1/sqrt(2) ≈ -3.01 dBFS
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
        # Exact integer-valued constant: (x - mean) is bit-exact zero → na=0.
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
                Path("video.mkv"), stream_index=1, channels=6,
                layout="5.1", start_s=10.0, dur_s=20.0,
            )
        assert out.shape == (0, 6)
        assert out.dtype == np.float32

    def test_empty_stdout_returns_empty(self) -> None:
        adapter = _adapter()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = b""  # zero samples
        mock_result.stderr = b""
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            out = adapter._decode_pcm_window(
                Path("video.mkv"), stream_index=1, channels=2,
                layout="stereo", start_s=0.0, dur_s=20.0,
            )
        assert out.shape == (0, 2)

    def test_valid_pcm_reshaped(self) -> None:
        adapter = _adapter()
        # Two stereo frames: [0.1, 0.2], [0.3, 0.4]
        pcm = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = pcm
        mock_result.stderr = b""
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            out = adapter._decode_pcm_window(
                Path("v.mkv"), stream_index=0, channels=2,
                layout="stereo", start_s=0.0, dur_s=20.0,
            )
        assert out.shape == (2, 2)
        assert abs(out[0, 0] - 0.1) < 1e-6
        assert abs(out[1, 1] - 0.4) < 1e-6

    def test_truncated_pcm_is_trimmed(self) -> None:
        """A trailing partial frame (not a multiple of channels*4 bytes) is
        dropped; reshape operates on ``n * channels`` samples only."""
        adapter = _adapter()
        # 5 floats for 2 channels → 2 full frames, 1 float trailing.
        pcm = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32).tobytes()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = pcm
        mock_result.stderr = b""
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            out = adapter._decode_pcm_window(
                Path("v.mkv"), stream_index=0, channels=2,
                layout="stereo", start_s=0.0, dur_s=20.0,
            )
        assert out.shape == (2, 2)

    def test_cmd_contains_expected_filter(self) -> None:
        """Verify the aformat filter argument carries layout and sample rate."""
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
                Path("v.mkv"), stream_index=3, channels=6,
                layout="5.1", start_s=12.5, dur_s=20.0,
            )
        cmd = captured_cmd[0]
        af_idx = cmd.index("-af")
        assert cmd[af_idx + 1] == "aformat=channel_layouts=5.1:sample_rates=48000"
        assert "0:3" in cmd
        assert "f32le" in cmd


class TestProfileAudioTrack:
    def test_unsupported_channels_raises(self) -> None:
        adapter = _adapter()
        with pytest.raises(ValueError, match="unsupported channels=3"):
            adapter.profile_audio_track(Path("v.mkv"), 1, 3, 60.0)

    def test_no_windows_decoded_raises(self) -> None:
        """Every window returns empty → RuntimeError."""
        adapter = _adapter()
        mock_result = MagicMock()
        mock_result.returncode = 2
        mock_result.stdout = b""
        mock_result.stderr = b"decode fail"
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="no windows decoded"):
                adapter.profile_audio_track(Path("v.mkv"), 0, 6, 60.0)

    def test_stereo_dispatch(self) -> None:
        """Stereo path: only L/R fields populated, multichannel are None."""
        adapter = _adapter()
        # 480 frames of 2-channel f32 zeros == digital silence => rms at -120.
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
        assert metrics.corr_lr == 0.0  # zero-norm → 0
        assert metrics.corr_ls_l is None

    def test_5_1_dispatch(self) -> None:
        """5.1 path: L/R/C/LFE/Ls/Rs populated, Lb/Rb None."""
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
        """7.1 path: every RMS/corr field populated."""
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

    def test_partial_windows_drop_empty_chunks(self) -> None:
        """If some windows fail and some succeed, only successful windows
        contribute to the concatenation — exercise the ``if window.size > 0``
        branch True and False in the same call."""
        adapter = _adapter()
        n = 480
        good_pcm = np.zeros(n * 6, dtype=np.float32).tobytes()

        call_results = iter([
            MagicMock(returncode=0, stdout=good_pcm, stderr=b""),   # 0.15 → good
            MagicMock(returncode=1, stdout=b"", stderr=b"err"),      # 0.35 → empty
            MagicMock(returncode=0, stdout=good_pcm, stderr=b""),   # 0.55 → good
            MagicMock(returncode=1, stdout=b"", stderr=b"err"),      # 0.75 → empty
        ])

        def fake_run(*_: Any, **__: Any) -> MagicMock:
            return next(call_results)

        with patch("furnace.adapters.ffmpeg.subprocess.run", side_effect=fake_run):
            metrics = adapter.profile_audio_track(Path("v.mkv"), 0, 6, 60.0)
        # 2 windows of 480 frames == 960 samples per channel, all zero => -120.
        assert metrics.rms_l == -120.0
