"""Integration test for FFmpegAdapter.profile_audio_track with a synthetic WAV."""

from __future__ import annotations

import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

from furnace.adapters.ffmpeg import FFmpegAdapter
from furnace.config import load_config


def _resolve_ffmpeg_paths() -> tuple[Path, Path]:
    """Return (ffmpeg, ffprobe) paths. Hard-fail if neither furnace.toml
    nor PATH yields working binaries — these tools are always required."""
    try:
        cfg = load_config()
    except (FileNotFoundError, KeyError):
        cfg = None
    if cfg is not None and Path(cfg.ffmpeg).exists() and Path(cfg.ffprobe).exists():
        return Path(cfg.ffmpeg), Path(cfg.ffprobe)
    which_ffmpeg = shutil.which("ffmpeg")
    which_ffprobe = shutil.which("ffprobe")
    if which_ffmpeg is None or which_ffprobe is None:
        raise RuntimeError(
            "ffmpeg/ffprobe not found via furnace.toml or PATH; these are required"
        )
    return Path(which_ffmpeg), Path(which_ffprobe)


def _write_synthetic_5_1_wav(path: Path, seconds: float = 2.0, sample_rate: int = 48000) -> None:
    """Write a tiny 5.1 WAV where only the center channel carries a 1 kHz tone.

    Other channels are digital silence. Exactly the pattern our detector
    should flag as a fake upmix.
    """
    n = int(seconds * sample_rate)
    # WAV channel order for 6-channel: L, R, C, LFE, Ls, Rs (standard extensible)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(6)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n):
            # Center channel: 1 kHz tone at -12 dBFS
            tone = int(0.25 * 32767 * math.sin(2 * math.pi * 1000 * i / sample_rate))
            samples = [0, 0, tone, 0, 0, 0]
            for s in samples:
                frames += struct.pack("<h", s)
        w.writeframes(bytes(frames))


@pytest.fixture
def adapter() -> FFmpegAdapter:
    ffmpeg_path, ffprobe_path = _resolve_ffmpeg_paths()
    return FFmpegAdapter(ffmpeg_path=ffmpeg_path, ffprobe_path=ffprobe_path)


def test_profile_audio_track_5_1_synthetic_wav(tmp_path: Path, adapter: FFmpegAdapter) -> None:
    wav_path = tmp_path / "synthetic.wav"
    _write_synthetic_5_1_wav(wav_path, seconds=2.0)

    metrics = adapter.profile_audio_track(
        path=wav_path,
        stream_index=0,
        channels=6,
        duration_s=2.0,
    )

    assert metrics.channels == 6
    # Center has the tone → not silent
    assert metrics.rms_c is not None
    assert metrics.rms_c > -30, f"expected loud center, got {metrics.rms_c}"
    # Surrounds are digital zero → clamped at -120 dB floor
    assert metrics.rms_ls is not None
    assert metrics.rms_ls < -80, f"expected silent Ls, got {metrics.rms_ls}"
    assert metrics.rms_rs is not None
    assert metrics.rms_rs < -80
    # LFE is silent
    assert metrics.rms_lfe is not None
    assert metrics.rms_lfe < -80
