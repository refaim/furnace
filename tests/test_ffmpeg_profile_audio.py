from __future__ import annotations

import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

from furnace.adapters.ffmpeg import FFmpegAdapter
from furnace.config import ToolPaths, load_config


def _resolve_ffmpeg_paths() -> tuple[Path, Path]:
    try:
        cfg = load_config()
    except (FileNotFoundError, KeyError):
        cfg = None
    if cfg is not None and Path(cfg.ffmpeg).exists() and Path(cfg.ffprobe).exists():
        return Path(cfg.ffmpeg), Path(cfg.ffprobe)
    which_ffmpeg = shutil.which("ffmpeg")
    which_ffprobe = shutil.which("ffprobe")
    if which_ffmpeg is None or which_ffprobe is None:
        raise RuntimeError("ffmpeg/ffprobe not found via furnace.toml or PATH; these are required")
    return Path(which_ffmpeg), Path(which_ffprobe)


def _tool_paths(ffmpeg: Path, ffprobe: Path) -> ToolPaths:
    unused = Path("unused")
    return ToolPaths(
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        mkvmerge=unused,
        mkvpropedit=unused,
        mkclean=unused,
        eac3to=unused,
        qaac64=unused,
        mpv=unused,
        makemkvcon=unused,
        nvencc=unused,
        dovi_tool=None,
    )


def test_resolve_ffmpeg_paths_prefers_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ffmpeg = tmp_path / "ffmpeg"
    ffprobe = tmp_path / "ffprobe"
    ffmpeg.touch()
    ffprobe.touch()
    monkeypatch.setattr(
        "tests.test_ffmpeg_profile_audio.load_config",
        lambda: _tool_paths(ffmpeg, ffprobe),
    )
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    assert _resolve_ffmpeg_paths() == (ffmpeg, ffprobe)


def test_resolve_ffmpeg_paths_hard_fails_when_tools_are_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    def _no_config() -> ToolPaths:
        raise FileNotFoundError("no furnace.toml")

    monkeypatch.setattr("tests.test_ffmpeg_profile_audio.load_config", _no_config)
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="these are required"):
        _resolve_ffmpeg_paths()


def _write_synthetic_5_1_wav(path: Path, seconds: float = 2.0, sample_rate: int = 48000) -> None:
    n = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(6)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n):
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
    assert metrics.rms_c is not None
    assert metrics.rms_c > -30, f"expected loud center, got {metrics.rms_c}"
    assert metrics.rms_ls is not None
    assert metrics.rms_ls < -80, f"expected silent Ls, got {metrics.rms_ls}"
    assert metrics.rms_rs is not None
    assert metrics.rms_rs < -80
    assert metrics.rms_lfe is not None
    assert metrics.rms_lfe < -80
