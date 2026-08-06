from __future__ import annotations

import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from furnace.adapters.ffmpeg import FFmpegAdapter
from furnace.config import ToolPaths, load_config
from furnace.core.audio_profile import Verdict, classify_audio
from furnace.core.downmix import DownmixMode


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


def _write_derived_center_wav(
    path: Path,
    channels: int,
    *,
    derived: bool = True,
    seconds: float = 2.0,
    sample_rate: int = 48000,
) -> None:
    n = int(seconds * sample_rate)
    freqs = [1000, 300, 700, 90, 1700, 2300, 3100, 3700][:channels]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n):
            tones = [0.25 * math.sin(2 * math.pi * f * i / sample_rate) for f in freqs]
            if derived:
                tones[2] = (tones[0] + tones[1]) / 2
            for value in tones:
                frames += struct.pack("<h", int(value * 32767))
        w.writeframes(bytes(frames))


@pytest.mark.parametrize("channels", [6, 8])
def test_a_center_derived_from_the_fronts_profiles_as_fake(
    tmp_path: Path,
    adapter: FFmpegAdapter,
    channels: int,
) -> None:
    wav_path = tmp_path / f"derived_center_{channels}.wav"
    _write_derived_center_wav(wav_path, channels)

    metrics = adapter.profile_audio_track(
        path=wav_path,
        stream_index=0,
        channels=channels,
        duration_s=2.0,
    )

    assert metrics.corr_c_lr is not None
    assert metrics.corr_c_lr > 0.95, f"expected a derived center, got corr={metrics.corr_c_lr}"
    profile = classify_audio(metrics)
    assert profile.verdict is Verdict.FAKE
    assert any("mix of the fronts" in r for r in profile.reasons)


@pytest.mark.parametrize("channels", [6, 8])
def test_an_independent_center_profiles_as_real(
    tmp_path: Path,
    adapter: FFmpegAdapter,
    channels: int,
) -> None:
    wav_path = tmp_path / f"real_center_{channels}.wav"
    _write_derived_center_wav(wav_path, channels, derived=False)

    metrics = adapter.profile_audio_track(
        path=wav_path,
        stream_index=0,
        channels=channels,
        duration_s=2.0,
    )

    assert metrics.corr_c_lr is not None
    assert abs(metrics.corr_c_lr) < 0.05
    assert classify_audio(metrics).verdict is Verdict.REAL


def _write_synthetic_three_channel_wav(
    path: Path,
    ffmpeg: Path,
    layout: str,
    *,
    third_silent: bool = True,
    seconds: float = 2.0,
) -> None:
    third = (
        f"anullsrc=r=48000:cl=mono:d={seconds}"
        if third_silent
        else f"sine=frequency=700:duration={seconds}:sample_rate=48000"
    )
    third_channel = "LFE" if layout == "2.1" else "FC"
    subprocess.run(
        [
            str(ffmpeg), "-v", "error", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=1000:duration={seconds}:sample_rate=48000",
            "-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}:sample_rate=48000",
            "-f", "lavfi", "-i", third,
            "-filter_complex",
            f"[0:a][1:a][2:a]join=inputs=3:channel_layout={layout}:"
            f"map=0.0-FL|1.0-FR|2.0-{third_channel}[a]",
            "-map", "[a]", "-c:a", "pcm_s16le", str(path),
        ],
        check=True,
        capture_output=True,
    )


def _write_synthetic_five_zero_wav(
    path: Path,
    ffmpeg: Path,
    layout: str,
    *,
    surrounds_silent: bool = True,
    seconds: float = 2.0,
) -> None:
    silence = f"anullsrc=r=48000:cl=mono:d={seconds}"
    left_surround = silence if surrounds_silent else f"sine=frequency=1700:duration={seconds}:sample_rate=48000"
    right_surround = silence if surrounds_silent else f"sine=frequency=2300:duration={seconds}:sample_rate=48000"
    left_back, right_back = ("SL", "SR") if layout.endswith("(side)") else ("BL", "BR")
    subprocess.run(
        [
            str(ffmpeg), "-v", "error", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=1000:duration={seconds}:sample_rate=48000",
            "-f", "lavfi", "-i", f"sine=frequency=300:duration={seconds}:sample_rate=48000",
            "-f", "lavfi", "-i", f"sine=frequency=700:duration={seconds}:sample_rate=48000",
            "-f", "lavfi", "-i", left_surround,
            "-f", "lavfi", "-i", right_surround,
            "-filter_complex",
            f"[0:a][1:a][2:a][3:a][4:a]join=inputs=5:channel_layout={layout}:"
            f"map=0.0-FL|1.0-FR|2.0-FC|3.0-{left_back}|4.0-{right_back}[a]",
            "-map", "[a]", "-c:a", "pcm_s16le", str(path),
        ],
        check=True,
        capture_output=True,
    )


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


def test_profile_audio_track_2_1_synthetic_wav(tmp_path: Path, adapter: FFmpegAdapter) -> None:
    wav_path = tmp_path / "synthetic_2_1.wav"
    _write_synthetic_three_channel_wav(wav_path, _resolve_ffmpeg_paths()[0], "2.1")

    metrics = adapter.profile_audio_track(
        path=wav_path,
        stream_index=0,
        channels=3,
        duration_s=2.0,
        channel_layout="2.1",
    )

    assert metrics.channels == 3
    assert metrics.rms_c is None
    assert metrics.rms_lfe is not None
    assert metrics.rms_lfe < -90, f"expected a dead LFE, got {metrics.rms_lfe}"
    assert metrics.rms_l > -30, f"expected a loud left, got {metrics.rms_l}"
    assert metrics.rms_r > -30, f"expected a loud right, got {metrics.rms_r}"

    profile = classify_audio(metrics)
    assert profile.verdict is Verdict.FAKE
    assert profile.suggested is DownmixMode.STEREO


def test_profile_audio_track_3_0_synthetic_wav(tmp_path: Path, adapter: FFmpegAdapter) -> None:
    wav_path = tmp_path / "synthetic_3_0.wav"
    _write_synthetic_three_channel_wav(wav_path, _resolve_ffmpeg_paths()[0], "3.0")

    metrics = adapter.profile_audio_track(
        path=wav_path,
        stream_index=0,
        channels=3,
        duration_s=2.0,
        channel_layout="3.0",
    )

    assert metrics.rms_lfe is None
    assert metrics.rms_c is not None
    assert metrics.rms_c < -90, f"expected a silent center, got {metrics.rms_c}"
    assert metrics.rms_l > -30
    assert metrics.rms_r > -30
    assert classify_audio(metrics).verdict is Verdict.FAKE


def test_profile_audio_track_3_0_center_is_a_mix_of_the_fronts(tmp_path: Path, adapter: FFmpegAdapter) -> None:
    wav_path = tmp_path / "matrix_3_0.wav"
    ffmpeg = _resolve_ffmpeg_paths()[0]
    subprocess.run(
        [
            str(ffmpeg), "-v", "error", "-y",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=2:sample_rate=48000",
            "-f", "lavfi", "-i", "sine=frequency=300:duration=2:sample_rate=48000",
            "-filter_complex",
            "[0:a][1:a]amerge=inputs=2[st];[st]asplit=2[a][b];"
            "[b]pan=mono|c0=0.5*c0+0.5*c1[mix];"
            "[a][mix]join=inputs=2:channel_layout=3.0[out]",
            "-map", "[out]", "-c:a", "pcm_s16le", str(wav_path),
        ],
        check=True,
        capture_output=True,
    )

    metrics = adapter.profile_audio_track(
        path=wav_path,
        stream_index=0,
        channels=3,
        duration_s=2.0,
        channel_layout="3.0",
    )

    assert metrics.corr_c_lr is not None
    assert metrics.corr_c_lr > 0.95, f"expected a derived center, got corr={metrics.corr_c_lr}"
    profile = classify_audio(metrics)
    assert profile.verdict is Verdict.FAKE
    assert any("mix of the fronts" in r for r in profile.reasons)


@pytest.mark.parametrize("layout", ["5.0", "5.0(side)"])
def test_profile_audio_track_5_0_synthetic_wav(tmp_path: Path, adapter: FFmpegAdapter, layout: str) -> None:
    wav_path = tmp_path / "synthetic_5_0.wav"
    _write_synthetic_five_zero_wav(wav_path, _resolve_ffmpeg_paths()[0], layout)

    metrics = adapter.profile_audio_track(
        path=wav_path,
        stream_index=0,
        channels=5,
        duration_s=2.0,
        channel_layout=layout,
    )

    assert metrics.channels == 5
    assert metrics.rms_lfe is None, "a 5.0 track has no LFE — none may be fabricated"
    assert metrics.rms_c is not None
    assert metrics.rms_c > -30, f"expected a loud center, got {metrics.rms_c}"
    assert metrics.rms_l > -30
    assert metrics.rms_r > -30
    assert metrics.rms_ls is not None
    assert metrics.rms_ls < -80, f"expected a silent Ls, got {metrics.rms_ls}"
    assert metrics.rms_rs is not None
    assert metrics.rms_rs < -80

    profile = classify_audio(metrics)
    assert profile.verdict is Verdict.SUSPICIOUS
    assert profile.suggested is DownmixMode.STEREO
    assert any("both surrounds are silent" in r for r in profile.reasons)
    assert not any("LFE" in r for r in profile.reasons)


def test_a_live_5_0_track_profiles_as_real(tmp_path: Path, adapter: FFmpegAdapter) -> None:
    wav_path = tmp_path / "live_5_0.wav"
    _write_synthetic_five_zero_wav(
        wav_path,
        _resolve_ffmpeg_paths()[0],
        "5.0(side)",
        surrounds_silent=False,
    )

    metrics = adapter.profile_audio_track(
        path=wav_path,
        stream_index=0,
        channels=5,
        duration_s=2.0,
        channel_layout="5.0(side)",
    )

    assert classify_audio(metrics).verdict is Verdict.REAL


def test_the_declared_layout_decides_how_the_third_channel_is_read(
    tmp_path: Path,
    adapter: FFmpegAdapter,
) -> None:
    wav_path = tmp_path / "real_2_1.wav"
    _write_synthetic_three_channel_wav(wav_path, _resolve_ffmpeg_paths()[0], "2.1", third_silent=False)

    as_declared = adapter.profile_audio_track(
        path=wav_path, stream_index=0, channels=3, duration_s=2.0, channel_layout="2.1"
    )
    mislabelled = adapter.profile_audio_track(
        path=wav_path, stream_index=0, channels=3, duration_s=2.0, channel_layout="3.0"
    )

    assert as_declared.rms_lfe is not None
    assert as_declared.rms_lfe > -30, "the real LFE must survive when the layout is declared correctly"
    assert mislabelled.rms_c is not None
    assert mislabelled.rms_c < as_declared.rms_lfe - 20, (
        "reading a 2.1 file as 3.0 must not silently produce the same third channel"
    )
