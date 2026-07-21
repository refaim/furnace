from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from furnace.adapters.ffmpeg import FFmpegAdapter, _parse_y4m_luma

_SD_H, _SD_W = 576, 720


def _window(n: int, noise_amp: float, h: int = _SD_H, w: int = _SD_W, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((h, w), 128.0, dtype=np.float32)
    return np.stack([np.clip(base + rng.normal(0.0, noise_amp, base.shape), 0, 255).astype(np.uint8) for _ in range(n)])


def _y4m(frames: np.ndarray) -> bytes:
    n, h, w = frames.shape
    header = f"YUV4MPEG2 W{w} H{h} F25:1 Ip A1:1 Cmono XCOLORRANGE=FULL\n".encode()
    return header + b"".join(b"FRAME\n" + frames[i].tobytes() for i in range(n))


def _raw_frames(n: int, noise_amp: float, h: int = _SD_H, w: int = _SD_W, seed: int = 7) -> bytes:
    return _y4m(_window(n, noise_amp, h, w, seed))


def _black_frames(n: int) -> bytes:
    return _y4m(np.zeros((n, _SD_H, _SD_W), dtype=np.uint8))


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
    assert cmd[cmd.index("-f") + 1] == "yuv4mpegpipe"
    assert cmd[cmd.index("-pix_fmt") + 1] == "gray"
    assert cmd[cmd.index("-frames:v") + 1] == "24"
    assert cmd[cmd.index("-i") + 1] == "v.mkv"
    assert cmd[cmd.index("-strict") + 1] == "-1"
    assert cmd[-1] == "-"


def test_sample_grain_never_downscales() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))

    with patch(
        "furnace.adapters.ffmpeg.subprocess.run",
        return_value=_fake_result(_raw_frames(24, 1.0)),
    ) as mock_run:
        adapter.sample_grain(Path("v.mkv"), duration_s=1000.0)

    for call in mock_run.call_args_list:
        cmd = call.args[0]
        assert cmd[cmd.index("-vf") + 1] == "format=gray"
        assert not any("scale" in str(arg) for arg in cmd)


def test_sample_grain_reads_the_resolution_ffmpeg_delivers() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    seen: list[tuple[int, ...]] = []

    def record(frames: np.ndarray) -> float:
        seen.append(frames.shape)
        return 1.0

    for height, width in ((576, 720), (1080, 1920)):
        with (
            patch(
                "furnace.adapters.ffmpeg.subprocess.run",
                return_value=_fake_result(_raw_frames(24, 2.0, height, width)),
            ),
            patch("furnace.adapters.ffmpeg._grain_window_value", side_effect=record),
        ):
            adapter.sample_grain(Path("v.mkv"), duration_s=1000.0)

    assert seen[:5] == [(24, 576, 720)] * 5
    assert seen[5:] == [(24, 1080, 1920)] * 5


def test_sample_grain_amplitude_tracks_noise() -> None:
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


def test_sample_grain_measures_the_same_grain_the_same_at_any_resolution() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    measured: list[float] = []

    for height, width in ((576, 720), (1080, 1920)):
        with patch(
            "furnace.adapters.ffmpeg.subprocess.run",
            return_value=_fake_result(_raw_frames(24, 2.0, height, width)),
        ):
            measured.append(adapter.sample_grain(Path("v.mkv"), duration_s=1000.0)[0])

    sd, hd = measured
    assert abs(hd - sd) / sd < 0.05


def test_sample_grain_skips_failed_window() -> None:
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
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    good = _raw_frames(24, 2.0)
    results = [
        _fake_result(good),
        _fake_result(_raw_frames(5, 2.0)),
        _fake_result(good),
        _fake_result(good),
        _fake_result(good),
    ]

    with patch("furnace.adapters.ffmpeg.subprocess.run", side_effect=results):
        values = adapter.sample_grain(Path("v.mkv"), duration_s=1000.0)

    assert len(values) == 4


def test_sample_grain_skips_unparsable_window() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    good = _raw_frames(24, 2.0)
    results = [
        _fake_result(good),
        _fake_result(b"not a y4m stream at all"),
        _fake_result(good),
        _fake_result(good),
        _fake_result(good),
    ]

    with patch("furnace.adapters.ffmpeg.subprocess.run", side_effect=results):
        values = adapter.sample_grain(Path("v.mkv"), duration_s=1000.0)

    assert len(values) == 4


def test_sample_grain_all_black_windows_return_empty() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))

    with patch(
        "furnace.adapters.ffmpeg.subprocess.run",
        return_value=_fake_result(_black_frames(24)),
    ):
        values = adapter.sample_grain(Path("v.mkv"), duration_s=1000.0)

    assert values == []


class TestParseY4mLuma:
    def test_returns_frames_shaped_by_the_header(self) -> None:
        frames = _window(12, 3.0, 240, 320)
        parsed = _parse_y4m_luma(_y4m(frames))
        assert parsed is not None
        assert parsed.shape == (12, 240, 320)
        assert np.array_equal(parsed, frames)

    def test_rejects_a_stream_without_the_magic(self) -> None:
        assert _parse_y4m_luma(b"RAWVIDEO W720 H576 \nFRAME\n" + b"\x80" * 16) is None

    def test_rejects_a_truncated_header(self) -> None:
        assert _parse_y4m_luma(b"YUV4MPEG2 W720 H576 F25:1") is None

    def test_rejects_zero_dimensions(self) -> None:
        assert _parse_y4m_luma(b"YUV4MPEG2 W0 H0 F25:1\nFRAME\n") is None

    def test_rejects_a_header_with_no_frames(self) -> None:
        assert _parse_y4m_luma(b"YUV4MPEG2 W8 H8 F25:1 Ip A1:1 Cmono\n") is None

    def test_drops_a_truncated_trailing_frame(self) -> None:
        full = _y4m(_window(4, 2.0, 32, 32))
        parsed = _parse_y4m_luma(full[: -32 * 16])
        assert parsed is not None
        assert parsed.shape == (3, 32, 32)

    def test_stops_at_an_unexpected_frame_header(self) -> None:
        frames = _window(2, 2.0, 32, 32)
        stream = _y4m(frames) + b"FRAME Xinterlace\n" + frames[0].tobytes()
        parsed = _parse_y4m_luma(stream)
        assert parsed is not None
        assert parsed.shape == (2, 32, 32)
