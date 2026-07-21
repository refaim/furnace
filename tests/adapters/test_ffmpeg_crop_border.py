from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from furnace.adapters.ffmpeg import FFmpegAdapter, _border_side_levels, _parse_y4m_luma
from furnace.core.progress import ProgressSample


def _y4m_420(frames: np.ndarray, tag: str = " C420mpeg2") -> bytes:
    n, h, w = frames.shape
    header = f"YUV4MPEG2 W{w} H{h} F25:1 Ip A1:1{tag}\n".encode()
    chroma = bytes([128]) * (2 * ((w + 1) // 2) * ((h + 1) // 2))
    return header + b"".join(b"FRAME\n" + frames[i].tobytes() + chroma for i in range(n))


def _bordered(level: int, inner: int = 200, n: int = 2, size: int = 64, ring: int = 8) -> np.ndarray:
    frame = np.full((size, size), inner, dtype=np.uint8)
    frame[:ring] = level
    frame[-ring:] = level
    frame[:, :ring] = level
    frame[:, -ring:] = level
    return np.stack([frame] * n)


def _adapter() -> FFmpegAdapter:
    return FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))


def _is_probe(cmd: list[str]) -> bool:
    return "yuv4mpegpipe" in cmd


def _crop_calls(call_args_list: list[Any]) -> list[list[str]]:
    return [list(call.args[0]) for call in call_args_list if not _is_probe(list(call.args[0]))]


def _probe_calls(call_args_list: list[Any]) -> list[list[str]]:
    return [list(call.args[0]) for call in call_args_list if _is_probe(list(call.args[0]))]


def _vf(cmd: list[str]) -> str:
    return cmd[cmd.index("-vf") + 1]


def _run_stub(
    probe_stdout: bytes,
    crop: str = "crop=702:484:10:46",
    probe_rc: int = 0,
) -> Callable[..., MagicMock]:
    def run(cmd: list[str], **kwargs: Any) -> MagicMock:
        result = MagicMock()
        if _is_probe(cmd):
            result.returncode = probe_rc
            result.stdout = probe_stdout
            result.stderr = b""
        else:
            result.returncode = 0
            result.stdout = b""
            result.stderr = f"[Parsed_cropdetect_0 @ 0x0] {crop}\n"
        return result

    return run


class TestParseY4mLuma:
    def test_extracts_the_luma_plane_of_a_420_stream(self) -> None:
        frames = _bordered(45, n=3, size=32)
        parsed = _parse_y4m_luma(_y4m_420(frames))
        assert parsed is not None
        assert parsed.shape == (3, 32, 32)
        assert np.array_equal(parsed, frames)

    def test_assumes_420_when_the_header_omits_the_chroma_tag(self) -> None:
        frames = _bordered(45, n=2, size=32)
        parsed = _parse_y4m_luma(_y4m_420(frames, tag=""))
        assert parsed is not None
        assert np.array_equal(parsed, frames)

    def test_rejects_an_unsupported_chroma_tag(self) -> None:
        frames = _bordered(45, n=2, size=32)
        assert _parse_y4m_luma(_y4m_420(frames, tag=" C444")) is None

    def test_odd_sized_420_frames_match_the_muxer_byte_layout(self) -> None:
        stream = _y4m_420(_bordered(45, n=1, size=63))
        assert len(stream) == len("YUV4MPEG2 W63 H63 F25:1 Ip A1:1 C420mpeg2\n") + len("FRAME\n") + 6017

    def test_parses_every_frame_of_an_odd_sized_stream(self) -> None:
        frames = _bordered(45, n=4, size=63)
        parsed = _parse_y4m_luma(_y4m_420(frames))
        assert parsed is not None
        assert parsed.shape == (4, 63, 63)
        assert np.array_equal(parsed, frames)


class TestBorderSideLevels:
    def test_reports_the_median_line_mean_of_each_side(self) -> None:
        levels = _border_side_levels(_bordered(45))
        assert levels == (45.0, 45.0, 45.0, 45.0)

    def test_one_bright_edge_row_does_not_move_the_level(self) -> None:
        frames = _bordered(45)
        frames[:, 0, :] = 250
        assert _border_side_levels(frames)[0] == 45.0

    def test_one_bright_edge_column_does_not_move_the_level(self) -> None:
        frames = _bordered(45)
        frames[:, :, 0] = 250
        assert _border_side_levels(frames)[2] == 45.0

    def test_takes_the_brightest_frame_of_the_window(self) -> None:
        frames = _bordered(45, n=3)
        frames[2, :8, :] = 60
        levels = _border_side_levels(frames)
        assert levels[0] == 60.0

    def test_sides_are_reported_top_bottom_left_right(self) -> None:
        base = np.full((2, 64, 64), 200, dtype=np.uint8)
        for index, painted in enumerate(
            (
                np.s_[:, :8, :],
                np.s_[:, -8:, :],
                np.s_[:, :, :8],
                np.s_[:, :, -8:],
            )
        ):
            frames = base.copy()
            frames[painted] = 30
            assert _border_side_levels(frames)[index] == 30.0

    def test_a_frame_thinner_than_the_ring_still_reports_levels(self) -> None:
        frames = np.full((2, 4, 4), 45, dtype=np.uint8)
        assert _border_side_levels(frames) == (45.0, 45.0, 45.0, 45.0)


class TestDetectCropAdaptiveLimit:
    def test_noisy_bars_raise_the_cropdetect_limit(self) -> None:
        adapter = _adapter()
        with patch(
            "furnace.adapters.ffmpeg.subprocess.run",
            side_effect=_run_stub(_y4m_420(_bordered(45))),
        ) as mock_run:
            adapter.detect_crop(Path("v.mkv"), duration_s=1000.0)
        for cmd in _crop_calls(mock_run.call_args_list):
            assert _vf(cmd) == "format=yuv420p,cropdetect=49:2:0"

    def test_full_frame_content_keeps_the_default_limit(self) -> None:
        adapter = _adapter()
        frames = np.full((2, 64, 64), 200, dtype=np.uint8)
        with patch(
            "furnace.adapters.ffmpeg.subprocess.run",
            side_effect=_run_stub(_y4m_420(frames)),
        ) as mock_run:
            adapter.detect_crop(Path("v.mkv"), duration_s=1000.0)
        for cmd in _crop_calls(mock_run.call_args_list):
            assert _vf(cmd) == "format=yuv420p,cropdetect=40:2:0"

    def test_clean_black_bars_keep_the_default_limit(self) -> None:
        adapter = _adapter()
        with patch(
            "furnace.adapters.ffmpeg.subprocess.run",
            side_effect=_run_stub(_y4m_420(_bordered(16))),
        ) as mock_run:
            adapter.detect_crop(Path("v.mkv"), duration_s=1000.0)
        for cmd in _crop_calls(mock_run.call_args_list):
            assert _vf(cmd) == "format=yuv420p,cropdetect=40:2:0"

    def test_failed_probe_keeps_the_default_limit(self) -> None:
        adapter = _adapter()
        with patch(
            "furnace.adapters.ffmpeg.subprocess.run",
            side_effect=_run_stub(_y4m_420(_bordered(45)), probe_rc=1),
        ) as mock_run:
            adapter.detect_crop(Path("v.mkv"), duration_s=1000.0)
        for cmd in _crop_calls(mock_run.call_args_list):
            assert _vf(cmd) == "format=yuv420p,cropdetect=40:2:0"

    def test_unreadable_probe_output_keeps_the_default_limit(self) -> None:
        adapter = _adapter()
        with patch(
            "furnace.adapters.ffmpeg.subprocess.run",
            side_effect=_run_stub(b"not y4m at all"),
        ) as mock_run:
            adapter.detect_crop(Path("v.mkv"), duration_s=1000.0)
        for cmd in _crop_calls(mock_run.call_args_list):
            assert _vf(cmd) == "format=yuv420p,cropdetect=40:2:0"

    def test_surviving_probe_points_still_set_the_limit(self) -> None:
        adapter = _adapter()
        bars = _y4m_420(_bordered(45))
        bright = _y4m_420(np.full((2, 64, 64), 200, dtype=np.uint8))
        seen = 0
        samples: list[ProgressSample] = []

        def run(cmd: list[str], **kwargs: Any) -> MagicMock:
            nonlocal seen
            result = MagicMock()
            if _is_probe(cmd):
                seen += 1
                failed = seen <= 6
                result.returncode = 1 if failed else 0
                result.stdout = bright if failed else bars
                result.stderr = b"decoder blew up"
            else:
                result.returncode = 0
                result.stdout = b""
                result.stderr = "[Parsed_cropdetect_0 @ 0x0] crop=702:484:10:46\n"
            return result

        with patch("furnace.adapters.ffmpeg.subprocess.run", side_effect=run) as mock_run:
            adapter.detect_crop(Path("v.mkv"), duration_s=1000.0, on_progress=samples.append)
        for cmd in _crop_calls(mock_run.call_args_list):
            assert _vf(cmd) == "format=yuv420p,cropdetect=49:2:0"
        assert len(samples) == 30

    def test_one_bright_sample_point_keeps_the_default_limit(self) -> None:
        adapter = _adapter()
        bars = _y4m_420(_bordered(45))
        bright = _y4m_420(np.full((2, 64, 64), 200, dtype=np.uint8))
        seen = 0

        def run(cmd: list[str], **kwargs: Any) -> MagicMock:
            nonlocal seen
            result = MagicMock()
            result.returncode = 0
            if _is_probe(cmd):
                seen += 1
                result.stdout = bright if seen == 3 else bars
            else:
                result.stdout = b""
                result.stderr = "[Parsed_cropdetect_0 @ 0x0] crop=702:484:10:46\n"
            return result

        with patch("furnace.adapters.ffmpeg.subprocess.run", side_effect=run) as mock_run:
            adapter.detect_crop(Path("v.mkv"), duration_s=1000.0)
        for cmd in _crop_calls(mock_run.call_args_list):
            assert _vf(cmd) == "format=yuv420p,cropdetect=40:2:0"


class TestBorderProbeCommand:
    def test_probe_precedes_every_cropdetect_run(self) -> None:
        adapter = _adapter()
        with patch(
            "furnace.adapters.ffmpeg.subprocess.run",
            side_effect=_run_stub(_y4m_420(_bordered(45))),
        ) as mock_run:
            adapter.detect_crop(Path("v.mkv"), duration_s=1000.0)
        flags = [_is_probe(list(call.args[0])) for call in mock_run.call_args_list]
        assert flags == [True] * 9 + [False] * 20

    def test_probe_shares_the_cropdetect_filter_chain(self) -> None:
        adapter = _adapter()
        with patch(
            "furnace.adapters.ffmpeg.subprocess.run",
            side_effect=_run_stub(_y4m_420(_bordered(45))),
        ) as mock_run:
            adapter.detect_crop(
                Path("v.mkv"),
                duration_s=1000.0,
                interlaced=True,
                hdr_transfer="smpte2084",
            )
        crop_vf = _vf(_crop_calls(mock_run.call_args_list)[0])
        probe_vf = _vf(_probe_calls(mock_run.call_args_list)[0])
        assert "cropdetect" not in probe_vf
        assert crop_vf == probe_vf + ",cropdetect=49:2:0"

    def test_probe_command_shape(self) -> None:
        adapter = _adapter()
        with patch(
            "furnace.adapters.ffmpeg.subprocess.run",
            side_effect=_run_stub(_y4m_420(_bordered(45))),
        ) as mock_run:
            adapter.detect_crop(Path("v.mkv"), duration_s=1000.0)
        cmd = _probe_calls(mock_run.call_args_list)[0]
        assert cmd[0] == "ffmpeg"
        assert cmd[cmd.index("-i") + 1] == "v.mkv"
        assert cmd.index("-ss") < cmd.index("-i")
        assert cmd[cmd.index("-frames:v") + 1] == "10"
        assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
        assert cmd[cmd.index("-f") + 1] == "yuv4mpegpipe"
        assert cmd[-1] == "-"

    def test_probe_spreads_seeks_across_the_runtime(self) -> None:
        adapter = _adapter()
        with patch(
            "furnace.adapters.ffmpeg.subprocess.run",
            side_effect=_run_stub(_y4m_420(_bordered(45))),
        ) as mock_run:
            adapter.detect_crop(Path("v.mkv"), duration_s=900.0)
        seeks = [float(cmd[cmd.index("-ss") + 1]) for cmd in _probe_calls(mock_run.call_args_list)]
        assert seeks == [50.0, 150.0, 250.0, 350.0, 450.0, 550.0, 650.0, 750.0, 850.0]


class TestBorderProbeProgress:
    def test_probe_points_count_towards_progress(self) -> None:
        adapter = _adapter()
        samples: list[ProgressSample] = []
        with patch(
            "furnace.adapters.ffmpeg.subprocess.run",
            side_effect=_run_stub(_y4m_420(_bordered(45))),
        ):
            adapter.detect_crop(Path("v.mkv"), duration_s=1000.0, on_progress=samples.append)
        assert len(samples) == 30
        assert samples[0].fraction == pytest.approx(1 / 49)
        assert samples[8].fraction == pytest.approx(9 / 49)
        assert samples[9].fraction == pytest.approx(10 / 49)
        assert samples[-1].fraction == 1.0
