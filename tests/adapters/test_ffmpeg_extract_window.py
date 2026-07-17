from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

from furnace.adapters._geometry import build_vf
from furnace.adapters.ffmpeg import FFmpegAdapter
from furnace.core.models import CropRect, VideoParams


def _adapter(log_dir: Path | None = None) -> FFmpegAdapter:
    return FFmpegAdapter(Path("ffmpeg.exe"), Path("ffprobe.exe"), log_dir=log_dir)


def _make_vp(*, crop: CropRect | None = None, deinterlace: bool = False) -> VideoParams:
    return VideoParams(
        cq=23,
        crop=crop,
        deinterlace=deinterlace,
        color_matrix="bt470bg",
        color_range="tv",
        color_transfer="bt470bg",
        color_primaries="bt470bg",
        hdr=None,
        gop=125,
        fps_num=25,
        fps_den=1,
        source_width=720,
        source_height=576,
        source_codec="mpeg2video",
        source_bitrate=6_000_000,
        sar_num=16,
        sar_den=15,
        grain=True,
    )


class TestExtractWindow:
    def test_extract_window_cmd(self) -> None:
        captured: list[str] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            captured.extend(str(c) for c in cmd)
            return 0, ""

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            rc = adapter.extract_window(Path("movie.mkv"), Path("window.mkv"), start_s=612.5, frames=480)
        assert rc == 0
        assert captured.index("-ss") < captured.index("-i")
        assert captured[captured.index("-ss") + 1] == "612.500"
        assert captured[captured.index("-i") + 1] == "movie.mkv"
        assert captured[captured.index("-map") + 1] == "0:v:0"
        assert captured[captured.index("-frames:v") + 1] == "480"
        assert captured[captured.index("-c:v") + 1] == "copy"
        assert "-an" in captured
        assert "-sn" in captured
        assert captured[captured.index("-y") + 1] == "window.mkv"
        assert captured[-1] == "window.mkv"

    def test_extract_window_return_code_propagates(self) -> None:
        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            return 1, ""

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            rc = adapter.extract_window(Path("movie.mkv"), Path("window.mkv"), start_s=0.0, frames=120)
        assert rc == 1

    def test_extract_window_log_path(self, tmp_path: Path) -> None:
        captured_kwargs: dict[str, Any] = {}

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            captured_kwargs["log_path"] = log_path
            return 0, ""

        adapter = _adapter(log_dir=tmp_path)
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            adapter.extract_window(Path("movie.mkv"), Path("window.mkv"), start_s=10.0, frames=48)
        assert captured_kwargs["log_path"] == tmp_path / "ffmpeg_extract_window.log"

    def test_extract_window_no_log_dir(self) -> None:
        captured_kwargs: dict[str, Any] = {}

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            captured_kwargs["log_path"] = log_path
            return 0, ""

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            adapter.extract_window(Path("movie.mkv"), Path("window.mkv"), start_s=10.0, frames=48)
        assert captured_kwargs["log_path"] is None


class TestBuildReference:
    @staticmethod
    def _capture(rc: int = 0) -> tuple[list[str], Any]:
        captured: list[str] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            captured.extend(str(c) for c in cmd)
            return rc, ""

        return captured, fake_run_tool

    def test_reference_uses_encode_filtergraph(self) -> None:
        captured, fake = self._capture()
        vp = _make_vp(crop=CropRect(w=716, h=572, x=2, y=2))
        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake):
            rc = adapter.build_reference(Path("window.mkv"), Path("ref.mkv"), vp)
        assert rc == 0
        assert captured[captured.index("-vf") + 1] == build_vf(vp)
        assert captured[captured.index("-c:v") + 1] == "ffv1"
        assert captured[captured.index("-i") + 1] == "window.mkv"
        assert captured[captured.index("-map") + 1] == "0:v:0"
        assert captured[-1] == "ref.mkv"
        assert captured[captured.index("-y") + 1] == "ref.mkv"

    def test_reference_pins_color_and_rate(self) -> None:
        captured, fake = self._capture()
        vp = _make_vp()
        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake):
            adapter.build_reference(Path("window.mkv"), Path("ref.mkv"), vp)
        assert captured[captured.index("-color_range") + 1] == "tv"
        assert captured[captured.index("-color_primaries") + 1] == "bt470bg"
        assert captured[captured.index("-color_trc") + 1] == "bt470bg"
        assert captured[captured.index("-colorspace") + 1] == "bt470bg"
        assert captured[captured.index("-r") + 1] == "25/1"
        assert "-an" in captured
        assert "-sn" in captured

    def test_reference_return_code_propagates(self) -> None:
        _captured, fake = self._capture(rc=1)
        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake):
            rc = adapter.build_reference(Path("window.mkv"), Path("ref.mkv"), _make_vp())
        assert rc == 1

    def test_reference_log_path(self, tmp_path: Path) -> None:
        captured_kwargs: dict[str, Any] = {}

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            captured_kwargs["log_path"] = log_path
            return 0, ""

        adapter = _adapter(log_dir=tmp_path)
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            adapter.build_reference(Path("window.mkv"), Path("ref.mkv"), _make_vp())
        assert captured_kwargs["log_path"] == tmp_path / "ffmpeg_build_reference.log"

    def test_reference_no_log_dir(self) -> None:
        captured_kwargs: dict[str, Any] = {}

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            captured_kwargs["log_path"] = log_path
            return 0, ""

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            adapter.build_reference(Path("window.mkv"), Path("ref.mkv"), _make_vp())
        assert captured_kwargs["log_path"] is None


class TestWindowBitrates:
    def _run(self, returncode: int, stdout: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")

    def test_bins_packets_into_windows(self) -> None:
        stdout = "0.0,1024\n5.0,1024\n12.0,2048\n25.0,512\n28.0,512"
        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=self._run(0, stdout)):
            result = adapter.window_bitrates(Path("m.mkv"), 10.0)
        assert result == [(0.0, 2.0), (10.0, 2.0), (20.0, 1.0)]

    def test_skips_unparseable_and_negative_pts(self) -> None:
        stdout = "N/A,1024\n1.0,notanint\n-1.0,1024\nnocomma\n5.0,2048"
        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=self._run(0, stdout)):
            result = adapter.window_bitrates(Path("m.mkv"), 10.0)
        assert result == [(0.0, 2.0)]

    def test_nonzero_returncode_returns_empty(self) -> None:
        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=self._run(1, "")):
            result = adapter.window_bitrates(Path("m.mkv"), 10.0)
        assert result == []


class TestWindowExtractorProtocol:
    def test_ffmpeg_adapter_is_window_extractor(self) -> None:
        from furnace.core.ports import WindowExtractor

        adapter = _adapter()
        assert isinstance(adapter, WindowExtractor)
