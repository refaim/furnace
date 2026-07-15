from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

from furnace.adapters.ffmpeg import FFmpegAdapter


def _adapter(log_dir: Path | None = None) -> FFmpegAdapter:
    return FFmpegAdapter(Path("ffmpeg.exe"), Path("ffprobe.exe"), log_dir=log_dir)


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
            rc = adapter.extract_window(
                Path("movie.mkv"), Path("window.mkv"), start_s=612.5, frames=480
            )
        assert rc == 0
        # `-ss` MUST precede `-i` (fast keyframe seek); `-frames:v` copies N packets.
        assert captured.index("-ss") < captured.index("-i")
        assert captured[captured.index("-ss") + 1] == "612.500"
        assert captured[captured.index("-i") + 1] == "movie.mkv"
        # Pin the first video stream so a multi-video-stream source (cover art)
        # can't make the window a different stream than the final encode.
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
            rc = adapter.extract_window(
                Path("movie.mkv"), Path("window.mkv"), start_s=0.0, frames=120
            )
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
            adapter.extract_window(
                Path("movie.mkv"), Path("window.mkv"), start_s=10.0, frames=48
            )
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
            adapter.extract_window(
                Path("movie.mkv"), Path("window.mkv"), start_s=10.0, frames=48
            )
        assert captured_kwargs["log_path"] is None


class TestWindowBitrates:
    def _run(self, returncode: int, stdout: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")

    def test_bins_packets_into_windows(self) -> None:
        # ffprobe CSV is "pts_time,size" per video packet; window_s=10 -> bins at 0/10/20.
        # packets: bin0 {0.0,5.0}, bin1 {12.0}, bin2 {25.0,28.0}.
        stdout = "0.0,1024\n5.0,1024\n12.0,2048\n25.0,512\n28.0,512"
        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=self._run(0, stdout)):
            result = adapter.window_bitrates(Path("m.mkv"), 10.0)
        # bin0 = 2048B = 2.0KB, bin1 = 2048B = 2.0KB, bin2 = 1024B = 1.0KB.
        assert result == [(0.0, 2.0), (10.0, 2.0), (20.0, 1.0)]

    def test_skips_unparseable_and_negative_pts(self) -> None:
        # Lines, in order: unparseable pts, unparseable size, negative pts, a line
        # with no comma, then the one real packet (-> bin 0).
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
