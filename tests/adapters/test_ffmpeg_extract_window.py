from __future__ import annotations

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


class TestWindowExtractorProtocol:
    def test_ffmpeg_adapter_is_window_extractor(self) -> None:
        from furnace.core.ports import WindowExtractor

        adapter = _adapter()
        assert isinstance(adapter, WindowExtractor)
