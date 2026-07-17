from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from furnace.adapters.ffmpeg import FFmpegAdapter
from furnace.core.progress import ProgressSample


def _adapter() -> FFmpegAdapter:
    return FFmpegAdapter(Path("ffmpeg.exe"), Path("ffprobe.exe"))


class TestCopyVideo:
    def test_copy_video_cmd(self) -> None:
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
            rc = adapter.copy_video(Path("video.mkv"), Path("video_out.mkv"))
        assert rc == 0
        assert captured[captured.index("-loglevel") + 1] == "fatal"
        assert captured[captured.index("-i") + 1] == "video.mkv"
        assert captured[captured.index("-map") + 1] == "0:v:0"
        assert captured[captured.index("-c:v") + 1] == "copy"
        assert captured[captured.index("-progress") + 1] == "pipe:1"
        assert captured[-1] == "video_out.mkv"
        assert captured[captured.index("-y") + 1] == "video_out.mkv"

    def test_copy_video_return_code_propagates(self) -> None:
        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            return 3, ""

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            rc = adapter.copy_video(Path("v.mkv"), Path("out.mkv"))
        assert rc == 3

    def test_copy_video_progress(self) -> None:
        samples: list[ProgressSample] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            on_progress_line("out_time_us=60000000")
            on_progress_line("speed=2.5x")
            on_progress_line("progress=continue")
            return 0, ""

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            adapter.copy_video(Path("v.mkv"), Path("out.mkv"), on_progress=samples.append)
        assert len(samples) == 1
        assert abs(samples[0].processed_s - 60.0) < 0.01  # type: ignore[operator]

    def test_copy_video_non_progress_line(self) -> None:
        results: list[bool] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            results.append(on_progress_line("no equals sign here"))
            return 0, ""

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            adapter.copy_video(Path("v.mkv"), Path("out.mkv"))
        assert results == [False]

    def test_copy_video_without_on_progress_skips_callback(self) -> None:
        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            assert on_progress_line("out_time_us=1000000") is True
            assert on_progress_line("speed=1.5x") is True
            assert on_progress_line("progress=continue") is True
            return 0, ""

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            rc = adapter.copy_video(Path("v.mkv"), Path("out.mkv"))
        assert rc == 0

    def test_copy_video_log_path(self, tmp_path: Path) -> None:
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

        adapter = FFmpegAdapter(Path("ffmpeg.exe"), Path("ffprobe.exe"), log_dir=tmp_path)
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            adapter.copy_video(Path("v.mkv"), Path("out.mkv"))
        assert captured_kwargs["log_path"] == tmp_path / "ffmpeg_copy_video.log"


class TestVideoCopierProtocol:
    def test_ffmpeg_adapter_is_video_copier(self) -> None:
        from furnace.core.ports import VideoCopier

        adapter = _adapter()
        assert isinstance(adapter, VideoCopier)
