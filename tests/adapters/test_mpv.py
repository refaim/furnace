"""Tests for MpvAdapter preview methods."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from furnace.adapters.mpv import MpvAdapter


class TestPreviewAudio:
    def test_audio_file_and_aid_in_cmd(self) -> None:
        adapter = MpvAdapter(Path("mpv.exe"))
        with patch("furnace.adapters.mpv.subprocess.run") as mock_run:
            adapter.preview_audio(Path("video.mkv"), Path("audio.flac"), 3)
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "mpv.exe"
        assert cmd[1] == "video.mkv"
        assert f"--audio-file={Path('audio.flac')}" in cmd
        assert "--aid=3" in cmd

    def test_check_false(self) -> None:
        adapter = MpvAdapter(Path("mpv.exe"))
        with patch("furnace.adapters.mpv.subprocess.run") as mock_run:
            adapter.preview_audio(Path("v.mkv"), Path("a.flac"), 1)
        assert mock_run.call_args[1]["check"] is False


class TestPreviewSubtitle:
    def test_sub_file_and_sid_in_cmd(self) -> None:
        adapter = MpvAdapter(Path("mpv.exe"))
        with patch("furnace.adapters.mpv.subprocess.run") as mock_run:
            adapter.preview_subtitle(Path("video.mkv"), Path("sub.sup"), 2)
        cmd = mock_run.call_args[0][0]
        assert f"--sub-file={Path('sub.sup')}" in cmd
        assert "--sid=2" in cmd

    def test_check_false(self) -> None:
        adapter = MpvAdapter(Path("mpv.exe"))
        with patch("furnace.adapters.mpv.subprocess.run") as mock_run:
            adapter.preview_subtitle(Path("v.mkv"), Path("s.srt"), 1)
        assert mock_run.call_args[1]["check"] is False


class TestPreviewFile:
    def test_no_aspect_override(self) -> None:
        adapter = MpvAdapter(Path("mpv.exe"))
        with patch("furnace.adapters.mpv.subprocess.run") as mock_run:
            adapter.preview_file(Path("video.mkv"))
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "mpv.exe"
        assert cmd[1] == "video.mkv"
        assert not any("--video-aspect-override" in c for c in cmd)

    def test_with_aspect_override(self) -> None:
        adapter = MpvAdapter(Path("mpv.exe"))
        with patch("furnace.adapters.mpv.subprocess.run") as mock_run:
            adapter.preview_file(Path("video.mkv"), aspect_override="16:9")
        cmd = mock_run.call_args[0][0]
        assert "--video-aspect-override=16:9" in cmd

    def test_check_false(self) -> None:
        adapter = MpvAdapter(Path("mpv.exe"))
        with patch("furnace.adapters.mpv.subprocess.run") as mock_run:
            adapter.preview_file(Path("v.mkv"))
        assert mock_run.call_args[1]["check"] is False
