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


class TestPreviewSingleTrackSet:
    def test_no_audio_file_when_track_lives_in_the_video(self) -> None:
        adapter = MpvAdapter(Path("mpv.exe"))
        with patch("furnace.adapters.mpv.subprocess.run") as mock_run:
            adapter.preview_audio(Path("video.mkv"), None, 2)
        cmd = mock_run.call_args[0][0]
        assert not any(c.startswith("--audio-file=") for c in cmd)
        assert "--aid=2" in cmd
        assert cmd.count("video.mkv") == 1

    def test_no_audio_file_when_paths_are_the_same(self) -> None:
        adapter = MpvAdapter(Path("mpv.exe"))
        with patch("furnace.adapters.mpv.subprocess.run") as mock_run:
            adapter.preview_audio(Path("video.mkv"), Path("video.mkv"), 1)
        cmd = mock_run.call_args[0][0]
        assert not any(c.startswith("--audio-file=") for c in cmd)

    def test_no_sub_file_when_track_lives_in_the_video(self) -> None:
        adapter = MpvAdapter(Path("mpv.exe"))
        with patch("furnace.adapters.mpv.subprocess.run") as mock_run:
            adapter.preview_subtitle(Path("video.mkv"), None, 3)
        cmd = mock_run.call_args[0][0]
        assert not any(c.startswith("--sub-file=") for c in cmd)
        assert "--sid=3" in cmd
        assert cmd.count("video.mkv") == 1

    def test_no_sub_file_when_paths_are_the_same(self) -> None:
        adapter = MpvAdapter(Path("mpv.exe"))
        with patch("furnace.adapters.mpv.subprocess.run") as mock_run:
            adapter.preview_subtitle(Path("video.mkv"), Path("video.mkv"), 1)
        cmd = mock_run.call_args[0][0]
        assert not any(c.startswith("--sub-file=") for c in cmd)

    def test_audio_preview_does_not_autoload_sidecars(self) -> None:
        adapter = MpvAdapter(Path("mpv.exe"))
        with patch("furnace.adapters.mpv.subprocess.run") as mock_run:
            adapter.preview_audio(Path("video.mkv"), None, 1)
        cmd = mock_run.call_args[0][0]
        assert "--audio-file-auto=no" in cmd
        assert "--sub-auto=no" in cmd

    def test_subtitle_preview_does_not_autoload_sidecars(self) -> None:
        adapter = MpvAdapter(Path("mpv.exe"))
        with patch("furnace.adapters.mpv.subprocess.run") as mock_run:
            adapter.preview_subtitle(Path("video.mkv"), None, 1)
        cmd = mock_run.call_args[0][0]
        assert "--audio-file-auto=no" in cmd
        assert "--sub-auto=no" in cmd


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
