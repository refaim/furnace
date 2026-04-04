from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class MpvAdapter:
    """Implements Previewer."""

    def __init__(self, mpv_path: Path) -> None:
        self._mpv = mpv_path

    def preview_audio(self, video_path: Path, audio_path: Path, stream_index: int) -> None:
        """mpv video --audio-file=audio --aid=index. Blocks until mpv closes."""
        cmd = [
            str(self._mpv),
            str(video_path),
            f"--audio-file={audio_path}",
            f"--aid={stream_index}",
        ]
        logger.info("mpv preview_audio cmd: %s", " ".join(cmd))
        subprocess.run(cmd)

    def preview_subtitle(self, video_path: Path, sub_path: Path, stream_index: int) -> None:
        """mpv video --sub-file=sub --sid=index. Blocks until mpv closes."""
        cmd = [
            str(self._mpv),
            str(video_path),
            f"--sub-file={sub_path}",
            f"--sid={stream_index}",
        ]
        logger.info("mpv preview_subtitle cmd: %s", " ".join(cmd))
        subprocess.run(cmd)

    def preview_file(self, path: Path, *, aspect_override: str | None = None) -> None:
        """mpv file. Blocks until mpv closes."""
        cmd = [str(self._mpv), str(path)]
        if aspect_override:
            cmd.append(f"--video-aspect-override={aspect_override}")
        logger.info("mpv preview_file cmd: %s", " ".join(cmd))
        subprocess.run(cmd)
