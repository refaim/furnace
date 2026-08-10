from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class MpvAdapter:
    def __init__(self, mpv_path: Path) -> None:
        self._mpv = mpv_path

    def _base_cmd(self, video_path: Path) -> list[str]:
        return [
            str(self._mpv),
            str(video_path),
            "--audio-file-auto=no",
            "--sub-auto=no",
        ]

    def preview_audio(self, video_path: Path, audio_path: Path | None, track_id: int) -> None:
        cmd = self._base_cmd(video_path)
        if audio_path is not None and audio_path != video_path:
            cmd.append(f"--audio-file={audio_path}")
        cmd.append(f"--aid={track_id}")
        logger.info("mpv preview_audio cmd: %s", " ".join(cmd))
        subprocess.run(cmd, check=False)

    def preview_subtitle(self, video_path: Path, sub_path: Path | None, track_id: int) -> None:
        cmd = self._base_cmd(video_path)
        if sub_path is not None and sub_path != video_path:
            cmd.append(f"--sub-file={sub_path}")
        cmd.append(f"--sid={track_id}")
        logger.info("mpv preview_subtitle cmd: %s", " ".join(cmd))
        subprocess.run(cmd, check=False)

    def preview_file(self, path: Path, *, aspect_override: str | None = None) -> None:
        cmd = [str(self._mpv), str(path)]
        if aspect_override:
            cmd.append(f"--video-aspect-override={aspect_override}")
        logger.info("mpv preview_file cmd: %s", " ".join(cmd))
        subprocess.run(cmd, check=False)
