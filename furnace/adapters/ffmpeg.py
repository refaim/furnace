from __future__ import annotations

import json
import logging
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from ..core.models import CropRect
from ._subprocess import OutputCallback, run_tool

logger = logging.getLogger(__name__)


class FFmpegAdapter:
    """Implements Prober + AudioExtractor."""

    def __init__(
        self, ffmpeg_path: Path, ffprobe_path: Path,
        on_output: OutputCallback = None, log_dir: Path | None = None,
    ) -> None:
        self._ffmpeg = ffmpeg_path
        self._ffprobe = ffprobe_path
        self._on_output = on_output
        self._log_dir = log_dir

    def set_log_dir(self, log_dir: Path | None) -> None:
        self._log_dir = log_dir

    def _get_ffmpeg_version(self) -> str:
        """Get ffmpeg version string (e.g. '7.1'). Cached after first call."""
        cached: str | None = getattr(self, "_ffmpeg_version_cached", None)
        if cached is not None:
            return cached
        try:
            result = subprocess.run(
                [str(self._ffmpeg), "-version"],
                capture_output=True, text=True, timeout=5,
            )
            m = re.match(r"ffmpeg version (\S+)", result.stdout)
            self._ffmpeg_version_cached: str = m.group(1) if m else ""
        except Exception:
            self._ffmpeg_version_cached = ""
        return self._ffmpeg_version_cached

    # ------------------------------------------------------------------
    # Prober
    # ------------------------------------------------------------------

    def probe(self, path: Path) -> dict[str, Any]:
        """ffprobe -v quiet -print_format json -show_format -show_streams -show_chapters path"""
        cmd = [
            str(self._ffprobe),
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            "-show_chapters",
            str(path),
        ]
        logger.debug("probe cmd: %s", cmd)
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            logger.error("ffprobe failed (rc=%d): %s", result.returncode, result.stderr)
            raise RuntimeError(f"ffprobe failed with return code {result.returncode}: {result.stderr}")
        data: dict[str, Any] = json.loads(result.stdout)
        return data

    _CROP_SAMPLE_POINTS: tuple[float, ...] = (
        0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
    )

    def detect_crop(self, path: Path, duration_s: float) -> CropRect | None:
        """Run cropdetect at 10 points across the timeline, 2 seconds each.

        Returns the mode crop value only if it appears in >50% of samples.
        This filters out false positives from dark scenes.
        Returns None if no crop detected or crop == full frame.
        """
        crop_values: list[str] = []

        for pct in self._CROP_SAMPLE_POINTS:
            seek = duration_s * pct
            cmd = [
                str(self._ffmpeg),
                "-hide_banner",
                "-ss", f"{seek:.2f}",
                "-i", str(path),
                "-t", "2",
                "-vf", "cropdetect=24:16:0",
                "-f", "null",
                "-",
            ]
            logger.debug("detect_crop cmd: %s", cmd)
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            # cropdetect writes to stderr — take last value per sample point
            last_crop: str | None = None
            for line in result.stderr.splitlines():
                m = re.search(r"crop=(\d+:\d+:\d+:\d+)", line)
                if m:
                    last_crop = m.group(1)
            if last_crop is not None:
                crop_values.append(last_crop)

        if not crop_values:
            return None

        # Mode: most frequent crop value, but only accept if >50% of samples agree
        counter = Counter(crop_values)
        mode_crop, mode_count = counter.most_common(1)[0]
        if mode_count <= len(crop_values) // 2:
            logger.info("Crop not reliable: mode %s appeared %d/%d times", mode_crop, mode_count, len(crop_values))
            return None

        parts = mode_crop.split(":")
        if len(parts) != 4:
            return None
        w, h, x, y = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        return CropRect(w=w, h=h, x=x, y=y)

    def get_encoder_tag(self, path: Path) -> str | None:
        """Read format.tags.ENCODER from probe output."""
        try:
            data = self.probe(path)
        except RuntimeError:
            return None
        tags = data.get("format", {}).get("tags", {})
        # Tags can be ENCODER or encoder (case varies)
        for key in ("ENCODER", "encoder"):
            if key in tags:
                return str(tags[key])
        return None

    def run_idet(self, path: Path, duration_s: float) -> float:
        """Run idet filter at multiple points across the timeline.

        Samples 1000 frames at 10%, 30%, 50%, 70%, 90% of duration.
        Returns the ratio of interlaced frames (0.0 to 1.0).
        """
        total_interlaced = 0
        total_prog = 0

        for pct in (0.10, 0.30, 0.50, 0.70, 0.90):
            seek = duration_s * pct
            cmd = [
                str(self._ffmpeg),
                "-hide_banner",
                "-ss", f"{seek:.2f}",
                "-i", str(path),
                "-vf", "idet",
                "-frames:v", "1000",
                "-f", "null",
                "-",
            ]
            logger.debug("run_idet cmd: %s", cmd)
            result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

            for line in result.stderr.splitlines():
                m = re.search(
                    r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*Progressive:\s*(\d+)",
                    line,
                )
                if m:
                    total_interlaced += int(m.group(1)) + int(m.group(2))
                    total_prog += int(m.group(3))

        total = total_interlaced + total_prog
        if total == 0:
            return 0.0

        return total_interlaced / total

    # ------------------------------------------------------------------
    # AudioExtractor
    # ------------------------------------------------------------------

    def extract_track(
        self,
        input_path: Path,
        stream_index: int,
        output_path: Path,
        codec: str,
    ) -> int:
        """ffmpeg -i input -map 0:{index} -c copy output"""
        cmd = [
            str(self._ffmpeg),
            "-hide_banner", "-loglevel", "warning",
            "-i", str(input_path),
            "-map", f"0:{stream_index}",
            "-c", "copy",
            "-y", str(output_path),
        ]
        log_path = self._log_dir / f"ffmpeg_extract_s{stream_index}.log" if self._log_dir else None
        rc, _out = run_tool(cmd, on_output=self._on_output, log_path=log_path)
        return rc

    def ffmpeg_to_wav(
        self,
        input_path: Path,
        stream_index: int,
        output_wav: Path,
    ) -> int:
        """ffmpeg -i input -map 0:{index} -f wav -rf64 auto output.wav"""
        cmd = [
            str(self._ffmpeg),
            "-hide_banner", "-loglevel", "warning",
            "-i", str(input_path),
            "-map", f"0:{stream_index}",
            "-f", "wav",
            "-rf64", "auto",
            "-y", str(output_wav),
        ]
        log_path = self._log_dir / f"ffmpeg_to_wav_s{stream_index}.log" if self._log_dir else None
        rc, _out = run_tool(cmd, on_output=self._on_output, log_path=log_path)
        return rc
