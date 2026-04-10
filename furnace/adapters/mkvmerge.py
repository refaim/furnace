from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ._subprocess import OutputCallback, run_tool
from furnace.core.progress import ProgressSample

logger = logging.getLogger(__name__)


_MKVMERGE_PROGRESS_RE = re.compile(r"^Progress:\s*(\d+)%\s*$")


def _parse_mkvmerge_progress_line(line: str) -> ProgressSample | None:
    """Parse an mkvmerge progress line into a sample.

    Format (confirmed by existing UI regex at run_tui.py:48): ``Progress: NN%``.
    """
    m = _MKVMERGE_PROGRESS_RE.match(line.strip())
    if not m:
        return None
    try:
        return ProgressSample(fraction=int(m.group(1)) / 100.0)
    except ValueError:
        return None


_COLOR_RANGE_MAP: dict[str, str] = {
    "tv": "1",       # broadcast / limited (16-235)
    "pc": "2",       # full (0-255)
}

_COLOR_PRIMARIES_MAP: dict[str, str] = {
    "bt709": "1",
    "bt470bg": "5",
    "smpte170m": "6",
    "smpte240m": "7",
    "bt2020": "9",
}

_COLOR_TRANSFER_MAP: dict[str, str] = {
    "bt709": "1",
    "smpte170m": "6",
    "smpte240m": "7",
    "linear": "8",
    "smpte2084": "16",    # HDR10 / PQ
    "arib-std-b67": "18", # HLG
}


class MkvmergeAdapter:
    """Implements Muxer."""

    def __init__(self, mkvmerge_path: Path, on_output: OutputCallback = None, log_dir: Path | None = None) -> None:
        self._mkvmerge = mkvmerge_path
        self._on_output = on_output
        self._log_dir = log_dir

    def set_log_dir(self, log_dir: Path | None) -> None:
        self._log_dir = log_dir

    def mux(
        self,
        video_path: Path,
        audio_files: list[tuple[Path, dict[str, Any]]],
        subtitle_files: list[tuple[Path, dict[str, Any]]],
        attachments: list[tuple[Path, str, str]],
        chapters_source: Path | None,
        output_path: Path,
        furnace_version: str,
        video_meta: dict[str, Any] | None = None,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int:
        """Build and run full mkvmerge command.

        audio_files: list of (path, {language, default, delay_ms})
        subtitle_files: list of (path, {language, default, forced, encoding})
        attachments: list of (path, filename, mime_type)
        video_meta: optional dict with color/HDR metadata for container-level flags

        Note: ENCODER tag is NOT set here. It is set separately via
        MkvpropeditAdapter after muxing.
        """
        cmd = self._build_mux_cmd(
            video_path, audio_files, subtitle_files, attachments,
            chapters_source, output_path, furnace_version, video_meta,
        )
        log_path = self._log_dir / "mkvmerge.log" if self._log_dir else None

        def _on_progress_line(line: str) -> bool:
            sample = _parse_mkvmerge_progress_line(line)
            if sample is None:
                return False
            if on_progress is not None:
                on_progress(sample)
            return True

        rc, stderr = run_tool(
            cmd,
            on_output=self._on_output,
            on_progress_line=_on_progress_line,
            log_path=log_path,
        )
        if rc >= 2:
            logger.error("mkvmerge mux failed (rc=%d): %s", rc, stderr[-500:])
        elif rc == 1:
            logger.warning("mkvmerge mux completed with warnings (rc=1)")
        return rc

    def _build_mux_cmd(
        self,
        video_path: Path,
        audio_files: list[tuple[Path, dict[str, Any]]],
        subtitle_files: list[tuple[Path, dict[str, Any]]],
        attachments: list[tuple[Path, str, str]],
        chapters_source: Path | None,
        output_path: Path,
        furnace_version: str,
        video_meta: dict[str, Any] | None = None,
    ) -> list[str]:
        cmd: list[str] = [
            str(self._mkvmerge),
            "--output", str(output_path),
            # Strip all existing tags and statistics
            "--no-track-tags",
            "--no-global-tags",
            "--disable-track-statistics-tags",
            # Clean title (remove source junk like "Ripped by...")
            "--title", "",
            # Normalize language codes (fre→fra, chi→zho)
            "--normalize-language-ietf", "canonical",
        ]

        # Video track: track 0 of video_path
        video_flags: list[str] = [
            "--track-name", "0:",          # remove track name
            "--language", "0:und",          # undetermined language for video
        ]

        # Color and HDR metadata at container level (duplicates VUI from stream)
        if video_meta:
            # --color-range
            cr = video_meta.get("color_range")
            if cr and cr in _COLOR_RANGE_MAP:
                video_flags += ["--color-range", f"0:{_COLOR_RANGE_MAP[cr]}"]

            # --color-primaries
            cp = video_meta.get("color_primaries")
            if cp and cp in _COLOR_PRIMARIES_MAP:
                video_flags += ["--color-primaries", f"0:{_COLOR_PRIMARIES_MAP[cp]}"]

            # --color-transfer-characteristics
            ct = video_meta.get("color_transfer")
            if ct and ct in _COLOR_TRANSFER_MAP:
                video_flags += ["--color-transfer-characteristics", f"0:{_COLOR_TRANSFER_MAP[ct]}"]

            # --max-content-light / --max-frame-light (HDR10 only)
            max_cll = video_meta.get("hdr_max_cll")
            max_fall = video_meta.get("hdr_max_fall")
            if max_cll is not None:
                video_flags += ["--max-content-light", f"0:{max_cll}"]
            if max_fall is not None:
                video_flags += ["--max-frame-light", f"0:{max_fall}"]

        video_flags += ["--no-chapters"]  # chapters come only from chapters_source
        video_flags.append(str(video_path))
        cmd += video_flags

        # Audio tracks
        for audio_path, audio_meta in audio_files:
            lang = audio_meta.get("language", "und")
            is_default = audio_meta.get("default", False)
            delay_ms = audio_meta.get("delay_ms", 0)

            cmd += ["--track-name", "0:"]
            cmd += ["--language", f"0:{lang}"]
            if is_default:
                cmd += ["--default-track-flag", "0:yes"]
            else:
                cmd += ["--default-track-flag", "0:no"]
            if delay_ms != 0:
                cmd += ["--sync", f"0:{delay_ms}"]
            cmd += ["--no-chapters"]
            cmd.append(str(audio_path))

        # Subtitle tracks
        for sub_path, sub_meta in subtitle_files:
            lang = sub_meta.get("language", "und")
            is_default = sub_meta.get("default", False)
            is_forced = sub_meta.get("forced", False)
            encoding = sub_meta.get("encoding", None)

            cmd += ["--track-name", "0:"]
            cmd += ["--language", f"0:{lang}"]
            if is_default:
                cmd += ["--default-track-flag", "0:yes"]
            else:
                cmd += ["--default-track-flag", "0:no"]
            if is_forced:
                cmd += ["--forced-display-flag", "0:yes"]
            if encoding:
                cmd += ["--sub-charset", f"0:{encoding}"]
            cmd += ["--no-chapters"]
            cmd.append(str(sub_path))

        # Attachments
        for att_path, att_filename, att_mime in attachments:
            cmd += [
                "--attachment-name", att_filename,
                "--attachment-mime-type", att_mime,
                "--attach-file", str(att_path),
            ]

        # Chapters (always OGM .txt file, prepared by executor)
        if chapters_source is not None:
            cmd += ["--chapters", str(chapters_source)]

        # Track order: video first (0:0), then audio, then subtitles
        audio_count = len(audio_files)
        track_order_parts: list[str] = [
            "0:0",
            *[f"{1 + i}:0" for i in range(audio_count)],
            *[f"{1 + audio_count + i}:0" for i in range(len(subtitle_files))],
        ]
        cmd += ["--track-order", ",".join(track_order_parts)]

        return cmd
