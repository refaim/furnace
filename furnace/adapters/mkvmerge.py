from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from furnace.core.color import CICP_MATRIX, CICP_PRIMARIES, CICP_TRANSFER
from furnace.core.progress import ProgressSample

from ._subprocess import OutputCallback, run_tool

logger = logging.getLogger(__name__)

MKVMERGE_ERROR_RC = 2

_MKVMERGE_PROGRESS_RE = re.compile(r"^Progress:\s*(\d+)%\s*$")


def _parse_mkvmerge_progress_line(line: str) -> ProgressSample | None:
    m = _MKVMERGE_PROGRESS_RE.match(line.strip())
    if not m:
        return None
    return ProgressSample(fraction=int(m.group(1)) / 100.0)


_COLOR_RANGE_MAP: dict[str, str] = {
    "tv": "1",
    "pc": "2",
}


class MkvmergeAdapter:
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
        video_meta: dict[str, Any] | None = None,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int:
        cmd = self._build_mux_cmd(
            video_path,
            audio_files,
            subtitle_files,
            attachments,
            chapters_source,
            output_path,
            video_meta,
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
        if rc >= MKVMERGE_ERROR_RC:
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
        video_meta: dict[str, Any] | None = None,
    ) -> list[str]:
        cmd: list[str] = [
            str(self._mkvmerge),
            "--output",
            str(output_path),
            "--no-track-tags",
            "--no-global-tags",
            "--disable-track-statistics-tags",
            "--title",
            "",
            "--normalize-language-ietf",
            "canonical",
        ]

        video_flags: list[str] = [
            "--track-name",
            "0:",
            "--language",
            "0:und",
        ]

        if video_meta:
            cr = video_meta.get("color_range")
            if cr and cr in _COLOR_RANGE_MAP:
                video_flags += ["--color-range", f"0:{_COLOR_RANGE_MAP[cr]}"]

            cp = video_meta.get("color_primaries")
            if cp and cp in CICP_PRIMARIES:
                video_flags += ["--color-primaries", f"0:{CICP_PRIMARIES[cp]}"]

            ct = video_meta.get("color_transfer")
            if ct and ct in CICP_TRANSFER:
                video_flags += ["--color-transfer-characteristics", f"0:{CICP_TRANSFER[ct]}"]

            cm = video_meta.get("color_matrix")
            if cm and cm in CICP_MATRIX:
                video_flags += ["--color-matrix-coefficients", f"0:{CICP_MATRIX[cm]}"]

            max_cll = video_meta.get("hdr_max_cll")
            max_fall = video_meta.get("hdr_max_fall")
            if max_cll is not None:
                video_flags += ["--max-content-light", f"0:{max_cll}"]
            if max_fall is not None:
                video_flags += ["--max-frame-light", f"0:{max_fall}"]

            fps_num = video_meta.get("fps_num")
            fps_den = video_meta.get("fps_den")
            if fps_num and fps_den:
                video_flags += ["--default-duration", f"0:{fps_num}/{fps_den}p"]

        video_flags += ["--no-chapters"]
        video_flags.append(str(video_path))
        cmd += video_flags

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

        for att_path, att_filename, att_mime in attachments:
            cmd += [
                "--attachment-name",
                att_filename,
                "--attachment-mime-type",
                att_mime,
                "--attach-file",
                str(att_path),
            ]

        if chapters_source is not None:
            cmd += ["--chapters", str(chapters_source)]

        audio_count = len(audio_files)
        track_order_parts: list[str] = [
            "0:0",
            *[f"{1 + i}:0" for i in range(audio_count)],
            *[f"{1 + audio_count + i}:0" for i in range(len(subtitle_files))],
        ]
        cmd += ["--track-order", ",".join(track_order_parts)]

        return cmd
