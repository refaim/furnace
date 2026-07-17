from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from rich.box import ASCII
from rich.console import Console
from rich.table import Table

from furnace.core.outdated import row_fix, row_severity
from furnace.core.scan import AudioTrackSummary, ScanRow, SubtitleTrackSummary, VideoSummary

_NONE = "—"

_NO_TRUNCATE_WIDTH = 10_000


def _rel_path(path: Path, root: Path) -> str:
    if path == root:
        return path.name
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _audio_line(track: AudioTrackSummary) -> str:
    lang = track.language or "und"
    parts = [lang, track.codec]
    if track.channels is not None:
        parts.append(f"{track.channels}ch")
    return " ".join(parts)


def _subtitle_line(track: SubtitleTrackSummary) -> str:
    lang = track.language or "und"
    return f"{lang} {track.codec}"


def _video_cell(video: VideoSummary) -> str:
    if video.codec is None:
        return _NONE
    if video.bit_depth is None:
        return video.codec
    return f"{video.codec} {video.bit_depth}bit"


def _status(row: ScanRow) -> str:
    if row.unreadable:
        return "unreadable"
    if row.furnace_version is not None:
        major, minor, patch = row.furnace_version
        return f"Furnace v{major}.{minor}.{patch}"
    return "not encoded"


def _cells(row: ScanRow, root: Path) -> tuple[str, str, str, str, str, str]:
    file_cell = _rel_path(row.path, root)
    status = _status(row)
    if row.unreadable:
        return file_cell, status, _NONE, _NONE, _NONE, _NONE
    video = _video_cell(row.video)
    hdr = row.video.hdr or _NONE
    audio = "\n".join(_audio_line(t) for t in row.audio) or _NONE
    subs = "\n".join(_subtitle_line(t) for t in row.subtitles) or _NONE
    return file_cell, status, video, hdr, audio, subs


def _outdated_cells(row: ScanRow, root: Path) -> tuple[str, str, str, str, str, str]:
    file_cell = _rel_path(row.path, root)
    severity = row_severity(row.defects).label
    fix = row_fix(row.defects).label
    reason = "\n".join(defect.reason for defect in row.defects)
    if row.unreadable:
        return file_cell, severity, fix, reason, _NONE, _NONE
    video = _video_cell(row.video)
    hdr = row.video.hdr or _NONE
    return file_cell, severity, fix, reason, video, hdr


def _build_table(rows: Sequence[ScanRow], root: Path, *, outdated: bool) -> Table:
    table = Table(box=ASCII)
    if outdated:
        for name in ("File", "Severity", "Fix", "Reason", "Video", "HDR"):
            table.add_column(name, no_wrap=True)
        display_rows = sorted(rows, key=lambda row: row_severity(row.defects).order)
        for row in display_rows:
            table.add_row(*_outdated_cells(row, root))
        return table

    for name in ("File", "Status", "Video", "HDR", "Audio", "Subs"):
        table.add_column(name, no_wrap=True)
    for row in rows:
        table.add_row(*_cells(row, root))
    return table


def render_scan_table(
    rows: Sequence[ScanRow],
    *,
    root: Path,
    total: int,
    warnings: Sequence[str] = (),
    file: TextIO | None = None,
    err: TextIO | None = None,
    outdated: bool = False,
) -> None:
    out = sys.stdout if file is None else file
    err_out = sys.stderr if err is None else err

    table = _build_table(rows, root, outdated=outdated)

    Console(file=out, width=_NO_TRUNCATE_WIDTH, highlight=False).print(table)

    err_console = Console(file=err_out, highlight=False)
    for warning in warnings:
        err_console.print(warning)
    if total == 0:
        err_console.print("no video files found")
    else:
        err_console.print(f"{len(rows)} of {total} shown")
