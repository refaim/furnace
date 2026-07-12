"""Redirect-safe table renderer for ``furnace scan``.

The inventory table is written to stdout and must survive redirection to a
file (``furnace scan DIR > out.txt`` yields a clean file):

- no ANSI color/control codes when stdout is not a TTY,
- ASCII box-drawing (consistent with the project's Windows-cmd rule),
- columns sized to content — long paths are never truncated.

Everything that is not the table — a one-line summary (``N of M shown``), any
warnings, and the "no video files found" note — goes to stderr, so a
redirected file stays pure table.
"""

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

# Placeholder for an absent value — no video stream, or any stream column on an
# unreadable row. The em-dash is content (not box-drawing), so it is allowed.
_NONE = "—"

# Console width for the table. Large enough that Rich never wraps or truncates a
# column; with ``expand=False`` the table still renders at its own content
# width, so this does not pad lines out to the full width.
_NO_TRUNCATE_WIDTH = 10_000


def _rel_path(path: Path, root: Path) -> str:
    """Path shown in the File column: relative to ``root`` where possible.

    A single-file ``root`` (``path == root``) shows the bare filename; a path
    that is not under ``root`` falls back to its absolute form.
    """
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
    """The Video column text: ``—`` with no stream, ``codec`` when the bit depth
    is unknown, else ``codec Nbit`` (e.g. ``hevc 10bit``)."""
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
    """The six table cells (File, Status, Video, HDR, Audio, Subs) for one row.

    An unreadable row shows its path and ``unreadable`` status, with ``—`` in
    every stream column regardless of any stale stream fields.
    """
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
    """The six ``--outdated`` cells (File, Severity, Fix, Reason, Video, HDR).

    Severity is the worst severity among the row's defects and Fix the strongest
    remedy; Reason stacks every defect label (already severity-ordered) on its
    own line. An unreadable row keeps its ``—`` stream columns.
    """
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
    """Build the ASCII table for either the normal inventory or outdated modes.

    Normal mode lists every row in discovery order across the full stream
    inventory. Outdated mode swaps in the defect columns and sorts rows
    worst-first by row severity (a stable sort, so ties keep discovery order).
    """
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
    """Render the scan inventory.

    The ASCII table of ``rows`` is written to ``file`` (default stdout). The
    summary ``N of M shown`` (``total`` is ``M``), any ``warnings``, and the
    "no video files found" note are written to ``err`` (default stderr), so a
    redirected stdout stays pure table. When ``total`` is ``0`` the empty
    table header still prints and the note replaces the summary.

    With ``outdated`` set the table shows the defect work-list columns
    (``File | Severity | Fix | Reason | Video | HDR``) with rows sorted
    worst-first; ``N`` is then the flagged count and ``M`` the total scanned.
    """
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
