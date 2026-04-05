"""Chapter encoding utilities — detect and fix mojibake in chapter titles."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def is_mojibake(text: str) -> bool:
    """Check if text appears to be UTF-8 bytes decoded as Latin-1/CP1252."""
    if not text or text.isascii():
        return False
    try:
        raw = text.encode("latin-1")
        raw.decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return False
    else:
        return True


def fix_mojibake(text: str) -> str:
    """Fix UTF-8 text that was incorrectly decoded as Latin-1/CP1252.

    Returns the original text unchanged if it's not mojibake.
    """
    if not text or text.isascii():
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def _seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm timestamp."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def chapters_have_mojibake(chapters: list[dict[str, Any]]) -> bool:
    """Check if any ffprobe chapter title contains mojibake."""
    return any(
        is_mojibake(ch.get("tags", {}).get("title", ""))
        for ch in chapters
    )


def write_ogm_chapters(chapters: list[dict[str, Any]], path: Path) -> None:
    """Write ffprobe chapters as OGM file with mojibake-fixed titles."""
    lines: list[str] = []
    for i, ch in enumerate(chapters, 1):
        start_s = float(ch.get("start_time", 0))
        title = ch.get("tags", {}).get("title", f"Chapter {i}")
        title = fix_mojibake(title)
        lines.append(f"CHAPTER{i:02d}={_seconds_to_timestamp(start_s)}")
        lines.append(f"CHAPTER{i:02d}NAME={title}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fix_chapters_file(path: Path) -> bool:
    """Read an OGM chapters file, fix mojibake in-place.

    Returns True if any fix was applied.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    fixed_lines: list[str] = []
    changed = False
    for line in text.splitlines():
        if line.startswith("CHAPTER") and "NAME=" in line:
            prefix, _, name = line.partition("NAME=")
            fixed_name = fix_mojibake(name)
            if fixed_name != name:
                changed = True
            fixed_lines.append(f"{prefix}NAME={fixed_name}")
        else:
            fixed_lines.append(line)
    if changed:
        path.write_text("\n".join(fixed_lines) + "\n", encoding="utf-8")
    return changed
