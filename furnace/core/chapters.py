"""Chapter encoding utilities — detect and fix mojibake in chapter titles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

# Decode candidates tried in order. Each entry is (encoding, required_markers):
# we only attempt the encoding when at least one marker character is present
# in the input, otherwise the candidate is skipped.
#
# cp1251 needs a guard because short clean Cyrillic strings (e.g. T-yo,
# V-yo) round-trip as valid UTF-8 and would be silently corrupted.
# Genuine cp1251 mojibake of Russian/Ukrainian text always contains one of
# the cp1251 chars 0xD0 / 0xD1 because those are the UTF-8 lead bytes for
# the U+0400..U+047F block (the entire mainstream alphabet). Decoded via
# cp1251 they render as U+0420 / U+0421 — used as marker characters below
# via \\u escapes to keep the source ASCII-only and ruff-clean.
#
# latin-1 has no marker requirement: it preserves the historical behaviour
# for Western mojibake and bytes outside cp1251 already raise on encode.
_CP1251_MOJIBAKE_MARKERS: Final[frozenset[str]] = frozenset({"\u0420", "\u0421"})

_DECODE_CHAIN: Final[tuple[tuple[str, frozenset[str] | None], ...]] = (
    ("cp1251", _CP1251_MOJIBAKE_MARKERS),
    ("latin-1", None),
)


def _try_unmangle(text: str) -> str | None:
    """Return UTF-8-recovered text if `text` round-trips through any known
    mojibake encoding (subject to per-encoding marker guards), otherwise None.
    """
    for enc, markers in _DECODE_CHAIN:
        if markers is not None and markers.isdisjoint(text):
            continue
        try:
            return text.encode(enc).decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            continue
    return None


def is_mojibake(text: str) -> bool:
    """Check if text appears to be UTF-8 bytes decoded as cp1251 or latin-1/CP1252."""
    if not text or text.isascii():
        return False
    return _try_unmangle(text) is not None


def fix_mojibake(text: str) -> str:
    """Fix UTF-8 text that was incorrectly decoded as cp1251 or latin-1/CP1252.

    Returns the original text unchanged if it's not mojibake.
    """
    if not text or text.isascii():
        return text
    recovered = _try_unmangle(text)
    return recovered if recovered is not None else text


def _seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm timestamp."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def chapters_have_mojibake(chapters: list[dict[str, Any]]) -> bool:
    """Check if any ffprobe chapter title contains mojibake."""
    return any(is_mojibake(ch.get("tags", {}).get("title", "")) for ch in chapters)


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
