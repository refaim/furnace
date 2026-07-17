from __future__ import annotations

from pathlib import Path
from typing import Any, Final

_CP1251_MOJIBAKE_MARKERS: Final[frozenset[str]] = frozenset({"\u0420", "\u0421"})

_DECODE_CHAIN: Final[tuple[tuple[str, frozenset[str] | None], ...]] = (
    ("cp1251", _CP1251_MOJIBAKE_MARKERS),
    ("latin-1", None),
)


def _try_unmangle(text: str) -> str | None:
    for enc, markers in _DECODE_CHAIN:
        if markers is not None and markers.isdisjoint(text):
            continue
        try:
            return text.encode(enc).decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            continue
    return None


def is_mojibake(text: str) -> bool:
    if not text or text.isascii():
        return False
    return _try_unmangle(text) is not None


def fix_mojibake(text: str) -> str:
    if not text or text.isascii():
        return text
    recovered = _try_unmangle(text)
    return recovered if recovered is not None else text


def _seconds_to_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def chapters_have_mojibake(chapters: list[dict[str, Any]]) -> bool:
    return any(is_mojibake(ch.get("tags", {}).get("title", "")) for ch in chapters)


def write_ogm_chapters(chapters: list[dict[str, Any]], path: Path) -> None:
    lines: list[str] = []
    for i, ch in enumerate(chapters, 1):
        start_s = float(ch.get("start_time", 0))
        title = ch.get("tags", {}).get("title", f"Chapter {i}")
        title = fix_mojibake(title)
        lines.append(f"CHAPTER{i:02d}={_seconds_to_timestamp(start_s)}")
        lines.append(f"CHAPTER{i:02d}NAME={title}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fix_chapters_file(path: Path) -> bool:
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
