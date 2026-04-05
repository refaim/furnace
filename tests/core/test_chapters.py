"""Tests for furnace.core.chapters — mojibake detection and fix."""

from __future__ import annotations

from pathlib import Path

from furnace.core.chapters import (
    chapters_have_mojibake,
    fix_chapters_file,
    fix_mojibake,
    is_mojibake,
    write_ogm_chapters,
)

# "Глава" encoded as UTF-8 then decoded as Latin-1 produces this mojibake:
MOJIBAKE_GLAVA = "\u0420\u0413\u0301\u0420\u00bb\u0420\u00b0\u0420\u0406\u0420\u00b0"


def _make_mojibake(text: str) -> str:
    """Simulate mojibake: encode as UTF-8, decode as Latin-1."""
    return text.encode("utf-8").decode("latin-1")


class TestIsMojibake:
    def test_ascii_not_mojibake(self) -> None:
        assert not is_mojibake("Chapter 1")

    def test_empty_not_mojibake(self) -> None:
        assert not is_mojibake("")

    def test_clean_cyrillic_not_mojibake(self) -> None:
        assert not is_mojibake("Глава 1")

    def test_mojibake_detected(self) -> None:
        mangled = _make_mojibake("Глава 1")
        assert is_mojibake(mangled)

    def test_japanese_not_mojibake(self) -> None:
        # Japanese text that can't round-trip through latin-1
        assert not is_mojibake("第1章")


class TestFixMojibake:
    def test_fixes_cyrillic(self) -> None:
        mangled = _make_mojibake("Глава 1")
        assert fix_mojibake(mangled) == "Глава 1"

    def test_leaves_clean_text(self) -> None:
        assert fix_mojibake("Chapter 1") == "Chapter 1"

    def test_leaves_clean_cyrillic(self) -> None:
        assert fix_mojibake("Глава 1") == "Глава 1"

    def test_empty(self) -> None:
        assert fix_mojibake("") == ""


class TestChaptersHaveMojibake:
    def test_clean_chapters(self) -> None:
        chapters = [
            {"start_time": "0.000000", "tags": {"title": "Chapter 1"}},
            {"start_time": "300.000000", "tags": {"title": "Chapter 2"}},
        ]
        assert not chapters_have_mojibake(chapters)

    def test_mojibake_chapters(self) -> None:
        mangled = _make_mojibake("Глава 1")
        chapters = [
            {"start_time": "0.000000", "tags": {"title": mangled}},
            {"start_time": "300.000000", "tags": {"title": "Chapter 2"}},
        ]
        assert chapters_have_mojibake(chapters)

    def test_no_tags(self) -> None:
        chapters = [{"start_time": "0.000000"}]
        assert not chapters_have_mojibake(chapters)


class TestWriteOgmChapters:
    def test_writes_correct_format(self, tmp_path: Path) -> None:
        chapters = [
            {"start_time": "0.000000", "tags": {"title": "Chapter 1"}},
            {"start_time": "312.680000", "tags": {"title": "Chapter 2"}},
        ]
        out = tmp_path / "chapters.txt"
        write_ogm_chapters(chapters, out)
        text = out.read_text(encoding="utf-8")
        assert "CHAPTER01=00:00:00.000" in text
        assert "CHAPTER01NAME=Chapter 1" in text
        assert "CHAPTER02=00:05:12.680" in text
        assert "CHAPTER02NAME=Chapter 2" in text

    def test_fixes_mojibake_on_write(self, tmp_path: Path) -> None:
        mangled = _make_mojibake("Глава 1")
        chapters = [
            {"start_time": "0.000000", "tags": {"title": mangled}},
        ]
        out = tmp_path / "chapters.txt"
        write_ogm_chapters(chapters, out)
        text = out.read_text(encoding="utf-8")
        assert "CHAPTER01NAME=Глава 1" in text

    def test_missing_title_uses_default(self, tmp_path: Path) -> None:
        chapters = [{"start_time": "0.000000"}]
        out = tmp_path / "chapters.txt"
        write_ogm_chapters(chapters, out)
        text = out.read_text(encoding="utf-8")
        assert "CHAPTER01NAME=Chapter 1" in text


class TestFixChaptersFile:
    def test_fixes_mojibake_in_file(self, tmp_path: Path) -> None:
        mangled = _make_mojibake("Глава")
        content = (
            f"CHAPTER01=00:00:00.000\n"
            f"CHAPTER01NAME={mangled} 1\n"
            f"CHAPTER02=00:05:12.680\n"
            f"CHAPTER02NAME={mangled} 2\n"
        )
        f = tmp_path / "chapters.txt"
        f.write_text(content, encoding="utf-8")
        assert fix_chapters_file(f)
        fixed = f.read_text(encoding="utf-8")
        assert "CHAPTER01NAME=Глава 1" in fixed
        assert "CHAPTER02NAME=Глава 2" in fixed

    def test_no_change_for_clean_file(self, tmp_path: Path) -> None:
        content = (
            "CHAPTER01=00:00:00.000\n"
            "CHAPTER01NAME=Chapter 1\n"
        )
        f = tmp_path / "chapters.txt"
        f.write_text(content, encoding="utf-8")
        assert not fix_chapters_file(f)
