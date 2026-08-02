from __future__ import annotations

from pathlib import Path

from fontTools.ttLib import TTCollection, TTFont

from furnace.core.fonts import FontFace

_BOLD_WEIGHT = 700


def _font_face(font: TTFont) -> FontFace:
    names = frozenset(
        value for record in font["name"].names if record.nameID in {1, 16} if (value := record.toUnicode().strip())
    )
    os2 = font["OS/2"]
    head = font["head"]
    bold = os2.usWeightClass >= _BOLD_WEIGHT or bool(os2.fsSelection & 32) or bool(head.macStyle & 1)
    italic = bool(os2.fsSelection & 1) or bool(head.macStyle & 2)
    return FontFace(names, bold=bold, italic=italic)


class FontToolsAdapter:
    def inspect(self, path: Path) -> tuple[FontFace, ...]:
        if path.suffix.casefold() in {".ttc", ".otc"}:
            collection = TTCollection(path, lazy=True)
            try:
                return tuple(_font_face(font) for font in collection.fonts)
            finally:
                collection.close()
        font = TTFont(path, lazy=True)
        try:
            return (_font_face(font),)
        finally:
            font.close()
