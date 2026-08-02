from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTCollection, TTFont

from furnace.adapters.fonts import FontToolsAdapter
from furnace.core.fonts import FontFace


def _build_font(path: Path, family: str, style: str, *, bold: bool = False, italic: bool = False) -> None:
    builder = FontBuilder(1024, isTTF=True)
    builder.setupGlyphOrder([".notdef"])
    pen = TTGlyphPen(None)
    builder.setupGlyf({".notdef": pen.glyph()})
    builder.setupHorizontalMetrics({".notdef": (500, 0)})
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupCharacterMap({})
    builder.setupNameTable(
        {
            "familyName": family,
            "styleName": style,
            "uniqueFontIdentifier": f"{family}-{style}",
            "fullName": f"{family} {style}",
            "psName": f"{family.replace(' ', '')}-{style}",
            "version": "Version 1.0",
        }
    )
    selection = (1 if italic else 0) | (32 if bold else 0) | (0 if bold or italic else 64)
    builder.setupOS2(
        sTypoAscender=800,
        sTypoDescender=-200,
        usWinAscent=800,
        usWinDescent=200,
        usWeightClass=700 if bold else 400,
        fsSelection=selection,
    )
    builder.setupPost()
    builder.setupMaxp()
    builder.save(path)


def test_inspect_ttf_reads_family_and_face(tmp_path: Path) -> None:
    path = tmp_path / "regular.ttf"
    _build_font(path, "Example Sans", "Regular")

    faces = FontToolsAdapter().inspect(path)

    assert faces == (FontFace(frozenset({"Example Sans"}), bold=False, italic=False),)


def test_inspect_ttf_reads_bold_italic(tmp_path: Path) -> None:
    path = tmp_path / "bold-italic.ttf"
    _build_font(path, "Example Sans", "Bold Italic", bold=True, italic=True)

    faces = FontToolsAdapter().inspect(path)

    assert faces == (FontFace(frozenset({"Example Sans"}), bold=True, italic=True),)


def test_inspect_collection_reads_every_face(tmp_path: Path) -> None:
    regular_path = tmp_path / "regular.ttf"
    bold_path = tmp_path / "bold.ttf"
    collection_path = tmp_path / "collection.ttc"
    _build_font(regular_path, "Example Sans", "Regular")
    _build_font(bold_path, "Example Sans", "Bold", bold=True)
    collection = TTCollection()
    collection.fonts = [TTFont(regular_path), TTFont(bold_path)]
    collection.save(collection_path)
    for font in collection.fonts:
        font.close()

    faces = FontToolsAdapter().inspect(collection_path)

    assert faces == (
        FontFace(frozenset({"Example Sans"}), bold=False, italic=False),
        FontFace(frozenset({"Example Sans"}), bold=True, italic=False),
    )
