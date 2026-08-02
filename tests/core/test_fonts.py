from furnace.core.fonts import (
    FontFace,
    FontRequirement,
    is_font_attachment,
    parse_ass_font_requirements,
    select_font_attachment_indices,
)

STYLE_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
    "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
    "Alignment, MarginL, MarginR, MarginV, Encoding"
)

ASS = (
    r"""[Script Info]
ScriptType: v4.00+

[V4+ Styles]
"""
    + STYLE_FORMAT
    + r"""
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1
Style: Fancy,Times New Roman,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,-1,0,0,100,100,0,0,1,1,0,2,10,10,10,1
Style: Unused,Never Used,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,Hello
Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\fnComic Sans MS\b1\i1}Fancy{\r} normal
Dialogue: 0,0:00:02.00,0:00:03.00,Fancy,,0,0,0,,Styled
Dialogue: 0,0:00:03.00,0:00:04.00,Default,,0,0,0,,{\p1}m 0 0 l 1 1
Comment: 0,0:00:04.00,0:00:05.00,Unused,,0,0,0,,Ignored
"""
)


def test_parse_ass_font_requirements_uses_rendered_fragments_only() -> None:
    requirements = parse_ass_font_requirements(ASS)

    assert requirements == frozenset(
        {
            FontRequirement("Arial", bold=False, italic=False),
            FontRequirement("Comic Sans MS", bold=True, italic=True),
            FontRequirement("Times New Roman", bold=True, italic=True),
        }
    )


def test_parse_ass_font_requirements_empty_events() -> None:
    ass = ASS.split("[Events]", 1)[0] + (
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    assert parse_ass_font_requirements(ass) == frozenset()


def test_select_font_attachment_indices_prefers_exact_faces() -> None:
    requirements = frozenset(
        {
            FontRequirement("Arial", bold=False, italic=False),
            FontRequirement("Arial", bold=True, italic=False),
        }
    )
    faces = {
        0: (FontFace(frozenset({"Arial"}), bold=False, italic=False),),
        1: (FontFace(frozenset({"Arial"}), bold=True, italic=False),),
        2: (FontFace(frozenset({"Arial"}), bold=False, italic=True),),
        3: (FontFace(frozenset({"Tahoma"}), bold=False, italic=False),),
    }

    selected, missing = select_font_attachment_indices(requirements, faces)

    assert selected == (0, 1)
    assert missing == frozenset()


def test_select_font_attachment_indices_uses_family_fallback() -> None:
    requirement = FontRequirement("Arial", bold=True, italic=True)
    faces = {4: (FontFace(frozenset({"Arial"}), bold=False, italic=False),)}

    selected, missing = select_font_attachment_indices(frozenset({requirement}), faces)

    assert selected == (4,)
    assert missing == frozenset()


def test_select_font_attachment_indices_reports_missing_family() -> None:
    requirement = FontRequirement("Missing Font", bold=False, italic=False)

    selected, missing = select_font_attachment_indices(frozenset({requirement}), {})

    assert selected == ()
    assert missing == frozenset({requirement})


def test_select_font_attachment_indices_matches_localized_vertical_family_name() -> None:
    requirement = FontRequirement("@kozuka gothic pr6n h", bold=False, italic=False)
    faces = {
        8: (FontFace(frozenset({"Kozuka Gothic Pr6N H", "小塚ゴシック Pr6N H"}), bold=False, italic=False),),
    }

    selected, missing = select_font_attachment_indices(frozenset({requirement}), faces)

    assert selected == (8,)
    assert missing == frozenset()


def test_is_font_attachment_uses_mime_or_extension() -> None:
    assert is_font_attachment("font.bin", "font/ttf") is True
    assert is_font_attachment("font.bin", "application/vnd.ms-opentype") is True
    assert is_font_attachment("font.TTC", "application/octet-stream") is True
    assert is_font_attachment("cover.jpg", "image/jpeg") is False
    assert is_font_attachment("notes.txt", "") is False
