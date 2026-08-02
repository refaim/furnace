from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from pysubs2 import SSAFile, SSAStyle
from pysubs2.formats.substation import parse_tags

from .models import Attachment


@dataclass(frozen=True)
class FontRequirement:
    family: str
    bold: bool
    italic: bool


@dataclass(frozen=True)
class FontFace:
    family_names: frozenset[str]
    bold: bool
    italic: bool


@dataclass(frozen=True)
class FontResolution:
    attachments: tuple[Attachment, ...]
    required: frozenset[FontRequirement]
    missing: frozenset[FontRequirement]


def _normalized_family(name: str) -> str:
    return " ".join(name.lstrip("@").split()).casefold()


def _has_visible_text(fragment: str) -> bool:
    return bool(fragment.replace(r"\N", "").replace(r"\n", "").replace(r"\h", "").strip())


def parse_ass_font_requirements(text: str) -> frozenset[FontRequirement]:
    subtitles = SSAFile.from_string(text, format_="ass")
    requirements: set[FontRequirement] = set()
    for event in subtitles.events:
        if event.is_comment:
            continue
        base_style = subtitles.styles.get(event.style, SSAStyle.DEFAULT_STYLE)
        for fragment, style in parse_tags(event.text, base_style, subtitles.styles, skip_empty_fragments=True):
            if not style.drawing and _has_visible_text(fragment):
                requirements.add(FontRequirement(style.fontname, bool(style.bold), bool(style.italic)))
    return frozenset(requirements)


def is_font_attachment(filename: str, mime_type: str) -> bool:
    extension = filename.rsplit(".", 1)[-1].casefold() if "." in filename else ""
    normalized_mime = mime_type.casefold()
    return (
        extension in {"ttf", "otf", "ttc", "otc"}
        or normalized_mime.startswith("font/")
        or "truetype" in normalized_mime
        or "opentype" in normalized_mime
        or "x-font" in normalized_mime
    )


def select_font_attachment_indices(
    requirements: frozenset[FontRequirement],
    faces_by_index: Mapping[int, tuple[FontFace, ...]],
) -> tuple[tuple[int, ...], frozenset[FontRequirement]]:
    selected: set[int] = set()
    missing: set[FontRequirement] = set()
    for requirement in sorted(requirements, key=lambda item: (_normalized_family(item.family), item.bold, item.italic)):
        family = _normalized_family(requirement.family)
        family_matches = [
            index
            for index, faces in faces_by_index.items()
            if any(family in {_normalized_family(name) for name in face.family_names} for face in faces)
        ]
        exact_matches = [
            index
            for index in family_matches
            if any(
                family in {_normalized_family(name) for name in face.family_names}
                and face.bold == requirement.bold
                and face.italic == requirement.italic
                for face in faces_by_index[index]
            )
        ]
        matches = exact_matches or family_matches
        if matches:
            selected.add(min(matches))
        else:
            missing.add(requirement)
    return tuple(sorted(selected)), frozenset(missing)
