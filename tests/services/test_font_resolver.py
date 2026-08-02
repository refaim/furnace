from pathlib import Path
from unittest.mock import MagicMock

import pytest

from furnace.core.fonts import FontFace, FontRequirement
from furnace.core.models import Attachment, SubtitleCodecId, Track, TrackType
from furnace.services.font_resolver import FontResolver
from tests.conftest import make_movie, make_track


def _ass(path: Path, font: str) -> Path:
    path.write_text(
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{font},20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,Hello\n",
        encoding="utf-8",
    )
    return path


def _sub(path: Path, index: int = 2) -> Track:
    return make_track(
        index=index,
        track_type=TrackType.SUBTITLE,
        codec_name="ass",
        codec_id=SubtitleCodecId.ASS,
        language="eng",
        source_file=path,
        channels=None,
        encoding="utf-8",
    )


def _attachments(source: Path) -> list[Attachment]:
    return [
        Attachment("Arial.ttf", "font/ttf", source, stream_index=3),
        Attachment("Times.ttf", "font/ttf", source, stream_index=4),
        Attachment("cover.jpg", "image/jpeg", source, stream_index=5),
    ]


def test_no_selected_subtitles_drops_fonts_and_keeps_non_font(tmp_path: Path) -> None:
    extractor = MagicMock()
    inspector = MagicMock()
    source = tmp_path / "movie.mkv"
    movie = make_movie(main_file=source, attachments=_attachments(source))

    result = FontResolver(extractor, inspector).resolve(movie, [])

    assert [attachment.filename for attachment in result.attachments] == ["cover.jpg"]
    assert result.required == frozenset()
    assert result.missing == frozenset()
    extractor.extract_attachment.assert_not_called()
    inspector.inspect.assert_not_called()


def test_non_ass_subtitles_drop_fonts(tmp_path: Path) -> None:
    extractor = MagicMock()
    inspector = MagicMock()
    source = tmp_path / "movie.mkv"
    srt = make_track(
        index=2,
        track_type=TrackType.SUBTITLE,
        codec_name="subrip",
        codec_id=SubtitleCodecId.SRT,
        language="eng",
        source_file=source,
        channels=None,
    )
    movie = make_movie(main_file=source, attachments=_attachments(source), subtitle_tracks=[srt])

    result = FontResolver(extractor, inspector).resolve(movie, [srt])

    assert [attachment.filename for attachment in result.attachments] == ["cover.jpg"]
    extractor.extract_attachment.assert_not_called()


def test_ass_without_rendered_text_drops_fonts_without_inspection(tmp_path: Path) -> None:
    extractor = MagicMock()
    inspector = MagicMock()
    source = tmp_path / "movie.mkv"
    ass_path = tmp_path / "empty.ass"
    ass_path.write_text(
        "[Script Info]\nScriptType: v4.00+\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
        encoding="utf-8",
    )
    subtitle = _sub(ass_path)
    movie = make_movie(main_file=source, attachments=_attachments(source), subtitle_tracks=[subtitle])

    result = FontResolver(extractor, inspector).resolve(movie, [subtitle])

    assert [attachment.filename for attachment in result.attachments] == ["cover.jpg"]
    extractor.extract_attachment.assert_not_called()
    inspector.inspect.assert_not_called()


def test_selected_ass_keeps_only_matching_font_and_non_font(tmp_path: Path) -> None:
    extractor = MagicMock()
    inspector = MagicMock()
    source = tmp_path / "movie.mkv"
    subtitle = _sub(_ass(tmp_path / "english.ass", "Arial"))
    attachments = _attachments(source)
    movie = make_movie(main_file=source, attachments=attachments, subtitle_tracks=[subtitle])

    def extract_attachment(input_path: Path, stream_index: int, output_path: Path) -> int:
        output_path.write_bytes(str(stream_index).encode())
        return 0

    extractor.extract_attachment.side_effect = extract_attachment

    def inspect(path: Path) -> tuple[FontFace, ...]:
        family = "Arial" if path.read_bytes() == b"3" else "Times New Roman"
        return (FontFace(frozenset({family}), bold=False, italic=False),)

    inspector.inspect.side_effect = inspect

    result = FontResolver(extractor, inspector).resolve(movie, [subtitle])

    assert [attachment.filename for attachment in result.attachments] == ["Arial.ttf", "cover.jpg"]
    assert result.required == frozenset({FontRequirement("Arial", bold=False, italic=False)})
    assert result.missing == frozenset()
    assert extractor.extract_attachment.call_count == 2


def test_selected_tracks_control_union(tmp_path: Path) -> None:
    extractor = MagicMock()
    inspector = MagicMock()
    source = tmp_path / "movie.mkv"
    english = _sub(_ass(tmp_path / "english.ass", "Arial"), index=2)
    signs = _sub(_ass(tmp_path / "signs.ass", "Times New Roman"), index=3)
    attachments = _attachments(source)
    movie = make_movie(main_file=source, attachments=attachments, subtitle_tracks=[english, signs])

    def extract_attachment(input_path: Path, stream_index: int, output_path: Path) -> int:
        output_path.write_bytes(str(stream_index).encode())
        return 0

    extractor.extract_attachment.side_effect = extract_attachment
    def inspect(path: Path) -> tuple[FontFace, ...]:
        family = "Arial" if path.read_bytes() == b"3" else "Times New Roman"
        return (FontFace(frozenset({family}), bold=False, italic=False),)

    inspector.inspect.side_effect = inspect
    resolver = FontResolver(extractor, inspector)

    english_only = resolver.resolve(movie, [english])
    both = resolver.resolve(movie, [english, signs])

    assert [attachment.filename for attachment in english_only.attachments] == ["Arial.ttf", "cover.jpg"]
    assert [attachment.filename for attachment in both.attachments] == ["Arial.ttf", "Times.ttf", "cover.jpg"]


def test_internal_ass_is_extracted_before_analysis(tmp_path: Path) -> None:
    extractor = MagicMock()
    inspector = MagicMock()
    source = tmp_path / "movie.mkv"
    subtitle = _sub(source)
    attachments = [Attachment("Arial.ttf", "font/ttf", source, stream_index=3)]
    movie = make_movie(main_file=source, attachments=attachments, subtitle_tracks=[subtitle])

    def extract_track(input_path: Path, stream_index: int, output_path: Path, on_progress: object = None) -> int:
        _ass(output_path, "Arial")
        return 0

    def extract_attachment(input_path: Path, stream_index: int, output_path: Path) -> int:
        output_path.write_bytes(b"font")
        return 0

    extractor.extract_track.side_effect = extract_track
    extractor.extract_attachment.side_effect = extract_attachment
    inspector.inspect.return_value = (FontFace(frozenset({"Arial"}), bold=False, italic=False),)

    result = FontResolver(extractor, inspector).resolve(movie, [subtitle])

    assert [attachment.filename for attachment in result.attachments] == ["Arial.ttf"]
    extractor.extract_track.assert_called_once()


def test_failed_attachment_extraction_raises(tmp_path: Path) -> None:
    extractor = MagicMock()
    inspector = MagicMock()
    source = tmp_path / "movie.mkv"
    subtitle = _sub(_ass(tmp_path / "english.ass", "Arial"))
    attachment = Attachment("Arial.ttf", "font/ttf", source, stream_index=3)
    movie = make_movie(main_file=source, attachments=[attachment], subtitle_tracks=[subtitle])
    extractor.extract_attachment.return_value = 1

    with pytest.raises(RuntimeError, match=r"Arial\.ttf"):
        FontResolver(extractor, inspector).resolve(movie, [subtitle])


def test_failed_internal_subtitle_extraction_raises(tmp_path: Path) -> None:
    extractor = MagicMock()
    inspector = MagicMock()
    source = tmp_path / "movie.mkv"
    subtitle = _sub(source)
    movie = make_movie(main_file=source, subtitle_tracks=[subtitle])
    extractor.extract_track.return_value = 1

    with pytest.raises(RuntimeError, match="subtitle stream 2"):
        FontResolver(extractor, inspector).resolve(movie, [subtitle])


def test_missing_font_is_reported(tmp_path: Path) -> None:
    extractor = MagicMock()
    inspector = MagicMock()
    source = tmp_path / "movie.mkv"
    subtitle = _sub(_ass(tmp_path / "english.ass", "Missing Font"))
    attachment = Attachment("Arial.ttf", "font/ttf", source, stream_index=3)
    movie = make_movie(main_file=source, attachments=[attachment], subtitle_tracks=[subtitle])

    def extract_attachment(input_path: Path, stream_index: int, output_path: Path) -> int:
        output_path.write_bytes(b"font")
        return 0

    extractor.extract_attachment.side_effect = extract_attachment
    inspector.inspect.return_value = (FontFace(frozenset({"Arial"}), bold=False, italic=False),)

    result = FontResolver(extractor, inspector).resolve(movie, [subtitle])

    assert result.attachments == ()
    assert result.missing == frozenset({FontRequirement("Missing Font", bold=False, italic=False)})
