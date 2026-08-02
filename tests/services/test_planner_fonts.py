from pathlib import Path
from unittest.mock import MagicMock

import pytest

from furnace.core.fonts import FontRequirement, FontResolution
from furnace.core.models import Attachment, SubtitleCodecId, TrackType
from furnace.services.planner import PlannerService
from tests.conftest import make_movie, make_track


def test_planner_resolves_fonts_from_selected_subtitles(tmp_path: Path) -> None:
    source = tmp_path / "movie.mkv"
    subtitle = make_track(
        index=2,
        track_type=TrackType.SUBTITLE,
        codec_name="ass",
        codec_id=SubtitleCodecId.ASS,
        language="eng",
        source_file=source,
        channels=None,
    )
    selected = Attachment("Arial.ttf", "font/ttf", source, stream_index=3)
    dropped = Attachment("Unused.ttf", "font/ttf", source, stream_index=4)
    movie = make_movie(
        main_file=source,
        subtitle_tracks=[subtitle],
        attachments=[selected, dropped],
    )
    resolver = MagicMock()
    resolver.resolve.return_value = FontResolution(
        attachments=(selected,),
        required=frozenset(),
        missing=frozenset(),
    )

    plan = PlannerService(previewer=None, font_resolver=resolver).create_plan(
        [(movie, tmp_path / "out.mkv")],
        audio_lang_filter=[],
        sub_lang_filter=["eng"],
    )

    resolver.resolve.assert_called_once_with(movie, [subtitle])
    assert plan.jobs[0].attachments == [
        {
            "filename": "Arial.ttf",
            "mime_type": "font/ttf",
            "source_file": str(source),
            "stream_index": 3,
        }
    ]


def test_planner_without_resolver_drops_fonts_when_no_subtitles(tmp_path: Path) -> None:
    source = tmp_path / "movie.mkv"
    cover = Attachment("cover.jpg", "image/jpeg", source, stream_index=3)
    font = Attachment("Arial.ttf", "font/ttf", source, stream_index=4)
    movie = make_movie(main_file=source, attachments=[cover, font])

    plan = PlannerService(previewer=None).create_plan(
        [(movie, tmp_path / "out.mkv")],
        audio_lang_filter=[],
        sub_lang_filter=[],
    )

    assert plan.jobs[0].attachments == [
        {
            "filename": "cover.jpg",
            "mime_type": "image/jpeg",
            "source_file": str(source),
            "stream_index": 3,
        }
    ]


def test_planner_without_resolver_conservatively_keeps_fonts_for_ass(tmp_path: Path) -> None:
    source = tmp_path / "movie.mkv"
    subtitle = make_track(
        index=2,
        track_type=TrackType.SUBTITLE,
        codec_name="ass",
        codec_id=SubtitleCodecId.ASS,
        language="eng",
        source_file=source,
        channels=None,
    )
    font = Attachment("Arial.ttf", "font/ttf", source)
    movie = make_movie(main_file=source, subtitle_tracks=[subtitle], attachments=[font])

    plan = PlannerService(previewer=None).create_plan(
        [(movie, tmp_path / "out.mkv")],
        audio_lang_filter=[],
        sub_lang_filter=["eng"],
    )

    assert plan.jobs[0].attachments == [
        {
            "filename": "Arial.ttf",
            "mime_type": "font/ttf",
            "source_file": str(source),
        }
    ]


def test_planner_logs_missing_fonts(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    source = tmp_path / "movie.mkv"
    subtitle = make_track(
        index=2,
        track_type=TrackType.SUBTITLE,
        codec_name="ass",
        codec_id=SubtitleCodecId.ASS,
        language="eng",
        source_file=source,
        channels=None,
    )
    movie = make_movie(main_file=source, subtitle_tracks=[subtitle])
    resolver = MagicMock()
    resolver.resolve.return_value = FontResolution(
        attachments=(),
        required=frozenset(),
        missing=frozenset(
            {
                FontRequirement("Arial", bold=False, italic=False),
                FontRequirement("Arial", bold=True, italic=True),
            }
        ),
    )

    PlannerService(previewer=None, font_resolver=resolver).create_plan(
        [(movie, tmp_path / "out.mkv")],
        audio_lang_filter=[],
        sub_lang_filter=["eng"],
    )

    assert "Arial (regular, roman)" in caplog.text
    assert "Arial (bold, italic)" in caplog.text
