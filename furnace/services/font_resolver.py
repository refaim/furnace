from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from furnace.core.fonts import (
    FontFace,
    FontRequirement,
    FontResolution,
    is_font_attachment,
    parse_ass_font_requirements,
    select_font_attachment_indices,
)
from furnace.core.models import Attachment, Movie, SubtitleCodecId, Track
from furnace.core.ports import FontInspector, MediaExtractor


def _is_ass(track: Track) -> bool:
    return track.codec_id is SubtitleCodecId.ASS or track.codec_name.casefold() in {"ass", "ssa"}


class FontResolver:
    def __init__(self, extractor: MediaExtractor, inspector: FontInspector) -> None:
        self._extractor = extractor
        self._inspector = inspector

    def resolve(self, movie: Movie, selected_subtitles: list[Track]) -> FontResolution:
        font_attachments = [
            attachment
            for attachment in movie.attachments
            if is_font_attachment(attachment.filename, attachment.mime_type)
        ]
        non_font_attachments = tuple(
            attachment
            for attachment in movie.attachments
            if not is_font_attachment(attachment.filename, attachment.mime_type)
        )
        ass_tracks = [track for track in selected_subtitles if _is_ass(track)]
        if not ass_tracks:
            return FontResolution(non_font_attachments, frozenset(), frozenset())

        with TemporaryDirectory(prefix="furnace-fonts-") as temp_name:
            temp_dir = Path(temp_name)
            requirements = self._collect_requirements(ass_tracks, temp_dir)
            if not requirements:
                return FontResolution(non_font_attachments, requirements, frozenset())
            faces_by_index = self._inspect_attachments(font_attachments, temp_dir)

        selected_indices, missing = select_font_attachment_indices(requirements, faces_by_index)
        selected_set = set(selected_indices)
        selected_fonts = {id(attachment) for index, attachment in enumerate(font_attachments) if index in selected_set}
        attachments = tuple(
            attachment
            for attachment in movie.attachments
            if id(attachment) in selected_fonts or not is_font_attachment(attachment.filename, attachment.mime_type)
        )
        return FontResolution(attachments, requirements, missing)

    def _collect_requirements(
        self,
        tracks: list[Track],
        temp_dir: Path,
    ) -> frozenset[FontRequirement]:
        requirements: set[FontRequirement] = set()
        for position, track in enumerate(tracks):
            subtitle_path = track.source_file
            if subtitle_path.suffix.casefold() not in {".ass", ".ssa"}:
                subtitle_path = temp_dir / f"subtitle_{position}.ass"
                rc = self._extractor.extract_track(track.source_file, track.index, subtitle_path)
                if rc != 0:
                    raise RuntimeError(f"Failed to extract subtitle stream {track.index} from {track.source_file}")
            encoding = track.encoding or "utf-8-sig"
            requirements.update(parse_ass_font_requirements(subtitle_path.read_text(encoding=encoding)))
        return frozenset(requirements)

    def _inspect_attachments(
        self,
        attachments: list[Attachment],
        temp_dir: Path,
    ) -> dict[int, tuple[FontFace, ...]]:
        faces_by_index: dict[int, tuple[FontFace, ...]] = {}
        for position, attachment in enumerate(attachments):
            safe_name = Path(attachment.filename).name
            font_path = temp_dir / f"attachment_{position}_{safe_name}"
            rc = self._extractor.extract_attachment(attachment.source_file, attachment.stream_index, font_path)
            if rc != 0:
                raise RuntimeError(f"Failed to extract attachment {attachment.filename} from {attachment.source_file}")
            faces_by_index[position] = self._inspector.inspect(font_path)
        return faces_by_index
