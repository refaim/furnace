from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from furnace.core.outdated import Defect, classify_outdated
from furnace.core.ports import Prober
from furnace.core.scan import (
    ScanRow,
    VideoSummary,
    parse_encoder_family,
    parse_furnace_version,
    row_matches,
    summarize_streams,
)
from furnace.services.scanner import VIDEO_EXTENSIONS

logger = logging.getLogger(__name__)


def _classify_row(row: ScanRow) -> tuple[Defect, ...]:
    return classify_outdated(
        unreadable=row.unreadable,
        version=row.furnace_version,
        encoder_family=row.encoder_family,
        codec=row.video.codec,
        height=row.video.height,
        color_matrix=row.video.color_matrix,
        audio_channels=tuple(track.channels for track in row.audio),
    )


def discover_video_files(root: Path) -> list[Path]:
    if root.is_file():
        if root.suffix.lower() in VIDEO_EXTENSIONS:
            return [root]
        return []

    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if ".furnace_demux" in path.parts:
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        files.append(path)
    return files


class ScanService:
    def __init__(self, prober: Prober) -> None:
        self._prober = prober

    def scan(
        self,
        root: Path,
        *,
        not_encoded: bool = False,
        encoded: bool = False,
        max_version: tuple[int, int, int] | None = None,
        outdated: bool = False,
    ) -> tuple[list[ScanRow], int]:
        files = discover_video_files(root)
        rows: list[ScanRow] = []
        for path in files:
            try:
                probe_json = self._prober.probe(path)
            except (OSError, RuntimeError, ValueError):
                logger.debug("Probe failed for %s", path, exc_info=True)
                unreadable_row = ScanRow(
                    path=path,
                    furnace_version=None,
                    video=VideoSummary(None, None, None),
                    audio=(),
                    subtitles=(),
                    unreadable=True,
                )
                if outdated:
                    unreadable_row = replace(unreadable_row, defects=_classify_row(unreadable_row))
                rows.append(unreadable_row)
                continue

            tags = probe_json.get("format", {}).get("tags", {})
            encoder_tag = tags.get("ENCODER", tags.get("encoder"))
            version = parse_furnace_version(encoder_tag)

            if outdated:
                settings = tags.get("ENCODER_SETTINGS", tags.get("encoder_settings"))
                family = parse_encoder_family(settings)
                video, audio, subtitles = summarize_streams(probe_json)
                row = ScanRow(
                    path=path,
                    furnace_version=version,
                    video=video,
                    audio=audio,
                    subtitles=subtitles,
                    encoder_family=family,
                )
                defects = _classify_row(row)
                if defects:
                    rows.append(replace(row, defects=defects))
                continue

            if not row_matches(
                version,
                not_encoded=not_encoded,
                encoded=encoded,
                max_version=max_version,
            ):
                continue

            video, audio, subtitles = summarize_streams(probe_json)
            rows.append(
                ScanRow(
                    path=path,
                    furnace_version=version,
                    video=video,
                    audio=audio,
                    subtitles=subtitles,
                )
            )
        return rows, len(files)
