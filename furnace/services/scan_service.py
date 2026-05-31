from __future__ import annotations

import logging
from pathlib import Path

from furnace.core.ports import Prober
from furnace.core.scan import ScanRow, parse_furnace_version, row_matches, summarize_streams
from furnace.services.scanner import VIDEO_EXTENSIONS

logger = logging.getLogger(__name__)


def discover_video_files(root: Path) -> list[Path]:
    """Discover video files under ``root`` in stable discovery order.

    A single-file ``root`` yields one entry (or none, if it is not a video
    file). A directory is walked recursively via sorted ``rglob``; the
    ``.furnace_demux`` working directory is skipped, mirroring ``Scanner``.
    """
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
    """Read-only inventory of video files and their Furnace-encode status.

    Reuses a single ``Prober.probe()`` call per file to obtain both the MKV
    ``ENCODER`` tag and all stream detail. Never modifies files.
    """

    def __init__(self, prober: Prober) -> None:
        self._prober = prober

    def scan(
        self,
        root: Path,
        *,
        not_encoded: bool = False,
        encoded: bool = False,
        max_version: tuple[int, int, int] | None = None,
    ) -> tuple[list[ScanRow], int]:
        """Return ``(rows, total)`` for the video files under ``root``.

        ``total`` is the count of every discovered video file — the ``M`` in
        the ``N of M shown`` summary — while ``rows`` is the subset surviving
        the encode-status filter, in discovery order. The tree is walked
        exactly once, so ``total`` and ``rows`` share a single filesystem
        snapshot (no skew from files appearing or vanishing mid-scan).

        Each file is probed; a parseable Furnace ``ENCODER`` tag yields its
        version, and the encode-status filter (``not_encoded`` / ``encoded`` /
        ``max_version``) is applied via ``row_matches``. A probe failure
        (``OSError`` / ``RuntimeError`` / ``ValueError``) becomes an
        ``unreadable`` row, which is always included (never silently dropped).
        """
        files = discover_video_files(root)
        rows: list[ScanRow] = []
        for path in files:
            try:
                probe_json = self._prober.probe(path)
            except (OSError, RuntimeError, ValueError):
                logger.debug("Probe failed for %s", path, exc_info=True)
                rows.append(
                    ScanRow(
                        path=path,
                        furnace_version=None,
                        video_codec=None,
                        audio=(),
                        subtitles=(),
                        unreadable=True,
                    )
                )
                continue

            # The ENCODER tag's key casing varies by muxer (ENCODER / encoder),
            # so fall back to the lowercase form — mirroring FFmpegAdapter.
            tags = probe_json.get("format", {}).get("tags", {})
            encoder_tag = tags.get("ENCODER", tags.get("encoder"))
            version = parse_furnace_version(encoder_tag)
            if not row_matches(
                version,
                not_encoded=not_encoded,
                encoded=encoded,
                max_version=max_version,
            ):
                continue

            video_codec, audio, subtitles = summarize_streams(probe_json)
            rows.append(
                ScanRow(
                    path=path,
                    furnace_version=version,
                    video_codec=video_codec,
                    audio=audio,
                    subtitles=subtitles,
                )
            )
        return rows, len(files)
