from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# A Furnace-stamped MKV carries an ENCODER tag of exactly ``Furnace vX.Y.Z``.
# Anything else (no tag, a foreign encoder, a malformed Furnace tag) is treated
# as "not encoded" — foreign encoder names are never surfaced.
_FURNACE_TAG_RE = re.compile(r"^Furnace v(\d+)\.(\d+)\.(\d+)$")

# The ``--max-version`` CLI argument must be a full ``X.Y.Z`` version.
_VERSION_ARG_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def parse_furnace_version(encoder_tag: str | None) -> tuple[int, int, int] | None:
    """Parse a Furnace ENCODER tag into ``(major, minor, patch)``.

    Only a tag matching ``^Furnace v(\\d+)\\.(\\d+)\\.(\\d+)$`` counts as Furnace.
    Returns ``None`` for a missing tag, a foreign encoder, or a malformed
    Furnace tag.
    """
    if encoder_tag is None:
        return None
    match = _FURNACE_TAG_RE.match(encoder_tag)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def parse_version_arg(s: str) -> tuple[int, int, int]:
    """Parse a ``--max-version`` argument into ``(major, minor, patch)``.

    Raises ``ValueError`` on anything that is not a full ``X.Y.Z`` version.
    """
    match = _VERSION_ARG_RE.match(s)
    if match is None:
        raise ValueError(f"Invalid version {s!r}: expected X.Y.Z")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


@dataclass(frozen=True)
class AudioTrackSummary:
    """One audio track, as shown in a scan table row."""

    language: str | None
    codec: str
    channels: int | None


@dataclass(frozen=True)
class SubtitleTrackSummary:
    """One subtitle track, as shown in a scan table row."""

    language: str | None
    codec: str


@dataclass(frozen=True)
class ScanRow:
    """A single scanned video file's inventory.

    ``furnace_version`` is the parsed ``(major, minor, patch)`` when the file
    carries a valid Furnace ENCODER tag, else ``None`` ("not encoded").
    ``unreadable`` marks a file whose probe failed; its stream fields are empty.
    """

    path: Path
    furnace_version: tuple[int, int, int] | None
    video_codec: str | None
    audio: tuple[AudioTrackSummary, ...]
    subtitles: tuple[SubtitleTrackSummary, ...]
    unreadable: bool = False


def summarize_streams(
    probe_json: dict[str, Any],
) -> tuple[str | None, tuple[AudioTrackSummary, ...], tuple[SubtitleTrackSummary, ...]]:
    """Reduce ffprobe JSON to (video_codec, audio tracks, subtitle tracks).

    The video codec is the first video stream's ``codec_name`` (or ``None`` when
    there is no video stream). Each audio/subtitle stream becomes one summary;
    a missing ``language`` tag yields ``None`` (the UI renders it as ``und``) and
    a missing ``codec_name`` yields ``"unknown"``.
    """
    streams = probe_json.get("streams", [])

    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    video_codec = video_streams[0].get("codec_name", "unknown") if video_streams else None

    audio = tuple(
        AudioTrackSummary(
            language=s.get("tags", {}).get("language"),
            codec=s.get("codec_name", "unknown"),
            channels=s.get("channels"),
        )
        for s in streams
        if s.get("codec_type") == "audio"
    )

    subtitles = tuple(
        SubtitleTrackSummary(
            language=s.get("tags", {}).get("language"),
            codec=s.get("codec_name", "unknown"),
        )
        for s in streams
        if s.get("codec_type") == "subtitle"
    )

    return video_codec, audio, subtitles


def row_matches(
    version: tuple[int, int, int] | None,
    *,
    not_encoded: bool,
    encoded: bool,
    max_version: tuple[int, int, int] | None,
) -> bool:
    """Decide whether a scanned file passes the encode-status filter.

    With no predicate set (``not_encoded`` and ``encoded`` both false and
    ``max_version`` ``None``), every file matches. Otherwise the predicates are
    OR-combined on the encode-status dimension:

    - ``not_encoded`` matches a file with no Furnace version (``version`` ``None``)
    - ``encoded`` matches any Furnace-encoded file (``version`` not ``None``)
    - ``max_version`` matches a Furnace version ``<=`` it; a not-encoded file
      (``version`` ``None``) never satisfies it.
    """
    if not not_encoded and not encoded and max_version is None:
        return True
    if not_encoded and version is None:
        return True
    if encoded and version is not None:
        return True
    return max_version is not None and version is not None and version <= max_version
