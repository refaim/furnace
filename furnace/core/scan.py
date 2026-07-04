from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .detect import detect_hdr

# A Furnace-stamped MKV carries an ENCODER tag of exactly ``Furnace vX.Y.Z``.
# Anything else (no tag, a foreign encoder, a malformed Furnace tag) is treated
# as "not encoded" — foreign encoder names are never surfaced.
_FURNACE_TAG_RE = re.compile(r"^Furnace v(\d+)\.(\d+)\.(\d+)$")

# The ``--max-version`` CLI argument must be a full ``X.Y.Z`` version.
_VERSION_ARG_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

# A trailing bit-depth token in a planar pix_fmt: the ``10`` in ``yuv420p10le``,
# the ``12`` in ``yuv420p12be``, the ``10`` in ``gbrp10le`` or ``p010le``. The
# ``p`` anchor is what keeps the subsampling digits (the ``420`` in ``yuv420p``)
# from being mistaken for a depth token.
_PIX_FMT_DEPTH_RE = re.compile(r"p(\d+)(?:[lb]e)?$")


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
class VideoSummary:
    """The video-stream fields shown in a scan row: codec, bit depth, HDR class."""

    codec: str | None  # first video stream's codec_name; None when no video stream
    bit_depth: int | None  # 8/10/12 from pix_fmt; None when no video stream / unknown
    hdr: str | None  # "SDR"|"HDR10"|"HLG"|"DV P{n}"|"DV"; None when no video stream


@dataclass(frozen=True)
class ScanRow:
    """A single scanned video file's inventory.

    ``furnace_version`` is the parsed ``(major, minor, patch)`` when the file
    carries a valid Furnace ENCODER tag, else ``None`` ("not encoded").
    ``video`` is the first video stream's codec, bit depth and HDR class (all
    ``None`` when there is no video stream). ``unreadable`` marks a file whose
    probe failed; its stream fields are empty.
    """

    path: Path
    furnace_version: tuple[int, int, int] | None
    video: VideoSummary
    audio: tuple[AudioTrackSummary, ...]
    subtitles: tuple[SubtitleTrackSummary, ...]
    unreadable: bool = False


def bit_depth_from_pix_fmt(pix_fmt: str | None) -> int | None:
    """Extract the video bit depth from an ffprobe ``pix_fmt``.

    Returns ``None`` for a missing (``None`` or empty) pix_fmt. Otherwise a
    trailing depth token after a ``p`` is read (``yuv420p10le`` -> 10,
    ``yuv420p12be`` -> 12, ``gbrp10le`` -> 10, ``p010le`` -> 10); a planar
    format with no depth token (``yuv420p``, ``yuv444p``) is 8-bit. The ``p``
    anchor avoids matching subsampling digits such as the ``420`` in ``yuv420p``.
    """
    if not pix_fmt:
        return None
    match = _PIX_FMT_DEPTH_RE.search(pix_fmt)
    if match is None:
        return 8
    return int(match.group(1))


def hdr_label(video_stream: dict[str, Any], side_data: list[dict[str, Any]]) -> str:
    """Classify a video stream's HDR type for the scan table.

    Dolby Vision wins over everything (``DV P{n}`` when the profile is known,
    else ``DV``). Failing that, the ``color_transfer`` decides: ``smpte2084`` is
    HDR10 (HDR10+ is intentionally not distinguished here), ``arib-std-b67`` is
    HLG, and anything else is SDR.
    """
    hdr = detect_hdr(video_stream, side_data)
    if hdr.is_dolby_vision:
        return f"DV P{hdr.dv_profile}" if hdr.dv_profile is not None else "DV"
    transfer = video_stream.get("color_transfer")
    if transfer == "smpte2084":
        return "HDR10"
    if transfer == "arib-std-b67":
        return "HLG"
    return "SDR"


def summarize_streams(
    probe_json: dict[str, Any],
) -> tuple[VideoSummary, tuple[AudioTrackSummary, ...], tuple[SubtitleTrackSummary, ...]]:
    """Reduce ffprobe JSON to (video summary, audio tracks, subtitle tracks).

    The video summary carries the first video stream's ``codec_name`` (or
    ``"unknown"`` when absent), its bit depth from ``pix_fmt`` and its HDR class;
    with no video stream it is ``VideoSummary(None, None, None)``. Each
    audio/subtitle stream becomes one summary; a missing ``language`` tag yields
    ``None`` (the UI renders it as ``und``) and a missing ``codec_name`` yields
    ``"unknown"``.
    """
    streams = probe_json.get("streams", [])

    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if video_streams:
        vs = video_streams[0]
        video = VideoSummary(
            codec=vs.get("codec_name", "unknown"),
            bit_depth=bit_depth_from_pix_fmt(vs.get("pix_fmt")),
            hdr=hdr_label(vs, vs.get("side_data_list") or []),
        )
    else:
        video = VideoSummary(None, None, None)

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

    return video, audio, subtitles


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
