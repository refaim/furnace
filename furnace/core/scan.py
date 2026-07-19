from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .detect import detect_hdr
from .outdated import Defect, EncoderFamily

_FURNACE_TAG_RE = re.compile(r"^Furnace v(\d+)\.(\d+)\.(\d+)$")

_VERSION_ARG_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

_PIX_FMT_DEPTH_RE = re.compile(r"p(\d+)(?:[lb]e)?$")


def parse_furnace_version(encoder_tag: str | None) -> tuple[int, int, int] | None:
    if encoder_tag is None:
        return None
    match = _FURNACE_TAG_RE.match(encoder_tag)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def parse_version_arg(s: str) -> tuple[int, int, int]:
    match = _VERSION_ARG_RE.match(s)
    if match is None:
        raise ValueError(f"Invalid version {s!r}: expected X.Y.Z")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


_PREFIX_ENCODER_FAMILIES: tuple[EncoderFamily, ...] = (
    EncoderFamily.HEVC_NVENC,
    EncoderFamily.AV1_NVENC,
    EncoderFamily.AV1_SVT,
)


def parse_encoder_family(settings: str | None) -> EncoderFamily:
    if settings is None:
        return EncoderFamily.UNKNOWN
    for family in _PREFIX_ENCODER_FAMILIES:
        if settings.startswith(family.value):
            return family
    if EncoderFamily.PASSTHROUGH.value in settings:
        return EncoderFamily.PASSTHROUGH
    return EncoderFamily.UNKNOWN


@dataclass(frozen=True)
class AudioTrackSummary:
    language: str | None
    codec: str
    channels: int | None


@dataclass(frozen=True)
class SubtitleTrackSummary:
    language: str | None
    codec: str


@dataclass(frozen=True)
class VideoSummary:
    codec: str | None
    bit_depth: int | None
    hdr: str | None
    width: int | None = None
    height: int | None = None
    color_matrix: str | None = None
    color_transfer: str | None = None


@dataclass(frozen=True)
class ScanRow:
    path: Path
    furnace_version: tuple[int, int, int] | None
    video: VideoSummary
    audio: tuple[AudioTrackSummary, ...]
    subtitles: tuple[SubtitleTrackSummary, ...]
    unreadable: bool = False
    encoder_family: EncoderFamily = EncoderFamily.UNKNOWN
    defects: tuple[Defect, ...] = ()


def bit_depth_from_pix_fmt(pix_fmt: str | None) -> int | None:
    if not pix_fmt:
        return None
    match = _PIX_FMT_DEPTH_RE.search(pix_fmt)
    if match is None:
        return 8
    return int(match.group(1))


def hdr_label(video_stream: dict[str, Any], side_data: list[dict[str, Any]]) -> str:
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
    streams = probe_json.get("streams", [])

    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if video_streams:
        vs = video_streams[0]
        video = VideoSummary(
            codec=vs.get("codec_name", "unknown"),
            bit_depth=bit_depth_from_pix_fmt(vs.get("pix_fmt")),
            hdr=hdr_label(vs, vs.get("side_data_list") or []),
            width=vs.get("width"),
            height=vs.get("height"),
            color_matrix=vs.get("color_space"),
            color_transfer=vs.get("color_transfer"),
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
    if not not_encoded and not encoded and max_version is None:
        return True
    if not_encoded and version is None:
        return True
    if encoded and version is not None:
        return True
    return max_version is not None and version is not None and version <= max_version
