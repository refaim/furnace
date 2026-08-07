from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .detect import detect_hdr, is_mastering_display_side_data
from .outdated import Defect, EncoderFamily
from .quality import ALIGNMENT

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


_CROP_TAG_RE = re.compile(r"(?:^|[\s/])crop=(\d+):(\d+):(\d+):(\d+)")

_NVENCC_FAMILIES: tuple[EncoderFamily, ...] = (
    EncoderFamily.AV1_NVENC,
    EncoderFamily.HEVC_NVENC,
)


def parse_crop_rescale(
    settings: str | None,
    family: EncoderFamily,
    output_size: tuple[int, int] | None,
) -> bool | None:
    """Whether the encode reached its final size by rescaling the frame.

    Cutting a few more pixels off the crop is free; rescaling to reach the same
    size resamples every pixel and skews the aspect. Returns None when the tag
    says too little to tell.

    SVT-AV1 records the rectangle it kept (``crop=w:h:x:y``), and the file
    itself carries what that rectangle became, so the two just get compared: an
    axis that came out smaller was squashed, one that came out larger was
    stretched to square pixels, which is legitimate, and one that matches was
    never resampled. Nothing about the source size is assumed -- only that a
    stretch outgrows the <8px alignment residue, which every real SAR clears by
    an order of magnitude. A SAR within a few pixels of unity, reachable when
    SAR Fix derives one from an almost-16:9 source, breaks that tie and reads
    as a rescale.

    NVEncC records only the pixels it removed (``crop=top:bottom:left:right``),
    never the source size, so the kept size can only be inferred by assuming
    the source itself sat on the 8px grid. That holds for every disc
    (720/1920/3840) but not for a web rip like 1920x804, where the answer
    inverts: a clean file reads as rescaled and a rescaled one reads as clean.
    It also cannot see a rescale that happened with no crop recorded at all.
    The tag carries nothing better; only the version gate in classify_outdated
    bounds it.
    """
    if settings is None:
        return None
    match = _CROP_TAG_RE.search(settings)
    if match is None:
        return None
    first, second, third, fourth = (int(group) for group in match.groups())
    if family in _NVENCC_FAMILIES:
        return (first + second) % ALIGNMENT != 0 or (third + fourth) % ALIGNMENT != 0
    if family is EncoderFamily.AV1_SVT:
        if output_size is None:
            return None
        return output_size[0] < first or output_size[1] < second
    return None


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
    container_mastering_display: bool = False


@dataclass(frozen=True)
class ScanRow:
    path: Path
    furnace_version: tuple[int, int, int] | None
    video: VideoSummary
    audio: tuple[AudioTrackSummary, ...]
    subtitles: tuple[SubtitleTrackSummary, ...]
    unreadable: bool = False
    encoder_family: EncoderFamily = EncoderFamily.UNKNOWN
    crop_rescaled: bool | None = None
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
        side_data = vs.get("side_data_list") or []
        video = VideoSummary(
            codec=vs.get("codec_name", "unknown"),
            bit_depth=bit_depth_from_pix_fmt(vs.get("pix_fmt")),
            hdr=hdr_label(vs, side_data),
            width=vs.get("width"),
            height=vs.get("height"),
            color_matrix=vs.get("color_space"),
            color_transfer=vs.get("color_transfer"),
            container_mastering_display=any(
                is_mastering_display_side_data(entry.get("side_data_type", ""))
                for entry in side_data
            ),
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
