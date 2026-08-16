from __future__ import annotations

import contextlib
import enum
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import AudioCodecId, CropRect, DvBlCompatibility, HdrMetadata, SubtitleCodecId, Track, VideoInfo

DV_PROFILE_FEL = 7


def classify_passthrough(video: VideoInfo, *, copy_video: bool) -> tuple[bool, str | None]:
    if not copy_video:
        return False, None
    if video.interlaced:
        return False, "interlaced"
    if video.hdr.is_dolby_vision and video.hdr.dv_profile == DV_PROFILE_FEL:
        return False, "DV P7 FEL"
    return True, None


class VideoSystem(enum.Enum):
    PAL = "pal"
    NTSC = "ntsc"
    HD = "hd"


_PAL_HEIGHTS = frozenset({576, 288})
_NTSC_HEIGHTS = frozenset({480, 486, 240})
_HD_MIN_HEIGHT = 720
_PAL_FRAME_RATES = frozenset({25, 50})
_NTSC_FRAME_RATES = frozenset({24, 30, 60})


def detect_video_system(height: int, fps_num: int, fps_den: int) -> VideoSystem:
    if height in _PAL_HEIGHTS:
        return VideoSystem.PAL
    if height in _NTSC_HEIGHTS:
        return VideoSystem.NTSC
    if height >= _HD_MIN_HEIGHT:
        return VideoSystem.HD
    rate = round(fps_num / fps_den)
    if rate in _PAL_FRAME_RATES:
        return VideoSystem.PAL
    if rate in _NTSC_FRAME_RATES:
        return VideoSystem.NTSC
    raise ValueError(
        f"Unknown SD video: height {height} at {fps_num}/{fps_den} fps cannot be classified PAL/NTSC. "
        f"Add the height to _PAL_HEIGHTS/_NTSC_HEIGHTS or the frame rate to "
        f"_PAL_FRAME_RATES/_NTSC_FRAME_RATES in detect.py"
    )


@dataclass(frozen=True)
class ResolvedColor:
    matrix: str
    transfer: str
    primaries: str


_BT2020_MATRICES = frozenset({"bt2020nc", "bt2020c"})
_BT601_MATRICES = frozenset({"bt470bg", "smpte170m"})

_TRANSFER_FROM_PRIMARIES: dict[str, str] = {
    "bt470bg": "smpte170m",
    "smpte170m": "smpte170m",
    "bt470m": "smpte170m",
    "bt709": "bt709",
}


def resolve_color_metadata(
    matrix_raw: str | None,
    transfer_raw: str | None,
    primaries_raw: str | None,
    system: VideoSystem,
    *,
    has_hdr: bool,
) -> ResolvedColor:
    if matrix_raw in _BT2020_MATRICES:
        family = "bt2020"
    elif matrix_raw == "bt709":
        family = "bt709"
    elif matrix_raw in _BT601_MATRICES:
        family = "bt601"
    elif matrix_raw is None:
        if has_hdr:
            family = "bt2020"
        elif system == VideoSystem.HD:
            family = "bt709"
        else:
            family = "bt601"
    else:
        raise ValueError(f"Unrecognized matrix_raw: {matrix_raw!r}")

    is_pal = system == VideoSystem.PAL

    if matrix_raw is not None:
        matrix = matrix_raw
    elif family == "bt2020":
        matrix = "bt2020nc"
    elif family == "bt709":
        matrix = "bt709"
    elif is_pal:
        matrix = "bt470bg"
    else:
        matrix = "smpte170m"

    if primaries_raw is not None:
        primaries = primaries_raw
    elif family == "bt2020":
        primaries = "bt2020"
    elif family == "bt709":
        primaries = "bt709"
    elif is_pal:
        primaries = "bt470bg"
    else:
        primaries = "smpte170m"

    if transfer_raw is not None:
        transfer = transfer_raw
    elif family == "bt2020":
        transfer = "smpte2084" if has_hdr else "bt709"
    elif family == "bt709":
        transfer = "bt709"
    elif primaries in _TRANSFER_FROM_PRIMARIES:
        transfer = _TRANSFER_FROM_PRIMARIES[primaries]
    else:
        transfer = "smpte170m"

    return ResolvedColor(matrix=matrix, transfer=transfer, primaries=primaries)


FORCED_FILENAME_KEYWORDS: list[str] = ["forced", "форсир", "только надписи", "forsed", "tolko nadpisi"]
FORCED_FILENAME_EXCLUDE: list[str] = ["normal"]
FORCED_TRACKNAME_KEYWORDS: list[str] = ["forced", "caption"]
FORCED_TRACKNAME_EXCLUDE: list[str] = ["sdh"]
FULL_TRACKNAME_KEYWORDS: list[str] = ["sdh"]


def detect_forced_subtitles(subtitle_tracks: list[Track]) -> None:
    for track in subtitle_tracks:
        filename_lower = track.source_file.name.lower()
        if any(kw in filename_lower for kw in FORCED_FILENAME_EXCLUDE):
            continue
        if any(kw in filename_lower for kw in FORCED_FILENAME_KEYWORDS):
            track.is_forced = True

    for track in subtitle_tracks:
        title_lower = track.title.lower()
        if any(kw in title_lower for kw in FORCED_TRACKNAME_EXCLUDE):
            continue
        if any(kw in title_lower for kw in FORCED_TRACKNAME_KEYWORDS):
            track.is_forced = True

    _binary_codecs = {SubtitleCodecId.PGS, SubtitleCodecId.VOBSUB}
    _text_codecs = {SubtitleCodecId.SRT, SubtitleCodecId.ASS}

    stat_tracks = [t for t in subtitle_tracks if t.language != "chi" and "sdh" not in t.title.lower()]

    binary_tracks = [t for t in stat_tracks if t.codec_id in _binary_codecs]
    text_tracks = [t for t in stat_tracks if t.codec_id in _text_codecs]

    def _apply_statistical(group: list[Track], metric_attr: str) -> None:
        lang_max: dict[str, int] = {}
        for track in group:
            value: int | None = getattr(track, metric_attr)
            if value is not None:
                current = lang_max.get(track.language, 0)
                if value > current:
                    lang_max[track.language] = value
        for track in group:
            max_val = lang_max.get(track.language)
            if max_val is None or max_val == 0:
                continue
            value = getattr(track, metric_attr)
            if value is not None and value < max_val * 0.5:
                track.is_forced = True

    _apply_statistical(binary_tracks, "num_frames")
    _apply_statistical(binary_tracks, "num_captions")

    _apply_statistical(text_tracks, "num_captions")
    _apply_statistical(text_tracks, "num_frames")


_DVD_RESOLUTIONS = {(720, 480), (720, 576)}


def is_dvd_resolution(width: int, height: int) -> bool:
    return (width, height) in _DVD_RESOLUTIONS


_HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})


def is_hdr_transfer(color_transfer: str | None) -> bool:
    return color_transfer in _HDR_TRANSFERS


def hdr_tonemap_transfer(color_transfer: str | None) -> str | None:
    return color_transfer if color_transfer in _HDR_TRANSFERS else None


CROP_EDGE_TOLERANCE = 8

_CROP_DETECT_LIMIT = 40
_CROP_BAR_CEILING = 48
_CROP_LIMIT_MARGIN = 4


def cropdetect_limit(border_levels: Sequence[float]) -> int:
    bar_levels = [level for level in border_levels if _CROP_DETECT_LIMIT < level <= _CROP_BAR_CEILING]
    if not bar_levels:
        return _CROP_DETECT_LIMIT
    limit = math.ceil(max(bar_levels)) + _CROP_LIMIT_MARGIN
    picture_levels = [level for level in border_levels if level > _CROP_BAR_CEILING]
    if not picture_levels:
        return limit
    return min(limit, math.floor(min(picture_levels)) - 1)


def _dominant_edge(values: list[int], tolerance: int) -> int:
    best: list[int] = []
    for anchor in values:
        members = [v for v in values if abs(v - anchor) <= tolerance]
        if len(members) > len(best):
            best = members
    best.sort()
    return best[len(best) // 2]


def aggregate_crop(
    crops: list[CropRect],
    tolerance: int = CROP_EDGE_TOLERANCE,
) -> CropRect:
    left = _dominant_edge([c.x for c in crops], tolerance)
    right = _dominant_edge([c.x + c.w for c in crops], tolerance)
    top = _dominant_edge([c.y for c in crops], tolerance)
    bottom = _dominant_edge([c.y + c.h for c in crops], tolerance)
    if right < left or bottom < top:
        raise ValueError(
            f"cropdetect samples too inconsistent to crop: x {left}..{right}, y {top}..{bottom}",
        )
    left -= left % 2
    top -= top % 2
    right += right % 2
    bottom += bottom % 2
    return CropRect(w=right - left, h=bottom - top, x=left, y=top)


_INTERLACED_FIELD_ORDERS = {"tt", "bb"}
_TV_FPS_THRESHOLD = 48.0
_IDET_INTERLACE_THRESHOLD = 0.05


def _is_hd(height: int) -> bool:
    return height >= _HD_MIN_HEIGHT


def needs_idet(field_order: str | None, fps: float, height: int) -> bool:
    if field_order not in _INTERLACED_FIELD_ORDERS:
        return False
    if fps >= _TV_FPS_THRESHOLD:
        return False
    return not _is_hd(height)


def should_deinterlace(field_order: str | None, fps: float, idet_ratio: float, height: int) -> bool:
    if field_order not in _INTERLACED_FIELD_ORDERS:
        return False
    if fps >= _TV_FPS_THRESHOLD or _is_hd(height):
        return True
    return idet_ratio > _IDET_INTERLACE_THRESHOLD


_FIELD_SEPARATED_PACKETS_PER_FRAME = 2.0
_FIELD_SEPARATED_RATIO_TOLERANCE = 0.02
_MIN_FIELD_PAIRING_SAMPLE = 100


def needs_field_rate_probe(field_order: str | None, fps_num: int, fps_den: int) -> bool:
    if field_order not in _INTERLACED_FIELD_ORDERS:
        return False
    return fps_num / fps_den >= _TV_FPS_THRESHOLD


def detect_field_separated(
    fps_num: int,
    fps_den: int,
    frames: int,
    packets: int,
) -> tuple[int, int] | None:
    if frames < _MIN_FIELD_PAIRING_SAMPLE:
        return None
    ratio = packets / frames
    if abs(ratio - _FIELD_SEPARATED_PACKETS_PER_FRAME) > _FIELD_SEPARATED_RATIO_TOLERANCE:
        return None
    num, den = fps_num, fps_den * 2
    common = math.gcd(num, den)
    return num // common, den // common


_NTSC_FPS_MIN = 29.9
_NTSC_FPS_MAX = 30.1
_PULLDOWN_TARGET_RATIO = 4 / 5
_PULLDOWN_RATIO_TOLERANCE = 0.02
_MIN_PULLDOWN_SAMPLE = 100


def needs_pulldown_probe(codec_name: str, fps_num: int, fps_den: int, height: int) -> bool:
    if codec_name != "mpeg2video":
        return False
    if _is_hd(height):
        return False
    fps = fps_num / fps_den
    return _NTSC_FPS_MIN <= fps <= _NTSC_FPS_MAX


def detect_soft_telecine(fps_num: int, fps_den: int, repeat_picts: Sequence[int]) -> tuple[int, int] | None:
    if len(repeat_picts) < _MIN_PULLDOWN_SAMPLE:
        return None
    fields = sum(2 + r for r in repeat_picts)
    ratio = 2 * len(repeat_picts) / fields
    if abs(ratio - _PULLDOWN_TARGET_RATIO) > _PULLDOWN_RATIO_TOLERANCE:
        return None
    num, den = fps_num * 4, fps_den * 5
    common = math.gcd(num, den)
    return num // common, den // common


_GRAIN_FLICKER_THRESHOLD = 0.40


def classify_grain(flicker_samples: Sequence[float]) -> bool:
    if not flicker_samples:
        return True
    ordered = sorted(flicker_samples)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return median >= _GRAIN_FLICKER_THRESHOLD


_HDR10_PLUS_SIDE_DATA_MARKERS = ("HDR10+", "SMPTE ST 2094", "SMPTE2094")
_DOLBY_VISION_SIDE_DATA_TYPES = ("Dolby Vision RPU Data", "Dolby Vision Metadata")
_MASTERING_DISPLAY_SIDE_DATA_MARKER = "Mastering display metadata"
_CONTENT_LIGHT_SIDE_DATA_MARKER = "Content light level metadata"


def is_hdr10_plus_side_data(side_data_type: str) -> bool:
    return any(marker in side_data_type for marker in _HDR10_PLUS_SIDE_DATA_MARKERS)


def is_dolby_vision_side_data(side_data_type: str) -> bool:
    return side_data_type in _DOLBY_VISION_SIDE_DATA_TYPES


def is_mastering_display_side_data(side_data_type: str) -> bool:
    return _MASTERING_DISPLAY_SIDE_DATA_MARKER in side_data_type


def is_content_light_side_data(side_data_type: str) -> bool:
    return _CONTENT_LIGHT_SIDE_DATA_MARKER in side_data_type


def _fraction_numerator(val: str) -> str:
    s = str(val)
    if "/" in s:
        return s.split("/", 1)[0]
    return s


def detect_hdr(stream_data: dict[str, Any], side_data: list[dict[str, Any]] | None) -> HdrMetadata:
    mastering_display: str | None = None
    content_light: str | None = None
    is_dolby_vision: bool = False
    is_hdr10_plus: bool = False
    dv_profile: int | None = None
    dv_bl_compatibility: DvBlCompatibility | None = None

    sd = side_data or []

    for entry in sd:
        side_type = entry.get("side_data_type", "")

        if is_mastering_display_side_data(side_type):
            mastering_display = (
                f"G({_fraction_numerator(entry.get('green_x', ''))},"
                f"{_fraction_numerator(entry.get('green_y', ''))})"
                f"B({_fraction_numerator(entry.get('blue_x', ''))},"
                f"{_fraction_numerator(entry.get('blue_y', ''))})"
                f"R({_fraction_numerator(entry.get('red_x', ''))},"
                f"{_fraction_numerator(entry.get('red_y', ''))})"
                f"WP({_fraction_numerator(entry.get('white_point_x', ''))},"
                f"{_fraction_numerator(entry.get('white_point_y', ''))})"
                f"L({_fraction_numerator(entry.get('max_luminance', ''))},"
                f"{_fraction_numerator(entry.get('min_luminance', ''))})"
            )

        elif is_content_light_side_data(side_type):
            max_cll = entry.get("max_content", "")
            max_fall = entry.get("max_average", "")
            content_light = f"MaxCLL={max_cll},MaxFALL={max_fall}"

        elif side_type == "DOVI configuration record":
            is_dolby_vision = True
            raw_profile = entry.get("dv_profile")
            if raw_profile is not None:
                dv_profile = int(raw_profile)
            raw_compat = entry.get("dv_bl_signal_compatibility_id")
            if raw_compat is not None:
                with contextlib.suppress(ValueError):
                    dv_bl_compatibility = DvBlCompatibility(int(raw_compat))

        elif is_dolby_vision_side_data(side_type):
            is_dolby_vision = True

        elif is_hdr10_plus_side_data(side_type):
            is_hdr10_plus = True

    codec_name = stream_data.get("codec_name", "")
    if codec_name in ("dvhe", "dvh1"):
        is_dolby_vision = True

    return HdrMetadata(
        mastering_display=mastering_display,
        content_light=content_light,
        is_dolby_vision=is_dolby_vision,
        is_hdr10_plus=is_hdr10_plus,
        dv_profile=dv_profile,
        dv_bl_compatibility=dv_bl_compatibility,
    )


def should_skip_file(
    output_path: Path,
    encoder_tag: str | None,
    *,
    force: bool = False,
) -> tuple[bool, str]:
    if force:
        return False, ""
    if output_path.exists():
        return True, f"output file already exists: {output_path}"
    if encoder_tag is not None and encoder_tag.startswith("Furnace"):
        return True, f"file already encoded by Furnace (tag: {encoder_tag})"
    return False, ""


def _describe_unsupported(track: Track, kind: str) -> str:
    # The profile is what separates a DTS variant we cannot classify from plain
    # DTS, so name it -- otherwise the message is unactionable.
    profile = f" profile {track.profile!r}" if track.profile is not None else ""
    return f"{kind} stream #{track.index} ({track.codec_name!r}{profile}, lang={track.language})"


def check_unsupported_codecs(
    audio_tracks: list[Track],
    subtitle_tracks: list[Track],
) -> str | None:
    unknown: list[str] = [
        _describe_unsupported(track, "audio")
        for track in audio_tracks
        if track.codec_id is AudioCodecId.UNKNOWN
    ]

    unknown.extend(
        _describe_unsupported(track, "subtitle")
        for track in subtitle_tracks
        if track.codec_id is SubtitleCodecId.UNKNOWN
    )

    if unknown:
        items = ", ".join(unknown)
        return f"unsupported codecs detected: {items}"

    return None
