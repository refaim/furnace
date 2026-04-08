from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import AudioCodecId, CropRect, DvBlCompatibility, HdrMetadata, SubtitleCodecId, Track

class VideoSystem(enum.Enum):
    """Video system determined from frame height."""
    PAL = "pal"
    NTSC = "ntsc"
    HD = "hd"


_PAL_HEIGHTS = frozenset({576, 288})
_NTSC_HEIGHTS = frozenset({480, 486, 240})


def detect_video_system(height: int) -> VideoSystem:
    """Determine video system from frame height.

    PAL:  576, 288
    NTSC: 480, 486, 240
    HD:   >= 720
    Other SD: ValueError
    """
    if height in _PAL_HEIGHTS:
        return VideoSystem.PAL
    if height in _NTSC_HEIGHTS:
        return VideoSystem.NTSC
    if height >= 720:
        return VideoSystem.HD
    raise ValueError(
        f"Unknown SD height {height}: cannot determine PAL/NTSC. "
        f"Add this height to _PAL_HEIGHTS or _NTSC_HEIGHTS in detect.py"
    )


@dataclass(frozen=True)
class ResolvedColor:
    """Resolved color metadata for NVEncC flags."""
    matrix: str      # --colormatrix
    transfer: str    # --transfer
    primaries: str   # --colorprim


_BT2020_MATRICES = frozenset({"bt2020nc", "bt2020c"})
_BT601_MATRICES = frozenset({"bt470bg", "smpte170m"})

_TRANSFER_FROM_PRIMARIES: dict[str, str] = {
    "bt470bg": "bt470bg",
    "smpte170m": "smpte170m",
    "bt470m": "bt470m",
    "bt709": "bt709",
}


def resolve_color_metadata(
    matrix_raw: str | None,
    transfer_raw: str | None,
    primaries_raw: str | None,
    system: VideoSystem,
    has_hdr: bool,
) -> ResolvedColor:
    """Resolve color metadata, filling in missing values per ITU standards.

    Raises ValueError for unrecognized matrix_raw values.
    """
    # Step 1: determine family
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

    # Step 2: resolve matrix
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

    # Step 3: resolve primaries
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

    # Step 4: resolve transfer
    if transfer_raw is not None:
        transfer = transfer_raw
    elif family == "bt2020":
        transfer = "smpte2084" if has_hdr else "bt709"
    elif family == "bt709":
        transfer = "bt709"
    elif primaries in _TRANSFER_FROM_PRIMARIES:
        # bt601: infer from resolved primaries
        transfer = _TRANSFER_FROM_PRIMARIES[primaries]
    elif is_pal:
        transfer = "bt470bg"
    else:
        transfer = "smpte170m"

    return ResolvedColor(matrix=matrix, transfer=transfer, primaries=primaries)


FORCED_FILENAME_KEYWORDS: list[str] = ["forced", "форсир", "только надписи", "forsed", "tolko nadpisi"]
FORCED_FILENAME_EXCLUDE: list[str] = ["normal"]
FORCED_TRACKNAME_KEYWORDS: list[str] = ["forced", "caption"]
FORCED_TRACKNAME_EXCLUDE: list[str] = ["sdh"]
FULL_TRACKNAME_KEYWORDS: list[str] = ["sdh"]


def detect_forced_subtitles(subtitle_tracks: list[Track]) -> None:
    """Трёхэтапный алгоритм (in-place мутация is_forced):
    1. Ключевые слова в имени файла (для satellite files) -- FORCED_FILENAME_KEYWORDS / FORCED_FILENAME_EXCLUDE
    2. Ключевые слова в имени дорожки -- FORCED_TRACKNAME_KEYWORDS (но исключить FORCED_TRACKNAME_EXCLUDE)
    3. Статистический анализ:
       a. Исключить дорожки с языком 'chi' и дорожки с 'sdh' в названии из статистического сравнения.
       b. Разделить оставшиеся дорожки на две группы:
          - binary (PGS, VOBSUB): сравнивать по num_frames
          - text (SRT, ASS): сравнивать по num_captions
       c. Внутри каждой группы: для каждого языка найти максимум метрики.
          Если дорожка < 50% от максимума на том же языке -> forced.
       d. Использовать обе метрики (num_frames И num_captions) когда обе доступны,
          достаточно одной метрики < 50% для пометки forced.
    """
    # Stage 1: filename keywords
    for track in subtitle_tracks:
        filename_lower = track.source_file.name.lower()
        if any(kw in filename_lower for kw in FORCED_FILENAME_EXCLUDE):
            continue
        if any(kw in filename_lower for kw in FORCED_FILENAME_KEYWORDS):
            track.is_forced = True

    # Stage 2: track name keywords
    for track in subtitle_tracks:
        title_lower = track.title.lower()
        if any(kw in title_lower for kw in FORCED_TRACKNAME_EXCLUDE):
            continue
        if any(kw in title_lower for kw in FORCED_TRACKNAME_KEYWORDS):
            track.is_forced = True

    # Stage 3: statistical analysis
    _binary_codecs = {SubtitleCodecId.PGS, SubtitleCodecId.VOBSUB}
    _text_codecs = {SubtitleCodecId.SRT, SubtitleCodecId.ASS}

    # a. Exclude chi language and sdh tracks from statistical comparison
    stat_tracks = [
        t for t in subtitle_tracks
        if t.language != "chi" and "sdh" not in t.title.lower()
    ]

    # b. Split into binary and text groups
    binary_tracks = [t for t in stat_tracks if t.codec_id in _binary_codecs]
    text_tracks = [t for t in stat_tracks if t.codec_id in _text_codecs]

    # c/d. Within each group, for each language find max metric; mark < 50% as forced
    def _apply_statistical(group: list[Track], metric_attr: str) -> None:
        # Build per-language max
        lang_max: dict[str, int] = {}
        for track in group:
            value: int | None = getattr(track, metric_attr)
            if value is not None:
                current = lang_max.get(track.language, 0)
                if value > current:
                    lang_max[track.language] = value
        # Mark tracks below 50% of their language max
        for track in group:
            max_val = lang_max.get(track.language)
            if max_val is None or max_val == 0:
                continue
            value = getattr(track, metric_attr)
            if value is not None and value < max_val * 0.5:
                track.is_forced = True

    # Binary group uses num_frames; also check num_captions if available
    _apply_statistical(binary_tracks, "num_frames")
    _apply_statistical(binary_tracks, "num_captions")

    # Text group uses num_captions; also check num_frames if available
    _apply_statistical(text_tracks, "num_captions")
    _apply_statistical(text_tracks, "num_frames")


_DVD_RESOLUTIONS = {(720, 480), (720, 576)}


def is_dvd_resolution(width: int, height: int) -> bool:
    """720x480 (NTSC) or 720x576 (PAL)."""
    return (width, height) in _DVD_RESOLUTIONS


def cluster_crop_values(
    crops: list[CropRect],
    tolerance: int = 16,
) -> tuple[CropRect, int]:
    """Find largest cluster of similar crop values.

    Two CropRect values are 'close' if all 4 coordinates differ by at most
    *tolerance* pixels.  Returns (per-coordinate median of cluster, cluster size).
    """
    best_members: list[CropRect] = []

    for anchor in crops:
        members = [
            c for c in crops
            if (abs(c.w - anchor.w) <= tolerance
                and abs(c.h - anchor.h) <= tolerance
                and abs(c.x - anchor.x) <= tolerance
                and abs(c.y - anchor.y) <= tolerance)
        ]
        if len(members) > len(best_members):
            best_members = members

    ws = sorted(c.w for c in best_members)
    hs = sorted(c.h for c in best_members)
    xs = sorted(c.x for c in best_members)
    ys = sorted(c.y for c in best_members)
    mid = len(best_members) // 2
    return CropRect(w=ws[mid], h=hs[mid], x=xs[mid], y=ys[mid]), len(best_members)


_INTERLACED_FIELD_ORDERS = {"tt", "bb"}
_TV_FPS_THRESHOLD = 48.0
_IDET_INTERLACE_THRESHOLD = 0.05


def needs_idet(field_order: str | None, fps: float) -> bool:
    """Determine if idet analysis is needed to confirm interlace.

    Returns False (no idet) when:
    - field_order is not tt/bb → clearly progressive
    - field_order is tt/bb but fps >= 48 → clearly TV interlace
    Returns True when field_order is tt/bb and fps < 48 → ambiguous (DVD soft telecine?)
    """
    if field_order not in _INTERLACED_FIELD_ORDERS:
        return False
    return fps < _TV_FPS_THRESHOLD


def should_deinterlace(field_order: str | None, fps: float, idet_ratio: float) -> bool:
    """Decide whether to deinterlace based on ffprobe metadata and idet result.

    - field_order not tt/bb → progressive
    - field_order tt/bb + fps >= 48 → TV interlace, always deinterlace
    - field_order tt/bb + fps < 48 → idet decides (>5% interlaced → deinterlace)
    """
    if field_order not in _INTERLACED_FIELD_ORDERS:
        return False
    if fps >= _TV_FPS_THRESHOLD:
        return True
    return idet_ratio > _IDET_INTERLACE_THRESHOLD


def _fraction_numerator(val: str) -> str:
    """Extract numerator from fraction string. '8500/50000' -> '8500'. No-op for non-fractions."""
    s = str(val)
    if "/" in s:
        return s.split("/", 1)[0]
    return s


def detect_hdr(stream_data: dict[str, Any], side_data: list[dict[str, Any]] | None) -> HdrMetadata:
    """Анализирует side_data_list из ffprobe для MDCV и CLL.
    Проверяет codec_name для Dolby Vision (dvhe/dvh1) и
    HDR10+ (наличие dynamic metadata в side_data)."""
    mastering_display: str | None = None
    content_light: str | None = None
    is_dolby_vision: bool = False
    is_hdr10_plus: bool = False
    dv_profile: int | None = None
    dv_bl_compatibility: DvBlCompatibility | None = None

    sd = side_data or []

    for entry in sd:
        side_type = entry.get("side_data_type", "")

        if "Mastering display metadata" in side_type:
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

        elif "Content light level metadata" in side_type:
            max_cll = entry.get("max_content", "")
            max_fall = entry.get("max_average", "")
            content_light = f"MaxCLL={max_cll},MaxFALL={max_fall}"

        elif "Dolby Vision configuration" in side_type:
            is_dolby_vision = True
            raw_profile = entry.get("dv_profile")
            if raw_profile is not None:
                dv_profile = int(raw_profile)
            raw_compat = entry.get("dv_bl_signal_compatibility_id")
            if raw_compat is not None:
                try:
                    dv_bl_compatibility = DvBlCompatibility(int(raw_compat))
                except ValueError:
                    pass

        elif "HDR10+" in side_type or "SMPTE ST 2094" in side_type:
            is_hdr10_plus = True

    # Check codec_name for Dolby Vision
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
) -> tuple[bool, str]:
    """Возвращает (skip, reason). Skip если:
    - output_path существует
    - encoder_tag начинается с 'Furnace'
    """
    if output_path.exists():
        return True, f"output file already exists: {output_path}"
    if encoder_tag is not None and encoder_tag.startswith("Furnace"):
        return True, f"file already encoded by Furnace (tag: {encoder_tag})"
    return False, ""


def check_unsupported_codecs(
    audio_tracks: list[Track],
    subtitle_tracks: list[Track],
) -> str | None:
    """Возвращает строку с предупреждением если есть неизвестные кодеки, или None."""
    unknown: list[str] = []

    for track in audio_tracks:
        if track.codec_id is AudioCodecId.UNKNOWN:
            unknown.append(
                f"audio stream #{track.index} ({track.codec_name!r}, lang={track.language})"
            )

    for track in subtitle_tracks:
        if track.codec_id is SubtitleCodecId.UNKNOWN:
            unknown.append(
                f"subtitle stream #{track.index} ({track.codec_name!r}, lang={track.language})"
            )

    if unknown:
        items = ", ".join(unknown)
        return f"unsupported codecs detected: {items}"

    return None
