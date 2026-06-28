from __future__ import annotations

import contextlib
import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import AudioCodecId, CropRect, DvBlCompatibility, HdrMetadata, SubtitleCodecId, Track, VideoInfo

DV_PROFILE_FEL = 7  # Dolby Vision FEL — needs a P7 → P8.1 re-encode (no passthrough)


def classify_passthrough(video: VideoInfo, *, copy_video: bool) -> tuple[bool, str | None]:
    """Decide whether a source video can be copied verbatim.

    (False, None)         -> copy_video not requested (normal encode)
    (False, "interlaced") -> must deinterlace
    (False, "DV P7 FEL")  -> P7 FEL needs the P7 -> P8.1 conversion
    (True, None)          -> copy the stream verbatim
    """
    if not copy_video:
        return False, None
    if video.interlaced:
        return False, "interlaced"
    if video.hdr.is_dolby_vision and video.hdr.dv_profile == DV_PROFILE_FEL:
        return False, "DV P7 FEL"
    return True, None


class VideoSystem(enum.Enum):
    """Video system determined from frame height."""

    PAL = "pal"
    NTSC = "ntsc"
    HD = "hd"


_PAL_HEIGHTS = frozenset({576, 288})
_NTSC_HEIGHTS = frozenset({480, 486, 240})
_HD_MIN_HEIGHT = 720  # anything at 720 or above is treated as HD


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
    if height >= _HD_MIN_HEIGHT:
        return VideoSystem.HD
    raise ValueError(
        f"Unknown SD height {height}: cannot determine PAL/NTSC. "
        f"Add this height to _PAL_HEIGHTS or _NTSC_HEIGHTS in detect.py"
    )


@dataclass(frozen=True)
class ResolvedColor:
    """Resolved color metadata for NVEncC flags."""

    matrix: str  # --colormatrix
    transfer: str  # --transfer
    primaries: str  # --colorprim


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
    *,
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
    """Three-stage algorithm that mutates is_forced in place:
    1. Filename keywords (for satellite files) -- FORCED_FILENAME_KEYWORDS / FORCED_FILENAME_EXCLUDE.
    2. Track-name keywords -- FORCED_TRACKNAME_KEYWORDS, minus FORCED_TRACKNAME_EXCLUDE.
    3. Statistical analysis:
       a. Exclude tracks with language 'chi' and tracks with 'sdh' in the title from the comparison.
       b. Split the remaining tracks into two groups:
          - binary (PGS, VOBSUB): compared by num_frames
          - text (SRT, ASS): compared by num_captions
       c. Within each group, find the per-language max of the metric.
          A track with < 50% of the max for its language is marked forced.
       d. Use both metrics (num_frames AND num_captions) when both are available;
          either one falling below 50% is enough to mark the track forced.
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
    stat_tracks = [t for t in subtitle_tracks if t.language != "chi" and "sdh" not in t.title.lower()]

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


_HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})


def hdr_transfer_for_cropdetect(color_transfer: str | None) -> str | None:
    """Return the transfer string when cropdetect needs HDR tonemapping.

    Maps PQ ('smpte2084') and HLG ('arib-std-b67') through unchanged so the
    adapter can plug them straight into ``zscale=tin=...``. Anything else
    (including ``None``) returns ``None`` -- SDR path unchanged.
    """
    return color_transfer if color_transfer in _HDR_TRANSFERS else None


CROP_EDGE_TOLERANCE = 8
"""Pixels: cropdetect's per-edge jitter merged into a single cluster.

Comfortably above the +-2px centering wobble seen in practice yet well below
``round=16`` (the cropdetect rounding), so genuinely distinct crops are never
merged."""


def _dominant_edge(values: list[int], tolerance: int) -> int:
    """Median of the largest cluster of *values* within +-``tolerance``.

    A cluster is every value within ``tolerance`` of an anchor; the anchor
    whose cluster is largest wins (first-seen breaks ties), and its members'
    upper median is returned. This is a 1-D mode: the consensus position the
    most samples agree on, robust to a scatter of outliers on either side.

    On exact ties the returned median can depend on input order, since a
    different anchor's cluster (with different members) may win. That is
    harmless here only because a real crop edge is constant across well-lit
    samples and so forms the single tightest, largest cluster, out-voting any
    gradient of scattered over-/under-crops.
    """
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
    """Combine per-sample cropdetect results into a single crop rectangle.

    cropdetect is noisy in two ways: dark scenes make it *over-crop* (a black
    bar is always black, but transiently dark content shrinks the detected
    picture), while stray bright pixels/logos in a bar make it *under-crop*.
    Both are minorities scattered around a consensus: the true picture edge is
    where the many well-lit samples agree.

    So each content-box edge (left=x, right=x+w, top=y, bottom=y+h) is reduced
    independently to the median of its densest cluster (see ``_dominant_edge``).
    Per-edge decomposition means a noisy axis (letterbox flicker) cannot drag a
    rock-solid axis (a constant pillarbox); taking the cluster *mode* rather
    than the plain median means a dark-majority episode -- where over-crops
    outnumber true samples but never agree with each other -- still resolves to
    the true edge.

    Raises ``ValueError`` if the dominant edges invert (left past right, or top
    past bottom): the independent per-edge medians carry no joint invariant, so
    a pathological set of wildly inconsistent samples could do this. Real
    cropdetect output (stable bars, ``x+w <= width``, w/h a multiple of 16)
    never does -- and the planner catches the ValueError and treats it as "no
    reliable crop" rather than letting a degenerate rectangle reach the encoder.

    Requires a non-empty list.
    """
    left = _dominant_edge([c.x for c in crops], tolerance)
    right = _dominant_edge([c.x + c.w for c in crops], tolerance)
    top = _dominant_edge([c.y for c in crops], tolerance)
    bottom = _dominant_edge([c.y + c.h for c in crops], tolerance)
    if right < left or bottom < top:
        raise ValueError(
            f"cropdetect samples too inconsistent to crop: "
            f"x {left}..{right}, y {top}..{bottom}",
        )
    return CropRect(w=right - left, h=bottom - top, x=left, y=top)


_INTERLACED_FIELD_ORDERS = {"tt", "bb"}
_TV_FPS_THRESHOLD = 48.0
_IDET_INTERLACE_THRESHOLD = 0.05


def _is_hd(height: int) -> bool:
    """True when the frame is HD (height >= 720).

    For HD, a tt/bb field_order is trusted outright and idet is never consulted.
    ``needs_idet`` and ``should_deinterlace`` must stay in lockstep on this: the
    former skips idet for HD (leaving idet_ratio at its 0.0 default), the latter
    ignores idet_ratio for HD — so the unmeasured ratio can never leak into a
    decision. Route both through this helper so the coupling can't silently drift.
    """
    return height >= _HD_MIN_HEIGHT


def needs_idet(field_order: str | None, fps: float, height: int) -> bool:
    """Determine if idet analysis is needed to confirm interlace.

    Returns False (no idet) when:
    - field_order is not tt/bb → clearly progressive
    - field_order is tt/bb but fps >= 48 → clearly TV interlace (field rate reported)
    - field_order is tt/bb but HD (height >= 720) → genuine HD interlace; soft
      telecine is an SD phenomenon, so the flag is authoritative and idet (which
      under-counts combing on low-motion HD) would only mislead.
    Returns True only when field_order is tt/bb, fps < 48 AND SD (height < 720)
    → ambiguous (DVD soft telecine vs real SD interlace), so idet must decide.
    """
    if field_order not in _INTERLACED_FIELD_ORDERS:
        return False
    if fps >= _TV_FPS_THRESHOLD:
        return False
    return not _is_hd(height)


def should_deinterlace(field_order: str | None, fps: float, idet_ratio: float, height: int) -> bool:
    """Decide whether to deinterlace based on ffprobe metadata and idet result.

    - field_order not tt/bb → progressive
    - field_order tt/bb + fps >= 48 → TV interlace (field rate), always deinterlace
    - field_order tt/bb + HD (height >= 720) → genuine HD interlace, always
      deinterlace. 1080i25 broadcast reports frame rate (25) not field rate (50),
      so the fps shortcut misses it, and idet under-counts combing on low-motion
      HD drama. HD soft telecine does not exist, so the tt/bb flag is trusted.
    - field_order tt/bb + SD + fps < 48 → idet decides (>5% interlaced → deinterlace)

    Tradeoff: the one HD case this loses on is progressive-segmented-frame (PsF)
    film mis-flagged tt/bb — it will be force-deinterlaced (nnedi discards a field
    and interpolates, a quality loss on truly progressive frames). This is
    intentional and idet cannot resolve it (PsF and low-motion 1080i both show
    near-zero combing, so they are metadata-indistinguishable). Real interlace
    vastly outnumbers mis-flagged PsF in this tool's broadcast/disc domain, and a
    user can override ``deinterlace`` to false in the plan JSON when it matters.
    """
    if field_order not in _INTERLACED_FIELD_ORDERS:
        return False
    if fps >= _TV_FPS_THRESHOLD or _is_hd(height):
        return True
    return idet_ratio > _IDET_INTERLACE_THRESHOLD


def _fraction_numerator(val: str) -> str:
    """Extract numerator from fraction string. '8500/50000' -> '8500'. No-op for non-fractions."""
    s = str(val)
    if "/" in s:
        return s.split("/", 1)[0]
    return s


def detect_hdr(stream_data: dict[str, Any], side_data: list[dict[str, Any]] | None) -> HdrMetadata:
    """Parse ffprobe side_data_list for MDCV and CLL.

    Also inspects codec_name for Dolby Vision (dvhe/dvh1) and detects
    HDR10+ by the presence of dynamic metadata in side_data.
    """
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

        elif side_type == "DOVI configuration record":
            # Stream (packet) level — HEVC dvcC box. Carries dv_profile and compat id.
            is_dolby_vision = True
            raw_profile = entry.get("dv_profile")
            if raw_profile is not None:
                dv_profile = int(raw_profile)
            raw_compat = entry.get("dv_bl_signal_compatibility_id")
            if raw_compat is not None:
                with contextlib.suppress(ValueError):
                    dv_bl_compatibility = DvBlCompatibility(int(raw_compat))

        elif side_type in ("Dolby Vision RPU Data", "Dolby Vision Metadata"):
            # Frame-level markers (no profile info).
            is_dolby_vision = True

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
    *,
    force: bool = False,
) -> tuple[bool, str]:
    """Return (skip, reason). Skip if:
    - output_path already exists, or
    - encoder_tag starts with 'Furnace'.

    When ``force`` is True, never skip (both conditions are bypassed).
    """
    if force:
        return False, ""
    if output_path.exists():
        return True, f"output file already exists: {output_path}"
    if encoder_tag is not None and encoder_tag.startswith("Furnace"):
        return True, f"file already encoded by Furnace (tag: {encoder_tag})"
    return False, ""


def check_unsupported_codecs(
    audio_tracks: list[Track],
    subtitle_tracks: list[Track],
) -> str | None:
    """Return a warning string if any unknown codecs are present, or None."""
    unknown: list[str] = [
        f"audio stream #{track.index} ({track.codec_name!r}, lang={track.language})"
        for track in audio_tracks
        if track.codec_id is AudioCodecId.UNKNOWN
    ]

    unknown.extend(
        f"subtitle stream #{track.index} ({track.codec_name!r}, lang={track.language})"
        for track in subtitle_tracks
        if track.codec_id is SubtitleCodecId.UNKNOWN
    )

    if unknown:
        items = ", ".join(unknown)
        return f"unsupported codecs detected: {items}"

    return None
