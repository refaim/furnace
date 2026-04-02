from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import AudioCodecId, FieldOrder, HdrMetadata, SubtitleCodecId, Track

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


def detect_interlace(field_order_raw: str | None) -> FieldOrder:
    """'tt' -> TFF, 'bb' -> BFF, всё остальное -> PROGRESSIVE."""
    if field_order_raw == "tt":
        return FieldOrder.TFF
    if field_order_raw == "bb":
        return FieldOrder.BFF
    return FieldOrder.PROGRESSIVE


def detect_hdr(stream_data: dict[str, Any], side_data: list[dict[str, Any]] | None) -> HdrMetadata:
    """Анализирует side_data_list из ffprobe для MDCV и CLL.
    Проверяет codec_name для Dolby Vision (dvhe/dvh1) и
    HDR10+ (наличие dynamic metadata в side_data)."""
    mastering_display: str | None = None
    content_light: str | None = None
    is_dolby_vision: bool = False
    is_hdr10_plus: bool = False

    sd = side_data or []

    for entry in sd:
        side_type = entry.get("side_data_type", "")

        if "Mastering display metadata" in side_type:
            # Build MDCV string from ffprobe fields
            mastering_display = (
                f"G({entry.get('green_x', '')},{entry.get('green_y', '')})"
                f"B({entry.get('blue_x', '')},{entry.get('blue_y', '')})"
                f"R({entry.get('red_x', '')},{entry.get('red_y', '')})"
                f"WP({entry.get('white_point_x', '')},{entry.get('white_point_y', '')})"
                f"L({entry.get('max_luminance', '')},{entry.get('min_luminance', '')})"
            )

        elif "Content light level metadata" in side_type:
            max_cll = entry.get("max_content", "")
            max_fall = entry.get("max_average", "")
            content_light = f"MaxCLL={max_cll},MaxFALL={max_fall}"

        elif "Dolby Vision configuration" in side_type:
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
    )


def should_skip_file(
    output_path: Path,
    encoder_tag: str | None,
) -> tuple[bool, str]:
    """Возвращает (skip, reason). Skip если:
    - output_path существует
    - encoder_tag начинается с 'Furnace/'
    """
    if output_path.exists():
        return True, f"output file already exists: {output_path}"
    if encoder_tag is not None and encoder_tag.startswith("Furnace/"):
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
