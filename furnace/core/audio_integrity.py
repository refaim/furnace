from __future__ import annotations

from typing import Any

_AUDIO_MIN_DURATION_RATIO = 0.97
_TIMESTAMP_PARTS = 3


def _parse_seconds(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0.0 else None


def _parse_timestamp(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    parts = value.split(":")
    if len(parts) != _TIMESTAMP_PARTS:
        return None
    try:
        hours, minutes, seconds = (float(part) for part in parts)
    except ValueError:
        return None
    total = hours * 3600.0 + minutes * 60.0 + seconds
    return total if total > 0.0 else None


def probe_audio_duration(
    probe: dict[str, Any],
    stream_index: int,
    *,
    allow_container_fallback: bool = True,
) -> float | None:
    for stream in probe.get("streams", []):
        if stream.get("index") == stream_index:
            direct = _parse_seconds(stream.get("duration"))
            if direct is not None:
                return direct
            tagged = _parse_timestamp((stream.get("tags") or {}).get("DURATION"))
            if tagged is not None:
                return tagged
            break
    if not allow_container_fallback:
        return None
    return _parse_seconds(probe.get("format", {}).get("duration"))


def audio_is_truncated(source_s: float, produced_s: float, *, ratio: float = _AUDIO_MIN_DURATION_RATIO) -> bool:
    return produced_s < source_s * ratio
