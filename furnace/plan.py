from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .core.models import (
    AudioAction,
    AudioInstruction,
    CropRect,
    DownmixMode,
    DvBlCompatibility,
    DvMode,
    HdrMetadata,
    Job,
    JobStatus,
    Plan,
    SubtitleAction,
    SubtitleInstruction,
    VideoParams,
)

PLAN_VERSION = "2"


class _PlanEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Path, Enum, and dataclass objects."""

    def default(self, obj: object) -> object:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, __import__("enum").Enum):
            return obj.value
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        return super().default(obj)


def atomic_write(path: Path, data: str) -> None:
    """Write data to a temp file in the same directory, then rename atomically."""
    dir_path = path.parent
    fd, tmp_path = tempfile.mkstemp(dir=str(dir_path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        Path(tmp_path).replace(path)
    except:
        with contextlib.suppress(OSError):
            Path(tmp_path).unlink()
        raise


def save_plan(plan: Plan, path: Path) -> None:
    """Serialize Plan to JSON using custom encoder. Pretty-print with indent=2."""
    data = json.dumps(dataclasses.asdict(plan), cls=_PlanEncoder, indent=2, ensure_ascii=False)
    atomic_write(path, data)


def _load_crop(raw: dict[str, Any] | None) -> CropRect | None:
    if raw is None:
        return None
    return CropRect(w=raw["w"], h=raw["h"], x=raw["x"], y=raw["y"])


def _load_hdr(raw: dict[str, Any] | None) -> HdrMetadata | None:
    if raw is None:
        return None
    dv_bl_compat: DvBlCompatibility | None = None
    raw_compat = raw.get("dv_bl_compatibility")
    if raw_compat is not None:
        with contextlib.suppress(ValueError, TypeError):
            dv_bl_compat = DvBlCompatibility(int(raw_compat))
    raw_profile = raw.get("dv_profile")
    dv_profile: int | None = int(raw_profile) if raw_profile is not None else None
    return HdrMetadata(
        mastering_display=raw.get("mastering_display"),
        content_light=raw.get("content_light"),
        is_dolby_vision=raw.get("is_dolby_vision", False),
        is_hdr10_plus=raw.get("is_hdr10_plus", False),
        dv_profile=dv_profile,
        dv_bl_compatibility=dv_bl_compat,
    )


def _load_video_params(raw: dict[str, Any]) -> VideoParams:
    dv_mode_raw = raw.get("dv_mode")
    dv_mode: DvMode | None = DvMode(dv_mode_raw) if dv_mode_raw is not None else None
    return VideoParams(
        cq=raw["cq"],
        crop=_load_crop(raw.get("crop")),
        deinterlace=raw["deinterlace"],
        color_matrix=raw["color_matrix"],
        color_range=raw["color_range"],
        color_transfer=raw["color_transfer"],
        color_primaries=raw["color_primaries"],
        hdr=_load_hdr(raw.get("hdr")),
        gop=raw["gop"],
        fps_num=raw["fps_num"],
        fps_den=raw["fps_den"],
        source_width=raw["source_width"],
        source_height=raw["source_height"],
        source_codec=raw.get("source_codec", ""),
        source_bitrate=raw.get("source_bitrate", 0),
        sar_num=raw.get("sar_num", 1),
        sar_den=raw.get("sar_den", 1),
        dv_mode=dv_mode,
    )


def _load_audio(raw: dict[str, Any]) -> AudioInstruction:
    downmix_raw = raw.get("downmix")
    downmix = DownmixMode(downmix_raw) if downmix_raw is not None else None
    return AudioInstruction(
        source_file=raw["source_file"],
        stream_index=raw["stream_index"],
        language=raw["language"],
        action=AudioAction(raw["action"]),
        delay_ms=raw["delay_ms"],
        is_default=raw["is_default"],
        codec_name=raw["codec_name"],
        channels=raw.get("channels"),
        bitrate=raw.get("bitrate"),
        downmix=downmix,
    )


def _load_subtitle(raw: dict[str, Any]) -> SubtitleInstruction:
    return SubtitleInstruction(
        source_file=raw["source_file"],
        stream_index=raw["stream_index"],
        language=raw["language"],
        action=SubtitleAction(raw["action"]),
        is_default=raw["is_default"],
        is_forced=raw["is_forced"],
        codec_name=raw["codec_name"],
        source_encoding=raw.get("source_encoding"),
    )


def _load_job(raw: dict[str, Any]) -> Job:
    return Job(
        id=raw["id"],
        source_files=raw["source_files"],
        output_file=raw["output_file"],
        video_params=_load_video_params(raw["video_params"]),
        audio=[_load_audio(a) for a in raw.get("audio", [])],
        subtitles=[_load_subtitle(s) for s in raw.get("subtitles", [])],
        attachments=raw.get("attachments", []),
        copy_chapters=raw["copy_chapters"],
        chapters_source=raw.get("chapters_source"),
        status=JobStatus(raw.get("status", "pending")),
        error=raw.get("error"),
        vmaf_score=raw.get("vmaf_score"),
        ssim_score=raw.get("ssim_score"),
        source_size=raw.get("source_size", 0),
        output_size=raw.get("output_size"),
        duration_s=raw.get("duration_s", 0.0),
    )


def load_plan(path: Path) -> Plan:
    """Read JSON, validate version, reconstruct all nested dataclasses."""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    version = raw.get("version")
    if version != PLAN_VERSION:
        raise ValueError(f"Unsupported plan version: {version!r} (expected {PLAN_VERSION!r})")

    return Plan(
        version=raw["version"],
        furnace_version=raw["furnace_version"],
        created_at=raw["created_at"],
        source=raw["source"],
        destination=raw["destination"],
        vmaf_enabled=raw["vmaf_enabled"],
        demux_dir=raw.get("demux_dir"),
        jobs=[_load_job(j) for j in raw.get("jobs", [])],
    )


def update_job_status(
    plan_path: Path,
    job_id: str,
    status: JobStatus,
    error: str | None = None,
    vmaf_score: float | None = None,
    ssim_score: float | None = None,
    output_size: int | None = None,
) -> None:
    """Read JSON, find job by id, update status fields, write back atomically."""
    with plan_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    found = False
    for job in raw.get("jobs", []):
        if job["id"] == job_id:
            job["status"] = status.value
            job["error"] = error
            if vmaf_score is not None:
                job["vmaf_score"] = vmaf_score
            if ssim_score is not None:
                job["ssim_score"] = ssim_score
            if output_size is not None:
                job["output_size"] = output_size
            found = True
            break

    if not found:
        raise KeyError(f"Job {job_id!r} not found in plan {plan_path}")

    data = json.dumps(raw, indent=2, ensure_ascii=False)
    atomic_write(plan_path, data)
