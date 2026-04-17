"""Shared test factories for furnace test suite.

Every factory produces a valid object with sensible defaults.
All parameters are keyword-only so call-sites are self-documenting.
"""
from __future__ import annotations

from pathlib import Path

from furnace.core.models import (
    Attachment,
    AudioAction,
    AudioCodecId,
    AudioInstruction,
    CropRect,
    DownmixMode,
    DvMode,
    HdrMetadata,
    Job,
    JobStatus,
    Movie,
    Plan,
    SubtitleAction,
    SubtitleCodecId,
    SubtitleInstruction,
    Track,
    TrackType,
    VideoInfo,
    VideoParams,
)

# ---------------------------------------------------------------------------
# VideoInfo
# ---------------------------------------------------------------------------

def make_video_info(
    *,
    index: int = 0,
    codec_name: str = "h264",
    width: int = 1920,
    height: int = 1080,
    pixel_area: int | None = None,
    fps_num: int = 24,
    fps_den: int = 1,
    duration_s: float = 5400.0,
    interlaced: bool = False,
    color_matrix_raw: str | None = "bt709",
    color_range: str | None = "tv",
    color_transfer: str | None = "bt709",
    color_primaries: str | None = "bt709",
    pix_fmt: str = "yuv420p",
    hdr: HdrMetadata | None = None,
    source_file: Path | None = None,
    bitrate: int = 0,
    sar_num: int = 1,
    sar_den: int = 1,
) -> VideoInfo:
    return VideoInfo(
        index=index,
        codec_name=codec_name,
        width=width,
        height=height,
        pixel_area=pixel_area if pixel_area is not None else width * height,
        fps_num=fps_num,
        fps_den=fps_den,
        duration_s=duration_s,
        interlaced=interlaced,
        color_matrix_raw=color_matrix_raw,
        color_range=color_range,
        color_transfer=color_transfer,
        color_primaries=color_primaries,
        pix_fmt=pix_fmt,
        hdr=hdr if hdr is not None else HdrMetadata(),
        source_file=source_file if source_file is not None else Path("/src/movie.mkv"),
        bitrate=bitrate,
        sar_num=sar_num,
        sar_den=sar_den,
    )


# ---------------------------------------------------------------------------
# VideoParams
# ---------------------------------------------------------------------------

def make_video_params(
    *,
    cq: int = 25,
    crop: CropRect | None = None,
    deinterlace: bool = False,
    color_matrix: str = "bt709",
    color_range: str = "tv",
    color_transfer: str = "bt709",
    color_primaries: str = "bt709",
    hdr: HdrMetadata | None = None,
    gop: int = 120,
    fps_num: int = 24,
    fps_den: int = 1,
    source_width: int = 1920,
    source_height: int = 1080,
    source_codec: str = "",
    source_bitrate: int = 0,
    sar_num: int = 1,
    sar_den: int = 1,
    dv_mode: DvMode | None = None,
) -> VideoParams:
    return VideoParams(
        cq=cq,
        crop=crop,
        deinterlace=deinterlace,
        color_matrix=color_matrix,
        color_range=color_range,
        color_transfer=color_transfer,
        color_primaries=color_primaries,
        hdr=hdr,
        gop=gop,
        fps_num=fps_num,
        fps_den=fps_den,
        source_width=source_width,
        source_height=source_height,
        source_codec=source_codec,
        source_bitrate=source_bitrate,
        sar_num=sar_num,
        sar_den=sar_den,
        dv_mode=dv_mode,
    )


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------

def make_track(
    *,
    index: int = 0,
    track_type: TrackType = TrackType.AUDIO,
    codec_name: str = "aac",
    codec_id: AudioCodecId | SubtitleCodecId | None = AudioCodecId.AAC_LC,
    language: str = "eng",
    title: str = "",
    is_default: bool = False,
    is_forced: bool = False,
    source_file: Path | None = None,
    channels: int | None = 2,
    channel_layout: str | None = None,
    bitrate: int | None = None,
    sample_rate: int | None = None,
    delay_ms: int = 0,
    profile: str | None = None,
    num_frames: int | None = None,
    num_captions: int | None = None,
    encoding: str | None = None,
) -> Track:
    return Track(
        index=index,
        track_type=track_type,
        codec_name=codec_name,
        codec_id=codec_id,
        language=language,
        title=title,
        is_default=is_default,
        is_forced=is_forced,
        source_file=source_file if source_file is not None else Path("/src/movie.mkv"),
        channels=channels,
        channel_layout=channel_layout,
        bitrate=bitrate,
        sample_rate=sample_rate,
        delay_ms=delay_ms,
        profile=profile,
        num_frames=num_frames,
        num_captions=num_captions,
        encoding=encoding,
    )


# ---------------------------------------------------------------------------
# AudioInstruction
# ---------------------------------------------------------------------------

def make_audio_instruction(
    *,
    source_file: str = "/src/movie.mkv",
    stream_index: int = 1,
    language: str = "eng",
    action: AudioAction = AudioAction.COPY,
    delay_ms: int = 0,
    is_default: bool = True,
    codec_name: str = "aac",
    channels: int | None = 2,
    bitrate: int | None = 192000,
    downmix: DownmixMode | None = None,
) -> AudioInstruction:
    return AudioInstruction(
        source_file=source_file,
        stream_index=stream_index,
        language=language,
        action=action,
        delay_ms=delay_ms,
        is_default=is_default,
        codec_name=codec_name,
        channels=channels,
        bitrate=bitrate,
        downmix=downmix,
    )


# ---------------------------------------------------------------------------
# SubtitleInstruction
# ---------------------------------------------------------------------------

def make_subtitle_instruction(
    *,
    source_file: str = "/src/movie.mkv",
    stream_index: int = 2,
    language: str = "eng",
    action: SubtitleAction = SubtitleAction.COPY,
    is_default: bool = False,
    is_forced: bool = False,
    codec_name: str = "hdmv_pgs_subtitle",
    source_encoding: str | None = None,
) -> SubtitleInstruction:
    return SubtitleInstruction(
        source_file=source_file,
        stream_index=stream_index,
        language=language,
        action=action,
        is_default=is_default,
        is_forced=is_forced,
        codec_name=codec_name,
        source_encoding=source_encoding,
    )


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

def make_job(
    *,
    job_id: str = "test-job-001",
    source_files: list[str] | None = None,
    output_file: str = "/out/movie.mkv",
    video_params: VideoParams | None = None,
    audio: list[AudioInstruction] | None = None,
    subtitles: list[SubtitleInstruction] | None = None,
    attachments: list[dict[str, str]] | None = None,
    copy_chapters: bool = True,
    chapters_source: str | None = None,
    status: JobStatus = JobStatus.PENDING,
    error: str | None = None,
    vmaf_score: float | None = None,
    ssim_score: float | None = None,
    source_size: int = 1_000_000,
    output_size: int | None = None,
    duration_s: float = 0.0,
) -> Job:
    return Job(
        id=job_id,
        source_files=source_files if source_files is not None else ["/src/movie.mkv"],
        output_file=output_file,
        video_params=video_params if video_params is not None else make_video_params(),
        audio=audio if audio is not None else [make_audio_instruction()],
        subtitles=subtitles if subtitles is not None else [make_subtitle_instruction()],
        attachments=attachments if attachments is not None else [],
        copy_chapters=copy_chapters,
        chapters_source=chapters_source,
        status=status,
        error=error,
        vmaf_score=vmaf_score,
        ssim_score=ssim_score,
        source_size=source_size,
        output_size=output_size,
        duration_s=duration_s,
    )


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

def make_plan(
    *,
    version: str = "2",
    furnace_version: str = "0.1.0",
    created_at: str = "2026-04-01T00:00:00",
    source: str = "/src",
    destination: str = "/out",
    vmaf_enabled: bool = False,
    demux_dir: str | None = None,
    jobs: list[Job] | None = None,
) -> Plan:
    return Plan(
        version=version,
        furnace_version=furnace_version,
        created_at=created_at,
        source=source,
        destination=destination,
        vmaf_enabled=vmaf_enabled,
        demux_dir=demux_dir,
        jobs=jobs if jobs is not None else [make_job()],
    )


# ---------------------------------------------------------------------------
# Movie
# ---------------------------------------------------------------------------

def make_movie(
    *,
    main_file: Path | None = None,
    satellite_files: list[Path] | None = None,
    video: VideoInfo | None = None,
    audio_tracks: list[Track] | None = None,
    subtitle_tracks: list[Track] | None = None,
    attachments: list[Attachment] | None = None,
    has_chapters: bool = False,
    file_size: int = 0,
) -> Movie:
    return Movie(
        main_file=main_file if main_file is not None else Path("/src/movie.mkv"),
        satellite_files=satellite_files if satellite_files is not None else [],
        video=video if video is not None else make_video_info(),
        audio_tracks=audio_tracks if audio_tracks is not None else [],
        subtitle_tracks=subtitle_tracks if subtitle_tracks is not None else [],
        attachments=attachments if attachments is not None else [],
        has_chapters=has_chapters,
        file_size=file_size,
    )
