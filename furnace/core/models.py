from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audio_profile import AudioProfile
from .downmix import STEREO_CHANNELS, SURROUND_5_1_CHANNELS, THREE_CHANNELS, DownmixMode

__all__ = [
    "STEREO_CHANNELS",
    "SURROUND_5_1_CHANNELS",
    "THREE_CHANNELS",
    "AnalysisOutcome",
    "AnalyzeStatus",
    "Attachment",
    "AudioAction",
    "AudioCodecId",
    "AudioInstruction",
    "AudioProfile",
    "CropRect",
    "DiscSource",
    "DiscTitle",
    "DiscType",
    "DownmixMode",
    "DvBlCompatibility",
    "DvMode",
    "EncodeResult",
    "HdrMetadata",
    "Job",
    "JobStatus",
    "MetricPool",
    "MetricScores",
    "Movie",
    "Plan",
    "ScanResult",
    "SubtitleAction",
    "SubtitleCodecId",
    "SubtitleInstruction",
    "Track",
    "TrackType",
    "VideoInfo",
    "VideoParams",
]


class TrackType(enum.Enum):
    VIDEO = "video"
    AUDIO = "audio"
    SUBTITLE = "subtitle"


class AudioCodecId(enum.Enum):
    AAC_LC = "aac_lc"
    AAC_HE = "aac_he"
    AAC_HE_V2 = "aac_he_v2"
    AC3 = "ac3"
    EAC3 = "eac3"
    DTS = "dts"
    DTS_ES = "dts_es"
    DTS_HRA = "dts_hra"
    DTS_MA = "dts_ma"
    TRUEHD = "truehd"
    FLAC = "flac"
    PCM_S16LE = "pcm_s16le"
    PCM_S24LE = "pcm_s24le"
    PCM_S16BE = "pcm_s16be"
    MP2 = "mp2"
    MP3 = "mp3"
    VORBIS = "vorbis"
    OPUS = "opus"
    WMA_V2 = "wmav2"
    WMA_PRO = "wmapro"
    AMR = "amr_nb"
    UNKNOWN = "unknown"


class SubtitleCodecId(enum.Enum):
    SRT = "subrip"
    ASS = "ass"
    PGS = "hdmv_pgs_subtitle"
    VOBSUB = "dvd_subtitle"
    UNKNOWN = "unknown"


class AudioAction(enum.Enum):
    COPY = "copy"
    DENORM = "denorm"
    DECODE_ENCODE = "decode_encode"
    FFMPEG_ENCODE = "ffmpeg_encode"


class SubtitleAction(enum.Enum):
    COPY = "copy"
    COPY_RECODE = "copy_recode"


class JobStatus(enum.Enum):
    PENDING = "pending"
    DONE = "done"
    ERROR = "error"


class AnalyzeStatus(enum.Enum):
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


class DvBlCompatibility(enum.IntEnum):
    NONE = 0
    HDR10 = 1
    SDR = 2
    HLG = 4


class DvMode(enum.IntEnum):
    COPY = 0
    TO_8_1 = 2


class DiscType(enum.Enum):
    DVD = "dvd"
    BLURAY = "bluray"


@dataclass(frozen=True)
class HdrMetadata:
    mastering_display: str | None = None
    content_light: str | None = None
    is_dolby_vision: bool = False
    is_hdr10_plus: bool = False
    dv_profile: int | None = None
    dv_bl_compatibility: DvBlCompatibility | None = None


@dataclass(frozen=True)
class CropRect:
    w: int
    h: int
    x: int
    y: int


@dataclass(frozen=True)
class EncodeResult:
    return_code: int
    encoder_settings: str


class MetricPool(enum.Enum):
    MEAN = "mean"
    LOW = "low"


@dataclass(frozen=True)
class MetricScores:
    ssimulacra2: float | None = None
    butteraugli: float | None = None
    cvvdp: float | None = None


METRIC_NAMES: frozenset[str] = frozenset({"ssimulacra2", "butteraugli", "cvvdp"})


@dataclass(frozen=True)
class ScanResult:
    main_file: Path
    satellite_files: list[Path]
    output_path: Path


@dataclass(frozen=True)
class DiscSource:
    path: Path
    disc_type: DiscType


@dataclass(frozen=True)
class DiscTitle:
    number: int
    duration_s: float
    raw_label: str


@dataclass
class Track:
    index: int
    track_type: TrackType
    codec_name: str
    codec_id: AudioCodecId | SubtitleCodecId | None
    language: str
    title: str
    is_default: bool
    is_forced: bool
    source_file: Path

    channels: int | None = None
    channel_layout: str | None = None
    bitrate: int | None = None
    sample_rate: int | None = None
    delay_ms: int = 0
    profile: str | None = None

    num_frames: int | None = None
    num_captions: int | None = None
    encoding: str | None = None

    audio_profile: AudioProfile | None = None


@dataclass
class VideoInfo:
    index: int
    codec_name: str
    width: int
    height: int
    pixel_area: int
    fps_num: int
    fps_den: int
    duration_s: float
    interlaced: bool
    color_matrix_raw: str | None
    color_range: str | None
    color_transfer: str | None
    color_primaries: str | None
    pix_fmt: str
    hdr: HdrMetadata
    source_file: Path
    bitrate: int = 0
    sar_num: int = 1
    sar_den: int = 1
    grainy: bool = False


@dataclass
class Attachment:
    filename: str
    mime_type: str
    source_file: Path
    stream_index: int = -1


@dataclass
class Movie:
    main_file: Path
    satellite_files: list[Path]
    video: VideoInfo
    audio_tracks: list[Track]
    subtitle_tracks: list[Track]
    attachments: list[Attachment]
    has_chapters: bool
    file_size: int


@dataclass(frozen=True)
class AnalysisOutcome:
    movie: Movie | None
    status: AnalyzeStatus
    detail: str


@dataclass
class AudioInstruction:
    source_file: str
    stream_index: int
    language: str
    action: AudioAction
    delay_ms: int
    is_default: bool
    codec_name: str
    channels: int | None
    bitrate: int | None
    downmix: DownmixMode | None = None


@dataclass
class SubtitleInstruction:
    source_file: str
    stream_index: int
    language: str
    action: SubtitleAction
    is_default: bool
    is_forced: bool
    codec_name: str
    source_encoding: str | None


@dataclass
class VideoParams:
    cq: int
    crop: CropRect | None
    deinterlace: bool
    color_matrix: str
    color_range: str
    color_transfer: str
    color_primaries: str
    hdr: HdrMetadata | None
    gop: int
    fps_num: int
    fps_den: int
    source_width: int
    source_height: int
    source_codec: str = ""
    source_bitrate: int = 0
    sar_num: int = 1
    sar_den: int = 1
    dv_mode: DvMode | None = None
    passthrough: bool = False
    grain: bool = False


@dataclass
class Job:
    id: str
    source_files: list[str]
    output_file: str
    video_params: VideoParams
    audio: list[AudioInstruction]
    subtitles: list[SubtitleInstruction]
    attachments: list[dict[str, Any]]
    copy_chapters: bool
    chapters_source: str | None
    status: JobStatus = JobStatus.PENDING
    error: str | None = None
    source_size: int = 0
    output_size: int | None = None
    duration_s: float = 0.0
    chosen_cq: int | None = None


@dataclass
class Plan:
    version: str
    furnace_version: str
    created_at: str
    source: str
    destination: str
    demux_dir: str | None = None
    jobs: list[Job] = field(default_factory=list)
