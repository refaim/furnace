from __future__ import annotations

import enum
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


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
    COPY = "copy"  # AAC -- copy as-is
    DENORM = "denorm"  # AC3/EAC3/DTS core -- eac3to denormalize
    DECODE_ENCODE = "decode_encode"  # lossless -> eac3to WAV -> qaac64 AAC
    FFMPEG_ENCODE = "ffmpeg_encode"  # exotic codecs -> ffmpeg WAV -> qaac64 AAC


# Channel-count landmarks referenced across planner/UI when deciding downmix.
STEREO_CHANNELS = 2  # a 2.0 track cannot be downmixed further
SURROUND_5_1_CHANNELS = 6  # DOWN6 only makes sense for >6 channels (7.1/6.1)


class DownmixMode(StrEnum):
    """Audio downmix target applied via eac3to flags.

    STEREO -> eac3to: -downStereo  (multichannel -> 2.0 AAC)
    DOWN6  -> eac3to: -down6       (7.1/6.1 -> 5.1 AAC)
    """

    STEREO = "stereo"
    DOWN6 = "down6"


class SubtitleAction(enum.Enum):
    COPY = "copy"  # PGS, VOBSUB -- binary, passthrough
    COPY_RECODE = "copy_recode"  # SRT, ASS -- recode to UTF-8


class JobStatus(enum.Enum):
    PENDING = "pending"
    DONE = "done"
    ERROR = "error"


class DvBlCompatibility(enum.IntEnum):
    """Dolby Vision base layer compatibility."""

    NONE = 0  # no fallback (Profile 5)
    HDR10 = 1  # HDR10 fallback
    SDR = 2  # SDR fallback
    HLG = 4  # HLG fallback


class DvMode(enum.IntEnum):
    """DV RPU extraction mode. Values match dovi_tool -m flag."""

    COPY = 0  # extract RPU as-is (no -m flag)
    TO_8_1 = 2  # convert P7 FEL -> P8.1 (-m 2)


class DiscType(enum.Enum):
    DVD = "dvd"
    BLURAY = "bluray"


@dataclass(frozen=True)
class HdrMetadata:
    mastering_display: str | None = None  # MDCV string: "G(...)B(...)R(...)WP(...)L(...)"
    content_light: str | None = None  # "MaxCLL=X,MaxFALL=Y"
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
    vmaf_score: float | None = None
    ssim_score: float | None = None


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
    """One playlist/VTS entry from eac3to listing."""

    number: int
    duration_s: float
    raw_label: str


@dataclass
class Track:
    index: int  # ffprobe stream index
    track_type: TrackType
    codec_name: str  # raw ffprobe codec_name
    codec_id: AudioCodecId | SubtitleCodecId | None  # parsed enum
    language: str  # ISO 639-3 (rus, eng, und)
    title: str
    is_default: bool
    is_forced: bool
    source_file: Path  # main file or a satellite

    # audio-specific
    channels: int | None = None
    channel_layout: str | None = None  # e.g. "5.1(side)"
    bitrate: int | None = None  # bps
    sample_rate: int | None = None
    delay_ms: int = 0  # derived from ffprobe start_pts
    profile: str | None = None  # AAC profile (LC, HE, HE-AAC v2)

    # subtitle-specific
    num_frames: int | None = None  # forced detection: frame count (binary subs)
    num_captions: int | None = None  # forced detection: caption count (text subs)
    encoding: str | None = None  # detected text-subtitle encoding


@dataclass
class VideoInfo:
    index: int
    codec_name: str  # h264, hevc, mpeg2video, ...
    width: int
    height: int
    pixel_area: int  # width * height, recalculated after crop for CQ
    fps_num: int
    fps_den: int
    duration_s: float
    interlaced: bool
    color_matrix_raw: str | None
    color_range: str | None  # "tv" | "pc" | None
    color_transfer: str | None
    color_primaries: str | None
    pix_fmt: str  # yuv420p, yuv420p10le, ...
    hdr: HdrMetadata
    source_file: Path
    bitrate: int = 0  # video stream bitrate in bps
    sar_num: int = 1  # sample aspect ratio numerator
    sar_den: int = 1  # sample aspect ratio denominator


@dataclass
class Attachment:
    filename: str
    mime_type: str
    source_file: Path


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


@dataclass
class AudioInstruction:
    source_file: str
    stream_index: int
    language: str
    action: AudioAction
    delay_ms: int
    is_default: bool
    codec_name: str  # source codec, informational
    channels: int | None
    bitrate: int | None
    downmix: DownmixMode | None = None  # None = no downmix applied


@dataclass
class SubtitleInstruction:
    source_file: str
    stream_index: int
    language: str
    action: SubtitleAction
    is_default: bool
    is_forced: bool
    codec_name: str
    source_encoding: str | None  # source encoding, for recoding


@dataclass
class VideoParams:
    cq: int
    crop: CropRect | None  # None = no crop
    deinterlace: bool
    color_matrix: str
    color_range: str  # always "tv"
    color_transfer: str
    color_primaries: str
    hdr: HdrMetadata | None  # HDR metadata for passthrough
    gop: int  # GOP size (fps * 5)
    fps_num: int
    fps_den: int
    source_width: int  # pre-crop, informational
    source_height: int
    source_codec: str = ""  # ffprobe codec_name (h264, hevc, mpeg2video...)
    source_bitrate: int = 0  # video stream bitrate in bps (from ffprobe)
    sar_num: int = 1  # sample aspect ratio numerator
    sar_den: int = 1  # sample aspect ratio denominator
    dv_mode: DvMode | None = None  # None=no DV, COPY=as-is, TO_8_1=P7->P8.1


@dataclass
class Job:
    id: str  # uuid4
    source_files: list[str]  # [main_file, *satellites]
    output_file: str
    video_params: VideoParams
    audio: list[AudioInstruction]
    subtitles: list[SubtitleInstruction]
    attachments: list[dict[str, str]]  # [{filename, mime_type, source_file}]
    copy_chapters: bool
    chapters_source: str | None  # path to chapters file
    status: JobStatus = JobStatus.PENDING
    error: str | None = None
    vmaf_score: float | None = None
    ssim_score: float | None = None
    source_size: int = 0
    output_size: int | None = None  # None until encoding completes
    duration_s: float = 0.0  # source video duration in seconds; 0.0 means unknown


@dataclass
class Plan:
    version: str  # "1"
    furnace_version: str  # "0.1.0"
    created_at: str  # ISO datetime
    source: str  # source path/directory
    destination: str  # output directory
    vmaf_enabled: bool
    demux_dir: str | None = None  # path to .furnace_demux/ or None
    jobs: list[Job] = field(default_factory=list)
