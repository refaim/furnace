from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path


class TrackType(enum.Enum):
    VIDEO = "video"
    AUDIO = "audio"
    SUBTITLE = "subtitle"


class AudioCodecId(enum.Enum):
    """Идентификаторы аудиокодеков из ffprobe codec_name + profile."""
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
    """Действие над аудиодорожкой при обработке."""
    COPY = "copy"                # AAC -- копировать как есть
    DENORM = "denorm"            # AC3/EAC3/DTS core -- eac3to denormalize
    DECODE_ENCODE = "decode_encode"  # lossless -> eac3to WAV -> qaac64 AAC
    FFMPEG_ENCODE = "ffmpeg_encode"  # экзотика -> ffmpeg WAV -> qaac64 AAC


class SubtitleAction(enum.Enum):
    COPY = "copy"                # PGS, VOBSUB -- бинарные
    COPY_RECODE = "copy_recode"  # SRT, ASS -- перекодировать в UTF-8


class JobStatus(enum.Enum):
    PENDING = "pending"
    DONE = "done"
    ERROR = "error"


class FieldOrder(enum.Enum):
    PROGRESSIVE = "progressive"
    TFF = "tt"   # top field first
    BFF = "bb"   # bottom field first


class ColorSpace(enum.Enum):
    BT601 = "bt601"
    BT709 = "bt709"
    BT2020 = "bt2020"


class DiscType(enum.Enum):
    DVD = "dvd"
    BLURAY = "bluray"


@dataclass(frozen=True)
class HdrMetadata:
    """HDR10 static metadata. None означает отсутствие."""
    mastering_display: str | None = None   # "G(...)B(...)R(...)WP(...)L(...)" строка MDCV
    content_light: str | None = None       # "MaxCLL=X,MaxFALL=Y"
    is_dolby_vision: bool = False
    is_hdr10_plus: bool = False


@dataclass(frozen=True)
class CropRect:
    """Значения crop после выравнивания по 16x8."""
    w: int
    h: int
    x: int
    y: int


@dataclass(frozen=True)
class ScanResult:
    """Результат сканирования одного видеофайла."""
    main_file: Path
    satellite_files: list[Path]
    output_path: Path


@dataclass(frozen=True)
class DiscSource:
    """A detected disc structure in the source directory."""
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
    """Одна дорожка из медиафайла."""
    index: int                         # stream index в ffprobe
    track_type: TrackType
    codec_name: str                    # raw codec_name из ffprobe
    codec_id: AudioCodecId | SubtitleCodecId | None  # parsed enum
    language: str                      # ISO 639-3 (rus, eng, und)
    title: str                         # название дорожки (может быть "")
    is_default: bool
    is_forced: bool
    source_file: Path                  # файл-источник (основной или satellite)

    # audio-specific
    channels: int | None = None        # количество каналов
    channel_layout: str | None = None  # e.g. "5.1(side)"
    bitrate: int | None = None         # bps
    sample_rate: int | None = None
    delay_ms: int = 0                  # из start_pts, в миллисекундах
    profile: str | None = None         # AAC profile (LC, HE, HE-AAC v2)

    # subtitle-specific
    num_frames: int | None = None      # для forced detection (frames/events count для binary subs)
    num_captions: int | None = None    # для forced detection (caption/event count для text subs)
    encoding: str | None = None        # определённая кодировка текстовых субтитров


@dataclass
class VideoInfo:
    """Информация о видеодорожке (всегда одна на файл)."""
    index: int
    codec_name: str                    # h264, hevc, mpeg2video, ...
    width: int
    height: int
    pixel_area: int                    # width * height (после crop будет пересчитан для CQ)
    fps_num: int
    fps_den: int
    duration_s: float
    field_order: FieldOrder
    color_space: ColorSpace | None
    color_range: str | None            # "tv" | "pc" | None
    color_transfer: str | None
    color_primaries: str | None
    pix_fmt: str                       # yuv420p, yuv420p10le, ...
    hdr: HdrMetadata
    source_file: Path
    bitrate: int = 0                   # video stream bitrate in bps
    sar_num: int = 1                   # sample aspect ratio numerator
    sar_den: int = 1                   # sample aspect ratio denominator


@dataclass
class Attachment:
    """Вложение (шрифт, изображение) из исходного MKV."""
    filename: str
    mime_type: str
    source_file: Path


@dataclass
class Movie:
    """Один фильм = основной видеофайл + satellite files."""
    main_file: Path
    satellite_files: list[Path]
    video: VideoInfo
    audio_tracks: list[Track]
    subtitle_tracks: list[Track]
    attachments: list[Attachment]
    has_chapters: bool
    file_size: int                     # размер основного файла в байтах


@dataclass
class AudioInstruction:
    """Инструкция обработки одной аудиодорожки в Job."""
    source_file: str                   # путь к файлу-источнику
    stream_index: int
    language: str
    action: AudioAction
    delay_ms: int
    is_default: bool
    codec_name: str                    # исходный кодек для информации
    channels: int | None
    bitrate: int | None


@dataclass
class SubtitleInstruction:
    """Инструкция обработки одной дорожки субтитров в Job."""
    source_file: str
    stream_index: int
    language: str
    action: SubtitleAction
    is_default: bool
    is_forced: bool
    codec_name: str
    source_encoding: str | None        # исходная кодировка (для перекодирования)


@dataclass
class VideoParams:
    """Параметры кодирования видео."""
    cq: int
    crop: CropRect | None              # None = без crop
    deinterlace: bool                  # нужен ли bwdif_cuda
    color_space: ColorSpace
    color_range: str                   # "tv" всегда
    color_transfer: str | None         # raw ffmpeg value для passthrough
    color_primaries: str | None
    hdr: HdrMetadata | None            # HDR metadata для passthrough
    gop: int                           # GOP size (fps * 5)
    fps_num: int
    fps_den: int
    source_width: int                  # до crop, для информации
    source_height: int
    source_codec: str = ""             # ffprobe codec_name (h264, hevc, mpeg2video...)
    source_bitrate: int = 0            # video stream bitrate in bps (from ffprobe)
    sar_num: int = 1                   # sample aspect ratio numerator
    sar_den: int = 1                   # sample aspect ratio denominator


@dataclass
class Job:
    """Одна задача = один выходной файл."""
    id: str                            # уникальный ID (uuid4)
    source_files: list[str]            # [main_file, *satellites]
    output_file: str
    video_params: VideoParams
    audio: list[AudioInstruction]
    subtitles: list[SubtitleInstruction]
    attachments: list[dict[str, str]]   # [{filename, mime_type, source_file}]
    copy_chapters: bool
    chapters_source: str | None        # путь к файлу с главами
    status: JobStatus = JobStatus.PENDING
    error: str | None = None
    vmaf_score: float | None = None
    source_size: int = 0               # размер исходного файла
    output_size: int | None = None     # размер выходного файла (после кодирования)


@dataclass
class Plan:
    """Весь план -- сериализуется в JSON."""
    version: str                       # "1"
    furnace_version: str               # "0.1.0"
    created_at: str                    # ISO datetime
    source: str                        # исходный путь/директория
    destination: str                   # выходная директория
    vmaf_enabled: bool
    demux_dir: str | None = None       # path to .furnace_demux/ or None
    jobs: list[Job] = field(default_factory=list)
