from __future__ import annotations

from .models import AudioAction, AudioCodecId, SubtitleAction, SubtitleCodecId

AUDIO_CODEC_ACTIONS: dict[AudioCodecId, AudioAction] = {
    AudioCodecId.AAC_LC: AudioAction.COPY,
    AudioCodecId.AAC_HE: AudioAction.COPY,
    AudioCodecId.AAC_HE_V2: AudioAction.COPY,
    AudioCodecId.AC3: AudioAction.DENORM,
    AudioCodecId.EAC3: AudioAction.DENORM,
    AudioCodecId.DTS: AudioAction.DENORM,
    AudioCodecId.DTS_ES: AudioAction.DECODE_ENCODE,
    AudioCodecId.DTS_HRA: AudioAction.DECODE_ENCODE,
    AudioCodecId.DTS_MA: AudioAction.DECODE_ENCODE,
    AudioCodecId.TRUEHD: AudioAction.DECODE_ENCODE,
    AudioCodecId.FLAC: AudioAction.DECODE_ENCODE,
    AudioCodecId.PCM_S16LE: AudioAction.DECODE_ENCODE,
    AudioCodecId.PCM_S24LE: AudioAction.DECODE_ENCODE,
    AudioCodecId.PCM_S16BE: AudioAction.DECODE_ENCODE,
    AudioCodecId.MP2: AudioAction.FFMPEG_ENCODE,
    AudioCodecId.MP3: AudioAction.FFMPEG_ENCODE,
    AudioCodecId.VORBIS: AudioAction.FFMPEG_ENCODE,
    AudioCodecId.OPUS: AudioAction.FFMPEG_ENCODE,
    AudioCodecId.WMA_V2: AudioAction.FFMPEG_ENCODE,
    AudioCodecId.WMA_PRO: AudioAction.FFMPEG_ENCODE,
    AudioCodecId.AMR: AudioAction.FFMPEG_ENCODE,
}

SUBTITLE_CODEC_ACTIONS: dict[SubtitleCodecId, SubtitleAction] = {
    SubtitleCodecId.SRT: SubtitleAction.COPY_RECODE,
    SubtitleCodecId.ASS: SubtitleAction.COPY_RECODE,
    SubtitleCodecId.PGS: SubtitleAction.COPY,
    SubtitleCodecId.VOBSUB: SubtitleAction.COPY,
}

_SUBTITLE_CODEC_MAP: dict[str, SubtitleCodecId] = {
    codec.value: codec for codec in SubtitleCodecId if codec is not SubtitleCodecId.UNKNOWN
}


def get_audio_action(codec_id: AudioCodecId) -> AudioAction | None:
    return AUDIO_CODEC_ACTIONS.get(codec_id)


def get_subtitle_action(codec_id: SubtitleCodecId) -> SubtitleAction | None:
    return SUBTITLE_CODEC_ACTIONS.get(codec_id)


def is_known_audio_codec(codec_id: AudioCodecId) -> bool:
    return codec_id in AUDIO_CODEC_ACTIONS


def is_known_subtitle_codec(codec_id: SubtitleCodecId) -> bool:
    return codec_id in SUBTITLE_CODEC_ACTIONS


def parse_audio_codec(codec_name: str, profile: str | None) -> AudioCodecId:
    if codec_name == "dts":
        match profile:
            case "DTS-HD MA":
                return AudioCodecId.DTS_MA
            case "DTS-HD HRA":
                return AudioCodecId.DTS_HRA
            case "DTS-ES":
                return AudioCodecId.DTS_ES
            case "DTS" | None:
                return AudioCodecId.DTS
            case _:
                return AudioCodecId.DTS

    if codec_name == "aac":
        match profile:
            case "HE-AAC":
                return AudioCodecId.AAC_HE
            case "HE-AAC v2":
                return AudioCodecId.AAC_HE_V2
            case _:
                return AudioCodecId.AAC_LC

    audio_codec_name_map: dict[str, AudioCodecId] = {
        "ac3": AudioCodecId.AC3,
        "eac3": AudioCodecId.EAC3,
        "truehd": AudioCodecId.TRUEHD,
        "flac": AudioCodecId.FLAC,
        "pcm_s16le": AudioCodecId.PCM_S16LE,
        "pcm_s24le": AudioCodecId.PCM_S24LE,
        "pcm_s16be": AudioCodecId.PCM_S16BE,
        "mp2": AudioCodecId.MP2,
        "mp3": AudioCodecId.MP3,
        "vorbis": AudioCodecId.VORBIS,
        "opus": AudioCodecId.OPUS,
        "wmav2": AudioCodecId.WMA_V2,
        "wmapro": AudioCodecId.WMA_PRO,
        "amr_nb": AudioCodecId.AMR,
    }
    return audio_codec_name_map.get(codec_name, AudioCodecId.UNKNOWN)


def parse_subtitle_codec(codec_name: str) -> SubtitleCodecId:
    return _SUBTITLE_CODEC_MAP.get(codec_name, SubtitleCodecId.UNKNOWN)
