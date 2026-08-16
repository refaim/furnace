from __future__ import annotations

import pytest

from furnace.core.models import AudioAction, AudioCodecId, SubtitleAction, SubtitleCodecId
from furnace.core.rules import (
    _DTS_PROFILE_PREFIXES,
    get_audio_action,
    get_subtitle_action,
    is_known_audio_codec,
    is_known_subtitle_codec,
    parse_audio_codec,
    parse_subtitle_codec,
)


class TestParseAudioCodec:
    def test_dts_core_no_profile(self) -> None:
        assert parse_audio_codec("dts", None) == AudioCodecId.DTS

    # Every name ffmpeg can emit for a DCA stream (libavcodec ff_dca_profiles).
    # The "+ DTS:X" rows are the ones that used to be misread as lossy core and
    # copied verbatim, which is how a 4.7 GB lossless track survived an encode.
    @pytest.mark.parametrize(
        ("profile", "expected"),
        [
            ("DTS", AudioCodecId.DTS),
            ("DTS-ES", AudioCodecId.DTS_ES),
            ("DTS 96/24", AudioCodecId.DTS),
            ("DTS-HD HRA", AudioCodecId.DTS_HRA),
            ("DTS-HD MA", AudioCodecId.DTS_MA),
            ("DTS Express", AudioCodecId.DTS),
            ("DTS-HD MA + DTS:X", AudioCodecId.DTS_MA),
            ("DTS-HD MA + DTS:X IMAX", AudioCodecId.DTS_MA),
        ],
    )
    def test_every_ffmpeg_dts_profile_is_classified(
        self,
        profile: str,
        expected: AudioCodecId,
    ) -> None:
        assert parse_audio_codec("dts", profile) == expected

    @pytest.mark.parametrize(
        "profile",
        [
            "DTS-X",
            # Shares the bare-core prefix, so it must not fall through to DTS.
            "DTS-HD Ultra",
            # ffprobe prints the raw profile integer when libavcodec has no
            # ff_dca_profiles entry for it -- the realistic way a DTS variant
            # newer than the linked ffmpeg build shows up here. 63 is unused;
            # 62 would arrive named, as "DTS-HD MA + DTS:X IMAX".
            "63",
            "unknown",
        ],
    )
    def test_unrecognised_dts_profile_is_unknown(self, profile: str) -> None:
        assert parse_audio_codec("dts", profile) == AudioCodecId.UNKNOWN

    def test_dts_prefixes_do_not_shadow_each_other(self) -> None:
        prefixes = [prefix for prefix, _ in _DTS_PROFILE_PREFIXES]
        for outer in prefixes:
            shadowed = [inner for inner in prefixes if inner != outer and inner.startswith(outer)]
            assert not shadowed, f"{outer!r} shadows {shadowed!r}; matching order would matter"

    def test_aac_lc_no_profile(self) -> None:
        assert parse_audio_codec("aac", None) == AudioCodecId.AAC_LC

    def test_aac_lc_explicit(self) -> None:
        assert parse_audio_codec("aac", "LC") == AudioCodecId.AAC_LC

    def test_aac_he(self) -> None:
        assert parse_audio_codec("aac", "HE-AAC") == AudioCodecId.AAC_HE

    def test_aac_he_v2(self) -> None:
        assert parse_audio_codec("aac", "HE-AAC v2") == AudioCodecId.AAC_HE_V2

    def test_ac3(self) -> None:
        assert parse_audio_codec("ac3", None) == AudioCodecId.AC3

    def test_eac3(self) -> None:
        assert parse_audio_codec("eac3", None) == AudioCodecId.EAC3

    def test_truehd(self) -> None:
        assert parse_audio_codec("truehd", None) == AudioCodecId.TRUEHD

    def test_flac(self) -> None:
        assert parse_audio_codec("flac", None) == AudioCodecId.FLAC

    def test_pcm_s16le(self) -> None:
        assert parse_audio_codec("pcm_s16le", None) == AudioCodecId.PCM_S16LE

    def test_pcm_s24le(self) -> None:
        assert parse_audio_codec("pcm_s24le", None) == AudioCodecId.PCM_S24LE

    def test_pcm_s16be(self) -> None:
        assert parse_audio_codec("pcm_s16be", None) == AudioCodecId.PCM_S16BE

    def test_mp3(self) -> None:
        assert parse_audio_codec("mp3", None) == AudioCodecId.MP3

    def test_mp2(self) -> None:
        assert parse_audio_codec("mp2", None) == AudioCodecId.MP2

    def test_unknown_codec_name(self) -> None:
        assert parse_audio_codec("someweirdcodec", None) == AudioCodecId.UNKNOWN

    def test_empty_codec_name(self) -> None:
        assert parse_audio_codec("", None) == AudioCodecId.UNKNOWN


class TestAudioActionRouting:
    @pytest.mark.parametrize(
        "codec_id",
        [
            AudioCodecId.AAC_LC,
            AudioCodecId.AAC_HE,
            AudioCodecId.AAC_HE_V2,
        ],
    )
    def test_aac_copy(self, codec_id: AudioCodecId) -> None:
        assert get_audio_action(codec_id) == AudioAction.COPY

    @pytest.mark.parametrize(
        "codec_id",
        [
            AudioCodecId.AC3,
            AudioCodecId.EAC3,
            AudioCodecId.DTS,
        ],
    )
    def test_denorm(self, codec_id: AudioCodecId) -> None:
        assert get_audio_action(codec_id) == AudioAction.DENORM

    @pytest.mark.parametrize(
        "codec_id",
        [
            AudioCodecId.DTS_ES,
            AudioCodecId.DTS_HRA,
            AudioCodecId.DTS_MA,
            AudioCodecId.TRUEHD,
            AudioCodecId.FLAC,
            AudioCodecId.PCM_S16LE,
            AudioCodecId.PCM_S24LE,
            AudioCodecId.PCM_S16BE,
        ],
    )
    def test_decode_encode(self, codec_id: AudioCodecId) -> None:
        assert get_audio_action(codec_id) == AudioAction.DECODE_ENCODE

    @pytest.mark.parametrize(
        "codec_id",
        [
            AudioCodecId.MP2,
            AudioCodecId.MP3,
            AudioCodecId.VORBIS,
            AudioCodecId.OPUS,
            AudioCodecId.WMA_V2,
            AudioCodecId.WMA_PRO,
            AudioCodecId.AMR,
        ],
    )
    def test_ffmpeg_encode(self, codec_id: AudioCodecId) -> None:
        assert get_audio_action(codec_id) == AudioAction.FFMPEG_ENCODE

    def test_unknown_returns_none(self) -> None:
        assert get_audio_action(AudioCodecId.UNKNOWN) is None


class TestParseSubtitleCodec:
    def test_subrip(self) -> None:
        assert parse_subtitle_codec("subrip") == SubtitleCodecId.SRT

    def test_ass(self) -> None:
        assert parse_subtitle_codec("ass") == SubtitleCodecId.ASS

    def test_hdmv_pgs_subtitle(self) -> None:
        assert parse_subtitle_codec("hdmv_pgs_subtitle") == SubtitleCodecId.PGS

    def test_dvd_subtitle(self) -> None:
        assert parse_subtitle_codec("dvd_subtitle") == SubtitleCodecId.VOBSUB

    def test_unknown_returns_unknown(self) -> None:
        assert parse_subtitle_codec("webvtt") == SubtitleCodecId.UNKNOWN

    def test_empty_returns_unknown(self) -> None:
        assert parse_subtitle_codec("") == SubtitleCodecId.UNKNOWN


class TestKnownCodecChecks:
    def test_known_audio_codecs_are_known(self) -> None:
        known = [
            AudioCodecId.AAC_LC,
            AudioCodecId.AAC_HE,
            AudioCodecId.AAC_HE_V2,
            AudioCodecId.AC3,
            AudioCodecId.EAC3,
            AudioCodecId.DTS,
            AudioCodecId.DTS_ES,
            AudioCodecId.DTS_HRA,
            AudioCodecId.DTS_MA,
            AudioCodecId.TRUEHD,
            AudioCodecId.FLAC,
            AudioCodecId.PCM_S16LE,
            AudioCodecId.PCM_S24LE,
            AudioCodecId.PCM_S16BE,
            AudioCodecId.MP2,
            AudioCodecId.MP3,
            AudioCodecId.VORBIS,
            AudioCodecId.OPUS,
            AudioCodecId.WMA_V2,
            AudioCodecId.WMA_PRO,
            AudioCodecId.AMR,
        ]
        for codec in known:
            assert is_known_audio_codec(codec), f"{codec} should be known"

    def test_unknown_audio_is_not_known(self) -> None:
        assert not is_known_audio_codec(AudioCodecId.UNKNOWN)

    def test_known_subtitle_codecs_are_known(self) -> None:
        known = [SubtitleCodecId.SRT, SubtitleCodecId.ASS, SubtitleCodecId.PGS, SubtitleCodecId.VOBSUB]
        for codec in known:
            assert is_known_subtitle_codec(codec), f"{codec} should be known"

    def test_unknown_subtitle_is_not_known(self) -> None:
        assert not is_known_subtitle_codec(SubtitleCodecId.UNKNOWN)

    def test_subtitle_action_pgs(self) -> None:
        assert get_subtitle_action(SubtitleCodecId.PGS) == SubtitleAction.COPY

    def test_subtitle_action_vobsub(self) -> None:
        assert get_subtitle_action(SubtitleCodecId.VOBSUB) == SubtitleAction.COPY

    def test_subtitle_action_srt(self) -> None:
        assert get_subtitle_action(SubtitleCodecId.SRT) == SubtitleAction.COPY_RECODE

    def test_subtitle_action_ass(self) -> None:
        assert get_subtitle_action(SubtitleCodecId.ASS) == SubtitleAction.COPY_RECODE

    def test_subtitle_action_unknown_returns_none(self) -> None:
        assert get_subtitle_action(SubtitleCodecId.UNKNOWN) is None
