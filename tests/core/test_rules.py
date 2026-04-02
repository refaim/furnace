from __future__ import annotations

import pytest

from furnace.core.models import AudioAction, AudioCodecId, SubtitleAction, SubtitleCodecId
from furnace.core.rules import (
    get_audio_action,
    get_subtitle_action,
    is_known_audio_codec,
    is_known_subtitle_codec,
    parse_audio_codec,
    parse_subtitle_codec,
)


# ---------------------------------------------------------------------------
# test_parse_audio_codec
# ---------------------------------------------------------------------------

class TestParseAudioCodec:
    # DTS variants
    def test_dts_core_no_profile(self):
        assert parse_audio_codec("dts", None) == AudioCodecId.DTS

    def test_dts_core_explicit_profile(self):
        assert parse_audio_codec("dts", "DTS") == AudioCodecId.DTS

    def test_dts_es(self):
        assert parse_audio_codec("dts", "DTS-ES") == AudioCodecId.DTS_ES

    def test_dts_hd_hra(self):
        assert parse_audio_codec("dts", "DTS-HD HRA") == AudioCodecId.DTS_HRA

    def test_dts_hd_ma(self):
        assert parse_audio_codec("dts", "DTS-HD MA") == AudioCodecId.DTS_MA

    def test_dts_unknown_profile_falls_back_to_dts(self):
        """Unknown DTS profile -> fall back to plain DTS."""
        assert parse_audio_codec("dts", "DTS-X") == AudioCodecId.DTS

    # AAC variants
    def test_aac_lc_no_profile(self):
        assert parse_audio_codec("aac", None) == AudioCodecId.AAC_LC

    def test_aac_lc_explicit(self):
        assert parse_audio_codec("aac", "LC") == AudioCodecId.AAC_LC

    def test_aac_he(self):
        assert parse_audio_codec("aac", "HE-AAC") == AudioCodecId.AAC_HE

    def test_aac_he_v2(self):
        assert parse_audio_codec("aac", "HE-AAC v2") == AudioCodecId.AAC_HE_V2

    # Other named codecs
    def test_ac3(self):
        assert parse_audio_codec("ac3", None) == AudioCodecId.AC3

    def test_eac3(self):
        assert parse_audio_codec("eac3", None) == AudioCodecId.EAC3

    def test_truehd(self):
        assert parse_audio_codec("truehd", None) == AudioCodecId.TRUEHD

    def test_flac(self):
        assert parse_audio_codec("flac", None) == AudioCodecId.FLAC

    def test_pcm_s16le(self):
        assert parse_audio_codec("pcm_s16le", None) == AudioCodecId.PCM_S16LE

    def test_pcm_s24le(self):
        assert parse_audio_codec("pcm_s24le", None) == AudioCodecId.PCM_S24LE

    def test_pcm_s16be(self):
        assert parse_audio_codec("pcm_s16be", None) == AudioCodecId.PCM_S16BE

    def test_mp3(self):
        assert parse_audio_codec("mp3", None) == AudioCodecId.MP3

    def test_mp2(self):
        assert parse_audio_codec("mp2", None) == AudioCodecId.MP2

    def test_unknown_codec_name(self):
        assert parse_audio_codec("someweirdcodec", None) == AudioCodecId.UNKNOWN

    def test_empty_codec_name(self):
        assert parse_audio_codec("", None) == AudioCodecId.UNKNOWN


# ---------------------------------------------------------------------------
# test_audio_action_routing
# ---------------------------------------------------------------------------

class TestAudioActionRouting:
    """Every AudioCodecId maps to exactly the right AudioAction."""

    @pytest.mark.parametrize("codec_id", [
        AudioCodecId.AAC_LC,
        AudioCodecId.AAC_HE,
        AudioCodecId.AAC_HE_V2,
    ])
    def test_aac_copy(self, codec_id):
        assert get_audio_action(codec_id) == AudioAction.COPY

    @pytest.mark.parametrize("codec_id", [
        AudioCodecId.AC3,
        AudioCodecId.EAC3,
        AudioCodecId.DTS,
    ])
    def test_denorm(self, codec_id):
        assert get_audio_action(codec_id) == AudioAction.DENORM

    @pytest.mark.parametrize("codec_id", [
        AudioCodecId.DTS_ES,
        AudioCodecId.DTS_HRA,
        AudioCodecId.DTS_MA,
        AudioCodecId.TRUEHD,
        AudioCodecId.FLAC,
        AudioCodecId.PCM_S16LE,
        AudioCodecId.PCM_S24LE,
        AudioCodecId.PCM_S16BE,
    ])
    def test_decode_encode(self, codec_id):
        assert get_audio_action(codec_id) == AudioAction.DECODE_ENCODE

    @pytest.mark.parametrize("codec_id", [
        AudioCodecId.MP2,
        AudioCodecId.MP3,
        AudioCodecId.VORBIS,
        AudioCodecId.OPUS,
        AudioCodecId.WMA_V2,
        AudioCodecId.WMA_PRO,
        AudioCodecId.AMR,
    ])
    def test_ffmpeg_encode(self, codec_id):
        assert get_audio_action(codec_id) == AudioAction.FFMPEG_ENCODE

    def test_unknown_returns_none(self):
        """UNKNOWN codec is not in the whitelist -> None."""
        assert get_audio_action(AudioCodecId.UNKNOWN) is None


# ---------------------------------------------------------------------------
# test_parse_subtitle_codec
# ---------------------------------------------------------------------------

class TestParseSubtitleCodec:
    def test_subrip(self):
        assert parse_subtitle_codec("subrip") == SubtitleCodecId.SRT

    def test_ass(self):
        assert parse_subtitle_codec("ass") == SubtitleCodecId.ASS

    def test_hdmv_pgs_subtitle(self):
        assert parse_subtitle_codec("hdmv_pgs_subtitle") == SubtitleCodecId.PGS

    def test_dvd_subtitle(self):
        assert parse_subtitle_codec("dvd_subtitle") == SubtitleCodecId.VOBSUB

    def test_unknown_returns_unknown(self):
        assert parse_subtitle_codec("webvtt") == SubtitleCodecId.UNKNOWN

    def test_empty_returns_unknown(self):
        assert parse_subtitle_codec("") == SubtitleCodecId.UNKNOWN


# ---------------------------------------------------------------------------
# test_known_codec_checks
# ---------------------------------------------------------------------------

class TestKnownCodecChecks:
    def test_known_audio_codecs_are_known(self):
        known = [
            AudioCodecId.AAC_LC, AudioCodecId.AAC_HE, AudioCodecId.AAC_HE_V2,
            AudioCodecId.AC3, AudioCodecId.EAC3,
            AudioCodecId.DTS, AudioCodecId.DTS_ES, AudioCodecId.DTS_HRA, AudioCodecId.DTS_MA,
            AudioCodecId.TRUEHD, AudioCodecId.FLAC,
            AudioCodecId.PCM_S16LE, AudioCodecId.PCM_S24LE, AudioCodecId.PCM_S16BE,
            AudioCodecId.MP2, AudioCodecId.MP3,
            AudioCodecId.VORBIS, AudioCodecId.OPUS,
            AudioCodecId.WMA_V2, AudioCodecId.WMA_PRO, AudioCodecId.AMR,
        ]
        for codec in known:
            assert is_known_audio_codec(codec), f"{codec} should be known"

    def test_unknown_audio_is_not_known(self):
        assert not is_known_audio_codec(AudioCodecId.UNKNOWN)

    def test_known_subtitle_codecs_are_known(self):
        known = [SubtitleCodecId.SRT, SubtitleCodecId.ASS, SubtitleCodecId.PGS, SubtitleCodecId.VOBSUB]
        for codec in known:
            assert is_known_subtitle_codec(codec), f"{codec} should be known"

    def test_unknown_subtitle_is_not_known(self):
        assert not is_known_subtitle_codec(SubtitleCodecId.UNKNOWN)

    def test_subtitle_action_pgs(self):
        assert get_subtitle_action(SubtitleCodecId.PGS) == SubtitleAction.COPY

    def test_subtitle_action_vobsub(self):
        assert get_subtitle_action(SubtitleCodecId.VOBSUB) == SubtitleAction.COPY

    def test_subtitle_action_srt(self):
        assert get_subtitle_action(SubtitleCodecId.SRT) == SubtitleAction.COPY_RECODE

    def test_subtitle_action_ass(self):
        assert get_subtitle_action(SubtitleCodecId.ASS) == SubtitleAction.COPY_RECODE

    def test_subtitle_action_unknown_returns_none(self):
        assert get_subtitle_action(SubtitleCodecId.UNKNOWN) is None
