from __future__ import annotations

from pathlib import Path

import pytest

from furnace.core.detect import (
    check_unsupported_codecs,
    detect_forced_subtitles,
    detect_hdr,
    should_skip_file,
)
from furnace.core.models import (
    AudioCodecId,
    DvBlCompatibility,
    SubtitleCodecId,
    Track,
    TrackType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sub_track(
    index: int = 0,
    codec_id: SubtitleCodecId = SubtitleCodecId.PGS,
    language: str = "eng",
    title: str = "",
    is_forced: bool = False,
    num_frames: int | None = None,
    num_captions: int | None = None,
    source_file: str = "movie.mkv",
) -> Track:
    return Track(
        index=index,
        track_type=TrackType.SUBTITLE,
        codec_name=codec_id.value,
        codec_id=codec_id,
        language=language,
        title=title,
        is_default=False,
        is_forced=is_forced,
        source_file=Path(source_file),
        num_frames=num_frames,
        num_captions=num_captions,
    )


def make_audio_track(
    index: int = 0,
    codec_id: AudioCodecId = AudioCodecId.AAC_LC,
    codec_name: str = "aac",
    language: str = "eng",
) -> Track:
    return Track(
        index=index,
        track_type=TrackType.AUDIO,
        codec_name=codec_name,
        codec_id=codec_id,
        language=language,
        title="",
        is_default=False,
        is_forced=False,
        source_file=Path("movie.mkv"),
    )


# ---------------------------------------------------------------------------
# test_forced_detection_keywords
# ---------------------------------------------------------------------------

class TestForcedDetectionKeywords:
    def test_filename_keyword_forced(self):
        track = make_sub_track(source_file="movie.forced.eng.srt")
        detect_forced_subtitles([track])
        assert track.is_forced

    def test_filename_keyword_forsed(self):
        """Russian transliteration 'forsed' in filename."""
        track = make_sub_track(source_file="movie.forsed.rus.srt")
        detect_forced_subtitles([track])
        assert track.is_forced

    def test_filename_keyword_tolko_nadpisi(self):
        """Russian transliteration 'tolko nadpisi' in filename."""
        track = make_sub_track(source_file="movie.tolko nadpisi.rus.srt")
        detect_forced_subtitles([track])
        assert track.is_forced

    def test_filename_keyword_cyrillic_only_nadpisi(self):
        """Cyrillic 'только надписи' in filename."""
        track = make_sub_track(source_file="movie.только надписи.rus.srt")
        detect_forced_subtitles([track])
        assert track.is_forced

    def test_filename_keyword_forsirovannye(self):
        """Partial cyrillic 'форсир' in filename."""
        track = make_sub_track(source_file="movie.форсированные.rus.srt")
        detect_forced_subtitles([track])
        assert track.is_forced

    def test_filename_keyword_normal_excluded(self):
        """'normal' in filename -> excluded from keyword matching."""
        track = make_sub_track(source_file="movie.normal.eng.srt")
        detect_forced_subtitles([track])
        assert not track.is_forced

    def test_trackname_keyword_forced(self):
        """'forced' in track title -> forced."""
        track = make_sub_track(title="Forced subtitles")
        detect_forced_subtitles([track])
        assert track.is_forced

    def test_trackname_keyword_caption(self):
        """'caption' in track title -> forced."""
        track = make_sub_track(title="Foreign captions")
        detect_forced_subtitles([track])
        assert track.is_forced

    def test_trackname_sdh_excluded(self):
        """'sdh' in track title excludes keyword detection for that track."""
        track = make_sub_track(title="English SDH (Forced)")
        detect_forced_subtitles([track])
        # SDH track is excluded from trackname keyword check even if 'forced' present
        assert not track.is_forced

    def test_no_keywords_not_forced(self):
        """Normal track with no keywords stays not-forced."""
        track = make_sub_track(title="English", source_file="movie.mkv")
        detect_forced_subtitles([track])
        assert not track.is_forced


# ---------------------------------------------------------------------------
# test_forced_detection_stats_binary
# ---------------------------------------------------------------------------

class TestForcedDetectionStatsBinary:
    def test_pgs_below_50_percent_is_forced(self):
        """PGS track with < 50% num_frames of same-language max -> forced."""
        full = make_sub_track(index=0, codec_id=SubtitleCodecId.PGS, language="eng", num_frames=1000)
        partial = make_sub_track(index=1, codec_id=SubtitleCodecId.PGS, language="eng", num_frames=400)
        detect_forced_subtitles([full, partial])
        assert not full.is_forced
        assert partial.is_forced

    def test_pgs_above_50_percent_not_forced(self):
        """PGS track with >= 50% num_frames -> not forced."""
        full = make_sub_track(index=0, codec_id=SubtitleCodecId.PGS, language="eng", num_frames=1000)
        partial = make_sub_track(index=1, codec_id=SubtitleCodecId.PGS, language="eng", num_frames=600)
        detect_forced_subtitles([full, partial])
        assert not full.is_forced
        assert not partial.is_forced

    def test_vobsub_below_50_percent_is_forced(self):
        """VOBSUB track with < 50% num_frames -> forced."""
        full = make_sub_track(index=0, codec_id=SubtitleCodecId.VOBSUB, language="rus", num_frames=800)
        partial = make_sub_track(index=1, codec_id=SubtitleCodecId.VOBSUB, language="rus", num_frames=100)
        detect_forced_subtitles([full, partial])
        assert not full.is_forced
        assert partial.is_forced

    def test_binary_different_languages_compared_separately(self):
        """Each language's threshold is computed independently."""
        eng_full = make_sub_track(index=0, codec_id=SubtitleCodecId.PGS, language="eng", num_frames=1000)
        eng_forced = make_sub_track(index=1, codec_id=SubtitleCodecId.PGS, language="eng", num_frames=100)
        rus_full = make_sub_track(index=2, codec_id=SubtitleCodecId.PGS, language="rus", num_frames=50)
        detect_forced_subtitles([eng_full, eng_forced, rus_full])
        assert not eng_full.is_forced
        assert eng_forced.is_forced
        assert not rus_full.is_forced  # only track for its language, no comparison

    def test_single_track_not_forced_by_stats(self):
        """Single binary track has no comparison partner -> not forced."""
        single = make_sub_track(codec_id=SubtitleCodecId.PGS, language="eng", num_frames=50)
        detect_forced_subtitles([single])
        assert not single.is_forced

    def test_pgs_exactly_50_percent_not_forced(self):
        """Track at exactly 50% (not strictly less) -> not forced."""
        full = make_sub_track(index=0, codec_id=SubtitleCodecId.PGS, language="eng", num_frames=1000)
        half = make_sub_track(index=1, codec_id=SubtitleCodecId.PGS, language="eng", num_frames=500)
        detect_forced_subtitles([full, half])
        assert not half.is_forced


# ---------------------------------------------------------------------------
# test_forced_detection_stats_text
# ---------------------------------------------------------------------------

class TestForcedDetectionStatsText:
    def test_srt_below_50_percent_is_forced(self):
        """SRT track with < 50% num_captions -> forced."""
        full = make_sub_track(index=0, codec_id=SubtitleCodecId.SRT, language="eng", num_captions=500)
        partial = make_sub_track(index=1, codec_id=SubtitleCodecId.SRT, language="eng", num_captions=200)
        detect_forced_subtitles([full, partial])
        assert not full.is_forced
        assert partial.is_forced

    def test_ass_below_50_percent_is_forced(self):
        """ASS track with < 50% num_captions -> forced."""
        full = make_sub_track(index=0, codec_id=SubtitleCodecId.ASS, language="rus", num_captions=600)
        partial = make_sub_track(index=1, codec_id=SubtitleCodecId.ASS, language="rus", num_captions=100)
        detect_forced_subtitles([full, partial])
        assert not full.is_forced
        assert partial.is_forced

    def test_text_above_50_percent_not_forced(self):
        """SRT track at 60% -> not forced."""
        full = make_sub_track(index=0, codec_id=SubtitleCodecId.SRT, language="eng", num_captions=1000)
        partial = make_sub_track(index=1, codec_id=SubtitleCodecId.SRT, language="eng", num_captions=600)
        detect_forced_subtitles([full, partial])
        assert not partial.is_forced


# ---------------------------------------------------------------------------
# test_forced_detection_exclude_chi
# ---------------------------------------------------------------------------

class TestForcedDetectionExcludeChi:
    def test_chi_excluded_from_stats(self):
        """Chi language tracks are excluded from statistical comparison."""
        eng_full = make_sub_track(index=0, codec_id=SubtitleCodecId.PGS, language="eng", num_frames=1000)
        chi_small = make_sub_track(index=1, codec_id=SubtitleCodecId.PGS, language="chi", num_frames=50)
        detect_forced_subtitles([eng_full, chi_small])
        # chi track is excluded from stats; only eng forms its own group
        assert not chi_small.is_forced  # not marked forced by stats (excluded)
        assert not eng_full.is_forced

    def test_chi_not_compared_with_eng(self):
        """Chi tracks form no comparison group so never get forced by stats."""
        chi_small = make_sub_track(index=0, codec_id=SubtitleCodecId.PGS, language="chi", num_frames=10)
        chi_large = make_sub_track(index=1, codec_id=SubtitleCodecId.PGS, language="chi", num_frames=1000)
        detect_forced_subtitles([chi_small, chi_large])
        # Both chi -> both excluded from stats -> neither forced by stats
        assert not chi_small.is_forced


# ---------------------------------------------------------------------------
# test_forced_detection_exclude_sdh
# ---------------------------------------------------------------------------

class TestForcedDetectionExcludeSdh:
    def test_sdh_track_excluded_from_stats(self):
        """Track with 'sdh' in title is excluded from statistical comparison."""
        full = make_sub_track(index=0, codec_id=SubtitleCodecId.PGS, language="eng",
                               title="English SDH", num_frames=2000)
        small = make_sub_track(index=1, codec_id=SubtitleCodecId.PGS, language="eng",
                                num_frames=100)
        detect_forced_subtitles([full, small])
        # 'full' (SDH) is excluded from stat group; 'small' has no comparison -> not forced
        assert not small.is_forced

    def test_sdh_case_insensitive(self):
        """SDH exclusion is case-insensitive."""
        sdh_track = make_sub_track(index=0, codec_id=SubtitleCodecId.PGS, language="eng",
                                    title="English SDH", num_frames=3000)
        normal = make_sub_track(index=1, codec_id=SubtitleCodecId.PGS, language="eng",
                                 num_frames=100)
        detect_forced_subtitles([sdh_track, normal])
        # SDH excluded, normal has no partner -> not forced
        assert not normal.is_forced


# ---------------------------------------------------------------------------
# test_hdr_detection
# ---------------------------------------------------------------------------

class TestHdrDetection:
    def test_sdr_no_side_data(self):
        """No side data -> all HDR fields None/False."""
        result = detect_hdr({}, None)
        assert result.mastering_display is None
        assert result.content_light is None
        assert not result.is_dolby_vision
        assert not result.is_hdr10_plus

    def test_hdr10_mastering_display(self):
        """Mastering display metadata side data -> mastering_display string set."""
        side_data = [{
            "side_data_type": "Mastering display metadata",
            "green_x": "0.2650", "green_y": "0.6900",
            "blue_x": "0.1500", "blue_y": "0.0600",
            "red_x": "0.6800", "red_y": "0.3200",
            "white_point_x": "0.3127", "white_point_y": "0.3290",
            "max_luminance": "1000.0000", "min_luminance": "0.0050",
        }]
        result = detect_hdr({}, side_data)
        assert result.mastering_display is not None
        assert "G(" in result.mastering_display
        assert "B(" in result.mastering_display
        assert "R(" in result.mastering_display
        assert "WP(" in result.mastering_display
        assert "L(" in result.mastering_display

    def test_hdr10_content_light(self):
        """Content light level metadata -> content_light string set."""
        side_data = [{
            "side_data_type": "Content light level metadata",
            "max_content": "1000",
            "max_average": "400",
        }]
        result = detect_hdr({}, side_data)
        assert result.content_light == "MaxCLL=1000,MaxFALL=400"

    def test_dolby_vision_side_data(self):
        """Dolby Vision configuration in side_data -> is_dolby_vision True."""
        side_data = [{"side_data_type": "Dolby Vision configuration record"}]
        result = detect_hdr({}, side_data)
        assert result.is_dolby_vision

    def test_dolby_vision_codec_name_dvhe(self):
        """codec_name 'dvhe' -> is_dolby_vision True."""
        result = detect_hdr({"codec_name": "dvhe"}, [])
        assert result.is_dolby_vision

    def test_dolby_vision_codec_name_dvh1(self):
        """codec_name 'dvh1' -> is_dolby_vision True."""
        result = detect_hdr({"codec_name": "dvh1"}, [])
        assert result.is_dolby_vision

    def test_hdr10_plus_side_data(self):
        """HDR10+ dynamic metadata in side_data -> is_hdr10_plus True."""
        side_data = [{"side_data_type": "HDR10+ Dynamic Metadata"}]
        result = detect_hdr({}, side_data)
        assert result.is_hdr10_plus

    def test_smpte_st2094_hdr10_plus(self):
        """SMPTE ST 2094 in side_data type -> is_hdr10_plus True."""
        side_data = [{"side_data_type": "SMPTE ST 2094-40 metadata"}]
        result = detect_hdr({}, side_data)
        assert result.is_hdr10_plus

    def test_plain_sdr_h264(self):
        """h264 codec with no side data -> all False/None."""
        result = detect_hdr({"codec_name": "h264"}, [])
        assert not result.is_dolby_vision
        assert not result.is_hdr10_plus
        assert result.mastering_display is None


# ---------------------------------------------------------------------------
# test_skip_logic
# ---------------------------------------------------------------------------

class TestSkipLogic:
    def test_file_exists_skip(self, tmp_path):
        """Output file exists -> should skip."""
        output = tmp_path / "output.mkv"
        output.touch()
        skip, reason = should_skip_file(output, None)
        assert skip is True
        assert "already exists" in reason

    def test_file_not_exists_no_skip(self, tmp_path):
        """Output file does not exist, no encoder tag -> do not skip."""
        output = tmp_path / "output.mkv"
        skip, reason = should_skip_file(output, None)
        assert skip is False
        assert reason == ""

    def test_encoder_tag_furnace_skip(self, tmp_path):
        """Encoder tag starts with 'Furnace/' -> skip."""
        output = tmp_path / "output.mkv"
        skip, reason = should_skip_file(output, "Furnace/0.1.0")
        assert skip is True
        assert "Furnace" in reason

    def test_encoder_tag_other_no_skip(self, tmp_path):
        """Encoder tag from another tool -> do not skip."""
        output = tmp_path / "output.mkv"
        skip, reason = should_skip_file(output, "HandBrake/1.6.0")
        assert skip is False

    def test_encoder_tag_empty_string_no_skip(self, tmp_path):
        """Empty string encoder tag -> do not skip."""
        output = tmp_path / "output.mkv"
        skip, reason = should_skip_file(output, "")
        assert skip is False


# ---------------------------------------------------------------------------
# test_unknown_codec_check
# ---------------------------------------------------------------------------

class TestUnknownCodecCheck:
    def test_no_unknowns_returns_none(self):
        audio = [make_audio_track(codec_id=AudioCodecId.AAC_LC)]
        subs = [make_sub_track(codec_id=SubtitleCodecId.PGS)]
        result = check_unsupported_codecs(audio, subs)
        assert result is None

    def test_unknown_audio_returns_warning(self):
        audio = [make_audio_track(index=2, codec_id=AudioCodecId.UNKNOWN, codec_name="somecodec", language="eng")]
        result = check_unsupported_codecs(audio, [])
        assert result is not None
        assert "audio stream #2" in result
        assert "somecodec" in result

    def test_unknown_subtitle_returns_warning(self):
        subs = [make_sub_track(index=3, codec_id=SubtitleCodecId.UNKNOWN, language="fra")]
        result = check_unsupported_codecs([], subs)
        assert result is not None
        assert "subtitle stream #3" in result

    def test_multiple_unknowns_all_listed(self):
        audio = [make_audio_track(index=1, codec_id=AudioCodecId.UNKNOWN, codec_name="x", language="eng")]
        subs = [make_sub_track(index=2, codec_id=SubtitleCodecId.UNKNOWN, language="rus")]
        result = check_unsupported_codecs(audio, subs)
        assert result is not None
        assert "audio stream #1" in result
        assert "subtitle stream #2" in result


# ---------------------------------------------------------------------------
# test_dv_profile_detection
# ---------------------------------------------------------------------------

class TestDvProfileDetection:
    def test_dv_profile_from_side_data(self) -> None:
        side_data = [{
            "side_data_type": "Dolby Vision configuration record",
            "dv_profile": 8,
            "dv_bl_signal_compatibility_id": 1,
        }]
        result = detect_hdr({}, side_data)
        assert result.is_dolby_vision
        assert result.dv_profile == 8
        assert result.dv_bl_compatibility == DvBlCompatibility.HDR10

    def test_dv_profile7_fel(self) -> None:
        side_data = [{
            "side_data_type": "Dolby Vision configuration record",
            "dv_profile": 7,
            "dv_bl_signal_compatibility_id": 1,
        }]
        result = detect_hdr({}, side_data)
        assert result.dv_profile == 7
        assert result.dv_bl_compatibility == DvBlCompatibility.HDR10

    def test_dv_profile5_no_compat(self) -> None:
        side_data = [{
            "side_data_type": "Dolby Vision configuration record",
            "dv_profile": 5,
            "dv_bl_signal_compatibility_id": 0,
        }]
        result = detect_hdr({}, side_data)
        assert result.dv_profile == 5
        assert result.dv_bl_compatibility == DvBlCompatibility.NONE

    def test_dv_codec_name_no_side_data_no_profile(self) -> None:
        result = detect_hdr({"codec_name": "dvhe"}, [])
        assert result.is_dolby_vision
        assert result.dv_profile is None
        assert result.dv_bl_compatibility is None

    def test_no_dv_fields_none(self) -> None:
        side_data = [{
            "side_data_type": "Mastering display metadata",
            "green_x": "0.265", "green_y": "0.690",
            "blue_x": "0.150", "blue_y": "0.060",
            "red_x": "0.680", "red_y": "0.320",
            "white_point_x": "0.3127", "white_point_y": "0.3290",
            "max_luminance": "1000", "min_luminance": "0.005",
        }]
        result = detect_hdr({}, side_data)
        assert result.dv_profile is None
        assert result.dv_bl_compatibility is None


# ---------------------------------------------------------------------------
# test_detect_interlace
