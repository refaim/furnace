from __future__ import annotations

from pathlib import Path

import pytest

from furnace.core.detect import (
    DV_PROFILE_FEL,
    VideoSystem,
    aggregate_crop,
    check_unsupported_codecs,
    classify_passthrough,
    cropdetect_limit,
    detect_forced_subtitles,
    detect_hdr,
    hdr_tonemap_transfer,
    is_content_light_side_data,
    is_dolby_vision_side_data,
    is_dvd_resolution,
    is_hdr10_plus_side_data,
    is_mastering_display_side_data,
    resolve_color_metadata,
    should_skip_file,
)
from furnace.core.models import (
    AudioCodecId,
    CropRect,
    DvBlCompatibility,
    HdrMetadata,
    SubtitleCodecId,
    Track,
    TrackType,
    VideoInfo,
)
from tests.conftest import make_track


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
    return make_track(
        index=index,
        track_type=TrackType.SUBTITLE,
        codec_name=codec_id.value,
        codec_id=codec_id,
        language=language,
        title=title,
        is_forced=is_forced,
        source_file=Path(source_file),
        channels=None,
        num_frames=num_frames,
        num_captions=num_captions,
    )


def make_audio_track(
    index: int = 0,
    codec_id: AudioCodecId = AudioCodecId.AAC_LC,
    codec_name: str = "aac",
    language: str = "eng",
    profile: str | None = None,
) -> Track:
    return make_track(
        index=index,
        track_type=TrackType.AUDIO,
        codec_name=codec_name,
        codec_id=codec_id,
        language=language,
        source_file=Path("movie.mkv"),
        profile=profile,
    )


class TestForcedDetectionKeywords:
    def test_filename_keyword_forced(self) -> None:
        track = make_sub_track(source_file="movie.forced.eng.srt")
        detect_forced_subtitles([track])
        assert track.is_forced

    def test_filename_keyword_forsed(self) -> None:
        track = make_sub_track(source_file="movie.forsed.rus.srt")
        detect_forced_subtitles([track])
        assert track.is_forced

    def test_filename_keyword_tolko_nadpisi(self) -> None:
        track = make_sub_track(source_file="movie.tolko nadpisi.rus.srt")
        detect_forced_subtitles([track])
        assert track.is_forced

    def test_filename_keyword_cyrillic_only_nadpisi(self) -> None:
        track = make_sub_track(source_file="movie.только надписи.rus.srt")
        detect_forced_subtitles([track])
        assert track.is_forced

    def test_filename_keyword_forsirovannye(self) -> None:
        track = make_sub_track(source_file="movie.форсированные.rus.srt")
        detect_forced_subtitles([track])
        assert track.is_forced

    def test_filename_keyword_normal_excluded(self) -> None:
        track = make_sub_track(source_file="movie.normal.eng.srt")
        detect_forced_subtitles([track])
        assert not track.is_forced

    def test_trackname_keyword_forced(self) -> None:
        track = make_sub_track(title="Forced subtitles")
        detect_forced_subtitles([track])
        assert track.is_forced

    def test_trackname_keyword_caption(self) -> None:
        track = make_sub_track(title="Foreign captions")
        detect_forced_subtitles([track])
        assert track.is_forced

    def test_trackname_sdh_excluded(self) -> None:
        track = make_sub_track(title="English SDH (Forced)")
        detect_forced_subtitles([track])
        assert not track.is_forced

    def test_no_keywords_not_forced(self) -> None:
        track = make_sub_track(title="English", source_file="movie.mkv")
        detect_forced_subtitles([track])
        assert not track.is_forced


class TestForcedDetectionStatsBinary:
    def test_pgs_below_50_percent_is_forced(self) -> None:
        full = make_sub_track(index=0, codec_id=SubtitleCodecId.PGS, language="eng", num_frames=1000)
        partial = make_sub_track(index=1, codec_id=SubtitleCodecId.PGS, language="eng", num_frames=400)
        detect_forced_subtitles([full, partial])
        assert not full.is_forced
        assert partial.is_forced

    def test_pgs_above_50_percent_not_forced(self) -> None:
        full = make_sub_track(index=0, codec_id=SubtitleCodecId.PGS, language="eng", num_frames=1000)
        partial = make_sub_track(index=1, codec_id=SubtitleCodecId.PGS, language="eng", num_frames=600)
        detect_forced_subtitles([full, partial])
        assert not full.is_forced
        assert not partial.is_forced

    def test_vobsub_below_50_percent_is_forced(self) -> None:
        full = make_sub_track(index=0, codec_id=SubtitleCodecId.VOBSUB, language="rus", num_frames=800)
        partial = make_sub_track(index=1, codec_id=SubtitleCodecId.VOBSUB, language="rus", num_frames=100)
        detect_forced_subtitles([full, partial])
        assert not full.is_forced
        assert partial.is_forced

    def test_binary_different_languages_compared_separately(self) -> None:
        eng_full = make_sub_track(index=0, codec_id=SubtitleCodecId.PGS, language="eng", num_frames=1000)
        eng_forced = make_sub_track(index=1, codec_id=SubtitleCodecId.PGS, language="eng", num_frames=100)
        rus_full = make_sub_track(index=2, codec_id=SubtitleCodecId.PGS, language="rus", num_frames=50)
        detect_forced_subtitles([eng_full, eng_forced, rus_full])
        assert not eng_full.is_forced
        assert eng_forced.is_forced
        assert not rus_full.is_forced

    def test_single_track_not_forced_by_stats(self) -> None:
        single = make_sub_track(codec_id=SubtitleCodecId.PGS, language="eng", num_frames=50)
        detect_forced_subtitles([single])
        assert not single.is_forced

    def test_pgs_exactly_50_percent_not_forced(self) -> None:
        full = make_sub_track(index=0, codec_id=SubtitleCodecId.PGS, language="eng", num_frames=1000)
        half = make_sub_track(index=1, codec_id=SubtitleCodecId.PGS, language="eng", num_frames=500)
        detect_forced_subtitles([full, half])
        assert not half.is_forced


class TestForcedDetectionStatsText:
    def test_srt_below_50_percent_is_forced(self) -> None:
        full = make_sub_track(index=0, codec_id=SubtitleCodecId.SRT, language="eng", num_captions=500)
        partial = make_sub_track(index=1, codec_id=SubtitleCodecId.SRT, language="eng", num_captions=200)
        detect_forced_subtitles([full, partial])
        assert not full.is_forced
        assert partial.is_forced

    def test_ass_below_50_percent_is_forced(self) -> None:
        full = make_sub_track(index=0, codec_id=SubtitleCodecId.ASS, language="rus", num_captions=600)
        partial = make_sub_track(index=1, codec_id=SubtitleCodecId.ASS, language="rus", num_captions=100)
        detect_forced_subtitles([full, partial])
        assert not full.is_forced
        assert partial.is_forced

    def test_text_above_50_percent_not_forced(self) -> None:
        full = make_sub_track(index=0, codec_id=SubtitleCodecId.SRT, language="eng", num_captions=1000)
        partial = make_sub_track(index=1, codec_id=SubtitleCodecId.SRT, language="eng", num_captions=600)
        detect_forced_subtitles([full, partial])
        assert not partial.is_forced


class TestForcedDetectionExcludeChi:
    def test_chi_excluded_from_stats(self) -> None:
        eng_full = make_sub_track(index=0, codec_id=SubtitleCodecId.PGS, language="eng", num_frames=1000)
        chi_small = make_sub_track(index=1, codec_id=SubtitleCodecId.PGS, language="chi", num_frames=50)
        detect_forced_subtitles([eng_full, chi_small])
        assert not chi_small.is_forced
        assert not eng_full.is_forced

    def test_chi_not_compared_with_eng(self) -> None:
        chi_small = make_sub_track(index=0, codec_id=SubtitleCodecId.PGS, language="chi", num_frames=10)
        chi_large = make_sub_track(index=1, codec_id=SubtitleCodecId.PGS, language="chi", num_frames=1000)
        detect_forced_subtitles([chi_small, chi_large])
        assert not chi_small.is_forced


class TestForcedDetectionExcludeSdh:
    def test_sdh_track_excluded_from_stats(self) -> None:
        full = make_sub_track(
            index=0, codec_id=SubtitleCodecId.PGS, language="eng", title="English SDH", num_frames=2000
        )
        small = make_sub_track(index=1, codec_id=SubtitleCodecId.PGS, language="eng", num_frames=100)
        detect_forced_subtitles([full, small])
        assert not small.is_forced

    def test_sdh_case_insensitive(self) -> None:
        sdh_track = make_sub_track(
            index=0, codec_id=SubtitleCodecId.PGS, language="eng", title="English SDH", num_frames=3000
        )
        normal = make_sub_track(index=1, codec_id=SubtitleCodecId.PGS, language="eng", num_frames=100)
        detect_forced_subtitles([sdh_track, normal])
        assert not normal.is_forced


class TestHdrSideDataMatchers:
    def test_hdr10_plus_ffprobe_av1_label(self) -> None:
        assert is_hdr10_plus_side_data("HDR Dynamic Metadata SMPTE2094-40 (HDR10+)")

    def test_hdr10_plus_spaced_st_2094_label(self) -> None:
        assert is_hdr10_plus_side_data("SMPTE ST 2094-40 metadata")

    def test_hdr10_plus_rejects_static_metadata(self) -> None:
        assert not is_hdr10_plus_side_data("Mastering display metadata")

    def test_dolby_vision_metadata(self) -> None:
        assert is_dolby_vision_side_data("Dolby Vision Metadata")

    def test_dolby_vision_rpu_data(self) -> None:
        assert is_dolby_vision_side_data("Dolby Vision RPU Data")

    def test_dolby_vision_rejects_configuration_record(self) -> None:
        assert not is_dolby_vision_side_data("DOVI configuration record")

    def test_mastering_display(self) -> None:
        assert is_mastering_display_side_data("Mastering display metadata")

    def test_mastering_display_rejects_content_light(self) -> None:
        assert not is_mastering_display_side_data("Content light level metadata")

    def test_content_light(self) -> None:
        assert is_content_light_side_data("Content light level metadata")

    def test_content_light_rejects_mastering_display(self) -> None:
        assert not is_content_light_side_data("Mastering display metadata")


class TestHdrDetection:
    def test_sdr_no_side_data(self) -> None:
        result = detect_hdr({}, None)
        assert result.mastering_display is None
        assert result.content_light is None
        assert not result.is_dolby_vision
        assert not result.is_hdr10_plus

    def test_hdr10_mastering_display(self) -> None:
        side_data = [
            {
                "side_data_type": "Mastering display metadata",
                "green_x": "0.2650",
                "green_y": "0.6900",
                "blue_x": "0.1500",
                "blue_y": "0.0600",
                "red_x": "0.6800",
                "red_y": "0.3200",
                "white_point_x": "0.3127",
                "white_point_y": "0.3290",
                "max_luminance": "1000.0000",
                "min_luminance": "0.0050",
            }
        ]
        result = detect_hdr({}, side_data)
        assert result.mastering_display is not None
        assert "G(" in result.mastering_display
        assert "B(" in result.mastering_display
        assert "R(" in result.mastering_display
        assert "WP(" in result.mastering_display
        assert "L(" in result.mastering_display

    def test_hdr10_content_light(self) -> None:
        side_data = [
            {
                "side_data_type": "Content light level metadata",
                "max_content": "1000",
                "max_average": "400",
            }
        ]
        result = detect_hdr({}, side_data)
        assert result.content_light == "MaxCLL=1000,MaxFALL=400"

    def test_dolby_vision_side_data(self) -> None:
        side_data = [{"side_data_type": "DOVI configuration record"}]
        result = detect_hdr({}, side_data)
        assert result.is_dolby_vision

    def test_dolby_vision_rpu_data_frame_marker(self) -> None:
        side_data = [{"side_data_type": "Dolby Vision RPU Data"}]
        result = detect_hdr({}, side_data)
        assert result.is_dolby_vision

    def test_dolby_vision_metadata_frame_marker(self) -> None:
        side_data = [{"side_data_type": "Dolby Vision Metadata"}]
        result = detect_hdr({}, side_data)
        assert result.is_dolby_vision

    def test_dolby_vision_codec_name_dvhe(self) -> None:
        result = detect_hdr({"codec_name": "dvhe"}, [])
        assert result.is_dolby_vision

    def test_dolby_vision_codec_name_dvh1(self) -> None:
        result = detect_hdr({"codec_name": "dvh1"}, [])
        assert result.is_dolby_vision

    def test_hdr10_plus_side_data(self) -> None:
        side_data = [{"side_data_type": "HDR10+ Dynamic Metadata"}]
        result = detect_hdr({}, side_data)
        assert result.is_hdr10_plus

    def test_smpte_st2094_hdr10_plus(self) -> None:
        side_data = [{"side_data_type": "SMPTE ST 2094-40 metadata"}]
        result = detect_hdr({}, side_data)
        assert result.is_hdr10_plus

    def test_smpte2094_av1_label_hdr10_plus(self) -> None:
        side_data = [{"side_data_type": "HDR Dynamic Metadata SMPTE2094-40"}]
        result = detect_hdr({}, side_data)
        assert result.is_hdr10_plus

    def test_plain_sdr_h264(self) -> None:
        result = detect_hdr({"codec_name": "h264"}, [])
        assert not result.is_dolby_vision
        assert not result.is_hdr10_plus
        assert result.mastering_display is None

    def test_hdr_metadata_ignores_unknown_side_data_type(self) -> None:
        side_data = [{"side_data_type": "Unknown foo bar"}]
        result = detect_hdr({}, side_data)
        assert not result.is_hdr10_plus
        assert not result.is_dolby_vision
        assert result.mastering_display is None
        assert result.content_light is None
        assert result.dv_profile is None
        assert result.dv_bl_compatibility is None


class TestSkipLogic:
    def test_file_exists_skip(self, tmp_path: Path) -> None:
        output = tmp_path / "output.mkv"
        output.touch()
        skip, reason = should_skip_file(output, None)
        assert skip is True
        assert "already exists" in reason

    def test_file_not_exists_no_skip(self, tmp_path: Path) -> None:
        output = tmp_path / "output.mkv"
        skip, reason = should_skip_file(output, None)
        assert skip is False
        assert reason == ""

    def test_encoder_tag_furnace_skip(self, tmp_path: Path) -> None:
        output = tmp_path / "output.mkv"
        skip, reason = should_skip_file(output, "Furnace/0.1.0")
        assert skip is True
        assert "Furnace" in reason

    def test_encoder_tag_other_no_skip(self, tmp_path: Path) -> None:
        output = tmp_path / "output.mkv"
        skip, _reason = should_skip_file(output, "HandBrake/1.6.0")
        assert skip is False

    def test_encoder_tag_empty_string_no_skip(self, tmp_path: Path) -> None:
        output = tmp_path / "output.mkv"
        skip, _reason = should_skip_file(output, "")
        assert skip is False

    def test_force_bypasses_output_exists(self, tmp_path: Path) -> None:
        output = tmp_path / "output.mkv"
        output.touch()
        skip, reason = should_skip_file(output, None, force=True)
        assert skip is False
        assert reason == ""

    def test_force_bypasses_furnace_tag(self, tmp_path: Path) -> None:
        output = tmp_path / "output.mkv"
        skip, reason = should_skip_file(output, "Furnace/1.17.0", force=True)
        assert skip is False
        assert reason == ""


class TestUnknownCodecCheck:
    def test_no_unknowns_returns_none(self) -> None:
        audio = [make_audio_track(codec_id=AudioCodecId.AAC_LC)]
        subs = [make_sub_track(codec_id=SubtitleCodecId.PGS)]
        result = check_unsupported_codecs(audio, subs)
        assert result is None

    def test_unknown_audio_returns_warning(self) -> None:
        audio = [make_audio_track(index=2, codec_id=AudioCodecId.UNKNOWN, codec_name="somecodec", language="eng")]
        result = check_unsupported_codecs(audio, [])
        assert result is not None
        assert "audio stream #2" in result
        assert "somecodec" in result

    def test_unknown_audio_warning_names_the_profile(self) -> None:
        # A raw profile integer is what ffprobe prints when libavcodec has no
        # name for the constant, and parse_audio_codec really does reject it --
        # unlike a made-up "DTS-HD MA + ..." string, which would prefix-match
        # its way to DTS_MA and never reach here.
        audio = [
            make_audio_track(
                index=7,
                codec_id=AudioCodecId.UNKNOWN,
                codec_name="dts",
                language="eng",
                profile="63",
            )
        ]
        result = check_unsupported_codecs(audio, [])
        assert result is not None
        assert "audio stream #7 ('dts' profile '63', lang=eng)" in result

    def test_unknown_subtitle_returns_warning(self) -> None:
        subs = [make_sub_track(index=3, codec_id=SubtitleCodecId.UNKNOWN, language="fra")]
        result = check_unsupported_codecs([], subs)
        assert result is not None
        assert "subtitle stream #3" in result

    def test_multiple_unknowns_all_listed(self) -> None:
        audio = [make_audio_track(index=1, codec_id=AudioCodecId.UNKNOWN, codec_name="x", language="eng")]
        subs = [make_sub_track(index=2, codec_id=SubtitleCodecId.UNKNOWN, language="rus")]
        result = check_unsupported_codecs(audio, subs)
        assert result is not None
        assert "audio stream #1" in result
        assert "subtitle stream #2" in result


class TestDvProfileDetection:
    def test_dv_profile_from_side_data(self) -> None:
        side_data = [
            {
                "side_data_type": "DOVI configuration record",
                "dv_profile": 8,
                "dv_bl_signal_compatibility_id": 1,
            }
        ]
        result = detect_hdr({}, side_data)
        assert result.is_dolby_vision
        assert result.dv_profile == 8
        assert result.dv_bl_compatibility == DvBlCompatibility.HDR10

    def test_dv_profile7_fel(self) -> None:
        side_data = [
            {
                "side_data_type": "DOVI configuration record",
                "dv_profile": 7,
                "dv_bl_signal_compatibility_id": 1,
            }
        ]
        result = detect_hdr({}, side_data)
        assert result.dv_profile == 7
        assert result.dv_bl_compatibility == DvBlCompatibility.HDR10

    def test_dv_profile5_no_compat(self) -> None:
        side_data = [
            {
                "side_data_type": "DOVI configuration record",
                "dv_profile": 5,
                "dv_bl_signal_compatibility_id": 0,
            }
        ]
        result = detect_hdr({}, side_data)
        assert result.dv_profile == 5
        assert result.dv_bl_compatibility == DvBlCompatibility.NONE

    def test_dv_codec_name_no_side_data_no_profile(self) -> None:
        result = detect_hdr({"codec_name": "dvhe"}, [])
        assert result.is_dolby_vision
        assert result.dv_profile is None
        assert result.dv_bl_compatibility is None

    def test_no_dv_fields_none(self) -> None:
        side_data = [
            {
                "side_data_type": "Mastering display metadata",
                "green_x": "0.265",
                "green_y": "0.690",
                "blue_x": "0.150",
                "blue_y": "0.060",
                "red_x": "0.680",
                "red_y": "0.320",
                "white_point_x": "0.3127",
                "white_point_y": "0.3290",
                "max_luminance": "1000",
                "min_luminance": "0.005",
            }
        ]
        result = detect_hdr({}, side_data)
        assert result.dv_profile is None
        assert result.dv_bl_compatibility is None


class TestIsDvdResolution:
    def test_ntsc_dvd(self) -> None:
        assert is_dvd_resolution(720, 480) is True

    def test_pal_dvd(self) -> None:
        assert is_dvd_resolution(720, 576) is True

    def test_hd_1080(self) -> None:
        assert is_dvd_resolution(1920, 1080) is False

    def test_hd_720(self) -> None:
        assert is_dvd_resolution(1280, 720) is False

    def test_uhd_4k(self) -> None:
        assert is_dvd_resolution(3840, 2160) is False


class TestAggregateCrop:
    def test_all_identical(self) -> None:
        crops = [CropRect(688, 432, 14, 72)] * 10
        assert aggregate_crop(crops) == CropRect(688, 432, 14, 72)

    def test_single_value(self) -> None:
        assert aggregate_crop([CropRect(704, 576, 0, 0)]) == CropRect(704, 576, 0, 0)

    def test_dominant_cluster_per_edge(self) -> None:
        crops = [
            CropRect(1424, 1072, 248, 4),
            CropRect(1424, 976, 248, 100),
            CropRect(1424, 928, 248, 74),
            CropRect(1424, 1072, 248, 4),
            CropRect(1424, 1072, 248, 4),
            CropRect(1104, 1072, 408, 4),
            CropRect(1408, 848, 246, 228),
            CropRect(1408, 800, 248, 4),
            CropRect(1424, 1072, 248, 4),
            CropRect(1424, 1072, 248, 4),
        ]
        assert aggregate_crop(crops) == CropRect(1424, 1072, 248, 4)

    def test_scattered_overcrops_lose_to_dominant_cluster(self) -> None:
        crops = [
            CropRect(1424, 1072, 248, 4),
            CropRect(1424, 1072, 248, 4),
            CropRect(1424, 1072, 248, 4),
            CropRect(1424, 1040, 248, 24),
            CropRect(1424, 1008, 248, 40),
            CropRect(1424, 976, 248, 56),
            CropRect(1424, 944, 248, 72),
        ]
        assert aggregate_crop(crops) == CropRect(1424, 1072, 248, 4)

    def test_jitter_within_tolerance_clusters_together(self) -> None:
        crops = [
            CropRect(1424, 1072, 248, 4),
            CropRect(1426, 1072, 246, 4),
            CropRect(1422, 1072, 250, 4),
            CropRect(1424, 1072, 248, 4),
            CropRect(1424, 1072, 248, 4),
        ]
        assert aggregate_crop(crops) == CropRect(1424, 1072, 248, 4)

    def test_all_full_frame_returns_full_frame(self) -> None:
        crops = [CropRect(1920, 1080, 0, 0)] * 10
        assert aggregate_crop(crops) == CropRect(1920, 1080, 0, 0)

    def test_odd_horizontal_offsets_snap_to_even(self) -> None:
        crops = [CropRect(1646, 1080, 137, 0)]
        assert aggregate_crop(crops) == CropRect(1648, 1080, 136, 0)

    def test_odd_vertical_offsets_snap_to_even(self) -> None:
        crops = [CropRect(1920, 802, 0, 139)]
        assert aggregate_crop(crops) == CropRect(1920, 804, 0, 138)

    def test_inconsistent_clusters_raise(self) -> None:
        crops = [
            CropRect(0, 0, 100, 100),
            CropRect(0, 0, 100, 100),
            CropRect(0, 0, 100, 100),
            CropRect(50, 50, 0, 0),
            CropRect(40, 40, 10, 10),
            CropRect(30, 30, 20, 20),
            CropRect(10, 10, 40, 40),
        ]
        with pytest.raises(ValueError, match="too inconsistent"):
            aggregate_crop(crops)


class TestCropdetectLimit:
    def test_no_measurements_keeps_the_default(self) -> None:
        assert cropdetect_limit([]) == 40

    def test_picture_on_every_side_keeps_the_default(self) -> None:
        assert cropdetect_limit([130.0, 98.0, 212.0, 176.0]) == 40

    def test_clean_black_bars_keep_the_default(self) -> None:
        assert cropdetect_limit([16.2, 16.0, 15.9, 16.1]) == 40

    def test_noisy_bars_raise_the_limit_above_their_level(self) -> None:
        assert cropdetect_limit([45.2, 43.8, 16.3, 16.4]) == 50

    def test_picture_sides_do_not_pull_the_limit_up(self) -> None:
        assert cropdetect_limit([45.0, 44.0, 130.0, 98.0]) == 49

    def test_bar_level_the_default_already_crops_changes_nothing(self) -> None:
        assert cropdetect_limit([40.0, 34.0, 16.0, 16.0]) == 40

    def test_level_just_above_the_default_raises_the_limit(self) -> None:
        assert cropdetect_limit([40.5, 16.0, 16.0, 16.0]) == 45

    def test_top_of_the_band_yields_the_highest_limit(self) -> None:
        assert cropdetect_limit([48.0, 200.0, 200.0, 200.0]) == 52

    def test_a_dark_picture_edge_above_the_band_is_not_a_bar(self) -> None:
        assert cropdetect_limit([48.1, 200.0, 200.0, 200.0]) == 40

    def test_a_dim_side_outside_the_band_caps_the_limit(self) -> None:
        assert cropdetect_limit([47.1, 16.0, 16.0, 50.0]) == 49

    def test_the_dimmest_side_outside_the_band_sets_the_cap(self) -> None:
        assert cropdetect_limit([47.1, 50.9, 60.0, 200.0]) == 49

    def test_dark_theatre_borders_keep_the_default(self) -> None:
        assert cropdetect_limit([58.4, 86.1, 111.9, 77.0]) == 40


class TestHdrDetectionFractions:
    def test_mastering_display_fractions(self) -> None:
        side_data = [
            {
                "side_data_type": "Mastering display metadata",
                "green_x": "8500/50000",
                "green_y": "39850/50000",
                "blue_x": "6550/50000",
                "blue_y": "2300/50000",
                "red_x": "35400/50000",
                "red_y": "14600/50000",
                "white_point_x": "15635/50000",
                "white_point_y": "16450/50000",
                "max_luminance": "10000000/10000",
                "min_luminance": "1/10000",
            }
        ]
        result = detect_hdr({}, side_data)
        assert result.mastering_display == ("G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(10000000,1)")

    def test_mastering_display_integers(self) -> None:
        side_data = [
            {
                "side_data_type": "Mastering display metadata",
                "green_x": "8500",
                "green_y": "39850",
                "blue_x": "6550",
                "blue_y": "2300",
                "red_x": "35400",
                "red_y": "14600",
                "white_point_x": "15635",
                "white_point_y": "16450",
                "max_luminance": "10000000",
                "min_luminance": "1",
            }
        ]
        result = detect_hdr({}, side_data)
        assert result.mastering_display == ("G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(10000000,1)")

    def test_mastering_display_decimal_passthrough(self) -> None:
        side_data = [
            {
                "side_data_type": "Mastering display metadata",
                "green_x": "0.2650",
                "green_y": "0.6900",
                "blue_x": "0.1500",
                "blue_y": "0.0600",
                "red_x": "0.6800",
                "red_y": "0.3200",
                "white_point_x": "0.3127",
                "white_point_y": "0.3290",
                "max_luminance": "1000.0000",
                "min_luminance": "0.0050",
            }
        ]
        result = detect_hdr({}, side_data)
        assert result.mastering_display is not None
        assert "G(0.2650,0.6900)" in result.mastering_display

    def test_content_light_integers(self) -> None:
        side_data = [
            {
                "side_data_type": "Content light level metadata",
                "max_content": 1000,
                "max_average": 180,
            }
        ]
        result = detect_hdr({}, side_data)
        assert result.content_light == "MaxCLL=1000,MaxFALL=180"


class TestResolveColorMetadataUnknownMatrix:
    def test_unknown_matrix_raises(self) -> None:
        with pytest.raises(ValueError, match="Unrecognized matrix_raw"):
            resolve_color_metadata(
                matrix_raw="foobar",
                transfer_raw=None,
                primaries_raw=None,
                system=VideoSystem.HD,
                has_hdr=False,
            )


class TestHdrTransferForCropdetect:
    @pytest.mark.parametrize(
        ("color_transfer", "expected"),
        [
            ("smpte2084", "smpte2084"),
            ("arib-std-b67", "arib-std-b67"),
            ("bt709", None),
            ("smpte170m", None),
            (None, None),
        ],
    )
    def test_hdr_tonemap_transfer(
        self,
        color_transfer: str | None,
        expected: str | None,
    ) -> None:
        assert hdr_tonemap_transfer(color_transfer) == expected


def _vi(
    *,
    interlaced: bool = False,
    dv: bool = False,
    dv_profile: int | None = None,
) -> VideoInfo:
    return VideoInfo(
        index=0,
        codec_name="hevc",
        width=1920,
        height=1080,
        pixel_area=1920 * 1080,
        fps_num=24,
        fps_den=1,
        duration_s=10.0,
        interlaced=interlaced,
        color_matrix_raw=None,
        color_range=None,
        color_transfer=None,
        color_primaries=None,
        pix_fmt="yuv420p10le",
        hdr=HdrMetadata(is_dolby_vision=dv, dv_profile=dv_profile),
        source_file=Path("x.mkv"),
    )


def test_classify_passthrough_disabled_returns_encode() -> None:
    assert classify_passthrough(_vi(), copy_video=False) == (False, None)


def test_classify_passthrough_interlaced_falls_back() -> None:
    assert classify_passthrough(_vi(interlaced=True), copy_video=True) == (False, "interlaced")


def test_classify_passthrough_dv_p7_fel_falls_back() -> None:
    assert DV_PROFILE_FEL == 7
    assert classify_passthrough(_vi(dv=True, dv_profile=7), copy_video=True) == (False, "DV P7 FEL")


def test_classify_passthrough_eligible_passes_through() -> None:
    assert classify_passthrough(_vi(dv=True, dv_profile=8), copy_video=True) == (True, None)
