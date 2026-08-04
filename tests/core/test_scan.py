from __future__ import annotations

import pytest

from furnace.core.models import CropRect, VideoParams
from furnace.core.outdated import EncoderFamily
from furnace.core.scan import (
    AudioTrackSummary,
    ScanRow,
    SubtitleTrackSummary,
    VideoSummary,
    bit_depth_from_pix_fmt,
    hdr_label,
    parse_crop_rescale,
    parse_encoder_family,
    parse_furnace_version,
    parse_version_arg,
    row_matches,
    summarize_streams,
)


def test_parse_furnace_version_valid_tag() -> None:
    assert parse_furnace_version("Furnace v1.19.3") == (1, 19, 3)


def test_parse_furnace_version_multi_digit() -> None:
    assert parse_furnace_version("Furnace v12.345.6789") == (12, 345, 6789)


def test_parse_furnace_version_none() -> None:
    assert parse_furnace_version(None) is None


def test_parse_furnace_version_foreign_tag() -> None:
    assert parse_furnace_version("Lavf60.16.100") is None


def test_parse_furnace_version_malformed_missing_patch() -> None:
    assert parse_furnace_version("Furnace v1.19") is None


def test_parse_furnace_version_malformed_no_v() -> None:
    assert parse_furnace_version("Furnace 1.19.3") is None


def test_parse_furnace_version_malformed_trailing() -> None:
    assert parse_furnace_version("Furnace v1.19.3-dirty") is None


def test_parse_furnace_version_malformed_leading() -> None:
    assert parse_furnace_version("x Furnace v1.19.3") is None


def test_parse_furnace_version_empty() -> None:
    assert parse_furnace_version("") is None


def test_parse_version_arg_valid() -> None:
    assert parse_version_arg("1.19.3") == (1, 19, 3)


def test_parse_version_arg_multi_digit() -> None:
    assert parse_version_arg("10.0.255") == (10, 0, 255)


@pytest.mark.parametrize(
    "bad",
    ["1.19", "1", "1.19.3.4", "v1.19.3", "1.19.x", "", "1.19.", "a.b.c"],
)
def test_parse_version_arg_raises(bad: str) -> None:
    with pytest.raises(ValueError, match="Invalid version"):
        parse_version_arg(bad)


def test_summarize_streams_multi_audio_and_subs() -> None:
    probe = {
        "streams": [
            {"codec_type": "video", "codec_name": "hevc"},
            {
                "codec_type": "audio",
                "codec_name": "ac3",
                "channels": 6,
                "tags": {"language": "rus"},
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "channels": 2,
                "tags": {"language": "eng"},
            },
            {
                "codec_type": "subtitle",
                "codec_name": "subrip",
                "tags": {"language": "eng"},
            },
            {
                "codec_type": "subtitle",
                "codec_name": "hdmv_pgs_subtitle",
                "tags": {"language": "rus"},
            },
        ]
    }
    video, audio, subs = summarize_streams(probe)
    assert video == VideoSummary(codec="hevc", bit_depth=None, hdr="SDR")
    assert audio == (
        AudioTrackSummary(language="rus", codec="ac3", channels=6),
        AudioTrackSummary(language="eng", codec="aac", channels=2),
    )
    assert subs == (
        SubtitleTrackSummary(language="eng", codec="subrip"),
        SubtitleTrackSummary(language="rus", codec="hdmv_pgs_subtitle"),
    )


def test_summarize_streams_container_mastering_display_present() -> None:
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "av1",
                "color_transfer": "smpte2084",
                "side_data_list": [
                    {"side_data_type": "Mastering display metadata", "max_luminance": "1000/1"}
                ],
            }
        ]
    }
    video, _, _ = summarize_streams(probe)
    assert video.container_mastering_display is True


def test_summarize_streams_container_mastering_display_absent() -> None:
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "av1",
                "color_transfer": "smpte2084",
                "side_data_list": [
                    {"side_data_type": "Content light level metadata", "max_content": 1330}
                ],
            }
        ]
    }
    video, _, _ = summarize_streams(probe)
    assert video.container_mastering_display is False


def test_summarize_streams_container_mastering_display_no_side_data() -> None:
    probe = {"streams": [{"codec_type": "video", "codec_name": "av1"}]}
    video, _, _ = summarize_streams(probe)
    assert video.container_mastering_display is False


def test_summarize_streams_no_video_stream_has_no_container_mastering() -> None:
    video, _, _ = summarize_streams({"streams": []})
    assert video.container_mastering_display is False


def test_summarize_streams_missing_language_is_none() -> None:
    probe = {
        "streams": [
            {"codec_type": "audio", "codec_name": "ac3", "channels": 2},
            {"codec_type": "subtitle", "codec_name": "subrip"},
        ]
    }
    video, audio, subs = summarize_streams(probe)
    assert video == VideoSummary(None, None, None)
    assert audio == (AudioTrackSummary(language=None, codec="ac3", channels=2),)
    assert subs == (SubtitleTrackSummary(language=None, codec="subrip"),)


def test_summarize_streams_no_video_stream() -> None:
    probe = {
        "streams": [
            {"codec_type": "audio", "codec_name": "flac", "channels": 8, "tags": {"language": "jpn"}},
        ]
    }
    video, audio, subs = summarize_streams(probe)
    assert video == VideoSummary(None, None, None)
    assert audio == (AudioTrackSummary(language="jpn", codec="flac", channels=8),)
    assert subs == ()


def test_summarize_streams_missing_channels_is_none() -> None:
    probe = {
        "streams": [
            {"codec_type": "audio", "codec_name": "ac3", "tags": {"language": "eng"}},
        ]
    }
    _, audio, _ = summarize_streams(probe)
    assert audio == (AudioTrackSummary(language="eng", codec="ac3", channels=None),)


def test_summarize_streams_missing_codec_name_is_unknown() -> None:
    probe = {
        "streams": [
            {"codec_type": "video"},
            {"codec_type": "audio", "tags": {"language": "eng"}},
            {"codec_type": "subtitle", "tags": {"language": "eng"}},
        ]
    }
    video, audio, subs = summarize_streams(probe)
    assert video == VideoSummary(codec="unknown", bit_depth=None, hdr="SDR")
    assert audio == (AudioTrackSummary(language="eng", codec="unknown", channels=None),)
    assert subs == (SubtitleTrackSummary(language="eng", codec="unknown"),)


def test_summarize_streams_empty_streams() -> None:
    assert summarize_streams({"streams": []}) == (VideoSummary(None, None, None), (), ())


def test_summarize_streams_missing_streams_key() -> None:
    assert summarize_streams({}) == (VideoSummary(None, None, None), (), ())


def test_summarize_streams_first_video_wins() -> None:
    probe = {
        "streams": [
            {"codec_type": "video", "codec_name": "mpeg2video"},
            {"codec_type": "video", "codec_name": "hevc"},
        ]
    }
    video, _, _ = summarize_streams(probe)
    assert video.codec == "mpeg2video"


def test_summarize_streams_video_with_bit_depth_and_hdr() -> None:
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "hevc",
                "pix_fmt": "yuv420p10le",
                "color_transfer": "smpte2084",
            },
        ]
    }
    video, _, _ = summarize_streams(probe)
    assert video == VideoSummary(codec="hevc", bit_depth=10, hdr="HDR10", color_transfer="smpte2084")


def test_summarize_streams_populates_color_transfer() -> None:
    probe = {
        "streams": [
            {"codec_type": "video", "codec_name": "av1", "color_transfer": "smpte2084"},
        ]
    }
    video, _, _ = summarize_streams(probe)
    assert video.color_transfer == "smpte2084"


def test_summarize_streams_missing_color_transfer_is_none() -> None:
    probe = {"streams": [{"codec_type": "video", "codec_name": "av1"}]}
    video, _, _ = summarize_streams(probe)
    assert video.color_transfer is None


def test_summarize_streams_video_dolby_vision_from_side_data() -> None:
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "hevc",
                "pix_fmt": "yuv420p10le",
                "side_data_list": [
                    {"side_data_type": "DOVI configuration record", "dv_profile": 8},
                ],
            },
        ]
    }
    video, _, _ = summarize_streams(probe)
    assert video == VideoSummary(codec="hevc", bit_depth=10, hdr="DV P8")


def test_summarize_streams_populates_width_height_matrix() -> None:
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "av1",
                "width": 1920,
                "height": 1080,
                "color_space": "bt709",
            },
        ]
    }
    video, _, _ = summarize_streams(probe)
    assert video.width == 1920
    assert video.height == 1080
    assert video.color_matrix == "bt709"


def test_summarize_streams_missing_geometry_is_none() -> None:
    probe = {"streams": [{"codec_type": "video", "codec_name": "av1"}]}
    video, _, _ = summarize_streams(probe)
    assert video.width is None
    assert video.height is None
    assert video.color_matrix is None


def test_summarize_streams_color_space_unknown_passed_through() -> None:
    probe = {
        "streams": [
            {"codec_type": "video", "codec_name": "av1", "color_space": "unknown"},
        ]
    }
    video, _, _ = summarize_streams(probe)
    assert video.color_matrix == "unknown"


def test_summarize_streams_no_video_stream_has_no_geometry() -> None:
    probe = {"streams": [{"codec_type": "audio", "codec_name": "aac", "channels": 2}]}
    video, _, _ = summarize_streams(probe)
    assert video == VideoSummary(None, None, None)
    assert video.width is None
    assert video.height is None
    assert video.color_matrix is None


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ("hevc_nvenc / NVEncC=7.00 / main", EncoderFamily.HEVC_NVENC),
        ("av1_nvenc / NVEncC=8.00 / main", EncoderFamily.AV1_NVENC),
        ("av1_svt / SVT-AV1 / preset=4", EncoderFamily.AV1_SVT),
        ("video stream copied (passthrough)", EncoderFamily.PASSTHROUGH),
        (None, EncoderFamily.UNKNOWN),
        ("x265 / crf=18", EncoderFamily.UNKNOWN),
        ("", EncoderFamily.UNKNOWN),
    ],
)
def test_parse_encoder_family(settings: str | None, expected: EncoderFamily) -> None:
    assert parse_encoder_family(settings) == expected


@pytest.mark.parametrize(
    "family",
    [
        EncoderFamily.HEVC_NVENC,
        EncoderFamily.AV1_NVENC,
        EncoderFamily.AV1_SVT,
        EncoderFamily.PASSTHROUGH,
    ],
)
def test_parse_encoder_family_matches_enum_value(family: EncoderFamily) -> None:
    assert parse_encoder_family(f"{family.value} / trailing tokens") is family


def test_scan_row_outdated_fields_default() -> None:
    from pathlib import Path

    row = ScanRow(
        path=Path("a.mkv"),
        furnace_version=(2, 9, 0),
        video=VideoSummary(codec="av1", bit_depth=10, hdr="SDR"),
        audio=(),
        subtitles=(),
    )
    assert row.encoder_family is EncoderFamily.UNKNOWN
    assert row.defects == ()


def test_bit_depth_from_pix_fmt_none() -> None:
    assert bit_depth_from_pix_fmt(None) is None


def test_bit_depth_from_pix_fmt_empty() -> None:
    assert bit_depth_from_pix_fmt("") is None


@pytest.mark.parametrize(
    ("pix_fmt", "expected"),
    [
        ("yuv420p", 8),
        ("yuv444p", 8),
        ("yuv420p10le", 10),
        ("yuv420p12le", 12),
        ("yuv420p12be", 12),
        ("gbrp10le", 10),
        ("p010le", 10),
    ],
)
def test_bit_depth_from_pix_fmt_values(pix_fmt: str, expected: int) -> None:
    assert bit_depth_from_pix_fmt(pix_fmt) == expected


def test_hdr_label_sdr_transfer_bt709() -> None:
    stream = {"codec_name": "hevc", "color_transfer": "bt709"}
    assert hdr_label(stream, []) == "SDR"


def test_hdr_label_sdr_transfer_absent() -> None:
    stream = {"codec_name": "hevc"}
    assert hdr_label(stream, []) == "SDR"


def test_hdr_label_hdr10() -> None:
    stream = {"codec_name": "hevc", "color_transfer": "smpte2084"}
    assert hdr_label(stream, []) == "HDR10"


def test_hdr_label_hlg() -> None:
    stream = {"codec_name": "hevc", "color_transfer": "arib-std-b67"}
    assert hdr_label(stream, []) == "HLG"


def test_hdr_label_dolby_vision_no_profile() -> None:
    stream = {"codec_name": "dvh1"}
    assert hdr_label(stream, []) == "DV"


def test_hdr_label_dolby_vision_with_profile() -> None:
    stream = {"codec_name": "hevc"}
    side_data = [{"side_data_type": "DOVI configuration record", "dv_profile": 8}]
    assert hdr_label(stream, side_data) == "DV P8"


def test_hdr_label_dolby_vision_wins_over_smpte2084() -> None:
    stream = {"codec_name": "dvh1", "color_transfer": "smpte2084"}
    assert hdr_label(stream, []) == "DV"


def test_scan_row_defaults_unreadable_false() -> None:
    from pathlib import Path

    row = ScanRow(
        path=Path("a.mkv"),
        furnace_version=(1, 19, 3),
        video=VideoSummary(codec="hevc", bit_depth=10, hdr="SDR"),
        audio=(),
        subtitles=(),
    )
    assert row.unreadable is False
    assert row.furnace_version == (1, 19, 3)
    assert row.video == VideoSummary(codec="hevc", bit_depth=10, hdr="SDR")


def test_scan_row_unreadable() -> None:
    from pathlib import Path

    row = ScanRow(
        path=Path("bad.mkv"),
        furnace_version=None,
        video=VideoSummary(None, None, None),
        audio=(),
        subtitles=(),
        unreadable=True,
    )
    assert row.unreadable is True


def test_row_matches_no_predicate_encoded() -> None:
    assert row_matches((1, 19, 3), not_encoded=False, encoded=False, max_version=None) is True


def test_row_matches_no_predicate_not_encoded() -> None:
    assert row_matches(None, not_encoded=False, encoded=False, max_version=None) is True


def test_row_matches_not_encoded_true_when_none() -> None:
    assert row_matches(None, not_encoded=True, encoded=False, max_version=None) is True


def test_row_matches_not_encoded_false_when_version() -> None:
    assert row_matches((1, 0, 0), not_encoded=True, encoded=False, max_version=None) is False


def test_row_matches_encoded_true_when_version() -> None:
    assert row_matches((1, 0, 0), not_encoded=False, encoded=True, max_version=None) is True


def test_row_matches_encoded_false_when_none() -> None:
    assert row_matches(None, not_encoded=False, encoded=True, max_version=None) is False


def test_row_matches_max_version_below() -> None:
    assert row_matches((1, 19, 2), not_encoded=False, encoded=False, max_version=(1, 19, 3)) is True


def test_row_matches_max_version_boundary_equal() -> None:
    assert row_matches((1, 19, 3), not_encoded=False, encoded=False, max_version=(1, 19, 3)) is True


def test_row_matches_max_version_above() -> None:
    assert row_matches((1, 19, 4), not_encoded=False, encoded=False, max_version=(1, 19, 3)) is False


def test_row_matches_max_version_none_version() -> None:
    assert row_matches(None, not_encoded=False, encoded=False, max_version=(1, 19, 3)) is False


def test_row_matches_union_not_encoded_or_max_version() -> None:
    assert row_matches(None, not_encoded=True, encoded=False, max_version=(1, 19, 3)) is True


def test_row_matches_union_encoded_or_max_version() -> None:
    assert row_matches((2, 0, 0), not_encoded=False, encoded=True, max_version=(1, 19, 3)) is True


def test_row_matches_union_all_false() -> None:
    assert row_matches((2, 0, 0), not_encoded=True, encoded=False, max_version=(1, 19, 3)) is False


_NVENC_PREFIX = "av1_nvenc / NVEncC=9.29 / main / output-depth=10 / qvbr=30"
_SVT_PREFIX = "av1_svt / SVT-AV1 / preset=4 / crf=23"


def _off_grid_uhd_vp() -> VideoParams:
    """A crop whose recorded fields are deliberately NOT all multiples of 8.

    Both tags then change meaning under a field-order swap, so the round-trip
    tests catch a writer that reorders them.
    """
    return _uhd_vp(CropRect(w=3840, h=1606, x=0, y=270))


def _uhd_vp(crop: CropRect) -> VideoParams:
    return VideoParams(
        cq=30,
        crop=crop,
        deinterlace=False,
        color_matrix="bt2020nc",
        color_range="tv",
        color_transfer="smpte2084",
        color_primaries="bt2020",
        hdr=None,
        gop=120,
        fps_num=24000,
        fps_den=1001,
        source_width=3840,
        source_height=2160,
        source_codec="hevc",
    )


def test_parse_crop_rescale_none_settings() -> None:
    assert parse_crop_rescale(None, EncoderFamily.AV1_NVENC, None) is None


def test_parse_crop_rescale_no_crop_tag() -> None:
    assert parse_crop_rescale(_NVENC_PREFIX, EncoderFamily.AV1_NVENC, None) is None


def test_parse_crop_rescale_nvencc_on_grid_crop() -> None:
    settings = f"{_NVENC_PREFIX} / crop=276:276:0:0 / dolby-vision=10.1"
    assert parse_crop_rescale(settings, EncoderFamily.AV1_NVENC, None) is False


def test_parse_crop_rescale_nvencc_off_grid_height() -> None:
    settings = f"{_NVENC_PREFIX} / crop=276:278:0:0 / dolby-vision=10.1"
    assert parse_crop_rescale(settings, EncoderFamily.AV1_NVENC, None) is True


def test_parse_crop_rescale_nvencc_off_grid_width() -> None:
    settings = f"{_NVENC_PREFIX} / crop=0:0:6:0"
    assert parse_crop_rescale(settings, EncoderFamily.AV1_NVENC, None) is True


def test_parse_crop_rescale_hevc_nvencc_reads_same_layout() -> None:
    settings = "hevc_nvenc / NVEncC=7.0 / main / crop=276:278:0:0"
    assert parse_crop_rescale(settings, EncoderFamily.HEVC_NVENC, None) is True


def test_parse_crop_rescale_unknown_family() -> None:
    assert parse_crop_rescale("x264 / crop=276:278:0:0", EncoderFamily.UNKNOWN, None) is None


def test_parse_crop_rescale_passthrough_family() -> None:
    settings = "video stream copied (passthrough) / crop=276:278:0:0"
    assert parse_crop_rescale(settings, EncoderFamily.PASSTHROUGH, None) is None


def test_parse_crop_rescale_malformed_crop_tag() -> None:
    settings = f"{_NVENC_PREFIX} / crop=276:278 / dolby-vision=10.1"
    assert parse_crop_rescale(settings, EncoderFamily.AV1_NVENC, None) is None


def test_parse_crop_rescale_ignores_lookalike_key() -> None:
    settings = f"{_NVENC_PREFIX} / autocrop=276:278:0:0"
    assert parse_crop_rescale(settings, EncoderFamily.AV1_NVENC, None) is None


def test_parse_crop_rescale_nvencc_asymmetric_but_on_grid() -> None:
    settings = f"{_NVENC_PREFIX} / crop=2:6:0:0"
    assert parse_crop_rescale(settings, EncoderFamily.AV1_NVENC, None) is False


def test_parse_crop_rescale_nvencc_misreads_an_off_grid_source() -> None:
    """Known limit: the NVEncC tag records removed pixels, not the kept size.

    A 1920x804 source cropped to 1920x720 removes 42+42 and was never rescaled,
    yet reads as a rescale. Only the version gate keeps this out of the scan.
    """
    settings = f"{_NVENC_PREFIX} / crop=42:42:0:0"
    assert parse_crop_rescale(settings, EncoderFamily.AV1_NVENC, None) is True


class TestSvtCropRescale:
    """SVT records the rectangle it kept, so the file's own size settles it.

    Every case below is a real shape measured in the library: an axis that came
    out smaller was squashed, one that came out larger was stretched to square
    pixels, and one that matches was never resampled.
    """

    def test_axis_shrank_was_squashed(self) -> None:
        settings = f"{_SVT_PREFIX} / crop=720:574:0:2"
        assert parse_crop_rescale(settings, EncoderFamily.AV1_SVT, (768, 568)) is True

    def test_width_shrank_was_squashed(self) -> None:
        settings = f"{_SVT_PREFIX} / crop=1910:800:5:140"
        assert parse_crop_rescale(settings, EncoderFamily.AV1_SVT, (1904, 800)) is True

    def test_axis_grew_is_an_anamorphic_stretch(self) -> None:
        settings = f"{_SVT_PREFIX} / crop=718:576:2:0"
        assert parse_crop_rescale(settings, EncoderFamily.AV1_SVT, (760, 576)) is False

    def test_size_unchanged_is_clean(self) -> None:
        settings = f"{_SVT_PREFIX} / crop=1920:800:0:140"
        assert parse_crop_rescale(settings, EncoderFamily.AV1_SVT, (1920, 800)) is False

    def test_one_axis_stretched_the_other_squashed(self) -> None:
        settings = f"{_SVT_PREFIX} / crop=718:430:2:73"
        assert parse_crop_rescale(settings, EncoderFamily.AV1_SVT, (1016, 424)) is True

    def test_off_grid_kept_size_alone_is_not_a_rescale(self) -> None:
        # 702 is off the grid, but the width was stretched and the height never
        # moved -- the old mod-8 rule called 16 library files rescaled here.
        settings = f"{_SVT_PREFIX} / crop=702:568:9:4"
        assert parse_crop_rescale(settings, EncoderFamily.AV1_SVT, (992, 568)) is False

    def test_without_the_output_size_it_cannot_tell(self) -> None:
        settings = f"{_SVT_PREFIX} / crop=1910:798:5:141"
        assert parse_crop_rescale(settings, EncoderFamily.AV1_SVT, None) is None


def test_parse_crop_rescale_round_trips_the_nvencc_writer() -> None:
    """The reader's field order has to match what the adapter actually writes."""
    from pathlib import Path

    from furnace.adapters.nvencc import NVEncCAdapter

    adapter = NVEncCAdapter(Path("NVEncC64.exe"))
    adapter._version_cached = "9.29"  # keep the version probe off the real binary
    written = adapter._build_encoder_settings(_off_grid_uhd_vp())
    assert "crop=274:286:0:0" in written
    assert parse_crop_rescale(written, EncoderFamily.AV1_NVENC, None) is False


def test_parse_crop_rescale_round_trips_the_svt_writer() -> None:
    from pathlib import Path

    from furnace.adapters.svtav1 import SvtAv1Adapter

    adapter = SvtAv1Adapter(Path("ffmpeg.exe"))
    written = adapter._build_encoder_settings(_off_grid_uhd_vp())
    assert "crop=3840:1600:0:274" in written
    assert parse_crop_rescale(written, EncoderFamily.AV1_SVT, (3840, 1600)) is False
