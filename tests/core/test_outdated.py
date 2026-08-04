from __future__ import annotations

from collections.abc import Sequence

import pytest

from furnace.core.outdated import (
    Defect,
    EncoderFamily,
    Fix,
    Severity,
    classify_outdated,
    row_fix,
    row_severity,
)
from furnace.core.scan import parse_encoder_family


def classify(**overrides: object) -> tuple[Defect, ...]:
    params: dict[str, object] = {
        "unreadable": False,
        "version": (2, 9, 0),
        "encoder_family": EncoderFamily.AV1_NVENC,
        "codec": "av1",
        "height": 1080,
        "color_matrix": "bt709",
        "color_transfer": None,
        "audio_channels": (6,),
        "container_mastering_display": False,
        "crop_rescaled": None,
    }
    params.update(overrides)
    return classify_outdated(**params)  # type: ignore[arg-type]


def _reasons(defects: Sequence[Defect]) -> list[str]:
    return [d.reason for d in defects]


class TestParseEncoderFamily:
    def test_hevc_nvenc_prefix(self) -> None:
        assert parse_encoder_family("hevc_nvenc / NVEncC=7.0 / main") == EncoderFamily.HEVC_NVENC

    def test_av1_nvenc_prefix(self) -> None:
        assert parse_encoder_family("av1_nvenc / NVEncC=8.0 / main") == EncoderFamily.AV1_NVENC

    def test_av1_svt_prefix(self) -> None:
        assert parse_encoder_family("av1_svt / SVT-AV1 / preset=4") == EncoderFamily.AV1_SVT

    def test_passthrough_string(self) -> None:
        assert parse_encoder_family("video stream copied (passthrough)") == EncoderFamily.PASSTHROUGH

    def test_none_is_unknown(self) -> None:
        assert parse_encoder_family(None) == EncoderFamily.UNKNOWN

    def test_unrecognized_is_unknown(self) -> None:
        assert parse_encoder_family("x264 / crf=18") == EncoderFamily.UNKNOWN

    def test_empty_is_unknown(self) -> None:
        assert parse_encoder_family("") == EncoderFamily.UNKNOWN


class TestEnums:
    def test_severity_order_worst_first(self) -> None:
        ordered = sorted(Severity, key=lambda s: s.order)
        assert ordered == [
            Severity.SYNC,
            Severity.FOREIGN,
            Severity.QUALITY,
            Severity.COSMETIC,
            Severity.UNREADABLE,
        ]

    def test_severity_labels(self) -> None:
        assert Severity.SYNC.label == "SYNC"
        assert Severity.FOREIGN.label == "FOREIGN"
        assert Severity.QUALITY.label == "QUALITY"
        assert Severity.COSMETIC.label == "COSMETIC"
        assert Severity.UNREADABLE.label == "UNREADABLE"

    def test_fix_strength_order(self) -> None:
        assert Fix.REMUX.strength < Fix.RE_ENCODE.strength < Fix.RE_RUN.strength
        assert Fix.NONE.strength < Fix.REMUX.strength

    def test_fix_labels(self) -> None:
        assert Fix.REMUX.label == "REMUX"
        assert Fix.RE_ENCODE.label == "RE-ENCODE"
        assert Fix.RE_RUN.label == "RE-RUN"
        assert Fix.NONE.label == "—"


class TestUnreadable:
    def test_unreadable_single_defect(self) -> None:
        defects = classify(unreadable=True)
        assert defects == (Defect(reason="unreadable", severity=Severity.UNREADABLE, fix=Fix.NONE),)

    def test_unreadable_wins_over_everything(self) -> None:
        defects = classify(unreadable=True, version=None, encoder_family=EncoderFamily.HEVC_NVENC)
        assert _reasons(defects) == ["unreadable"]


class TestForeign:
    def test_foreign_uses_video_codec_as_reason(self) -> None:
        defects = classify(version=None, codec="h264")
        assert defects == (Defect(reason="h264", severity=Severity.FOREIGN, fix=Fix.RE_ENCODE),)

    def test_foreign_no_video_stream_reason_unknown(self) -> None:
        defects = classify(version=None, codec=None)
        assert defects == (Defect(reason="unknown", severity=Severity.FOREIGN, fix=Fix.RE_ENCODE),)

    def test_foreign_ignores_encoder_family(self) -> None:
        defects = classify(version=None, codec="vp9", encoder_family=EncoderFamily.AV1_NVENC)
        assert _reasons(defects) == ["vp9"]


class TestSupersededCodec:
    def test_hevc_encode_flagged(self) -> None:
        defects = classify(encoder_family=EncoderFamily.HEVC_NVENC, version=(1, 19, 3))
        assert defects == (Defect(reason="superseded codec", severity=Severity.QUALITY, fix=Fix.RE_ENCODE),)

    def test_hevc_encode_subsumes_all_other_defects(self) -> None:
        defects = classify(
            encoder_family=EncoderFamily.HEVC_NVENC,
            version=(1, 19, 3),
            height=480,
            color_matrix=None,
        )
        assert _reasons(defects) == ["superseded codec"]

    def test_hevc_encode_subsumes_mono_downmix(self) -> None:
        defects = classify(
            encoder_family=EncoderFamily.HEVC_NVENC,
            version=(2, 0, 0),
            audio_channels=(1,),
        )
        assert _reasons(defects) == ["superseded codec"]

    def test_av1_not_superseded(self) -> None:
        assert _reasons(classify(encoder_family=EncoderFamily.AV1_NVENC)) == []


class TestCrop4px:
    @pytest.mark.parametrize("family", [EncoderFamily.AV1_NVENC, EncoderFamily.AV1_SVT])
    def test_fires_below_threshold(self, family: EncoderFamily) -> None:
        defects = classify(encoder_family=family, version=(2, 1, 1))
        assert Defect("crop 4px", Severity.QUALITY, Fix.RE_ENCODE) in defects

    def test_boundary_exactly_2_1_2_does_not_fire(self) -> None:
        assert "crop 4px" not in _reasons(classify(version=(2, 1, 2)))

    def test_not_av1_does_not_fire(self) -> None:
        assert "crop 4px" not in _reasons(classify(encoder_family=EncoderFamily.PASSTHROUGH, version=(2, 1, 1)))


class TestFpsDrift:
    @pytest.mark.parametrize("family", [EncoderFamily.AV1_NVENC, EncoderFamily.AV1_SVT])
    def test_fires_below_threshold(self, family: EncoderFamily) -> None:
        defects = classify(encoder_family=family, version=(2, 1, 3))
        assert Defect("fps drift", Severity.SYNC, Fix.REMUX) in defects

    def test_boundary_exactly_2_1_4_does_not_fire(self) -> None:
        assert "fps drift" not in _reasons(classify(version=(2, 1, 4)))

    def test_not_av1_does_not_fire(self) -> None:
        assert "fps drift" not in _reasons(classify(encoder_family=EncoderFamily.PASSTHROUGH, version=(2, 1, 3)))


class TestSoftTelecine:
    @pytest.mark.parametrize("height", [480, 486, 240])
    def test_fires_for_ntsc_heights(self, height: int) -> None:
        defects = classify(encoder_family=EncoderFamily.AV1_NVENC, version=(2, 5, 0), height=height)
        assert Defect("soft telecine", Severity.SYNC, Fix.REMUX) in defects

    def test_boundary_start_2_1_4_fires(self) -> None:
        assert "soft telecine" in _reasons(classify(version=(2, 1, 4), height=480))

    def test_boundary_end_2_6_0_does_not_fire(self) -> None:
        assert "soft telecine" not in _reasons(classify(version=(2, 6, 0), height=480))

    def test_below_2_1_4_does_not_fire_telecine(self) -> None:
        defects = classify(version=(2, 1, 3), height=480)
        assert "soft telecine" not in _reasons(defects)
        assert "fps drift" in _reasons(defects)

    def test_non_ntsc_height_does_not_fire(self) -> None:
        assert "soft telecine" not in _reasons(classify(version=(2, 5, 0), height=576))

    def test_none_height_does_not_fire(self) -> None:
        assert "soft telecine" not in _reasons(classify(version=(2, 5, 0), height=None))

    def test_svt_family_does_not_fire(self) -> None:
        assert "soft telecine" not in _reasons(
            classify(encoder_family=EncoderFamily.AV1_SVT, version=(2, 5, 0), height=480)
        )


class TestSoftQvbr:
    def test_fires_for_high_res(self) -> None:
        defects = classify(encoder_family=EncoderFamily.AV1_NVENC, version=(2, 1, 5), height=2160)
        assert Defect("soft QVBR", Severity.QUALITY, Fix.RE_ENCODE) in defects

    def test_boundary_height_1440_fires(self) -> None:
        assert "soft QVBR" in _reasons(classify(version=(2, 1, 5), height=1440))

    def test_boundary_height_1439_does_not_fire(self) -> None:
        assert "soft QVBR" not in _reasons(classify(version=(2, 1, 5), height=1439))

    def test_boundary_version_2_2_0_does_not_fire(self) -> None:
        assert "soft QVBR" not in _reasons(classify(version=(2, 2, 0), height=2160))

    def test_none_height_does_not_fire(self) -> None:
        assert "soft QVBR" not in _reasons(classify(version=(2, 1, 5), height=None))

    def test_svt_family_does_not_fire(self) -> None:
        assert "soft QVBR" not in _reasons(
            classify(encoder_family=EncoderFamily.AV1_SVT, version=(2, 1, 5), height=2160)
        )


class TestSoftQvbrHdr:
    def test_fires_for_pre_target_quality_hdr10(self) -> None:
        defects = classify(
            encoder_family=EncoderFamily.AV1_NVENC,
            version=(2, 9, 0),
            height=1600,
            color_matrix="bt2020nc",
            color_transfer="smpte2084",
        )
        assert Defect("soft QVBR (HDR)", Severity.QUALITY, Fix.RE_ENCODE) in defects

    def test_fires_for_hlg(self) -> None:
        assert "soft QVBR (HDR)" in _reasons(classify(version=(2, 9, 0), color_transfer="arib-std-b67"))

    def test_boundary_version_2_11_0_does_not_fire(self) -> None:
        assert "soft QVBR (HDR)" not in _reasons(classify(version=(2, 11, 0), color_transfer="smpte2084"))

    def test_above_target_quality_does_not_fire(self) -> None:
        assert "soft QVBR (HDR)" not in _reasons(classify(version=(2, 19, 1), color_transfer="smpte2084"))

    def test_sdr_transfer_does_not_fire(self) -> None:
        assert "soft QVBR (HDR)" not in _reasons(classify(version=(2, 9, 0), color_transfer="bt709"))

    def test_none_transfer_does_not_fire(self) -> None:
        assert "soft QVBR (HDR)" not in _reasons(classify(version=(2, 9, 0), color_transfer=None))

    def test_svt_family_does_not_fire(self) -> None:
        assert "soft QVBR (HDR)" not in _reasons(
            classify(encoder_family=EncoderFamily.AV1_SVT, version=(2, 9, 0), color_transfer="smpte2084")
        )

    def test_passthrough_does_not_fire(self) -> None:
        assert "soft QVBR (HDR)" not in _reasons(
            classify(encoder_family=EncoderFamily.PASSTHROUGH, version=(2, 9, 0), color_transfer="smpte2084")
        )

    def test_fires_independent_of_height(self) -> None:
        assert "soft QVBR (HDR)" in _reasons(classify(version=(2, 9, 0), height=None, color_transfer="smpte2084"))


class TestCropRescale:
    def test_fires_for_off_grid_crop_before_the_fix(self) -> None:
        defects = classify(version=(2, 31, 0), crop_rescaled=True)
        assert Defect("crop rescale", Severity.QUALITY, Fix.RE_ENCODE) in defects

    def test_boundary_version_2_32_0_does_not_fire(self) -> None:
        assert "crop rescale" not in _reasons(classify(version=(2, 32, 0), crop_rescaled=True))

    def test_a_crop_that_was_not_rescaled_does_not_fire(self) -> None:
        assert "crop rescale" not in _reasons(classify(version=(2, 31, 0), crop_rescaled=False))

    def test_unknown_crop_does_not_fire(self) -> None:
        assert "crop rescale" not in _reasons(classify(version=(2, 31, 0), crop_rescaled=None))

    def test_fires_for_svt_family(self) -> None:
        assert "crop rescale" in _reasons(
            classify(
                encoder_family=EncoderFamily.AV1_SVT,
                version=(2, 31, 0),
                crop_rescaled=True,
            )
        )

    def test_passthrough_family_does_not_fire(self) -> None:
        assert "crop rescale" not in _reasons(
            classify(
                encoder_family=EncoderFamily.PASSTHROUGH,
                version=(2, 31, 0),
                crop_rescaled=True,
            )
        )


class TestContainerMasteringDisplay:
    def test_fires_for_pre_fix_hdr10(self) -> None:
        defects = classify(
            encoder_family=EncoderFamily.AV1_NVENC,
            version=(2, 28, 0),
            color_transfer="smpte2084",
        )
        assert Defect("container mastering", Severity.COSMETIC, Fix.REMUX) in defects

    def test_fires_for_hlg(self) -> None:
        assert "container mastering" in _reasons(
            classify(version=(2, 28, 0), color_transfer="arib-std-b67")
        )

    def test_boundary_version_2_29_0_does_not_fire(self) -> None:
        assert "container mastering" not in _reasons(
            classify(version=(2, 29, 0), color_transfer="smpte2084")
        )

    def test_hand_patched_old_file_does_not_fire(self) -> None:
        assert "container mastering" not in _reasons(
            classify(
                version=(2, 28, 0),
                color_transfer="smpte2084",
                container_mastering_display=True,
            )
        )

    def test_sdr_transfer_does_not_fire(self) -> None:
        assert "container mastering" not in _reasons(
            classify(version=(2, 28, 0), color_transfer="bt709")
        )

    def test_none_transfer_does_not_fire(self) -> None:
        assert "container mastering" not in _reasons(
            classify(version=(2, 28, 0), color_transfer=None)
        )

    def test_svt_family_does_not_fire(self) -> None:
        assert "container mastering" not in _reasons(
            classify(
                encoder_family=EncoderFamily.AV1_SVT,
                version=(2, 28, 0),
                color_transfer="smpte2084",
            )
        )

    def test_passthrough_does_not_fire(self) -> None:
        assert "container mastering" not in _reasons(
            classify(
                encoder_family=EncoderFamily.PASSTHROUGH,
                version=(2, 28, 0),
                color_transfer="smpte2084",
            )
        )

    def test_remux_only_does_not_force_reencode(self) -> None:
        defects = classify(version=(2, 28, 0), color_transfer="smpte2084")
        assert row_fix(defects) is Fix.REMUX


class TestSoftQvbrHdrStacking:
    def test_pre_2_2_0_hdr_uhd_stacks_both_qvbr_reasons(self) -> None:
        defects = classify(
            encoder_family=EncoderFamily.AV1_NVENC,
            version=(2, 1, 5),
            height=2160,
            color_matrix="bt2020nc",
            color_transfer="smpte2084",
        )
        assert _reasons(defects) == ["soft QVBR", "soft QVBR (HDR)", "container mastering"]


class TestGrainLoss:
    def test_fires_for_sd(self) -> None:
        defects = classify(encoder_family=EncoderFamily.AV1_NVENC, version=(2, 6, 0), height=480)
        assert Defect("grain loss", Severity.QUALITY, Fix.RE_ENCODE) in defects

    def test_boundary_height_719_fires(self) -> None:
        assert "grain loss" in _reasons(classify(version=(2, 6, 0), height=719))

    def test_boundary_height_720_does_not_fire(self) -> None:
        assert "grain loss" not in _reasons(classify(version=(2, 6, 0), height=720))

    def test_boundary_version_2_7_0_does_not_fire(self) -> None:
        assert "grain loss" not in _reasons(classify(version=(2, 7, 0), height=480))

    def test_none_height_does_not_fire(self) -> None:
        assert "grain loss" not in _reasons(classify(version=(2, 6, 0), height=None))

    def test_svt_family_does_not_fire(self) -> None:
        assert "grain loss" not in _reasons(
            classify(encoder_family=EncoderFamily.AV1_SVT, version=(2, 6, 0), height=480)
        )


class TestColorTags:
    def test_fires_when_matrix_missing(self) -> None:
        defects = classify(version=(2, 7, 1), color_matrix=None)
        assert Defect("color tags", Severity.COSMETIC, Fix.REMUX) in defects

    def test_fires_when_matrix_literal_unknown(self) -> None:
        assert "color tags" in _reasons(classify(version=(2, 7, 1), color_matrix="unknown"))

    def test_fires_when_matrix_empty_string(self) -> None:
        assert "color tags" in _reasons(classify(version=(2, 7, 1), color_matrix=""))

    def test_present_matrix_does_not_fire(self) -> None:
        assert "color tags" not in _reasons(classify(version=(2, 7, 1), color_matrix="bt709"))

    def test_boundary_version_2_7_2_does_not_fire(self) -> None:
        assert "color tags" not in _reasons(classify(version=(2, 7, 2), color_matrix=None))

    def test_fires_regardless_of_encoder_family(self) -> None:
        defects = classify(encoder_family=EncoderFamily.PASSTHROUGH, version=(2, 7, 1), color_matrix=None)
        assert "color tags" in _reasons(defects)


class TestMonoDownmix:
    def test_fires_for_mono_at_2_0_0(self) -> None:
        defects = classify(version=(2, 0, 0), audio_channels=(2, 1))
        assert Defect("mono downmix", Severity.QUALITY, Fix.RE_RUN) in defects

    def test_no_mono_track_does_not_fire(self) -> None:
        assert "mono downmix" not in _reasons(classify(version=(2, 0, 0), audio_channels=(2, 6)))

    def test_no_audio_does_not_fire(self) -> None:
        assert "mono downmix" not in _reasons(classify(version=(2, 0, 0), audio_channels=()))

    def test_other_version_does_not_fire(self) -> None:
        assert "mono downmix" not in _reasons(classify(version=(2, 0, 1), audio_channels=(1,)))

    def test_lower_boundary_1_19_0_does_not_fire(self) -> None:
        defects = classify(encoder_family=EncoderFamily.PASSTHROUGH, version=(1, 19, 0), audio_channels=(1,))
        assert "mono downmix" not in _reasons(defects)

    def test_channels_none_is_ignored(self) -> None:
        assert "mono downmix" not in _reasons(classify(version=(2, 0, 0), audio_channels=(None,)))


class TestMonoDrc:
    def test_fires_for_mono_at_1_19_3(self) -> None:
        defects = classify(encoder_family=EncoderFamily.PASSTHROUGH, version=(1, 19, 3), audio_channels=(2, 1))
        assert defects == (Defect(reason="mono DRC", severity=Severity.QUALITY, fix=Fix.RE_RUN),)

    def test_fires_for_older_version(self) -> None:
        assert "mono DRC" in _reasons(
            classify(encoder_family=EncoderFamily.PASSTHROUGH, version=(1, 15, 0), audio_channels=(1,))
        )

    def test_boundary_1_19_4_does_not_fire(self) -> None:
        assert "mono DRC" not in _reasons(
            classify(encoder_family=EncoderFamily.PASSTHROUGH, version=(1, 19, 4), audio_channels=(1,))
        )

    def test_no_mono_track_does_not_fire(self) -> None:
        assert "mono DRC" not in _reasons(
            classify(encoder_family=EncoderFamily.PASSTHROUGH, version=(1, 19, 3), audio_channels=(2, 6))
        )

    def test_no_audio_does_not_fire(self) -> None:
        assert "mono DRC" not in _reasons(
            classify(encoder_family=EncoderFamily.PASSTHROUGH, version=(1, 19, 3), audio_channels=())
        )

    def test_channels_none_is_ignored(self) -> None:
        assert "mono DRC" not in _reasons(
            classify(encoder_family=EncoderFamily.PASSTHROUGH, version=(1, 19, 3), audio_channels=(None,))
        )

    def test_fires_independent_of_encoder_family(self) -> None:
        assert "mono DRC" in _reasons(
            classify(encoder_family=EncoderFamily.AV1_SVT, version=(1, 19, 0), audio_channels=(1,))
        )

    def test_1_19_3_mono_is_drc_not_downmix(self) -> None:
        reasons = _reasons(classify(encoder_family=EncoderFamily.PASSTHROUGH, version=(1, 19, 3), audio_channels=(1,)))
        assert "mono DRC" in reasons
        assert "mono downmix" not in reasons

    def test_2_0_0_mono_is_downmix_not_drc(self) -> None:
        reasons = _reasons(classify(encoder_family=EncoderFamily.PASSTHROUGH, version=(2, 0, 0), audio_channels=(1,)))
        assert "mono downmix" in reasons
        assert "mono DRC" not in reasons

    def test_hevc_subsumes_mono_drc(self) -> None:
        defects = classify(encoder_family=EncoderFamily.HEVC_NVENC, version=(1, 19, 3), audio_channels=(1,))
        assert _reasons(defects) == ["superseded codec"]

    def test_stacks_sorted_and_rolls_up_re_run(self) -> None:
        defects = classify(
            encoder_family=EncoderFamily.AV1_NVENC,
            version=(1, 19, 3),
            height=480,
            color_matrix=None,
            audio_channels=(1,),
        )
        assert _reasons(defects) == ["fps drift", "crop 4px", "grain loss", "mono DRC", "color tags"]
        assert row_fix(defects) is Fix.RE_RUN


def test_clean_current_file_has_no_defects() -> None:
    assert classify() == ()


def test_clean_passthrough_file_has_no_defects() -> None:
    defects = classify(
        encoder_family=EncoderFamily.PASSTHROUGH,
        version=(2, 1, 0),
        height=1080,
        color_matrix="bt709",
        audio_channels=(6,),
    )
    assert defects == ()


class TestStacking:
    def test_av1_sd_ntsc_pre_2_1_2_stacks_and_sorts(self) -> None:
        defects = classify(
            encoder_family=EncoderFamily.AV1_NVENC,
            version=(2, 0, 5),
            height=480,
            color_matrix=None,
            audio_channels=(6,),
        )
        assert _reasons(defects) == ["fps drift", "crop 4px", "grain loss", "color tags"]
        assert [d.severity for d in defects] == [
            Severity.SYNC,
            Severity.QUALITY,
            Severity.QUALITY,
            Severity.COSMETIC,
        ]

    def test_stack_is_stable_within_same_severity(self) -> None:
        defects = classify(encoder_family=EncoderFamily.AV1_NVENC, version=(2, 0, 5), height=480)
        quality = [d.reason for d in defects if d.severity is Severity.QUALITY]
        assert quality == ["crop 4px", "grain loss"]


class TestRowRollup:
    def test_row_severity_is_worst(self) -> None:
        defects = (
            Defect("color tags", Severity.COSMETIC, Fix.REMUX),
            Defect("fps drift", Severity.SYNC, Fix.REMUX),
            Defect("grain loss", Severity.QUALITY, Fix.RE_ENCODE),
        )
        assert row_severity(defects) is Severity.SYNC

    def test_row_fix_is_strongest(self) -> None:
        defects = (
            Defect("fps drift", Severity.SYNC, Fix.REMUX),
            Defect("grain loss", Severity.QUALITY, Fix.RE_ENCODE),
            Defect("mono downmix", Severity.QUALITY, Fix.RE_RUN),
        )
        assert row_fix(defects) is Fix.RE_RUN

    def test_row_fix_remux_only(self) -> None:
        defects = (
            Defect("fps drift", Severity.SYNC, Fix.REMUX),
            Defect("color tags", Severity.COSMETIC, Fix.REMUX),
        )
        assert row_fix(defects) is Fix.REMUX

    def test_unreadable_rollup(self) -> None:
        defects = (Defect("unreadable", Severity.UNREADABLE, Fix.NONE),)
        assert row_severity(defects) is Severity.UNREADABLE
        assert row_fix(defects) is Fix.NONE

    def test_empty_defects_falls_back_safely(self) -> None:
        assert row_severity(()) is Severity.UNREADABLE
        assert row_fix(()) is Fix.NONE
