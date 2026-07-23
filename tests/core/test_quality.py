from __future__ import annotations

from furnace.core.models import CropRect, VideoParams
from furnace.core.quality import (
    CQ_ANCHORS,
    align_dimensions,
    calculate_gop,
    correct_sar,
    final_output_dimensions,
    force_16_9_sar,
    interpolate_cq,
)


class TestInterpolateCq:
    def test_sd_anchor(self) -> None:
        assert interpolate_cq(409_920) == 35

    def test_720p_anchor(self) -> None:
        assert interpolate_cq(921_600) == 35

    def test_1080p_anchor(self) -> None:
        assert interpolate_cq(2_073_600) == 36

    def test_1440p_anchor(self) -> None:
        assert interpolate_cq(3_686_400) == 35

    def test_4k_anchor(self) -> None:
        assert interpolate_cq(8_294_400) == 34

    def test_below_sd_clamps_to_sd(self) -> None:
        assert interpolate_cq(1) == CQ_ANCHORS[0][1]
        assert interpolate_cq(0) == CQ_ANCHORS[0][1]

    def test_above_4k_clamps_to_4k(self) -> None:
        assert interpolate_cq(99_000_000) == CQ_ANCHORS[-1][1]

    def test_intermediate_sd_to_720p(self) -> None:
        x0, y0 = CQ_ANCHORS[0]
        x1, y1 = CQ_ANCHORS[1]
        mid = (x0 + x1) // 2
        result = interpolate_cq(mid)
        assert y0 <= result <= y1

    def test_intermediate_720p_to_1080p(self) -> None:
        x0, y0 = CQ_ANCHORS[1]
        x1, y1 = CQ_ANCHORS[2]
        mid = (x0 + x1) // 2
        result = interpolate_cq(mid)
        assert min(y0, y1) <= result <= max(y0, y1)

    def test_intermediate_1080p_to_1440p(self) -> None:
        x0, y0 = CQ_ANCHORS[2]
        x1, y1 = CQ_ANCHORS[3]
        mid = (x0 + x1) // 2
        result = interpolate_cq(mid)
        assert min(y0, y1) <= result <= max(y0, y1)

    def test_intermediate_1440p_to_4k(self) -> None:
        x0, y0 = CQ_ANCHORS[3]
        x1, y1 = CQ_ANCHORS[4]
        mid = (x0 + x1) // 2
        result = interpolate_cq(mid)
        assert min(y0, y1) <= result <= max(y0, y1)

    def test_anchor_curve_shape(self) -> None:
        areas = [409_920, 921_600, 2_073_600, 3_686_400, 8_294_400]
        cqs = [interpolate_cq(a) for a in areas]
        assert cqs == [35, 35, 36, 35, 34]

    def test_4k_not_more_aggressive_than_1080p(self) -> None:
        assert interpolate_cq(8_294_400) <= interpolate_cq(2_073_600)

    def test_just_above_sd_anchor(self) -> None:
        assert interpolate_cq(409_921) == 35

    def test_just_below_4k_anchor(self) -> None:
        assert interpolate_cq(8_294_399) == 34

    def test_midpoint_sd_720p_exact(self) -> None:
        x0, _ = CQ_ANCHORS[0]
        x1, _ = CQ_ANCHORS[1]
        mid = (x0 + x1) // 2
        assert interpolate_cq(mid) == 35

    def test_midpoint_1440p_4k_exact(self) -> None:
        x0, _ = CQ_ANCHORS[3]
        x1, _ = CQ_ANCHORS[4]
        mid = (x0 + x1) // 2
        assert interpolate_cq(mid) == 34

    def test_quarter_720p_to_1080p(self) -> None:
        x0, _ = CQ_ANCHORS[1]
        x1, _ = CQ_ANCHORS[2]
        q = x0 + (x1 - x0) // 4
        assert interpolate_cq(q) == 35


class TestAlignDimensions:
    def test_already_aligned(self) -> None:
        result = align_dimensions(1920, 1080, 0, 0)
        assert result == CropRect(w=1920, h=1080, x=0, y=0)

    def test_8x8_alignment(self) -> None:
        result = align_dimensions(1922, 1082, 10, 20)
        assert result.w == 1920
        assert result.h == 1080
        assert result.x == 11
        assert result.y == 21

    def test_default_zero_offset(self) -> None:
        result = align_dimensions(1922, 1082)
        assert result.x == 1
        assert result.y == 1

    def test_centering_integer_division(self) -> None:
        result = align_dimensions(1921, 1080)
        assert result.w == 1920
        assert result.x == 0

    def test_trim_7(self) -> None:
        result = align_dimensions(1007, 1080, 5, 0)
        assert result.w == 1000
        assert result.x == 5 + 3

    def test_zero_values(self) -> None:
        result = align_dimensions(0, 0, 0, 0)
        assert result == CropRect(w=0, h=0, x=0, y=0)


class TestCalculateGop:
    def test_24fps(self) -> None:
        assert calculate_gop(24, 1) == 120

    def test_25fps(self) -> None:
        assert calculate_gop(25, 1) == 125

    def test_30fps(self) -> None:
        assert calculate_gop(30, 1) == 150

    def test_23_976fps(self) -> None:
        assert calculate_gop(24000, 1001) == 120

    def test_29_97fps(self) -> None:
        assert calculate_gop(30000, 1001) == 150

    def test_60fps(self) -> None:
        assert calculate_gop(60, 1) == 300


class TestCorrectSar:
    def test_square_pixels_unchanged(self) -> None:
        assert correct_sar(720, 480, 1, 1) == (720, 480)

    def test_sar_wider_scales_width(self) -> None:
        w, h = correct_sar(720, 480, 32, 27)
        assert w == 853
        assert h == 480

    def test_sar_taller_scales_height(self) -> None:
        w, h = correct_sar(720, 480, 5, 6)
        assert w == 720
        assert h == 576

    def test_dvd_pal_sar(self) -> None:
        w, h = correct_sar(720, 576, 64, 45)
        assert w == 1024
        assert h == 576

    def test_dvd_ntsc_sar_8_9(self) -> None:
        w, h = correct_sar(720, 480, 8, 9)
        assert w == 720
        assert h == 540


class TestForce16By9Sar:
    def test_pal_720x576(self) -> None:
        assert force_16_9_sar(720, 576) == (64, 45)

    def test_ntsc_720x480(self) -> None:
        assert force_16_9_sar(720, 480) == (32, 27)

    def test_produces_16_9_display(self) -> None:
        num, den = force_16_9_sar(720, 480)
        w, h = correct_sar(720, 480, num, den)
        assert round(w / h, 2) == 1.78

    def test_already_16_9_square_pixels(self) -> None:
        assert force_16_9_sar(1024, 576) == (1, 1)


def _vp(
    *,
    source_width: int,
    source_height: int,
    crop: CropRect | None = None,
    sar_num: int = 1,
    sar_den: int = 1,
) -> VideoParams:
    return VideoParams(
        cq=22,
        crop=crop,
        deinterlace=False,
        color_matrix="bt709",
        color_range="tv",
        color_transfer="bt709",
        color_primaries="bt709",
        hdr=None,
        gop=120,
        fps_num=24,
        fps_den=1,
        source_width=source_width,
        source_height=source_height,
        source_codec="h264",
        source_bitrate=10_000_000,
        sar_num=sar_num,
        sar_den=sar_den,
        dv_mode=None,
    )


class TestFinalOutputDimensions:
    def test_no_crop_square_sar_mod8_passthrough(self) -> None:
        vp = _vp(source_width=1920, source_height=1080)
        assert final_output_dimensions(vp) == (1920, 1080)

    def test_no_crop_square_sar_non_mod8_aligned(self) -> None:
        vp = _vp(source_width=1916, source_height=802)
        assert final_output_dimensions(vp) == (1912, 800)

    def test_crop_square_sar_mod8_passthrough(self) -> None:
        vp = _vp(
            source_width=1920,
            source_height=1080,
            crop=CropRect(w=1920, h=800, x=0, y=140),
        )
        assert final_output_dimensions(vp) == (1920, 800)

    def test_crop_square_sar_non_mod8_aligned(self) -> None:
        vp = _vp(
            source_width=1920,
            source_height=1080,
            crop=CropRect(w=1916, h=802, x=2, y=139),
        )
        assert final_output_dimensions(vp) == (1912, 800)

    def test_no_crop_pal_dvd_anamorphic(self) -> None:
        vp = _vp(source_width=720, source_height=576, sar_num=16, sar_den=15)
        assert final_output_dimensions(vp) == (768, 576)

    def test_crop_pal_dvd_anamorphic_bug_case(self) -> None:
        vp = _vp(
            source_width=720,
            source_height=576,
            sar_num=16,
            sar_den=15,
            crop=CropRect(w=704, h=400, x=8, y=88),
        )
        assert final_output_dimensions(vp) == (744, 400)

    def test_anamorphic_height_grows(self) -> None:
        vp = _vp(source_width=1024, source_height=576, sar_num=4, sar_den=5)
        assert final_output_dimensions(vp) == (1024, 720)
