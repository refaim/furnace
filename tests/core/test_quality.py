from __future__ import annotations

from furnace.core.models import CropRect
from furnace.core.quality import (
    CQ_ANCHORS,
    align_dimensions,
    calculate_gop,
    correct_sar,
    interpolate_cq,
)

# ---------------------------------------------------------------------------
# test_interpolate_cq
# ---------------------------------------------------------------------------

class TestInterpolateCq:
    def test_sd_anchor(self) -> None:
        """Exact SD anchor -> CQ 22."""
        assert interpolate_cq(409_920) == 22

    def test_720p_anchor(self) -> None:
        """Exact 720p anchor -> CQ 24."""
        assert interpolate_cq(921_600) == 24

    def test_1080p_anchor(self) -> None:
        """Exact 1080p anchor -> CQ 25."""
        assert interpolate_cq(2_073_600) == 25

    def test_1440p_anchor(self) -> None:
        """Exact 1440p anchor -> CQ 28."""
        assert interpolate_cq(3_686_400) == 28

    def test_4k_anchor(self) -> None:
        """Exact 4K anchor -> CQ 31."""
        assert interpolate_cq(8_294_400) == 31

    def test_below_sd_clamps_to_sd(self) -> None:
        """Pixel area below SD -> returns SD CQ (clamped at bottom)."""
        assert interpolate_cq(1) == CQ_ANCHORS[0][1]
        assert interpolate_cq(0) == CQ_ANCHORS[0][1]

    def test_above_4k_clamps_to_4k(self) -> None:
        """Pixel area above 4K -> returns 4K CQ (clamped at top)."""
        assert interpolate_cq(99_000_000) == CQ_ANCHORS[-1][1]

    def test_intermediate_sd_to_720p(self) -> None:
        """Midpoint between SD and 720p is interpolated."""
        x0, y0 = CQ_ANCHORS[0]
        x1, y1 = CQ_ANCHORS[1]
        mid = (x0 + x1) // 2
        result = interpolate_cq(mid)
        # Should be between y0 and y1 (inclusive)
        assert y0 <= result <= y1

    def test_intermediate_720p_to_1080p(self) -> None:
        """Midpoint between 720p and 1080p is interpolated."""
        x0, y0 = CQ_ANCHORS[1]
        x1, y1 = CQ_ANCHORS[2]
        mid = (x0 + x1) // 2
        result = interpolate_cq(mid)
        assert y0 <= result <= y1

    def test_intermediate_1080p_to_1440p(self) -> None:
        """Midpoint between 1080p and 1440p is interpolated."""
        x0, y0 = CQ_ANCHORS[2]
        x1, y1 = CQ_ANCHORS[3]
        mid = (x0 + x1) // 2
        result = interpolate_cq(mid)
        assert y0 <= result <= y1

    def test_intermediate_1440p_to_4k(self) -> None:
        """Midpoint between 1440p and 4K is interpolated."""
        x0, y0 = CQ_ANCHORS[3]
        x1, y1 = CQ_ANCHORS[4]
        mid = (x0 + x1) // 2
        result = interpolate_cq(mid)
        assert y0 <= result <= y1

    def test_monotone_increasing(self) -> None:
        """CQ is non-decreasing as pixel_area increases."""
        areas = [409_920, 921_600, 2_073_600, 3_686_400, 8_294_400]
        cqs = [interpolate_cq(a) for a in areas]
        assert cqs == sorted(cqs)

    def test_just_above_sd_anchor(self) -> None:
        """Pixel area one above SD anchor -> still CQ 22 (close to y0=22)."""
        # At x0+1, t ~ 0, round(22 + ~0*(24-22)) = 22
        assert interpolate_cq(409_921) == 22

    def test_just_below_4k_anchor(self) -> None:
        """Pixel area one below 4K anchor -> CQ 31 (very close to y1=31)."""
        # At x1-1, t ~ 1, round(28 + ~1*(31-28)) = 31
        assert interpolate_cq(8_294_399) == 31

    def test_midpoint_sd_720p_exact(self) -> None:
        """Midpoint between SD and 720p: t=0.5 -> round(22 + 0.5*2) = 23."""
        x0, _ = CQ_ANCHORS[0]
        x1, _ = CQ_ANCHORS[1]
        mid = (x0 + x1) // 2
        # t ~= 0.5, y = 22 + 0.5*2 = 23.0
        assert interpolate_cq(mid) == 23

    def test_midpoint_1440p_4k_exact(self) -> None:
        """Midpoint between 1440p and 4K: t=0.5 -> round(28 + 0.5*3) = round(29.5) = 30."""
        x0, _ = CQ_ANCHORS[3]
        x1, _ = CQ_ANCHORS[4]
        mid = (x0 + x1) // 2
        # t ~= 0.5, y = 28 + 0.5*3 = 29.5 -> banker's rounding -> 30
        assert interpolate_cq(mid) == 30

    def test_quarter_720p_to_1080p(self) -> None:
        """Quarter point between 720p (24) and 1080p (25): t=0.25 -> round(24.25) = 24."""
        x0, _ = CQ_ANCHORS[1]
        x1, _ = CQ_ANCHORS[2]
        q = x0 + (x1 - x0) // 4
        assert interpolate_cq(q) == 24


# ---------------------------------------------------------------------------
# test_align_dimensions
# ---------------------------------------------------------------------------

class TestAlignDimensions:
    def test_already_aligned(self) -> None:
        """Values already aligned to 8 -> unchanged."""
        result = align_dimensions(1920, 1080, 0, 0)
        assert result == CropRect(w=1920, h=1080, x=0, y=0)

    def test_8x8_alignment(self) -> None:
        """Both w and h trimmed to multiples of 8, offset centered."""
        # w=1922 -> trim=2 -> new_w=1920, new_x=x+1
        # h=1082 -> trim=2 -> new_h=1080, new_y=y+1
        result = align_dimensions(1922, 1082, 10, 20)
        assert result.w == 1920
        assert result.h == 1080
        assert result.x == 11
        assert result.y == 21

    def test_default_zero_offset(self) -> None:
        """x and y default to 0."""
        result = align_dimensions(1922, 1082)
        assert result.x == 1
        assert result.y == 1

    def test_centering_integer_division(self) -> None:
        """trim=1 -> trim//2=0 (floors)."""
        result = align_dimensions(1921, 1080)
        assert result.w == 1920
        assert result.x == 0

    def test_trim_7(self) -> None:
        """w=1007 -> trim=7 -> new_w=1000, offset=3."""
        result = align_dimensions(1007, 1080, 5, 0)
        assert result.w == 1000
        assert result.x == 5 + 3

    def test_zero_values(self) -> None:
        """All zeros -> CropRect(0,0,0,0)."""
        result = align_dimensions(0, 0, 0, 0)
        assert result == CropRect(w=0, h=0, x=0, y=0)


# ---------------------------------------------------------------------------
# test_calculate_gop
# ---------------------------------------------------------------------------

class TestCalculateGop:
    def test_24fps(self) -> None:
        """24fps -> ceil(24/1)*5 = 120."""
        assert calculate_gop(24, 1) == 120

    def test_25fps(self) -> None:
        """25fps -> ceil(25/1)*5 = 125."""
        assert calculate_gop(25, 1) == 125

    def test_30fps(self) -> None:
        """30fps -> ceil(30/1)*5 = 150."""
        assert calculate_gop(30, 1) == 150

    def test_23_976fps(self) -> None:
        """23.976fps (24000/1001) -> ceil(23.976...) = 24 -> 24*5 = 120."""
        assert calculate_gop(24000, 1001) == 120

    def test_29_97fps(self) -> None:
        """29.97fps (30000/1001) -> ceil(29.97...) = 30 -> 30*5 = 150."""
        assert calculate_gop(30000, 1001) == 150

    def test_60fps(self) -> None:
        """60fps -> ceil(60/1)*5 = 300."""
        assert calculate_gop(60, 1) == 300


# ---------------------------------------------------------------------------
# test_correct_sar
# ---------------------------------------------------------------------------

class TestCorrectSar:
    def test_square_pixels_unchanged(self) -> None:
        """SAR 1:1 -> no change."""
        assert correct_sar(720, 480, 1, 1) == (720, 480)

    def test_sar_wider_scales_width(self) -> None:
        """SAR > 1 (wide pixels) -> stretch width."""
        # SAR 32:27 = ~1.185, 720 * 32/27 = 853.33 -> 853
        w, h = correct_sar(720, 480, 32, 27)
        assert w == 853
        assert h == 480

    def test_sar_taller_scales_height(self) -> None:
        """SAR < 1 (tall pixels) -> stretch height."""
        # SAR 5:6, 480 * 6/5 = 576
        w, h = correct_sar(720, 480, 5, 6)
        assert w == 720
        assert h == 576

    def test_dvd_pal_sar(self) -> None:
        """PAL DVD 720x576 SAR 64:45 -> 1024x576."""
        w, h = correct_sar(720, 576, 64, 45)
        assert w == 1024
        assert h == 576

    def test_dvd_ntsc_sar_8_9(self) -> None:
        """NTSC DVD 720x480 SAR 8:9 -> stretch height."""
        w, h = correct_sar(720, 480, 8, 9)
        assert w == 720
        assert h == 540


