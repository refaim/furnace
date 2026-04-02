from __future__ import annotations

import pytest

from furnace.core.models import ColorSpace, CropRect
from furnace.core.quality import (
    CQ_ANCHORS,
    align_crop,
    calculate_gop,
    determine_color_space,
    interpolate_cq,
)


# ---------------------------------------------------------------------------
# test_interpolate_cq
# ---------------------------------------------------------------------------

class TestInterpolateCq:
    def test_sd_anchor(self):
        """Exact SD anchor -> CQ 22."""
        assert interpolate_cq(409_920) == 22

    def test_720p_anchor(self):
        """Exact 720p anchor -> CQ 24."""
        assert interpolate_cq(921_600) == 24

    def test_1080p_anchor(self):
        """Exact 1080p anchor -> CQ 25."""
        assert interpolate_cq(2_073_600) == 25

    def test_1440p_anchor(self):
        """Exact 1440p anchor -> CQ 28."""
        assert interpolate_cq(3_686_400) == 28

    def test_4k_anchor(self):
        """Exact 4K anchor -> CQ 31."""
        assert interpolate_cq(8_294_400) == 31

    def test_below_sd_clamps_to_sd(self):
        """Pixel area below SD -> returns SD CQ (clamped at bottom)."""
        assert interpolate_cq(1) == CQ_ANCHORS[0][1]
        assert interpolate_cq(0) == CQ_ANCHORS[0][1]

    def test_above_4k_clamps_to_4k(self):
        """Pixel area above 4K -> returns 4K CQ (clamped at top)."""
        assert interpolate_cq(99_000_000) == CQ_ANCHORS[-1][1]

    def test_intermediate_sd_to_720p(self):
        """Midpoint between SD and 720p is interpolated."""
        x0, y0 = CQ_ANCHORS[0]
        x1, y1 = CQ_ANCHORS[1]
        mid = (x0 + x1) // 2
        result = interpolate_cq(mid)
        # Should be between y0 and y1 (inclusive)
        assert y0 <= result <= y1

    def test_intermediate_720p_to_1080p(self):
        """Midpoint between 720p and 1080p is interpolated."""
        x0, y0 = CQ_ANCHORS[1]
        x1, y1 = CQ_ANCHORS[2]
        mid = (x0 + x1) // 2
        result = interpolate_cq(mid)
        assert y0 <= result <= y1

    def test_intermediate_1080p_to_1440p(self):
        """Midpoint between 1080p and 1440p is interpolated."""
        x0, y0 = CQ_ANCHORS[2]
        x1, y1 = CQ_ANCHORS[3]
        mid = (x0 + x1) // 2
        result = interpolate_cq(mid)
        assert y0 <= result <= y1

    def test_intermediate_1440p_to_4k(self):
        """Midpoint between 1440p and 4K is interpolated."""
        x0, y0 = CQ_ANCHORS[3]
        x1, y1 = CQ_ANCHORS[4]
        mid = (x0 + x1) // 2
        result = interpolate_cq(mid)
        assert y0 <= result <= y1

    def test_monotone_increasing(self):
        """CQ is non-decreasing as pixel_area increases."""
        areas = [409_920, 921_600, 2_073_600, 3_686_400, 8_294_400]
        cqs = [interpolate_cq(a) for a in areas]
        assert cqs == sorted(cqs)


# ---------------------------------------------------------------------------
# test_align_crop
# ---------------------------------------------------------------------------

class TestAlignCrop:
    def test_already_aligned(self):
        """Values already aligned -> unchanged w/h, same x/y."""
        result = align_crop(1920, 1080, 0, 0)
        assert result == CropRect(w=1920, h=1080, x=0, y=0)

    def test_16x8_alignment(self):
        """w not multiple of 16, h not multiple of 8 -> aligned down, x/y adjusted."""
        # w=1922 -> dw=2 -> new_w=1920, new_x=x+1
        # h=1082 -> dh=2 -> new_h=1080, new_y=y+1
        result = align_crop(1922, 1082, 10, 20)
        assert result.w == 1920
        assert result.h == 1080
        assert result.x == 10 + 1   # dw=2 -> dw//2=1
        assert result.y == 20 + 1   # dh=2 -> dh//2=1

    def test_zero_xy(self):
        """Zero x and y with alignment needed."""
        result = align_crop(1920, 1082, 0, 0)
        assert result.w == 1920
        assert result.h == 1080
        assert result.x == 0       # dw=0, x stays 0
        assert result.y == 1       # dh=2, dh//2=1

    def test_centering_via_dw_half(self):
        """dw=1 -> dw//2=0 (integer division floors), not 0.5."""
        result = align_crop(1921, 1080, 0, 0)
        assert result.w == 1920    # dw=1 -> 1921-1=1920
        assert result.x == 0      # dw//2=0 (1//2=0)

    def test_large_alignment_offset(self):
        """w=1935 -> dw=15 -> new_w=1920, new_x=x+7."""
        result = align_crop(1935, 1080, 5, 5)
        assert result.w == 1920
        assert result.x == 5 + 7   # dw=15 -> 15//2=7

    def test_h_alignment_dh_7(self):
        """h=1087 -> dh=7 -> new_h=1080, new_y=y+3."""
        result = align_crop(1920, 1087, 0, 10)
        assert result.h == 1080
        assert result.y == 10 + 3  # dh=7 -> 7//2=3

    def test_zero_values(self):
        """All zeros -> CropRect(0,0,0,0)."""
        result = align_crop(0, 0, 0, 0)
        assert result == CropRect(w=0, h=0, x=0, y=0)


# ---------------------------------------------------------------------------
# test_calculate_gop
# ---------------------------------------------------------------------------

class TestCalculateGop:
    def test_24fps(self):
        """24fps -> ceil(24/1)*5 = 120."""
        assert calculate_gop(24, 1) == 120

    def test_25fps(self):
        """25fps -> ceil(25/1)*5 = 125."""
        assert calculate_gop(25, 1) == 125

    def test_30fps(self):
        """30fps -> ceil(30/1)*5 = 150."""
        assert calculate_gop(30, 1) == 150

    def test_23_976fps(self):
        """23.976fps (24000/1001) -> ceil(23.976...) = 24 -> 24*5 = 120."""
        assert calculate_gop(24000, 1001) == 120

    def test_29_97fps(self):
        """29.97fps (30000/1001) -> ceil(29.97...) = 30 -> 30*5 = 150."""
        assert calculate_gop(30000, 1001) == 150

    def test_60fps(self):
        """60fps -> ceil(60/1)*5 = 300."""
        assert calculate_gop(60, 1) == 300


# ---------------------------------------------------------------------------
# test_determine_color_space
# ---------------------------------------------------------------------------

class TestDetermineColorSpace:
    def test_hd_1080p_returns_bt709(self):
        """1920x1080 -> BT.709."""
        result = determine_color_space(1920, 1080, None)
        assert result == ColorSpace.BT709

    def test_hd_720p_returns_bt709(self):
        """1280x720 -> BT.709 (height == 720, boundary)."""
        result = determine_color_space(1280, 720, None)
        assert result == ColorSpace.BT709

    def test_sd_480p_returns_bt601(self):
        """854x480 -> BT.601 (height < 720)."""
        result = determine_color_space(854, 480, None)
        assert result == ColorSpace.BT601

    def test_sd_576p_returns_bt601(self):
        """1024x576 -> BT.601."""
        result = determine_color_space(1024, 576, None)
        assert result == ColorSpace.BT601

    def test_below_720_returns_bt601(self):
        """Height 719 (one below threshold) -> BT.601."""
        result = determine_color_space(1280, 719, None)
        assert result == ColorSpace.BT601

    def test_bt2020_passthrough(self):
        """Source with BT.2020 color space -> passthrough BT.2020 regardless of resolution."""
        result = determine_color_space(3840, 2160, ColorSpace.BT2020)
        assert result == ColorSpace.BT2020

    def test_bt2020_passthrough_on_sd(self):
        """BT.2020 source even at SD resolution -> passthrough."""
        result = determine_color_space(854, 480, ColorSpace.BT2020)
        assert result == ColorSpace.BT2020

    def test_4k_without_bt2020_returns_bt709(self):
        """4K without BT.2020 source -> BT.709 (height >= 720)."""
        result = determine_color_space(3840, 2160, None)
        assert result == ColorSpace.BT709
