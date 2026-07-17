from __future__ import annotations

from furnace.core.detect import needs_idet, should_deinterlace


class TestNeedsIdet:
    def test_progressive_field_order_no_idet(self) -> None:
        assert needs_idet(field_order="progressive", fps=25.0, height=576) is False

    def test_none_field_order_no_idet(self) -> None:
        assert needs_idet(field_order=None, fps=25.0, height=576) is False

    def test_unknown_field_order_no_idet(self) -> None:
        assert needs_idet(field_order="unknown", fps=25.0, height=576) is False

    def test_tt_high_fps_no_idet(self) -> None:
        assert needs_idet(field_order="tt", fps=50.0, height=576) is False

    def test_bb_high_fps_no_idet(self) -> None:
        assert needs_idet(field_order="bb", fps=50.0, height=576) is False

    def test_tt_fps_boundary(self) -> None:
        assert needs_idet(field_order="tt", fps=48.0, height=576) is False

    def test_tt_low_fps_sd_needs_idet(self) -> None:
        assert needs_idet(field_order="tt", fps=25.0, height=576) is True

    def test_bb_low_fps_sd_needs_idet(self) -> None:
        assert needs_idet(field_order="bb", fps=24.0, height=480) is True

    def test_tt_fps_just_below_boundary_sd_needs_idet(self) -> None:
        assert needs_idet(field_order="tt", fps=47.99, height=576) is True

    def test_sd_height_boundary_needs_idet(self) -> None:
        assert needs_idet(field_order="tt", fps=25.0, height=719) is True

    def test_tt_low_fps_hd_no_idet(self) -> None:
        assert needs_idet(field_order="tt", fps=25.0, height=1080) is False

    def test_hd_height_boundary_no_idet(self) -> None:
        assert needs_idet(field_order="tt", fps=25.0, height=720) is False


class TestShouldDeinterlace:
    def test_progressive_field_order(self) -> None:
        assert should_deinterlace(field_order="progressive", fps=25.0, idet_ratio=0.0, height=576) is False

    def test_none_field_order(self) -> None:
        assert should_deinterlace(field_order=None, fps=25.0, idet_ratio=0.0, height=576) is False

    def test_tt_high_fps_always_deinterlace(self) -> None:
        assert should_deinterlace(field_order="tt", fps=50.0, idet_ratio=0.0, height=576) is True

    def test_bb_high_fps_always_deinterlace(self) -> None:
        assert should_deinterlace(field_order="bb", fps=50.0, idet_ratio=0.0, height=576) is True

    def test_tt_high_fps_deinterlace_even_idet_says_no(self) -> None:
        assert should_deinterlace(field_order="tt", fps=50.0, idet_ratio=0.001, height=576) is True

    def test_tt_hd_high_fps_always_deinterlace(self) -> None:
        assert should_deinterlace(field_order="tt", fps=50.0, idet_ratio=0.0, height=1080) is True

    def test_tt_hd_low_fps_always_deinterlace(self) -> None:
        assert should_deinterlace(field_order="tt", fps=25.0, idet_ratio=0.0, height=1080) is True

    def test_hd_height_boundary_always_deinterlace(self) -> None:
        assert should_deinterlace(field_order="tt", fps=25.0, idet_ratio=0.0, height=720) is True

    def test_tt_sd_low_fps_idet_confirms_interlace(self) -> None:
        assert should_deinterlace(field_order="tt", fps=25.0, idet_ratio=0.10, height=576) is True

    def test_tt_sd_low_fps_idet_denies_interlace(self) -> None:
        assert should_deinterlace(field_order="tt", fps=25.0, idet_ratio=0.02, height=576) is False

    def test_tt_sd_low_fps_idet_at_threshold(self) -> None:
        assert should_deinterlace(field_order="tt", fps=25.0, idet_ratio=0.05, height=576) is False

    def test_tt_sd_low_fps_idet_above_threshold(self) -> None:
        assert should_deinterlace(field_order="tt", fps=25.0, idet_ratio=0.051, height=576) is True

    def test_sd_height_boundary_idet_decides(self) -> None:
        assert should_deinterlace(field_order="tt", fps=25.0, idet_ratio=0.0, height=719) is False

    def test_bb_sd_low_fps_idet_zero(self) -> None:
        assert should_deinterlace(field_order="bb", fps=24.0, idet_ratio=0.0, height=480) is False

    def test_bb_sd_low_fps_idet_high(self) -> None:
        assert should_deinterlace(field_order="bb", fps=25.0, idet_ratio=0.30, height=576) is True
