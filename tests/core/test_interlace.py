from __future__ import annotations

from furnace.core.detect import needs_idet, should_deinterlace


class TestNeedsIdet:
    # --- field_order rules out interlace ---

    def test_progressive_field_order_no_idet(self) -> None:
        """field_order=progressive → no idet needed."""
        assert needs_idet(field_order="progressive", fps=25.0, height=576) is False

    def test_none_field_order_no_idet(self) -> None:
        """field_order=None → no idet needed."""
        assert needs_idet(field_order=None, fps=25.0, height=576) is False

    def test_unknown_field_order_no_idet(self) -> None:
        """field_order=unknown → no idet needed."""
        assert needs_idet(field_order="unknown", fps=25.0, height=576) is False

    # --- high fps: TV interlace, no idet ---

    def test_tt_high_fps_no_idet(self) -> None:
        """field_order=tt + fps >= 48 → TV format, no idet needed."""
        assert needs_idet(field_order="tt", fps=50.0, height=576) is False

    def test_bb_high_fps_no_idet(self) -> None:
        """field_order=bb + fps >= 48 → TV format, no idet needed."""
        assert needs_idet(field_order="bb", fps=50.0, height=576) is False

    def test_tt_fps_boundary(self) -> None:
        """field_order=tt + fps=48 → TV format, no idet needed."""
        assert needs_idet(field_order="tt", fps=48.0, height=576) is False

    # --- SD + low fps: ambiguous (soft telecine?), idet needed ---

    def test_tt_low_fps_sd_needs_idet(self) -> None:
        """field_order=tt + fps < 48 + SD → ambiguous (DVD?), idet needed."""
        assert needs_idet(field_order="tt", fps=25.0, height=576) is True

    def test_bb_low_fps_sd_needs_idet(self) -> None:
        """field_order=bb + fps < 48 + SD → ambiguous, idet needed."""
        assert needs_idet(field_order="bb", fps=24.0, height=480) is True

    def test_tt_fps_just_below_boundary_sd_needs_idet(self) -> None:
        """field_order=tt + fps=47.99 + SD → needs idet."""
        assert needs_idet(field_order="tt", fps=47.99, height=576) is True

    def test_sd_height_boundary_needs_idet(self) -> None:
        """height=719 (just under HD) + tt + low fps → SD, idet needed."""
        assert needs_idet(field_order="tt", fps=25.0, height=719) is True

    # --- HD + low fps: soft telecine doesn't exist in HD, no idet ---

    def test_tt_low_fps_hd_no_idet(self) -> None:
        """field_order=tt + fps < 48 + HD → genuine interlace, no idet needed."""
        assert needs_idet(field_order="tt", fps=25.0, height=1080) is False

    def test_hd_height_boundary_no_idet(self) -> None:
        """height=720 (HD threshold) + tt + low fps → HD, no idet needed."""
        assert needs_idet(field_order="tt", fps=25.0, height=720) is False


class TestShouldDeinterlace:
    # --- ffprobe says progressive ---

    def test_progressive_field_order(self) -> None:
        """field_order=progressive → never deinterlace."""
        assert should_deinterlace(field_order="progressive", fps=25.0, idet_ratio=0.0, height=576) is False

    def test_none_field_order(self) -> None:
        """field_order=None → never deinterlace."""
        assert should_deinterlace(field_order=None, fps=25.0, idet_ratio=0.0, height=576) is False

    # --- TV format (high fps) → always deinterlace ---

    def test_tt_high_fps_always_deinterlace(self) -> None:
        """field_order=tt + fps >= 48 → always deinterlace (TV)."""
        assert should_deinterlace(field_order="tt", fps=50.0, idet_ratio=0.0, height=576) is True

    def test_bb_high_fps_always_deinterlace(self) -> None:
        """field_order=bb + fps >= 48 → always deinterlace."""
        assert should_deinterlace(field_order="bb", fps=50.0, idet_ratio=0.0, height=576) is True

    def test_tt_high_fps_deinterlace_even_idet_says_no(self) -> None:
        """TV format: deinterlace regardless of idet result."""
        assert should_deinterlace(field_order="tt", fps=50.0, idet_ratio=0.001, height=576) is True

    def test_tt_hd_high_fps_always_deinterlace(self) -> None:
        """1080i50 (tt, HD, field rate reported) → both shortcuts agree, deinterlace."""
        assert should_deinterlace(field_order="tt", fps=50.0, idet_ratio=0.0, height=1080) is True

    # --- HD interlace (low fps) → always deinterlace, idet not consulted ---

    def test_tt_hd_low_fps_always_deinterlace(self) -> None:
        """field_order=tt + fps < 48 + HD → always deinterlace even with low idet.

        1080i25 broadcast reports frame rate (25), not field rate (50), and
        idet under-counts combing on low-motion HD; HD soft telecine does not
        exist, so the tt/bb flag is authoritative.
        """
        assert should_deinterlace(field_order="tt", fps=25.0, idet_ratio=0.0, height=1080) is True

    def test_hd_height_boundary_always_deinterlace(self) -> None:
        """height=720 (HD threshold) + tt + low fps + zero idet → deinterlace."""
        assert should_deinterlace(field_order="tt", fps=25.0, idet_ratio=0.0, height=720) is True

    # --- SD ambiguous (low fps + tt/bb) → idet decides ---

    def test_tt_sd_low_fps_idet_confirms_interlace(self) -> None:
        """field_order=tt + low fps + SD + idet > 5% → deinterlace."""
        assert should_deinterlace(field_order="tt", fps=25.0, idet_ratio=0.10, height=576) is True

    def test_tt_sd_low_fps_idet_denies_interlace(self) -> None:
        """field_order=tt + low fps + SD + idet < 5% → progressive (soft telecine DVD)."""
        assert should_deinterlace(field_order="tt", fps=25.0, idet_ratio=0.02, height=576) is False

    def test_tt_sd_low_fps_idet_at_threshold(self) -> None:
        """field_order=tt + low fps + SD + idet exactly 5% → not enough, progressive."""
        assert should_deinterlace(field_order="tt", fps=25.0, idet_ratio=0.05, height=576) is False

    def test_tt_sd_low_fps_idet_above_threshold(self) -> None:
        """field_order=tt + low fps + SD + idet 5.1% → deinterlace."""
        assert should_deinterlace(field_order="tt", fps=25.0, idet_ratio=0.051, height=576) is True

    def test_sd_height_boundary_idet_decides(self) -> None:
        """height=719 (just under HD) + tt + low fps + low idet → progressive (idet decides)."""
        assert should_deinterlace(field_order="tt", fps=25.0, idet_ratio=0.0, height=719) is False

    def test_bb_sd_low_fps_idet_zero(self) -> None:
        """field_order=bb + low fps + SD + idet 0% → progressive."""
        assert should_deinterlace(field_order="bb", fps=24.0, idet_ratio=0.0, height=480) is False

    def test_bb_sd_low_fps_idet_high(self) -> None:
        """field_order=bb + low fps + SD + idet 30% → deinterlace (real MPEG2 interlace)."""
        assert should_deinterlace(field_order="bb", fps=25.0, idet_ratio=0.30, height=576) is True
