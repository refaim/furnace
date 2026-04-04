from __future__ import annotations

from furnace.core.detect import needs_idet, should_deinterlace


class TestNeedsIdet:
    def test_progressive_field_order_no_idet(self):
        """field_order=progressive → no idet needed."""
        assert needs_idet(field_order="progressive", fps=25.0) is False

    def test_none_field_order_no_idet(self):
        """field_order=None → no idet needed."""
        assert needs_idet(field_order=None, fps=25.0) is False

    def test_unknown_field_order_no_idet(self):
        """field_order=unknown → no idet needed."""
        assert needs_idet(field_order="unknown", fps=25.0) is False

    def test_tt_high_fps_no_idet(self):
        """field_order=tt + fps >= 48 → TV format, no idet needed."""
        assert needs_idet(field_order="tt", fps=50.0) is False

    def test_bb_high_fps_no_idet(self):
        """field_order=bb + fps >= 48 → TV format, no idet needed."""
        assert needs_idet(field_order="bb", fps=50.0) is False

    def test_tt_low_fps_needs_idet(self):
        """field_order=tt + fps < 48 → ambiguous (DVD?), idet needed."""
        assert needs_idet(field_order="tt", fps=25.0) is True

    def test_bb_low_fps_needs_idet(self):
        """field_order=bb + fps < 48 → ambiguous, idet needed."""
        assert needs_idet(field_order="bb", fps=24.0) is True

    def test_tt_fps_boundary(self):
        """field_order=tt + fps=48 → TV format, no idet needed."""
        assert needs_idet(field_order="tt", fps=48.0) is False

    def test_tt_fps_just_below_boundary(self):
        """field_order=tt + fps=47.99 → needs idet."""
        assert needs_idet(field_order="tt", fps=47.99) is True


class TestShouldDeinterlace:
    # --- ffprobe says progressive ---

    def test_progressive_field_order(self):
        """field_order=progressive → never deinterlace."""
        assert should_deinterlace(field_order="progressive", fps=25.0, idet_ratio=0.0) is False

    def test_none_field_order(self):
        """field_order=None → never deinterlace."""
        assert should_deinterlace(field_order=None, fps=25.0, idet_ratio=0.0) is False

    # --- TV format (high fps) ---

    def test_tt_high_fps_always_deinterlace(self):
        """field_order=tt + fps >= 48 → always deinterlace (TV)."""
        assert should_deinterlace(field_order="tt", fps=50.0, idet_ratio=0.0) is True

    def test_bb_high_fps_always_deinterlace(self):
        """field_order=bb + fps >= 48 → always deinterlace."""
        assert should_deinterlace(field_order="bb", fps=50.0, idet_ratio=0.0) is True

    def test_tt_high_fps_deinterlace_even_idet_says_no(self):
        """TV format: deinterlace regardless of idet result."""
        assert should_deinterlace(field_order="tt", fps=50.0, idet_ratio=0.001) is True

    # --- Ambiguous (low fps + tt/bb) → idet decides ---

    def test_tt_low_fps_idet_confirms_interlace(self):
        """field_order=tt + low fps + idet > 5% → deinterlace."""
        assert should_deinterlace(field_order="tt", fps=25.0, idet_ratio=0.10) is True

    def test_tt_low_fps_idet_denies_interlace(self):
        """field_order=tt + low fps + idet < 5% → progressive (soft telecine DVD)."""
        assert should_deinterlace(field_order="tt", fps=25.0, idet_ratio=0.02) is False

    def test_tt_low_fps_idet_at_threshold(self):
        """field_order=tt + low fps + idet exactly 5% → not enough, progressive."""
        assert should_deinterlace(field_order="tt", fps=25.0, idet_ratio=0.05) is False

    def test_tt_low_fps_idet_above_threshold(self):
        """field_order=tt + low fps + idet 5.1% → deinterlace."""
        assert should_deinterlace(field_order="tt", fps=25.0, idet_ratio=0.051) is True

    def test_bb_low_fps_idet_zero(self):
        """field_order=bb + low fps + idet 0% → progressive."""
        assert should_deinterlace(field_order="bb", fps=24.0, idet_ratio=0.0) is False

    def test_bb_low_fps_idet_high(self):
        """field_order=bb + low fps + idet 30% → deinterlace (real MPEG2 interlace)."""
        assert should_deinterlace(field_order="bb", fps=25.0, idet_ratio=0.30) is True
