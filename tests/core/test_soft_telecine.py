from __future__ import annotations

from furnace.core.detect import detect_soft_telecine, needs_pulldown_probe


class TestNeedsPulldownProbe:
    def test_ntsc_sd_mpeg2_needs_probe(self) -> None:
        assert needs_pulldown_probe("mpeg2video", 30000, 1001, height=480) is True

    def test_exact_30fps_sd_mpeg2_needs_probe(self) -> None:
        assert needs_pulldown_probe("mpeg2video", 30, 1, height=480) is True

    def test_pal_mpeg2_no_probe(self) -> None:
        assert needs_pulldown_probe("mpeg2video", 25, 1, height=576) is False

    def test_film_rate_mpeg2_no_probe(self) -> None:
        assert needs_pulldown_probe("mpeg2video", 24000, 1001, height=480) is False

    def test_h264_no_probe(self) -> None:
        assert needs_pulldown_probe("h264", 30000, 1001, height=480) is False

    def test_hd_mpeg2_no_probe(self) -> None:
        assert needs_pulldown_probe("mpeg2video", 30000, 1001, height=1080) is False


class TestDetectSoftTelecine:
    def test_clean_23_pulldown_yields_film_rate(self) -> None:
        flags = [0, 1] * 250
        assert detect_soft_telecine(30000, 1001, flags) == (24000, 1001)

    def test_whole_rate_source_yields_whole_film_rate(self) -> None:
        flags = [0, 1] * 250
        assert detect_soft_telecine(30, 1, flags) == (24, 1)

    def test_no_rff_flags_is_not_telecine(self) -> None:
        assert detect_soft_telecine(30000, 1001, [0] * 500) is None

    def test_all_rff_is_not_23_pulldown(self) -> None:
        assert detect_soft_telecine(30000, 1001, [1] * 500) is None

    def test_empty_sample_is_none(self) -> None:
        assert detect_soft_telecine(30000, 1001, []) is None

    def test_short_sample_is_none(self) -> None:
        flags = [0, 1] * 40
        assert detect_soft_telecine(30000, 1001, flags) is None

    def test_noisy_cadence_within_tolerance_detected(self) -> None:
        flags = [1] * 260 + [0] * 240
        assert detect_soft_telecine(30000, 1001, flags) == (24000, 1001)

    def test_hybrid_content_outside_tolerance_rejected(self) -> None:
        flags = [1] * 300 + [0] * 200
        assert detect_soft_telecine(30000, 1001, flags) is None
