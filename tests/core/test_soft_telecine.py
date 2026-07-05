"""Soft-telecine (2:3 pulldown) detection for NTSC MPEG-2 DVD sources.

A soft-telecined DVD stores progressive film frames (24000/1001) with
repeat_first_field flags; ffprobe reports the *display* rate (30000/1001).
NVEncC's avsw decode ignores the pulldown flags and encodes the coded film
frames, so the plan must carry the coded rate or the mux pins the wrong
--default-duration and the video drifts 25% fast against the audio.
"""

from __future__ import annotations

from furnace.core.detect import detect_soft_telecine, needs_pulldown_probe


class TestNeedsPulldownProbe:
    # --- the one shape that needs probing: NTSC SD MPEG-2 ---

    def test_ntsc_sd_mpeg2_needs_probe(self) -> None:
        """mpeg2video 30000/1001 SD (NTSC DVD) → probe for pulldown flags."""
        assert needs_pulldown_probe("mpeg2video", 30000, 1001, height=480) is True

    def test_exact_30fps_sd_mpeg2_needs_probe(self) -> None:
        """mpeg2video 30/1 SD → probe (whole-rate NTSC variant)."""
        assert needs_pulldown_probe("mpeg2video", 30, 1, height=480) is True

    # --- everything else is out of scope ---

    def test_pal_mpeg2_no_probe(self) -> None:
        """PAL DVD (25 fps) has no pulldown concept → no probe."""
        assert needs_pulldown_probe("mpeg2video", 25, 1, height=576) is False

    def test_film_rate_mpeg2_no_probe(self) -> None:
        """Already 24000/1001 → nothing to undo → no probe."""
        assert needs_pulldown_probe("mpeg2video", 24000, 1001, height=480) is False

    def test_h264_no_probe(self) -> None:
        """Non-MPEG-2 codec → no probe even at NTSC rate."""
        assert needs_pulldown_probe("h264", 30000, 1001, height=480) is False

    def test_hd_mpeg2_no_probe(self) -> None:
        """HD MPEG-2 (ATSC broadcast) is out of the DVD domain → no probe."""
        assert needs_pulldown_probe("mpeg2video", 30000, 1001, height=1080) is False


class TestDetectSoftTelecine:
    def test_clean_23_pulldown_yields_film_rate(self) -> None:
        """Perfect 2:3 cadence (RFF on every other frame) → 24000/1001."""
        flags = [0, 1] * 250
        assert detect_soft_telecine(30000, 1001, flags) == (24000, 1001)

    def test_whole_rate_source_yields_whole_film_rate(self) -> None:
        """30/1 display rate with 2:3 cadence → exact 24/1."""
        flags = [0, 1] * 250
        assert detect_soft_telecine(30, 1, flags) == (24, 1)

    def test_no_rff_flags_is_not_telecine(self) -> None:
        """Hard telecine / true interlace carries no RFF flags → None."""
        assert detect_soft_telecine(30000, 1001, [0] * 500) is None

    def test_all_rff_is_not_23_pulldown(self) -> None:
        """RFF on every frame (ratio 2/3) is not a 2:3 cadence → None."""
        assert detect_soft_telecine(30000, 1001, [1] * 500) is None

    def test_empty_sample_is_none(self) -> None:
        assert detect_soft_telecine(30000, 1001, []) is None

    def test_short_sample_is_none(self) -> None:
        """Fewer than 100 sampled frames is too noisy to trust."""
        flags = [0, 1] * 40  # 80 frames, perfect cadence
        assert detect_soft_telecine(30000, 1001, flags) is None

    def test_noisy_cadence_within_tolerance_detected(self) -> None:
        """52% RFF (scene-boundary jitter) still reads as 2:3 pulldown."""
        flags = [1] * 260 + [0] * 240
        assert detect_soft_telecine(30000, 1001, flags) == (24000, 1001)

    def test_hybrid_content_outside_tolerance_rejected(self) -> None:
        """60% RFF (hybrid film/video disc) → None, keep the display rate."""
        flags = [1] * 300 + [0] * 200
        assert detect_soft_telecine(30000, 1001, flags) is None
