"""Field-separated storage detection for interlaced sources.

Some muxers store an interlaced stream's two FIELDS as separate container
blocks (MediaInfo: "Scan type, store method: Separated fields"). The block
count then measures the FIELD rate, and ffprobe reports 1080i25 as 50 fps.
Every decoder pairs the fields back into 25 frames, and furnace's
deinterlacers are single-rate, so the plan must carry the coded frame rate —
with the field rate the mux pins --default-duration at 50 and the video plays
twice as fast, running out halfway through the audio.
"""

from __future__ import annotations

import pytest

from furnace.core.detect import (
    detect_field_separated,
    needs_field_rate_probe,
    needs_pulldown_probe,
)


class TestNeedsFieldRateProbe:
    # --- the one shape that needs probing: interlaced at a field rate ---

    def test_tt_at_50_needs_probe(self) -> None:
        """1080i25 stored as separated fields reports tt + 50 fps → probe."""
        assert needs_field_rate_probe("tt", 50, 1) is True

    def test_bb_at_50_needs_probe(self) -> None:
        """Bottom-field-first is the same shape → probe."""
        assert needs_field_rate_probe("bb", 50, 1) is True

    def test_ntsc_field_rate_needs_probe(self) -> None:
        """60000/1001 (NTSC field rate) with tt → probe."""
        assert needs_field_rate_probe("tt", 60000, 1001) is True

    def test_exactly_48_needs_probe(self) -> None:
        """The threshold itself counts as a field rate."""
        assert needs_field_rate_probe("tt", 48, 1) is True

    # --- everything else is out of scope ---

    def test_progressive_no_probe(self) -> None:
        """A progressive 50p stream reports its true frame rate → no probe."""
        assert needs_field_rate_probe("progressive", 50, 1) is False

    def test_none_field_order_no_probe(self) -> None:
        """Absent field_order → not interlaced → no probe."""
        assert needs_field_rate_probe(None, 50, 1) is False

    def test_tt_at_25_no_probe(self) -> None:
        """Ordinary 1080i25 already reports the frame rate → nothing to undo."""
        assert needs_field_rate_probe("tt", 25, 1) is False

    def test_tt_below_threshold_no_probe(self) -> None:
        """Just under 48 is a frame rate, not a field rate."""
        assert needs_field_rate_probe("tt", 47, 1) is False

    # --- the exclusion the analyzer's stage ordering leans on ---

    @pytest.mark.parametrize("fps", [(24, 1), (24000, 1001), (25, 1), (30, 1),
                                     (30000, 1001), (48, 1), (50, 1), (60, 1), (60000, 1001)])
    @pytest.mark.parametrize("codec", ["mpeg2video", "h264"])
    @pytest.mark.parametrize("height", [480, 576, 720, 1080])
    @pytest.mark.parametrize("field_order", ["tt", "bb", "progressive", None])
    def test_never_fires_together_with_the_pulldown_gate(
        self, fps: tuple[int, int], codec: str, height: int, field_order: str | None,
    ) -> None:
        """No source may take both fps-rewriting paths.

        The analyzer computes both gates BEFORE either probe runs, so a rate
        this probe halves can never be re-examined for pulldown. That is only
        safe while the two gates cannot both fire — lower ``_TV_FPS_THRESHOLD``
        into the NTSC window (or widen that window past 48) and this fails.
        """
        fps_num, fps_den = fps
        assert not (
            needs_field_rate_probe(field_order, fps_num, fps_den)
            and needs_pulldown_probe(codec, fps_num, fps_den, height)
        )


class TestDetectFieldSeparated:
    def test_two_packets_per_frame_halves_the_rate(self) -> None:
        """Every frame arrives as two field packets → 50/1 is really 25/1."""
        assert detect_field_separated(50, 1, frames=1500, packets=3000) == (25, 1)

    def test_ntsc_field_rate_halves_to_frame_rate(self) -> None:
        """60000/1001 field rate → 30000/1001 frame rate, gcd-reduced."""
        assert detect_field_separated(60000, 1001, frames=1500, packets=3000) == (30000, 1001)

    def test_odd_numerator_halves_via_denominator(self) -> None:
        """An odd numerator halves by doubling the denominator instead."""
        assert detect_field_separated(50, 3, frames=1500, packets=3000) == (25, 3)
        assert detect_field_separated(49, 1, frames=1500, packets=3000) == (49, 2)

    def test_frame_coded_source_keeps_its_rate(self) -> None:
        """One packet per frame → the container rate is already the frame rate."""
        assert detect_field_separated(50, 1, frames=1500, packets=1500) is None

    def test_mixed_paff_rejected(self) -> None:
        """A stream that mixes frame- and field-coded pictures → keep the rate.

        No single halving is correct for it, so returning None is the safe
        answer: the container rate at least matches the packet cadence.
        """
        assert detect_field_separated(50, 1, frames=1500, packets=2250) is None

    def test_window_boundary_jitter_within_tolerance_detected(self) -> None:
        """A trailing field whose frame never decoded still reads as separated."""
        assert detect_field_separated(50, 1, frames=1500, packets=3001) == (25, 1)

    def test_empty_sample_is_none(self) -> None:
        """A probe that measured nothing (fail-soft (0, 0)) → keep the rate."""
        assert detect_field_separated(50, 1, frames=0, packets=0) is None

    def test_short_sample_is_none(self) -> None:
        """Fewer than 100 decoded frames is too small to trust."""
        assert detect_field_separated(50, 1, frames=80, packets=160) is None

    # --- Boundaries. Each constant is pinned from both sides, so it cannot
    # drift in either direction without a failure here. These four carry all
    # the weight; the recorded measurement at the end pins nothing.

    def test_ratio_at_the_top_of_the_jitter_band_detected(self) -> None:
        """The largest packet count still inside the band → still separated.

        3029/1500 = 2.0193 — the largest count the 0.02 band takes, since 3030
        is nominally exactly 2.02 but floats land it a hair over. Paired with
        the case below, this traps the tolerance between 0.0193 and 0.0207:
        tight enough that no plausible drift survives.
        """
        assert detect_field_separated(50, 1, frames=1500, packets=3029) == (25, 1)

    def test_ratio_past_the_jitter_band_rejected(self) -> None:
        """3031/1500 = 2.0207, clear of the band → keep the reported rate.

        Deliberately not 3030, the first count the implementation rejects:
        that one is nominally the band's inclusive edge (exactly 2.02) and
        falls outside only by float rounding, so it would pin the rejection
        on that rounding instead of on the tolerance.
        """
        assert detect_field_separated(50, 1, frames=1500, packets=3031) is None

    def test_minimum_sample_is_trusted(self) -> None:
        """Exactly 100 decoded frames is the smallest sample that counts.

        Paired with the 99-frame case below this pins the minimum exactly:
        both feed a ratio of 2.0, so sample size is the only discriminator.
        """
        assert detect_field_separated(50, 1, frames=100, packets=200) == (25, 1)

    def test_one_frame_below_minimum_is_rejected(self) -> None:
        """99 frames is one short → None."""
        assert detect_field_separated(50, 1, frames=99, packets=198) is None

    def test_the_real_chapaev_sample(self) -> None:
        """A recorded field measurement, kept as a documentary anchor.

        ``ffprobe -count_frames -count_packets -read_intervals %+60`` on
        Чапаев.1934.HDTV-ylnian.mkv returns exactly these counters: the file
        this probe exists for, at a dead-exact ratio of 2.0. It pins no
        threshold — the boundary cases above do that — and kills no mutant they
        miss. It is here so the next reader sees the shape the real world hands
        us instead of taking the prose on faith.
        """
        assert detect_field_separated(50, 1, frames=1499, packets=2998) == (25, 1)
