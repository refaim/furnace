from __future__ import annotations

import pytest

from furnace.core.detect import (
    detect_field_separated,
    needs_field_rate_probe,
    needs_pulldown_probe,
)


class TestNeedsFieldRateProbe:
    def test_tt_at_50_needs_probe(self) -> None:
        assert needs_field_rate_probe("tt", 50, 1) is True

    def test_bb_at_50_needs_probe(self) -> None:
        assert needs_field_rate_probe("bb", 50, 1) is True

    def test_ntsc_field_rate_needs_probe(self) -> None:
        assert needs_field_rate_probe("tt", 60000, 1001) is True

    def test_exactly_48_needs_probe(self) -> None:
        assert needs_field_rate_probe("tt", 48, 1) is True

    def test_progressive_no_probe(self) -> None:
        assert needs_field_rate_probe("progressive", 50, 1) is False

    def test_none_field_order_no_probe(self) -> None:
        assert needs_field_rate_probe(None, 50, 1) is False

    def test_tt_at_25_no_probe(self) -> None:
        assert needs_field_rate_probe("tt", 25, 1) is False

    def test_tt_below_threshold_no_probe(self) -> None:
        assert needs_field_rate_probe("tt", 47, 1) is False

    @pytest.mark.parametrize(
        "fps", [(24, 1), (24000, 1001), (25, 1), (30, 1), (30000, 1001), (48, 1), (50, 1), (60, 1), (60000, 1001)]
    )
    @pytest.mark.parametrize("codec", ["mpeg2video", "h264"])
    @pytest.mark.parametrize("height", [480, 576, 720, 1080])
    @pytest.mark.parametrize("field_order", ["tt", "bb", "progressive", None])
    def test_never_fires_together_with_the_pulldown_gate(
        self,
        fps: tuple[int, int],
        codec: str,
        height: int,
        field_order: str | None,
    ) -> None:
        fps_num, fps_den = fps
        assert not (
            needs_field_rate_probe(field_order, fps_num, fps_den)
            and needs_pulldown_probe(codec, fps_num, fps_den, height)
        )


class TestDetectFieldSeparated:
    def test_two_packets_per_frame_halves_the_rate(self) -> None:
        assert detect_field_separated(50, 1, frames=1500, packets=3000) == (25, 1)

    def test_ntsc_field_rate_halves_to_frame_rate(self) -> None:
        assert detect_field_separated(60000, 1001, frames=1500, packets=3000) == (30000, 1001)

    def test_odd_numerator_halves_via_denominator(self) -> None:
        assert detect_field_separated(50, 3, frames=1500, packets=3000) == (25, 3)
        assert detect_field_separated(49, 1, frames=1500, packets=3000) == (49, 2)

    def test_frame_coded_source_keeps_its_rate(self) -> None:
        assert detect_field_separated(50, 1, frames=1500, packets=1500) is None

    def test_mixed_paff_rejected(self) -> None:
        assert detect_field_separated(50, 1, frames=1500, packets=2250) is None

    def test_window_boundary_jitter_within_tolerance_detected(self) -> None:
        assert detect_field_separated(50, 1, frames=1500, packets=3001) == (25, 1)

    def test_empty_sample_is_none(self) -> None:
        assert detect_field_separated(50, 1, frames=0, packets=0) is None

    def test_short_sample_is_none(self) -> None:
        assert detect_field_separated(50, 1, frames=80, packets=160) is None

    def test_ratio_at_the_top_of_the_jitter_band_detected(self) -> None:
        assert detect_field_separated(50, 1, frames=1500, packets=3029) == (25, 1)

    def test_ratio_past_the_jitter_band_rejected(self) -> None:
        assert detect_field_separated(50, 1, frames=1500, packets=3031) is None

    def test_minimum_sample_is_trusted(self) -> None:
        assert detect_field_separated(50, 1, frames=100, packets=200) == (25, 1)

    def test_one_frame_below_minimum_is_rejected(self) -> None:
        assert detect_field_separated(50, 1, frames=99, packets=198) is None

    def test_the_real_chapaev_sample(self) -> None:
        assert detect_field_separated(50, 1, frames=1499, packets=2998) == (25, 1)
