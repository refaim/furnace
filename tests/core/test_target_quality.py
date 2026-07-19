from __future__ import annotations

import itertools
import math

import pytest

from furnace.core.models import CropRect
from furnace.core.target_quality import (
    GRAIN_POOL_PERCENTILE,
    KnobSearchResult,
    TargetSpec,
    fixed_grain_knob,
    grain_uses_svt,
    interior_windows,
    linear_interpolate,
    natural_cubic_spline,
    pchip_interpolate,
    pool_grain_windows,
    predict_knob,
    probe_windows,
    resolve_target,
    search_knob,
    select_hard_windows,
    source_is_variable_bitrate,
)
from tests.conftest import make_video_params


class TestLinearInterpolate:
    def test_exact_endpoints(self) -> None:
        x = [82.502861, 87.600777]
        y = [20.0, 10.0]
        assert linear_interpolate(x, y, 82.502861) == 20.0
        assert linear_interpolate(x, y, 87.600777) == 10.0

    def test_midpoint(self) -> None:
        x = [82.502861, 87.600777]
        y = [20.0, 10.0]
        result = linear_interpolate(x, y, 85.051819)
        assert result is not None
        assert abs(result - 15.0) < 0.1

    def test_between(self) -> None:
        x = [82.502861, 87.600777]
        y = [20.0, 10.0]
        result = linear_interpolate(x, y, 84.0)
        assert result is not None
        assert 15.0 < result < 20.0

    def test_second_dataset(self) -> None:
        x = [78.737953, 89.179634]
        y = [15.0, 5.0]
        result = linear_interpolate(x, y, 83.958794)
        assert result is not None
        assert abs(result - 10.0) < 0.1

    def test_non_increasing_x_is_none(self) -> None:
        assert linear_interpolate([87.600777, 82.502861], [10.0, 20.0], 85.0) is None

    def test_equal_x_is_none(self) -> None:
        assert linear_interpolate([85.0, 85.0], [20.0, 10.0], 85.0) is None


class TestNaturalCubicSpline:
    def test_exact_points(self) -> None:
        x = [72.812233, 78.517479, 84.872162]
        y = [30.0, 20.0, 10.0]
        for xi, expected in zip(x, y, strict=True):
            result = natural_cubic_spline(x, y, xi)
            assert result is not None
            assert abs(result - expected) < 1e-6

    def test_interpolation_in_range(self) -> None:
        x = [72.812233, 78.517479, 84.872162]
        y = [30.0, 20.0, 10.0]
        result = natural_cubic_spline(x, y, 81.0)
        assert result is not None
        assert 10.0 < result < 20.0

    def test_second_dataset(self) -> None:
        x = [72.134048, 80.161186, 84.864449]
        y = [35.0, 25.0, 15.0]
        result = natural_cubic_spline(x, y, 82.0)
        assert result is not None
        assert 15.0 < result < 25.0

    def test_third_dataset(self) -> None:
        x = [67.3447, 77.7812, 83.0155]
        y = [40.0, 30.0, 20.0]
        result = natural_cubic_spline(x, y, 80.0)
        assert result is not None
        assert 20.0 < result < 30.0

    def test_too_few_points_is_none(self) -> None:
        assert natural_cubic_spline([87.0715, 90.0064], [20.0, 10.0], 88.0) is None

    def test_mismatched_lengths_is_none(self) -> None:
        assert natural_cubic_spline([83.8, 87.07, 90.0], [30.0, 20.0], 85.0) is None

    def test_extrapolation_below_is_none(self) -> None:
        x = [72.812233, 78.517479, 84.872162]
        y = [30.0, 20.0, 10.0]
        assert natural_cubic_spline(x, y, 70.0) is None

    def test_extrapolation_above_is_none(self) -> None:
        x = [72.812233, 78.517479, 84.872162]
        y = [30.0, 20.0, 10.0]
        assert natural_cubic_spline(x, y, 90.0) is None

    def test_non_increasing_interior_is_none(self) -> None:
        assert natural_cubic_spline([75.0, 75.0, 80.0], [30.0, 20.0, 10.0], 77.0) is None


class TestPchipInterpolate:
    def test_exact_points(self) -> None:
        x = [72.9709, 80.5088, 85.7452, 92.4354]
        y = [35.0, 25.0, 15.0, 5.0]
        for xi, expected in zip(x, y, strict=True):
            result = pchip_interpolate(x, y, xi)
            assert result is not None
            assert abs(result - expected) < 1e-6

    def test_interpolation_in_first_segment(self) -> None:
        x = [72.9709, 80.5088, 85.7452, 92.4354]
        y = [35.0, 25.0, 15.0, 5.0]
        result = pchip_interpolate(x, y, 89.0)
        assert result is not None
        assert 5.0 < result < 15.0

    def test_interpolation_middle_segment(self) -> None:
        x = [37.30312, 50.740498, 57.916622, 66.699707]
        y = [55.0, 50.0, 45.0, 40.0]
        result = pchip_interpolate(x, y, 54.0)
        assert result is not None
        assert 45.0 < result < 50.0

    def test_local_extremum_zeroes_derivative(self) -> None:
        x = [4.944567, 5.270722, 5.345044, 5.575547]
        y = [65.0, 66.0, 64.0, 63.0]
        result = pchip_interpolate(x, y, 5.1)
        assert result is not None

    def test_flat_segment_zero_slope(self) -> None:
        x = [70.0, 75.0, 80.0, 85.0]
        y = [30.0, 30.0, 20.0, 10.0]
        result = pchip_interpolate(x, y, 72.0)
        assert result is not None

    def test_steep_segment_triggers_tau_rescale(self) -> None:
        x = [10.0, 11.0, 11.001, 50.0]
        y = [40.0, 39.0, 10.0, 5.0]
        result = pchip_interpolate(x, y, 10.5)
        assert result is not None
        assert math.isfinite(result)

    def test_extrapolation_uses_first_segment(self) -> None:
        x = [72.9709, 80.5088, 85.7452, 92.4354]
        y = [35.0, 25.0, 15.0, 5.0]
        result = pchip_interpolate(x, y, 100.0)
        assert result is not None
        assert math.isfinite(result)

    def test_non_increasing_x_is_none(self) -> None:
        assert pchip_interpolate([72.9709, 88.0, 85.7452, 92.4354], [35.0, 12.0, 15.0, 5.0], 87.0) is None

    def test_wrong_length_is_none(self) -> None:
        assert pchip_interpolate([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 2.0) is None


class TestLinearInterpolateArity:
    def test_wrong_length_is_none(self) -> None:
        assert linear_interpolate([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 2.0) is None
        assert linear_interpolate([1.0], [1.0], 1.0) is None


class TestPredictKnob:
    def test_empty_history_is_binary_midpoint(self) -> None:
        assert predict_knob(10, 40, [], 80.0, 80.0) == 25

    def test_single_probe_is_binary_midpoint(self) -> None:
        assert predict_knob(10, 40, [(25, 70.0)], 80.0, 80.0) == 25

    def test_two_probes_linear(self) -> None:
        history = [(26, 74.0), (13, 87.0)]
        assert predict_knob(1, 50, history, 79.0, 81.0) == 20

    def test_two_probes_equal_scores_fall_back_to_binary(self) -> None:
        history = [(20, 80.0), (30, 80.0)]
        assert predict_knob(10, 40, history, 79.0, 81.0) == 25

    def test_three_probes_natural_spline(self) -> None:
        history = [(35, 78.05), (17, 85.81), (30, 80.92)]
        result = predict_knob(31, 34, history, 79.5, 80.5)
        assert result == 32

    def test_four_probes_pchip(self) -> None:
        history = [(35, 72.97), (25, 80.51), (15, 85.75), (5, 92.44)]
        result = predict_knob(5, 35, history, 80.0, 81.0)
        assert result == 25

    def test_five_probes_fall_back_to_binary(self) -> None:
        history = [(10, 90.0), (20, 85.0), (30, 80.0), (40, 75.0), (50, 70.0)]
        assert predict_knob(12, 48, history, 82.0, 83.0) == 30

    def test_prediction_clamped_to_upper(self) -> None:
        history = [(30, 79.0), (20, 80.0)]
        result = predict_knob(18, 25, history, 70.0, 70.0)
        assert result == 25

    def test_prediction_clamped_to_lower(self) -> None:
        history = [(30, 79.0), (20, 80.0)]
        result = predict_knob(22, 40, history, 90.0, 90.0)
        assert result == 22


class TestSearchKnob:
    def test_converges_on_monotone_probe(self) -> None:
        result = search_knob(
            lambda knob: 100.0 - knob,
            target_lo=79.0,
            target_hi=81.0,
            lo=1,
            hi=50,
            max_probes=10,
        )
        assert 19 <= result.knob <= 21
        assert result.hit is True
        assert 79.0 <= result.score <= 81.0

    def test_result_records_probe_history(self) -> None:
        result = search_knob(
            lambda knob: 100.0 - knob,
            target_lo=79.0,
            target_hi=81.0,
            lo=1,
            hi=50,
            max_probes=10,
        )
        assert isinstance(result, KnobSearchResult)
        assert result.probes[0][0] == 26
        assert result.knob in {k for k, _ in result.probes}

    def test_max_probes_cutoff_returns_closest(self) -> None:
        result = search_knob(
            lambda knob: 100.0 - knob,
            target_lo=10.0,
            target_hi=12.0,
            lo=1,
            hi=50,
            max_probes=2,
        )
        assert result.hit is False
        assert len(result.probes) == 2

    def test_constant_high_score_raises_floor_then_dedups(self) -> None:
        result = search_knob(
            lambda _knob: 95.0,
            target_lo=79.0,
            target_hi=81.0,
            lo=1,
            hi=2,
            max_probes=10,
        )
        assert result.hit is False
        assert result.knob == 2

    def test_constant_low_score_lowers_ceiling_then_dedups(self) -> None:
        result = search_knob(
            lambda _knob: 50.0,
            target_lo=79.0,
            target_hi=81.0,
            lo=1,
            hi=2,
            max_probes=10,
        )
        assert result.hit is False
        assert result.probes == ((2, 50.0), (1, 50.0))
        assert result.knob == 2

    def test_nonlinear_probe_drives_spline_and_pchip_rounds(self) -> None:
        result = search_knob(
            lambda knob: 100.0 - 0.01 * knob * knob,
            target_lo=79.9,
            target_hi=80.1,
            lo=1,
            hi=63,
            max_probes=6,
        )
        assert result.hit is False
        assert 43 <= result.knob <= 46
        assert len(result.probes) >= 4

    def test_non_finite_probe_score_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            search_knob(
                lambda _k: math.nan,
                target_lo=79.0,
                target_hi=81.0,
                lo=1,
                hi=50,
                max_probes=4,
            )

    def test_negative_lo_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            search_knob(lambda _k: 80.0, target_lo=79.0, target_hi=81.0, lo=-1, hi=40, max_probes=4)

    def test_hit_on_first_probe(self) -> None:
        result = search_knob(
            lambda _knob: 80.0,
            target_lo=79.0,
            target_hi=81.0,
            lo=1,
            hi=50,
            max_probes=5,
        )
        assert result.hit is True
        assert len(result.probes) == 1

    def test_lo_gt_hi_raises(self) -> None:
        with pytest.raises(ValueError, match="lower bound"):
            search_knob(lambda _k: 80.0, target_lo=79.0, target_hi=81.0, lo=40, hi=10, max_probes=4)

    def test_target_lo_gt_hi_raises(self) -> None:
        with pytest.raises(ValueError, match="target lower bound"):
            search_knob(lambda _k: 80.0, target_lo=81.0, target_hi=79.0, lo=10, hi=40, max_probes=4)

    def test_max_probes_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="max_probes"):
            search_knob(lambda _k: 80.0, target_lo=79.0, target_hi=81.0, lo=10, hi=40, max_probes=0)


_SCORE_MAPS: dict[int, dict[int, float]] = {
    1: {35: 80.08},
    2: {17: 80.03, 35: 65.73},
    3: {17: 83.15, 22: 80.02, 35: 71.94},
    4: {17: 85.81, 30: 80.92, 32: 80.01, 35: 78.05},
    5: {35: 83.31, 53: 81.22, 55: 80.03, 61: 73.56, 64: 67.56},
    6: {
        35: 86.99,
        53: 84.41,
        57: 82.47,
        59: 81.14,
        60: 80.09,
        61: 78.58,
        69: 68.57,
        70: 64.90,
    },
}


def _simulate_av1an(score_map: dict[int, float]) -> list[tuple[int, float]]:
    history: list[tuple[int, float]] = []
    lo, hi = 1, 70
    target_lo, target_hi = 79.5, 80.5
    while True:
        assert len(history) < 10, f"no convergence within 10 probes: {history}"
        predicted = predict_knob(lo, hi, history, target_lo, target_hi)
        knob = min(score_map, key=lambda q: abs(q - predicted))
        score = score_map[knob]
        history.append((knob, score))
        if target_lo <= score <= target_hi:
            return history
        if score > target_hi:
            lo = min(knob + 1, hi)
        else:
            hi = max(knob - 1, lo)


class TestAv1anTableCases:
    @pytest.mark.parametrize("case", [1, 2, 3, 4, 5, 6])
    def test_converges_within_range(self, case: int) -> None:
        history = _simulate_av1an(_SCORE_MAPS[case])
        assert history, f"case {case} produced no probes"
        final_score = history[-1][1]
        assert 79.5 <= final_score <= 80.5, f"case {case} final score {final_score} out of range"


class TestResolveTarget:
    def test_hdr_pq_uses_cvvdp(self) -> None:
        vp = make_video_params(color_transfer="smpte2084", color_matrix="bt2020nc")
        spec = resolve_target(vp)
        assert spec.metric == "cvvdp"
        assert spec.target_lo < 9.5 < spec.target_hi
        assert spec.knob_lo < spec.knob_hi

    def test_hlg_uses_cvvdp(self) -> None:
        vp = make_video_params(color_transfer="arib-std-b67", color_matrix="bt2020nc")
        assert resolve_target(vp).metric == "cvvdp"

    def test_sdr_1080p_uses_ssimulacra2_high(self) -> None:
        vp = make_video_params(source_width=1920, source_height=1080)
        spec = resolve_target(vp)
        assert spec.metric == "ssimulacra2"
        assert spec.target_lo < 81.0 < spec.target_hi

    def test_sdr_720p_is_hd_bucket(self) -> None:
        vp = make_video_params(source_width=1280, source_height=720)
        spec = resolve_target(vp)
        assert spec.metric == "ssimulacra2"
        assert spec.target_lo < 81.0 < spec.target_hi

    def test_sdr_dvd_uses_ssimulacra2_low(self) -> None:
        vp = make_video_params(source_width=720, source_height=576)
        spec = resolve_target(vp)
        assert spec.metric == "ssimulacra2"
        assert spec.target_lo < 72.0 < spec.target_hi
        assert spec.target_hi < 81.0

    def test_target_band_brackets_center(self) -> None:
        spec = resolve_target(make_video_params(source_width=1920, source_height=1080))
        centre = (spec.target_lo + spec.target_hi) / 2
        assert abs(centre - 81.0) < 1e-9
        assert spec.target_hi - spec.target_lo < 4.0

    def test_returns_target_spec(self) -> None:
        spec = resolve_target(make_video_params())
        assert isinstance(spec, TargetSpec)
        assert spec.max_probes >= 1

    def test_grain_uses_crf_bounds_and_ssimulacra2(self) -> None:
        spec = resolve_target(make_video_params(grain=True, source_width=720, source_height=576))
        assert spec.metric == "ssimulacra2"
        assert spec.knob_lo == 26
        assert spec.knob_hi == 34
        assert spec.knob_lo < spec.knob_hi
        assert spec.target_lo < 70.0 < spec.target_hi

    def test_grain_samples_ten_windows(self) -> None:
        spec = resolve_target(make_video_params(grain=True, source_width=720, source_height=576))
        assert spec.window_count == 10

    def test_nvenc_samples_three_windows(self) -> None:
        for vp in (
            make_video_params(source_width=1920, source_height=1080),
            make_video_params(source_width=720, source_height=576),
            make_video_params(color_transfer="smpte2084", color_matrix="bt2020nc"),
        ):
            assert resolve_target(vp).window_count == 3

    def test_grain_hd_is_not_resolvable_here(self) -> None:
        with pytest.raises(ValueError, match="SD-only"):
            resolve_target(make_video_params(grain=True, source_width=1920, source_height=1080))

    def test_grain_hdr_raises_loudly(self) -> None:
        for transfer in ("smpte2084", "arib-std-b67"):
            vp = make_video_params(grain=True, color_transfer=transfer, color_matrix="bt2020nc")
            with pytest.raises(ValueError, match="grain target-quality is unsupported on HDR"):
                resolve_target(vp)


class TestGrainRouting:
    def test_grain_sd_uses_svt(self) -> None:
        assert grain_uses_svt(make_video_params(grain=True, source_width=720, source_height=576)) is True

    def test_grain_hd_does_not_use_svt(self) -> None:
        assert grain_uses_svt(make_video_params(grain=True, source_width=1920, source_height=1080)) is False

    def test_height_720_counts_as_hd(self) -> None:
        assert grain_uses_svt(make_video_params(grain=True, source_width=1280, source_height=720)) is False
        assert grain_uses_svt(make_video_params(grain=True, source_width=1024, source_height=719)) is True

    def test_non_grain_never_uses_svt(self) -> None:
        assert grain_uses_svt(make_video_params(source_width=720, source_height=576)) is False
        assert grain_uses_svt(make_video_params(source_width=1920, source_height=1080)) is False

    def test_fixed_knob_only_for_hd_grain(self) -> None:
        assert fixed_grain_knob(make_video_params(grain=True, source_width=1920, source_height=1080)) == 32

    def test_no_fixed_knob_for_sd_grain(self) -> None:
        assert fixed_grain_knob(make_video_params(grain=True, source_width=720, source_height=576)) is None

    def test_no_fixed_knob_for_non_grain(self) -> None:
        assert fixed_grain_knob(make_video_params(source_width=1920, source_height=1080)) is None
        assert fixed_grain_knob(make_video_params(source_width=720, source_height=576)) is None

    def test_svt_and_fixed_are_complementary_for_grain(self) -> None:
        for w, h in ((720, 576), (1920, 1080), (1280, 720)):
            vp = make_video_params(grain=True, source_width=w, source_height=h)
            assert grain_uses_svt(vp) == (fixed_grain_knob(vp) is None)

    def test_routing_uses_final_not_source_height(self) -> None:
        vp = make_video_params(
            grain=True,
            source_width=1920,
            source_height=1080,
            crop=CropRect(w=1920, h=704, x=0, y=188),
        )
        assert grain_uses_svt(vp) is True
        assert fixed_grain_knob(vp) is None


class TestProbeWindows:
    def test_even_spacing_exact(self) -> None:
        offsets = probe_windows(100.0, count=3, window_s=20.0)
        assert offsets == [10.0, 40.0, 70.0]

    def test_windows_stay_within_duration(self) -> None:
        offsets = probe_windows(6000.0, count=3, window_s=18.0)
        assert offsets is not None
        assert offsets[0] >= 0.0
        assert offsets[-1] + 18.0 <= 6000.0
        for a, b in itertools.pairwise(offsets):
            assert b - a >= 18.0

    def test_full_pass_fallback_short_source(self) -> None:
        assert probe_windows(50.0, count=3, window_s=18.0) is None

    def test_long_source_gets_windows(self) -> None:
        assert probe_windows(7200.0, count=3, window_s=18.0) is not None

    def test_full_pass_boundary(self) -> None:
        assert probe_windows(63.5, count=3, window_s=18.0) is None
        assert probe_windows(64.0, count=3, window_s=18.0) is not None

    def test_invalid_count_raises(self) -> None:
        with pytest.raises(ValueError, match="count"):
            probe_windows(1000.0, count=0, window_s=18.0)

    def test_invalid_window_s_raises(self) -> None:
        with pytest.raises(ValueError, match="window"):
            probe_windows(1000.0, count=3, window_s=0.0)

    def test_invalid_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="duration"):
            probe_windows(0.0, count=3, window_s=18.0)


class TestSourceIsVariableBitrate:
    def test_vbr_spread_is_variable(self) -> None:
        assert source_is_variable_bitrate([1.6, 2.8, 3.5, 5.0, 5.2, 2.1, 4.4]) is True

    def test_flat_is_not_variable(self) -> None:
        assert source_is_variable_bitrate([3.6, 3.7, 3.7, 3.8, 3.7, 3.7]) is False

    def test_fewer_than_two_is_not_variable(self) -> None:
        assert source_is_variable_bitrate([]) is False
        assert source_is_variable_bitrate([4.0]) is False

    def test_zero_mean_is_not_variable(self) -> None:
        assert source_is_variable_bitrate([0.0, 0.0, 0.0]) is False

    def test_threshold_is_the_boundary(self) -> None:
        assert source_is_variable_bitrate([0.9, 1.1], threshold=0.1) is True
        assert source_is_variable_bitrate([0.95, 1.05], threshold=0.1) is False


class TestInteriorWindows:
    def test_drops_leading_and_trailing_edges(self) -> None:
        scored = [
            (10.0, 5.0),
            (60.0, 1.0),
            (500.0, 9.0),
            (930.0, 2.0),
            (950.0, 8.0),
        ]
        assert interior_windows(scored, duration_s=1000.0, window_s=10.0, edge_skip=0.06) == [
            (60.0, 1.0),
            (500.0, 9.0),
            (930.0, 2.0),
        ]

    def test_preserves_order(self) -> None:
        scored = [(500.0, 9.0), (100.0, 1.0), (300.0, 5.0)]
        assert interior_windows(scored, duration_s=1000.0, window_s=10.0) == scored

    def test_empty_input(self) -> None:
        assert interior_windows([], duration_s=1000.0, window_s=10.0) == []


class TestSelectHardWindows:
    def test_picks_highest_value_windows(self) -> None:
        scored = [(0.0, 1.0), (100.0, 5.0), (200.0, 3.0), (300.0, 9.0), (400.0, 2.0)]
        assert select_hard_windows(scored, count=2, min_gap_s=10.0) == [100.0, 300.0]

    def test_min_gap_skips_neighbours(self) -> None:
        scored = [(100.0, 5.0), (300.0, 9.0), (305.0, 8.0)]
        assert select_hard_windows(scored, count=2, min_gap_s=50.0) == [100.0, 300.0]

    def test_fewer_candidates_than_count(self) -> None:
        assert select_hard_windows([(0.0, 1.0), (100.0, 2.0)], count=5, min_gap_s=10.0) == [0.0, 100.0]

    def test_default_gap_applies(self) -> None:
        assert select_hard_windows([(0.0, 1.0), (50.0, 9.0), (200.0, 5.0)], count=3) == [50.0, 200.0]

    def test_invalid_count_raises(self) -> None:
        with pytest.raises(ValueError, match="count"):
            select_hard_windows([(0.0, 1.0)], count=0, min_gap_s=10.0)

    def test_negative_gap_raises(self) -> None:
        with pytest.raises(ValueError, match="gap"):
            select_hard_windows([(0.0, 1.0)], count=1, min_gap_s=-1.0)


class TestPoolGrainWindows:
    def test_low_percentile_of_the_window_scores(self) -> None:
        scores = [90.0, 60.0, 61.0, 70.0, 80.0, 85.0, 88.0, 92.0, 95.0, 99.0]
        assert pool_grain_windows(scores) == pytest.approx(68.2)

    def test_denser_sampling_barely_moves_p20_but_collapses_min(self) -> None:
        sparse = [70.0 + i * 0.5 for i in range(10)]
        dense = [*[70.0 + i * 0.125 for i in range(40)], 60.0, 61.0]
        assert min(dense) < min(sparse) - 8.0
        assert abs(pool_grain_windows(dense) - pool_grain_windows(sparse)) < 2.0

    def test_one_outlier_window_does_not_govern(self) -> None:
        healthy = [70.0 + i for i in range(10)]
        assert pool_grain_windows([*healthy[1:], 20.0]) > 60.0

    def test_is_monotonic_in_the_knob(self) -> None:
        at_low_crf = [72.0, 75.0, 79.0, 81.0, 90.0]
        at_high_crf = [s - 3.0 for s in at_low_crf]
        assert pool_grain_windows(at_high_crf) < pool_grain_windows(at_low_crf)

    def test_single_window_returns_that_window(self) -> None:
        assert pool_grain_windows([73.5]) == pytest.approx(73.5)

    def test_two_windows_interpolate(self) -> None:
        assert pool_grain_windows([80.0, 70.0]) == pytest.approx(72.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            pool_grain_windows([])

    def test_percentile_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="percentile"):
            pool_grain_windows([70.0], percentile=150.0)
        with pytest.raises(ValueError, match="percentile"):
            pool_grain_windows([70.0], percentile=-1.0)

    def test_percentile_is_low(self) -> None:
        assert 0.0 < GRAIN_POOL_PERCENTILE <= 25.0
