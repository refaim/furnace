from __future__ import annotations

import itertools
import math

import pytest

from furnace.core.target_quality import (
    KnobSearchResult,
    TargetSpec,
    interior_windows,
    linear_interpolate,
    natural_cubic_spline,
    pchip_interpolate,
    predict_knob,
    probe_windows,
    resolve_target,
    search_knob,
    select_hard_windows,
    source_is_variable_bitrate,
)
from tests.conftest import make_video_params

# ---------------------------------------------------------------------------
# linear_interpolate  (ported from Av1an interpol.rs::linear_interpolate)
# ---------------------------------------------------------------------------


class TestLinearInterpolate:
    def test_exact_endpoints(self) -> None:
        """xi at a knot returns that knot's y exactly."""
        x = [82.502861, 87.600777]
        y = [20.0, 10.0]
        assert linear_interpolate(x, y, 82.502861) == 20.0
        assert linear_interpolate(x, y, 87.600777) == 10.0

    def test_midpoint(self) -> None:
        """Score midway between the two points -> CRF ~15."""
        x = [82.502861, 87.600777]
        y = [20.0, 10.0]
        result = linear_interpolate(x, y, 85.051819)
        assert result is not None
        assert abs(result - 15.0) < 0.1

    def test_between(self) -> None:
        """A score strictly inside the interval lands strictly between the knobs."""
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
        """x[1] < x[0] -> None (cannot interpolate a descending domain)."""
        assert linear_interpolate([87.600777, 82.502861], [10.0, 20.0], 85.0) is None

    def test_equal_x_is_none(self) -> None:
        """x[1] == x[0] -> None (zero-width interval)."""
        assert linear_interpolate([85.0, 85.0], [20.0, 10.0], 85.0) is None


# ---------------------------------------------------------------------------
# natural_cubic_spline  (ported from Av1an interpol.rs::natural_cubic_spline)
# ---------------------------------------------------------------------------


class TestNaturalCubicSpline:
    def test_exact_points(self) -> None:
        """The spline passes through every knot."""
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
        """Fewer than three knots -> None (needs at least a cubic segment set)."""
        assert natural_cubic_spline([87.0715, 90.0064], [20.0, 10.0], 88.0) is None

    def test_mismatched_lengths_is_none(self) -> None:
        """len(x) != len(y) -> None."""
        assert natural_cubic_spline([83.8, 87.07, 90.0], [30.0, 20.0], 85.0) is None

    def test_extrapolation_below_is_none(self) -> None:
        """xi below the observed range -> None (no extrapolation)."""
        x = [72.812233, 78.517479, 84.872162]
        y = [30.0, 20.0, 10.0]
        assert natural_cubic_spline(x, y, 70.0) is None

    def test_extrapolation_above_is_none(self) -> None:
        """xi above the observed range -> None (no extrapolation)."""
        x = [72.812233, 78.517479, 84.872162]
        y = [30.0, 20.0, 10.0]
        assert natural_cubic_spline(x, y, 90.0) is None

    def test_non_increasing_interior_is_none(self) -> None:
        """A non-positive interior gap (tied scores) -> None, even when xi is in
        range. This is the real "two probes scored equal" path."""
        assert natural_cubic_spline([75.0, 75.0, 80.0], [30.0, 20.0, 10.0], 77.0) is None


# ---------------------------------------------------------------------------
# pchip_interpolate  (ported from Av1an interpol.rs::pchip_interpolate)
# ---------------------------------------------------------------------------


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
        """xi in the middle segment exercises _find_interval returning k=1."""
        x = [37.30312, 50.740498, 57.916622, 66.699707]
        y = [55.0, 50.0, 45.0, 40.0]
        result = pchip_interpolate(x, y, 54.0)
        assert result is not None
        assert 45.0 < result < 50.0

    def test_local_extremum_zeroes_derivative(self) -> None:
        """Non-monotone knobs (up then down) hit the s_prev*s_next <= 0 branch."""
        x = [4.944567, 5.270722, 5.345044, 5.575547]
        y = [65.0, 66.0, 64.0, 63.0]
        result = pchip_interpolate(x, y, 5.1)
        assert result is not None

    def test_flat_segment_zero_slope(self) -> None:
        """Two equal knobs -> slope 0 -> the slopes[i] == 0 branch."""
        x = [70.0, 75.0, 80.0, 85.0]
        y = [30.0, 30.0, 20.0, 10.0]
        result = pchip_interpolate(x, y, 72.0)
        assert result is not None

    def test_steep_segment_triggers_tau_rescale(self) -> None:
        """A very steep interior slope next to a shallow endpoint pushes the
        normalised endpoint derivatives past tau^2 = 9 -> the monotonicity
        rescale branch fires."""
        x = [10.0, 11.0, 11.001, 50.0]
        y = [40.0, 39.0, 10.0, 5.0]
        result = pchip_interpolate(x, y, 10.5)
        assert result is not None
        assert math.isfinite(result)

    def test_extrapolation_uses_first_segment(self) -> None:
        """No extrapolation guard: xi beyond the last knot extrapolates via
        segment 0 (exercises the _find_interval fallthrough) and still returns
        a finite value."""
        x = [72.9709, 80.5088, 85.7452, 92.4354]
        y = [35.0, 25.0, 15.0, 5.0]
        result = pchip_interpolate(x, y, 100.0)
        assert result is not None
        assert math.isfinite(result)

    def test_non_increasing_x_is_none(self) -> None:
        assert pchip_interpolate([72.9709, 88.0, 85.7452, 92.4354], [35.0, 12.0, 15.0, 5.0], 87.0) is None

    def test_wrong_length_is_none(self) -> None:
        """PCHIP needs exactly four points; anything else -> None."""
        assert pchip_interpolate([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 2.0) is None


class TestLinearInterpolateArity:
    def test_wrong_length_is_none(self) -> None:
        """Linear interpolation needs exactly two points; anything else -> None
        (mirrors the arity guards on the spline and PCHIP helpers)."""
        assert linear_interpolate([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 2.0) is None
        assert linear_interpolate([1.0], [1.0], 1.0) is None


# ---------------------------------------------------------------------------
# predict_knob
# ---------------------------------------------------------------------------


class TestPredictKnob:
    def test_empty_history_is_binary_midpoint(self) -> None:
        """No probes -> midpoint of [lo, hi], rounded half-up."""
        assert predict_knob(10, 40, [], 80.0, 80.0) == 25

    def test_single_probe_is_binary_midpoint(self) -> None:
        """One probe -> still the midpoint (need >= 2 points to interpolate)."""
        assert predict_knob(10, 40, [(25, 70.0)], 80.0, 80.0) == 25

    def test_two_probes_linear(self) -> None:
        """Two probes -> linear interpolation to the target score."""
        history = [(26, 74.0), (13, 87.0)]
        assert predict_knob(1, 50, history, 79.0, 81.0) == 20

    def test_two_probes_equal_scores_fall_back_to_binary(self) -> None:
        """Tied scores make linear interpolation degenerate -> binary fallback."""
        history = [(20, 80.0), (30, 80.0)]
        assert predict_knob(10, 40, history, 79.0, 81.0) == 25

    def test_three_probes_natural_spline(self) -> None:
        """Three probes -> natural cubic spline (X-Men-like descending scores)."""
        history = [(35, 78.05), (17, 85.81), (30, 80.92)]
        result = predict_knob(31, 34, history, 79.5, 80.5)
        assert result == 32

    def test_four_probes_pchip(self) -> None:
        """Four probes -> PCHIP interpolation. Target midpoint 80.5 sits a hair
        below the knob-25 knot (score 80.51), so PCHIP returns just above 25 and
        rounds to 25 (within [lo, hi])."""
        history = [(35, 72.97), (25, 80.51), (15, 85.75), (5, 92.44)]
        result = predict_knob(5, 35, history, 80.0, 81.0)
        assert result == 25

    def test_five_probes_fall_back_to_binary(self) -> None:
        """Five or more probes -> no interpolation, bisection fallback."""
        history = [(10, 90.0), (20, 85.0), (30, 80.0), (40, 75.0), (50, 70.0)]
        assert predict_knob(12, 48, history, 82.0, 83.0) == 30

    def test_prediction_clamped_to_upper(self) -> None:
        """A target score below both probes extrapolates to a high knob (lower
        quality) that overshoots hi and is clamped to hi."""
        history = [(30, 79.0), (20, 80.0)]
        result = predict_knob(18, 25, history, 70.0, 70.0)
        assert result == 25

    def test_prediction_clamped_to_lower(self) -> None:
        """A target score above both probes extrapolates to a low (here
        negative) knob that undershoots lo and is clamped to lo."""
        history = [(30, 79.0), (20, 80.0)]
        result = predict_knob(22, 40, history, 90.0, 90.0)
        assert result == 22


# ---------------------------------------------------------------------------
# search_knob — end-to-end behaviour with synthetic monotone probes
# ---------------------------------------------------------------------------


class TestSearchKnob:
    def test_converges_on_monotone_probe(self) -> None:
        """score = 100 - knob, target score 80 -> knob 20, hit."""
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
        assert result.probes[0][0] == 26  # first probe is the midpoint of [1, 50]
        assert result.knob in {k for k, _ in result.probes}

    def test_max_probes_cutoff_returns_closest(self) -> None:
        """An unreachable target stops at max_probes and returns the closest
        probe with hit=False."""
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
        """Always-too-high score keeps raising the floor until the predicted
        knob repeats (dedup break); exercises the score > target_hi branch."""
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
        """Always-too-low score keeps lowering the ceiling until dedup;
        exercises the score < target_lo branch. Both probes tie on distance to
        the target midpoint, so the earlier one (the [1, 2] midpoint 2) wins."""
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
        """A convex monotone probe (score = 100 - 0.01*knob^2) that no integer
        knob places inside a tight target forces the search past the linear
        rounds: the 4th probe comes from the n=3 natural-spline prediction and a
        further n=4 PCHIP prediction runs before the search dedups. Guards the
        real search_knob loop (its bound updates + stop logic), which the linear
        synthetic probes converge too early to reach."""
        result = search_knob(
            lambda knob: 100.0 - 0.01 * knob * knob,
            target_lo=79.9,
            target_hi=80.1,
            lo=1,
            hi=63,
            max_probes=6,
        )
        # True knob for score 80 is sqrt(2000) ~ 44.7; no integer lands in the
        # tight band, so the search settles on the closest knob without a hit.
        assert result.hit is False
        assert 43 <= result.knob <= 46
        # >= 4 probes proves the n=3 spline prediction produced a fresh probe
        # (linear alone would have stalled at 3).
        assert len(result.probes) >= 4

    def test_non_finite_probe_score_raises(self) -> None:
        """A probe that returns NaN/inf fails loudly and early rather than
        crashing later inside the interpolation/rounding math."""
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
        """A negative lower bound is rejected: the half-up rounding only matches
        Av1an's half-away-from-zero on the non-negative knob domain."""
        with pytest.raises(ValueError, match="non-negative"):
            search_knob(lambda _k: 80.0, target_lo=79.0, target_hi=81.0, lo=-1, hi=40, max_probes=4)

    def test_hit_on_first_probe(self) -> None:
        """If the midpoint already lands in range, one probe suffices."""
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


# ---------------------------------------------------------------------------
# Av1an reference table cases (get_score_map 1..6 from target_quality.rs).
# Faithful port of run_av1an_simulation: the sparse score map stands in for a
# real encoder, so the predicted knob is snapped to the nearest available data
# point before probing. Each case must converge to a score in [79.5, 80.5].
# ---------------------------------------------------------------------------

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
    """Probe until the score lands in the target band, or fail loudly.

    The lo/hi updates are clamped against each other, so ``lo <= hi`` holds by
    construction; a search that stalls or wanders trips the probe cap instead
    of spinning.
    """
    history: list[tuple[int, float]] = []
    lo, hi = 1, 70
    target_lo, target_hi = 79.5, 80.5
    while True:
        assert len(history) < 10, f"no convergence within 10 probes: {history}"
        predicted = predict_knob(lo, hi, history, target_lo, target_hi)
        # Snap to the nearest available data point in the sparse map.
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


# ---------------------------------------------------------------------------
# resolve_target — content domain -> (metric, target band, knob bounds)
# ---------------------------------------------------------------------------


class TestResolveTarget:
    def test_hdr_pq_uses_cvvdp(self) -> None:
        """PQ (smpte2084) HDR -> CVVDP metric, calibrated band around 9.5 JOD."""
        vp = make_video_params(color_transfer="smpte2084", color_matrix="bt2020nc")
        spec = resolve_target(vp)
        assert spec.metric == "cvvdp"
        assert spec.target_lo < 9.5 < spec.target_hi
        assert spec.knob_lo < spec.knob_hi

    def test_hlg_uses_cvvdp(self) -> None:
        """HLG (arib-std-b67) is HDR too -> CVVDP."""
        vp = make_video_params(color_transfer="arib-std-b67", color_matrix="bt2020nc")
        assert resolve_target(vp).metric == "cvvdp"

    def test_sdr_1080p_uses_ssimulacra2_high(self) -> None:
        """1080p SDR -> SSIMULACRA2, calibrated band around 81."""
        vp = make_video_params(source_width=1920, source_height=1080)
        spec = resolve_target(vp)
        assert spec.metric == "ssimulacra2"
        assert spec.target_lo < 81.0 < spec.target_hi

    def test_sdr_720p_is_hd_bucket(self) -> None:
        """720p (height == 720) is the HD bucket, not SD."""
        vp = make_video_params(source_width=1280, source_height=720)
        spec = resolve_target(vp)
        assert spec.metric == "ssimulacra2"
        assert spec.target_lo < 81.0 < spec.target_hi

    def test_sdr_dvd_uses_ssimulacra2_low(self) -> None:
        """SD/DVD (576-line) -> SSIMULACRA2, lower target band around 72."""
        vp = make_video_params(source_width=720, source_height=576)
        spec = resolve_target(vp)
        assert spec.metric == "ssimulacra2"
        assert spec.target_lo < 72.0 < spec.target_hi
        assert spec.target_hi < 81.0  # SD bucket sits below the HD SDR target

    def test_target_band_brackets_center(self) -> None:
        """The band is a small tolerance either side of the centre target."""
        spec = resolve_target(make_video_params(source_width=1920, source_height=1080))
        centre = (spec.target_lo + spec.target_hi) / 2
        assert abs(centre - 81.0) < 1e-9
        assert spec.target_hi - spec.target_lo < 4.0  # tight (~1%) band

    def test_returns_target_spec(self) -> None:
        spec = resolve_target(make_video_params())
        assert isinstance(spec, TargetSpec)
        assert spec.max_probes >= 1

    def test_grain_uses_crf_bounds_and_ssimulacra2(self) -> None:
        """A grain source takes the SVT-AV1 CRF path: SSIMULACRA2, CRF-scale
        bounds (bracketing the default 23), worst-case target band."""
        spec = resolve_target(make_video_params(grain=True, source_width=720, source_height=576))
        assert spec.metric == "ssimulacra2"
        assert spec.knob_lo == 14
        assert spec.knob_hi == 34
        assert spec.knob_lo < 23 < spec.knob_hi  # brackets the default CRF
        assert spec.target_lo < 71.0 < spec.target_hi

    def test_grain_samples_ten_windows(self) -> None:
        """The grain (CRF) path samples 10 windows: CRF is one value for the whole
        movie, so the search must SEE the hard scenes -- 3 windows miss them and the
        search rails to too-high a CRF (мыло)."""
        spec = resolve_target(make_video_params(grain=True, source_width=720, source_height=576))
        assert spec.window_count == 10

    def test_nvenc_samples_three_windows(self) -> None:
        """The NVEnc (QVBR) path samples 3 windows (mean pooling; QVBR is
        scene-adaptive)."""
        for vp in (
            make_video_params(source_width=1920, source_height=1080),  # HD SDR
            make_video_params(source_width=720, source_height=576),  # SD SDR
            make_video_params(color_transfer="smpte2084", color_matrix="bt2020nc"),  # HDR
        ):
            assert resolve_target(vp).window_count == 3

    def test_grain_overrides_resolution_bucket(self) -> None:
        """grain wins over the SDR height buckets (grain is always the SVT path)."""
        # A 1080p source flagged grain still gets CRF bounds, not QVBR [16,44].
        spec = resolve_target(make_video_params(grain=True, source_width=1920, source_height=1080))
        assert spec.knob_hi == 34

    def test_grain_hdr_raises_loudly(self) -> None:
        """A grain override on an HDR (PQ/HLG) source is refused loudly: grain
        routes to SSIMULACRA2, whose absolute scale is compressed on PQ, so
        scoring it there would silently mis-target. Reachable only via a manual
        grain override on HDR content."""
        for transfer in ("smpte2084", "arib-std-b67"):
            vp = make_video_params(
                grain=True, color_transfer=transfer, color_matrix="bt2020nc"
            )
            with pytest.raises(ValueError, match="grain target-quality is unsupported on HDR"):
                resolve_target(vp)


# ---------------------------------------------------------------------------
# probe_windows — evenly-spaced sample offsets, full-pass fallback
# ---------------------------------------------------------------------------


class TestProbeWindows:
    def test_even_spacing_exact(self) -> None:
        """D=100, n=3, w=20 -> gaps of 10: offsets [10, 40, 70]."""
        offsets = probe_windows(100.0, count=3, window_s=20.0)
        assert offsets == [10.0, 40.0, 70.0]

    def test_windows_stay_within_duration(self) -> None:
        offsets = probe_windows(6000.0, count=3, window_s=18.0)
        assert offsets is not None
        assert offsets[0] >= 0.0
        assert offsets[-1] + 18.0 <= 6000.0
        # strictly increasing, non-overlapping
        for a, b in itertools.pairwise(offsets):
            assert b - a >= 18.0

    def test_full_pass_fallback_short_source(self) -> None:
        """When the windows would cover >= 85% of the source, return None
        (just encode the whole short thing)."""
        assert probe_windows(50.0, count=3, window_s=18.0) is None

    def test_long_source_gets_windows(self) -> None:
        assert probe_windows(7200.0, count=3, window_s=18.0) is not None

    def test_full_pass_boundary(self) -> None:
        """At exactly 85% coverage the fallback triggers (>=)."""
        # n*w = 54; 54 / 0.85 = 63.529...; at duration where 54 == 0.85*D -> D=63.529
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


# ---------------------------------------------------------------------------
# source_is_variable_bitrate — VBR (guide by bitrate) vs CBR/flat (even sampling)
# ---------------------------------------------------------------------------


class TestSourceIsVariableBitrate:
    def test_vbr_spread_is_variable(self) -> None:
        """A wide spread of per-window bitrates (VBR) is above the threshold."""
        assert source_is_variable_bitrate([1.6, 2.8, 3.5, 5.0, 5.2, 2.1, 4.4]) is True

    def test_flat_is_not_variable(self) -> None:
        """A nearly-flat distribution (CBR) is below the threshold."""
        assert source_is_variable_bitrate([3.6, 3.7, 3.7, 3.8, 3.7, 3.7]) is False

    def test_fewer_than_two_is_not_variable(self) -> None:
        """Too few samples to judge -> False (caller falls back to even sampling)."""
        assert source_is_variable_bitrate([]) is False
        assert source_is_variable_bitrate([4.0]) is False

    def test_zero_mean_is_not_variable(self) -> None:
        """A degenerate all-zero list -> False (no division by zero)."""
        assert source_is_variable_bitrate([0.0, 0.0, 0.0]) is False

    def test_threshold_is_the_boundary(self) -> None:
        """CoV exactly at the threshold counts as variable (>=), just below does not.
        [1-d, 1+d] has mean 1 and population stdev d, so CoV == d."""
        assert source_is_variable_bitrate([0.9, 1.1], threshold=0.1) is True
        assert source_is_variable_bitrate([0.95, 1.05], threshold=0.1) is False


# ---------------------------------------------------------------------------
# interior_windows — drop the leading/trailing edge (intros/credits)
# ---------------------------------------------------------------------------


class TestInteriorWindows:
    def test_drops_leading_and_trailing_edges(self) -> None:
        """Windows in the first/last edge_skip fraction are dropped; interior kept.
        duration 1000, window 10, edge 0.06 -> keep [60, 930]."""
        scored = [
            (10.0, 5.0),    # leading edge (< 60) -> drop
            (60.0, 1.0),    # exactly lo -> keep
            (500.0, 9.0),   # interior -> keep
            (930.0, 2.0),   # exactly hi (1000*0.94 - 10) -> keep
            (950.0, 8.0),   # trailing edge (> 930) -> drop
        ]
        assert interior_windows(scored, duration_s=1000.0, window_s=10.0, edge_skip=0.06) == [
            (60.0, 1.0), (500.0, 9.0), (930.0, 2.0),
        ]

    def test_preserves_order(self) -> None:
        """Kept windows retain input order (not re-sorted); default edge_skip used."""
        scored = [(500.0, 9.0), (100.0, 1.0), (300.0, 5.0)]
        assert interior_windows(scored, duration_s=1000.0, window_s=10.0) == scored

    def test_empty_input(self) -> None:
        assert interior_windows([], duration_s=1000.0, window_s=10.0) == []


# ---------------------------------------------------------------------------
# select_hard_windows — top-N by value with a minimum spacing (NMS)
# ---------------------------------------------------------------------------


class TestSelectHardWindows:
    def test_picks_highest_value_windows(self) -> None:
        """The top ``count`` by value, returned in ascending time order."""
        scored = [(0.0, 1.0), (100.0, 5.0), (200.0, 3.0), (300.0, 9.0), (400.0, 2.0)]
        assert select_hard_windows(scored, count=2, min_gap_s=10.0) == [100.0, 300.0]

    def test_min_gap_skips_neighbours(self) -> None:
        """A high-value window too close to an already-picked one is skipped for the
        next spread-out candidate (so the picks can't cluster in one sequence)."""
        # 300 is highest; 305 is 2nd but within the 50s gap of 300 -> next pick is 100.
        scored = [(100.0, 5.0), (300.0, 9.0), (305.0, 8.0)]
        assert select_hard_windows(scored, count=2, min_gap_s=50.0) == [100.0, 300.0]

    def test_fewer_candidates_than_count(self) -> None:
        """Returns all it can when the candidates run out before ``count``."""
        assert select_hard_windows([(0.0, 1.0), (100.0, 2.0)], count=5, min_gap_s=10.0) == [
            0.0, 100.0
        ]

    def test_default_gap_applies(self) -> None:
        """The default min_gap (90s) is used when the argument is omitted."""
        # 0 and 50 are within 90s: only the higher (50) is kept, plus a far one (200).
        assert select_hard_windows([(0.0, 1.0), (50.0, 9.0), (200.0, 5.0)], count=3) == [
            50.0, 200.0
        ]

    def test_invalid_count_raises(self) -> None:
        with pytest.raises(ValueError, match="count"):
            select_hard_windows([(0.0, 1.0)], count=0, min_gap_s=10.0)

    def test_negative_gap_raises(self) -> None:
        with pytest.raises(ValueError, match="gap"):
            select_hard_windows([(0.0, 1.0)], count=1, min_gap_s=-1.0)
