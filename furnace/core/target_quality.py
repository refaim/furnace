"""Target-quality knob search (pure, no I/O).

Given a probe callback that encodes a short window at a candidate quality knob
(NVEnc QVBR or SVT CRF) and returns an aggregated perceptual score, find the
knob whose score lands inside a target range -- so the final full encode can run
at that knob with NO metrics attached.

The interpolation search is ported from Av1an's ``predict_quantizer``
(av1an-core/src/target_quality.rs) and its interpolation helpers
(av1an-core/src/interpol.rs): binary search for the first two probes, then
linear / natural-cubic-spline / PCHIP interpolation as history grows, always
falling back to bisection when interpolation is degenerate. The knob is assumed
monotonic -- the score decreases as the knob rises (higher QVBR/CRF -> lower
quality) -- so the search runs in score->knob space (x = observed scores in
ascending order, y = knobs) and evaluates the inverse at the target score.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from .detect import is_hdr_transfer
from .models import VideoParams
from .quality import final_output_dimensions

# Number of probed points that select each interpolation method.
_LINEAR_POINTS = 2
_MIN_SPLINE_POINTS = 3
_PCHIP_POINTS = 4

# PCHIP monotonicity guard: if the squared sum of the normalised endpoint
# derivatives exceeds this, scale them down (Fritsch-Carlson). From Av1an.
_PCHIP_MAX_TAU_SQUARED = 9.0


def _round_half_up(value: float) -> int:
    """Round to the nearest int, ties away from zero for the non-negative knob
    domain (``search_knob`` enforces ``lo >= 0``).

    Matches Rust ``f64::round`` used by Av1an; Python's built-in ``round`` uses
    banker's rounding, which would diverge on exact ``.5`` knob predictions.
    (Half-up and half-away-from-zero agree only for non-negative inputs, which
    is why the search rejects a negative lower bound.)
    """
    return math.floor(value + 0.5)


def _find_interval(x: list[float], xi: float, segments: int) -> int:
    """Index of the segment [x[k], x[k+1]] containing xi, else 0 (extrapolate)."""
    for i in range(segments):
        if x[i] <= xi <= x[i + 1]:
            return i
    return 0


def linear_interpolate(x: list[float], y: list[float], xi: float) -> float | None:
    """Interpolate y at xi across two (x, y) points; None if there are not
    exactly two points or the domain is flat/descending."""
    if len(x) != _LINEAR_POINTS or len(y) != _LINEAR_POINTS:
        return None
    if x[1] <= x[0]:
        return None
    t = (xi - x[0]) / (x[1] - x[0])
    return y[0] + t * (y[1] - y[0])


def natural_cubic_spline(x: list[float], y: list[float], xi: float) -> float | None:
    """Natural cubic spline of y at xi. None unless there are >= 3 strictly
    increasing knots and xi lies within the observed range (no extrapolation)."""
    n = len(x)
    if n < _MIN_SPLINE_POINTS or n != len(y):
        return None
    if xi < x[0] or xi > x[n - 1]:
        return None

    h = [x[i + 1] - x[i] for i in range(n - 1)]
    if any(gap <= 0.0 for gap in h):
        return None

    a = [0.0] * n
    b = [2.0] * n
    c = [0.0] * n
    d = [0.0] * n
    # Natural boundary conditions: second derivative = 0 at both endpoints.
    b[0] = 1.0
    b[n - 1] = 1.0
    for i in range(1, n - 1):
        a[i] = h[i - 1]
        b[i] = 2.0 * (h[i - 1] + h[i])
        c[i] = h[i]
        d[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    # Thomas algorithm for the tridiagonal system of second-derivative moments.
    # The strict-increase guard above makes the matrix diagonally dominant, so
    # every pivot is strictly positive -- Av1an's singular-pivot->None fallback
    # is unreachable here and is deliberately omitted.
    pivot = [0.0] * n
    z = [0.0] * n
    m = [0.0] * n
    pivot[0] = b[0]
    for i in range(1, n):
        pivot[i] = b[i] - a[i] * c[i - 1] / pivot[i - 1]
        z[i] = (d[i] - a[i] * z[i - 1]) / pivot[i]
    m[n - 1] = z[n - 1]
    for i in range(n - 2, -1, -1):
        m[i] = z[i] - c[i] * m[i + 1] / pivot[i]

    k = _find_interval(x, xi, n - 1)
    dx = xi - x[k]
    hk = h[k]
    a_coeff = y[k]
    b_coeff = (y[k + 1] - y[k]) / hk - hk * (2.0 * m[k] + m[k + 1]) / 3.0
    c_coeff = m[k]
    d_coeff = (m[k + 1] - m[k]) / (3.0 * hk)
    return a_coeff + b_coeff * dx + c_coeff * dx**2 + d_coeff * dx**3


def pchip_interpolate(x: list[float], y: list[float], xi: float) -> float | None:
    """Monotone piecewise cubic Hermite (PCHIP) of y at xi over exactly four
    strictly increasing knots; None if that precondition is not met."""
    if len(x) != _PCHIP_POINTS or len(y) != _PCHIP_POINTS:
        return None
    if any(x[i + 1] <= x[i] for i in range(3)):
        return None

    k = _find_interval(x, xi, 3)

    s0 = (y[1] - y[0]) / (x[1] - x[0])
    s1 = (y[2] - y[1]) / (x[2] - x[1])
    s2 = (y[3] - y[2]) / (x[3] - x[2])
    d = [s0, 0.0, 0.0, s2]

    # Interior derivatives: weighted harmonic mean, zeroed at local extrema.
    for i in (1, 2):
        if i == 1:
            s_prev, s_next = s0, s1
            h_prev, h_next = x[1] - x[0], x[2] - x[1]
        else:
            s_prev, s_next = s1, s2
            h_prev, h_next = x[2] - x[1], x[3] - x[2]
        if s_prev * s_next <= 0.0:
            d[i] = 0.0
        else:
            w1 = 2.0 * h_next + h_prev
            w2 = 2.0 * h_prev + h_next
            d[i] = (w1 + w2) / (w1 / s_prev + w2 / s_next)

    # Fritsch-Carlson monotonicity constraint.
    slopes = [s0, s1, s2]
    for i in range(3):
        if slopes[i] == 0.0:
            d[i] = 0.0
            d[i + 1] = 0.0
        else:
            alpha = d[i] / slopes[i]
            beta = d[i + 1] / slopes[i]
            tau = alpha * alpha + beta * beta
            if tau > _PCHIP_MAX_TAU_SQUARED:
                scale = 3.0 / math.sqrt(tau)
                d[i] = scale * alpha * slopes[i]
                d[i + 1] = scale * beta * slopes[i]

    hseg = x[k + 1] - x[k]
    t = (xi - x[k]) / hseg
    t2 = t * t
    t3 = t2 * t
    return (
        (2.0 * t3 - 3.0 * t2 + 1.0) * y[k]
        + (t3 - 2.0 * t2 + t) * hseg * d[k]
        + (-2.0 * t3 + 3.0 * t2) * y[k + 1]
        + (t3 - t2) * hseg * d[k + 1]
    )


def predict_knob(
    lo: int,
    hi: int,
    history: list[tuple[int, float]],
    target_lo: float,
    target_hi: float,
) -> int:
    """Predict the next knob to probe, given the probes so far.

    0-1 probes -> bisection midpoint of [lo, hi]. 2 -> linear, 3 -> natural
    cubic spline, 4 -> PCHIP, all in score->knob space at the target midpoint.
    Any degenerate interpolation (or 5+ probes) falls back to bisection. The
    result is rounded to an integer knob and clamped to [lo, hi].
    """
    target = (target_lo + target_hi) / 2.0
    binary = (lo + hi) / 2.0
    n = len(history)

    predicted: float | None
    if n < _LINEAR_POINTS:
        predicted = binary
    else:
        ordered = sorted(history, key=lambda ks: ks[1])
        scores = [float(s) for _, s in ordered]
        knobs = [float(k) for k, _ in ordered]
        if n == _LINEAR_POINTS:
            predicted = linear_interpolate(scores, knobs, target)
        elif n == _MIN_SPLINE_POINTS:
            predicted = natural_cubic_spline(scores, knobs, target)
        elif n == _PCHIP_POINTS:
            predicted = pchip_interpolate(scores[:_PCHIP_POINTS], knobs[:_PCHIP_POINTS], target)
        else:
            predicted = None
        if predicted is None:
            predicted = binary

    return max(lo, min(hi, _round_half_up(predicted)))


@dataclass(frozen=True, slots=True)
class KnobSearchResult:
    """Outcome of a knob search.

    ``knob`` is the chosen quality knob; ``score`` its measured metric; ``hit``
    whether that score landed inside the target range; ``probes`` the full
    (knob, score) history in probe order.
    """

    knob: int
    score: float
    hit: bool
    probes: tuple[tuple[int, float], ...]


def _select(
    history: list[tuple[int, float]],
    target_lo: float,
    target_hi: float,
) -> KnobSearchResult:
    """Choose the final knob: the highest in-range knob (smallest file); if none
    landed in range, the probe whose score is closest to the target midpoint."""
    in_range = [(k, s) for k, s in history if target_lo <= s <= target_hi]
    if in_range:
        knob, score = max(in_range, key=lambda ks: ks[0])
        hit = True
    else:
        mid = (target_lo + target_hi) / 2.0
        knob, score = min(history, key=lambda ks: abs(ks[1] - mid))
        hit = False
    return KnobSearchResult(knob=knob, score=score, hit=hit, probes=tuple(history))


def search_knob(
    probe: Callable[[int], float],
    *,
    target_lo: float,
    target_hi: float,
    lo: int,
    hi: int,
    max_probes: int,
) -> KnobSearchResult:
    """Interpolation-search the quality knob to hit [target_lo, target_hi].

    ``probe(knob)`` encodes a window at ``knob`` and returns its aggregated
    perceptual score. At most ``max_probes`` probes run; the search stops early
    when a probe lands in range or the prediction stalls on an already-probed
    knob. Assumes the score decreases monotonically as the knob rises.
    """
    if lo < 0:
        raise ValueError(f"knob lower bound {lo} must be non-negative")
    if lo > hi:
        raise ValueError(f"knob lower bound {lo} exceeds upper bound {hi}")
    if target_lo > target_hi:
        raise ValueError(f"target lower bound {target_lo} exceeds upper bound {target_hi}")
    if max_probes < 1:
        raise ValueError(f"max_probes must be at least 1, got {max_probes}")

    history: list[tuple[int, float]] = []
    probed: set[int] = set()
    lower, upper = lo, hi
    while True:
        knob = predict_knob(lower, upper, history, target_lo, target_hi)
        if knob in probed:
            break
        score = probe(knob)
        if not math.isfinite(score):
            raise ValueError(f"probe returned a non-finite score {score!r} at knob {knob}")
        history.append((knob, score))
        probed.add(knob)
        if target_lo <= score <= target_hi:
            break
        if len(history) >= max_probes:
            break
        if score > target_hi:
            lower = min(knob + 1, upper)
        else:
            upper = max(knob - 1, lower)

    return _select(history, target_lo, target_hi)


# ---------------------------------------------------------------------------
# Furnace domain policy: content -> (metric, target band, knob bounds) and the
# probe-window layout. Pure rules -- the service orchestrates the I/O around
# them.
# ---------------------------------------------------------------------------

# HDR (PQ / HLG) is decided by ``core.detect.is_hdr_transfer`` -- the single
# source of truth shared with the grain probe gate and the planner's grain
# routing, so those can never disagree with this domain split. HDR routes to
# CVVDP, the only metric that scores PQ correctly (SSIMULACRA2's absolute scale is
# compressed on PQ).
# CALIBRATE/VERIFY: NVEncC's --vship-cvvdp auto-picks the PQ model on smpte2084
# (confirmed live); it is UNVERIFIED whether it selects the HLG EOTF on
# arib-std-b67. HLG is vanishingly rare in a movie archive (remuxes are PQ), but
# an HLG source scored under a PQ EOTF would bias its chosen QVBR -- verify
# during calibration if HLG content ever appears.

# Below this final height the source is SD/DVD-class. Splits the NON-grain SDR
# target only (SD DVDs cap lower on SSIMULACRA2); the grain path uses one target
# for every resolution and ``core.detect.needs_grain_probe`` no longer gates on
# height at all.
_SD_MAX_HEIGHT = 720

# Calibrated centre targets (2026-07-14, judged on HARD scenes -- max-complexity
# for non-grain where detail/blocking fails, dark+detailed for grain where grain
# fails -- with the metric pooled the way the search pools it). CVVDP is JOD (0-10,
# higher better); SSIMULACRA2 is 0-100 (higher better).
#   HDR (CVVDP, NVEnc mean): 9.5 -- near-transparent on 4K PQ. Below this the fine
#     grain that masks dark-gradient banding gets eaten and banding shows; 9.5 is
#     the floor. Lands QVBR ~24-28.
#   1080p SDR (SSIMULACRA2, NVEnc mean): 81 -- fine detail holds on hard (dark,
#     detailed) scenes at QVBR ~34; the mean over sample windows there is ~81.
#   SD/DVD SDR (SSIMULACRA2, NVEnc mean): 72 -- PROVISIONAL, uncalibrated: no
#     non-grain SD source was available (a denoised DVD is a rare edge case).
_HDR_TARGET = 9.5
_HD_SDR_TARGET = 81.0
_SD_SDR_TARGET = 72.0
# +/-1% band around the centre target. CALIBRATION CONSTRAINT: the band must stay
# at least as wide as one integer-knob step's metric delta, or no integer knob
# can land in-range and the search always returns the closest probe with
# ``hit=False``. Measured per-knob-step deltas near the calibrated centres (~0.03
# JOD/QVBR for CVVDP, ~1 SSIMULACRA2/QVBR, ~0.5 SSIMULACRA2/CRF for grain) sit
# inside these bands (0.19 / 1.6 / 1.4), so hits are expected.
_TARGET_TOLERANCE = 0.01

# NVEnc QVBR search bounds and probe budget. The default recipe anchors sit
# around 34-36; the band is wide enough to reach both transparency and thrift.
# NOTE: these bounds are QVBR-specific (NVEnc). The SVT-AV1 grain path drives CRF
# over a different scale (0-63, default 23) and uses its own bounds below.
_QVBR_LO = 16
_QVBR_HI = 44
_MAX_PROBES = 4

# SVT-AV1 grain path: CRF knob (0-63, load-bearing default 23). SSIMULACRA2 is
# pooled worst-case WITHIN a window (low-percentile p5 frames) since CRF is
# constant between scenes; ACROSS windows the service takes the MIN (the hardest
# sampled scene governs, because the window selection already targets the hard
# scenes). The target sits below a mean target (worst-case frame pooling AND
# grain's stochastic irreproducibility both pull it down). Bounds bracket the
# default 23. Calibrated across the whole SD-DVD + BD grain collection (2026-07-15):
# a p5 of ~71 = last-transparent (below it detail mushes), and the governing
# whole-movie CRF landed 20-28 per title -- one target serves HD and SD (no
# resolution split, unlike the SDR non-grain path).
_CRF_LO = 14
_CRF_HI = 34
_GRAIN_TARGET = 71.0

# Probe windows would-cover >= this fraction of the source -> just encode the
# whole (short) source instead of windowing it.
_FULL_PASS_FRACTION = 0.85

# Probe-window layout. The window LENGTH is shared; the window COUNT is per-path
# policy carried on the TargetSpec (see resolve_target). Public so the service and
# its tests share one source of truth.
PROBE_WINDOW_SECONDS = 18.0

# Grain (SVT-AV1 CRF) samples 10 windows; NVEnc (QVBR) samples 3. CRF is one value
# for the whole movie, so the grain search must SEE the hard scenes -- 3 windows
# miss them and the search rails to too-high a CRF (мыло). Grain pools worst-case
# (min) across windows; NVEnc mean-pools (QVBR is scene-adaptive).
_GRAIN_WINDOW_COUNT = 10
_NVENC_WINDOW_COUNT = 3

# Grain window SELECTION is regime-dependent (see the service). Measured across the
# collection: on a VBR source the encoder spent bits on the hard scenes, so the
# highest-bitrate windows ARE the hard scenes -- sample those. On a CBR source the
# bitrate is flat and says nothing about difficulty (it even points at the easy
# scenes), but hard scenes are common there, so evenly-spaced sampling catches them.
# The regime is read from the coefficient of variation (stdev/mean) of the per-window
# source bitrate: the collection splits cleanly (CBR ~0.01-0.02, VBR ~0.11-0.25).
_VBR_COV_THRESHOLD = 0.05
# Minimum spacing between selected hard windows, so the top-N by bitrate can't all
# cluster in one intense sequence and miss other hard scenes elsewhere.
_HARD_WINDOW_MIN_GAP_S = 90.0
# A coefficient of variation needs at least two windows.
_MIN_COV_SAMPLES = 2
# Ignore the leading/trailing fraction of the timeline (intros, credits) when reading
# source complexity for grain window selection, so a static logo or credit roll is
# neither read as difficulty nor picked as a hard window.
_EDGE_SKIP_FRACTION = 0.06


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """Resolved target-quality plan for one job: which perceptual ``metric`` to
    drive, the acceptable ``[target_lo, target_hi]`` score band, the knob search
    bounds ``[knob_lo, knob_hi]``, the probe budget ``max_probes``, and how many
    probe windows to sample (``window_count``)."""

    metric: str
    target_lo: float
    target_hi: float
    knob_lo: int
    knob_hi: int
    max_probes: int
    window_count: int


def resolve_target(vp: VideoParams) -> TargetSpec:
    """Map a job's content domain to its target-quality spec.

    Grain sources take the SVT-AV1 path: SSIMULACRA2 over a CRF knob (worst-case
    pooling is applied by the service). Otherwise the NVEnc path: HDR (PQ/HLG
    transfer) -> CVVDP; SDR splits by the *encoded* height into an HD bucket
    (>= 720p) and an SD/DVD bucket (< 720p), both driven by SSIMULACRA2 at
    different centre targets. Targets are calibrated (see the constants above;
    the SD/DVD SDR non-grain target is the one exception, still provisional).
    The knob bounds differ by path (CRF for grain, QVBR for NVEnc).

    Raises ValueError for a grain source on an HDR (PQ/HLG) transfer: grain routes
    to SSIMULACRA2, whose absolute scale is compressed on PQ, so scoring it there
    would silently mis-target. HDR content belongs on the NVEnc/CVVDP path.

    DO NOT DELETE THIS REFUSAL AS UNREACHABLE. The planner already clears ``grain``
    when the resolved transfer is HDR (``_build_video_params``), so no plan it builds
    can land here — but that is precisely why this must stay: it is the backstop for
    a ``VideoParams`` built OUTSIDE the planner, i.e. a hand-edited plan JSON, whose
    loader (``furnace.plan``) obeys the file rather than re-deciding. The planner
    coerces; this refuses loudly.
    """
    if vp.grain:
        if is_hdr_transfer(vp.color_transfer):
            raise ValueError(
                f"grain target-quality is unsupported on HDR sources "
                f"(transfer {vp.color_transfer!r}): SSIMULACRA2 does not score PQ/HLG "
                f"correctly; HDR belongs on the NVEnc/CVVDP path"
            )
        return _spec(
            "ssimulacra2", _GRAIN_TARGET, _CRF_LO, _CRF_HI,
            window_count=_GRAIN_WINDOW_COUNT,
        )

    _, final_h = final_output_dimensions(vp)
    if is_hdr_transfer(vp.color_transfer):
        metric, centre = "cvvdp", _HDR_TARGET
    elif final_h < _SD_MAX_HEIGHT:
        metric, centre = "ssimulacra2", _SD_SDR_TARGET
    else:
        metric, centre = "ssimulacra2", _HD_SDR_TARGET
    return _spec(
        metric, centre, _QVBR_LO, _QVBR_HI,
        window_count=_NVENC_WINDOW_COUNT,
    )


def _spec(
    metric: str,
    centre: float,
    knob_lo: int,
    knob_hi: int,
    *,
    window_count: int,
) -> TargetSpec:
    tol = centre * _TARGET_TOLERANCE
    return TargetSpec(
        metric=metric,
        target_lo=centre - tol,
        target_hi=centre + tol,
        knob_lo=knob_lo,
        knob_hi=knob_hi,
        max_probes=_MAX_PROBES,
        window_count=window_count,
    )


def probe_windows(duration_s: float, *, count: int, window_s: float) -> list[float] | None:
    """Start offsets (seconds) for ``count`` evenly-spaced windows of ``window_s``.

    Windows and the gaps between them (including the leading and trailing gap)
    are equal, so the samples span the whole timeline without clustering. Returns
    ``None`` -- the full-pass fallback -- when the windows would cover at least
    :data:`_FULL_PASS_FRACTION` of the source (a short extra is cheaper to encode
    whole than to window). Raises for non-positive inputs.
    """
    if count < 1:
        raise ValueError(f"probe window count must be >= 1, got {count}")
    if window_s <= 0.0:
        raise ValueError(f"probe window length must be positive, got {window_s}")
    if duration_s <= 0.0:
        raise ValueError(f"source duration must be positive, got {duration_s}")

    total = window_s * count
    if total >= _FULL_PASS_FRACTION * duration_s:
        return None
    gap = (duration_s - total) / (count + 1)
    return [gap * (k + 1) + window_s * k for k in range(count)]


def source_is_variable_bitrate(
    bitrates: list[float], *, threshold: float = _VBR_COV_THRESHOLD
) -> bool:
    """Whether the per-window source bitrate varies enough to guide hard-scene
    selection.

    True (VBR): the source encoder concentrated bits on the hard scenes, so the
    highest-bitrate windows ARE the hard scenes -- select by bitrate. False (CBR /
    flat, or fewer than two samples): the bitrate says nothing about difficulty (it
    can even point at the easy scenes), so the caller falls back to even sampling.

    The signal is the coefficient of variation (population stdev / mean) of
    ``bitrates``; the grain collection splits cleanly around ``threshold``
    (CBR ~0.01-0.02, VBR ~0.11-0.25).
    """
    n = len(bitrates)
    if n < _MIN_COV_SAMPLES:
        return False
    mean = sum(bitrates) / n
    if mean <= 0.0:
        return False
    variance = sum((b - mean) ** 2 for b in bitrates) / n
    return math.sqrt(variance) / mean >= threshold


def interior_windows(
    scored: list[tuple[float, float]],
    *,
    duration_s: float,
    window_s: float,
    edge_skip: float = _EDGE_SKIP_FRACTION,
) -> list[tuple[float, float]]:
    """Keep only the candidate windows in the interior of the timeline, dropping the
    leading and trailing ``edge_skip`` fraction (intros, credits) -- so a static logo
    or credit roll is neither read as difficulty nor picked as a hard window.
    ``scored`` is ``(start_s, value)`` per candidate; the filtered order is preserved.
    """
    lo = duration_s * edge_skip
    hi = duration_s * (1 - edge_skip) - window_s
    return [(start, value) for start, value in scored if lo <= start <= hi]


def select_hard_windows(
    scored: list[tuple[float, float]], *, count: int, min_gap_s: float = _HARD_WINDOW_MIN_GAP_S
) -> list[float]:
    """Pick up to ``count`` window start offsets, greedily taking the highest-value
    (highest source bitrate = hardest) candidates while keeping every pick at least
    ``min_gap_s`` from the ones already chosen -- so the hardest windows can't all
    cluster in one intense sequence and miss hard scenes elsewhere.

    ``scored`` is ``(start_s, value)`` per candidate window (value = source bytes /
    bitrate). Returns the chosen offsets in ascending time order. Ties keep input
    order (a stable sort), so the result is deterministic for a given candidate list.
    """
    if count < 1:
        raise ValueError(f"hard window count must be >= 1, got {count}")
    if min_gap_s < 0.0:
        raise ValueError(f"min gap must be non-negative, got {min_gap_s}")
    chosen: list[float] = []
    for start, _value in sorted(scored, key=lambda sv: -sv[1]):
        if all(abs(start - c) >= min_gap_s for c in chosen):
            chosen.append(start)
            if len(chosen) == count:
                break
    return sorted(chosen)
