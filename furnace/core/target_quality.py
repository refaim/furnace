from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from .detect import is_hdr_transfer
from .models import VideoParams
from .quality import final_output_dimensions

_LINEAR_POINTS = 2
_MIN_SPLINE_POINTS = 3
_PCHIP_POINTS = 4

_PCHIP_MAX_TAU_SQUARED = 9.0


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def _find_interval(x: list[float], xi: float, segments: int) -> int:
    for i in range(segments):
        if x[i] <= xi <= x[i + 1]:
            return i
    return 0


def linear_interpolate(x: list[float], y: list[float], xi: float) -> float | None:
    if len(x) != _LINEAR_POINTS or len(y) != _LINEAR_POINTS:
        return None
    if x[1] <= x[0]:
        return None
    t = (xi - x[0]) / (x[1] - x[0])
    return y[0] + t * (y[1] - y[0])


def natural_cubic_spline(x: list[float], y: list[float], xi: float) -> float | None:
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
    b[0] = 1.0
    b[n - 1] = 1.0
    for i in range(1, n - 1):
        a[i] = h[i - 1]
        b[i] = 2.0 * (h[i - 1] + h[i])
        c[i] = h[i]
        d[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

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
    if len(x) != _PCHIP_POINTS or len(y) != _PCHIP_POINTS:
        return None
    if any(x[i + 1] <= x[i] for i in range(3)):
        return None

    k = _find_interval(x, xi, 3)

    s0 = (y[1] - y[0]) / (x[1] - x[0])
    s1 = (y[2] - y[1]) / (x[2] - x[1])
    s2 = (y[3] - y[2]) / (x[3] - x[2])
    d = [s0, 0.0, 0.0, s2]

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
    knob: int
    score: float
    hit: bool
    probes: tuple[tuple[int, float], ...]


def _select(
    history: list[tuple[int, float]],
    target_lo: float,
    target_hi: float,
) -> KnobSearchResult:
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
    seed: int | None = None,
) -> KnobSearchResult:
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
        if seed is not None and not history:
            knob = max(lower, min(upper, seed))
        else:
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


_SD_MAX_HEIGHT = 720

_HDR_TARGET = 9.5
_HD_SDR_TARGET = 81.0
_SD_SDR_TARGET = 72.0
_TARGET_TOLERANCE = 0.01

_QVBR_LO = 16
_QVBR_HI = 44
_MAX_PROBES = 4

_CRF_HI = 34
_GRAIN_TARGET = 70.0
_GRAIN_SD_CRF_FLOOR = 26
_GRAIN_HD_MIN_HEIGHT = 720
_GRAIN_HD_QVBR = 32

_FULL_PASS_FRACTION = 0.85

PROBE_WINDOW_SECONDS = 18.0

_GRAIN_WINDOW_COUNT = 10
# Three windows sampled the runtime too thinly: the knob tracked whichever
# stretches the grid happened to land on rather than the file, so sibling
# episodes of one series landed anywhere from QVBR 23 to 32.
_NVENC_WINDOW_COUNT = 10

GRAIN_POOL_PERCENTILE = 20.0
_PERCENTILE_MAX = 100.0

# How far the next source's bitrate may sit from an already-solved one and still
# borrow its answer as a starting point. Dark's first season spans 27.9-30.9 Mbit/s
# (well inside), while its second season jumps to ~40 and starts over.
_SEED_BITRATE_TOLERANCE = 0.20

_VBR_COV_THRESHOLD = 0.05
_HARD_WINDOW_MIN_GAP_S = 90.0
_MIN_COV_SAMPLES = 2
_EDGE_SKIP_FRACTION = 0.06


@dataclass(frozen=True, slots=True)
class TargetSpec:
    metric: str
    target_lo: float
    target_hi: float
    knob_lo: int
    knob_hi: int
    max_probes: int
    window_count: int


def resolve_target(vp: VideoParams) -> TargetSpec:
    if vp.grain:
        if is_hdr_transfer(vp.color_transfer):
            raise ValueError(
                f"grain target-quality is unsupported on HDR sources "
                f"(transfer {vp.color_transfer!r}): SSIMULACRA2 does not score PQ/HLG "
                f"correctly; HDR belongs on the NVEnc/CVVDP path"
            )
        if not grain_uses_svt(vp):
            raise ValueError(
                f"grain target-quality search is SD-only; HD grain "
                f"(final height >= {_GRAIN_HD_MIN_HEIGHT}) encodes at a fixed NVENC QVBR "
                f"and must not reach resolve_target"
            )
        return _spec(
            "ssimulacra2",
            _GRAIN_TARGET,
            _GRAIN_SD_CRF_FLOOR,
            _CRF_HI,
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
        metric,
        centre,
        _QVBR_LO,
        _QVBR_HI,
        window_count=_NVENC_WINDOW_COUNT,
    )


def grain_uses_svt(vp: VideoParams) -> bool:
    if not vp.grain:
        return False
    _, final_h = final_output_dimensions(vp)
    return final_h < _GRAIN_HD_MIN_HEIGHT


def fixed_grain_knob(vp: VideoParams) -> int | None:
    if vp.grain and not grain_uses_svt(vp):
        return _GRAIN_HD_QVBR
    return None


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


def pool_grain_windows(scores: list[float], *, percentile: float = GRAIN_POOL_PERCENTILE) -> float:
    if not scores:
        raise ValueError("pool_grain_windows needs at least one window score")
    if not 0.0 <= percentile <= _PERCENTILE_MAX:
        raise ValueError(f"percentile must be within [0, {_PERCENTILE_MAX}], got {percentile}")
    ordered = sorted(scores)
    rank = (len(ordered) - 1) * percentile / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (rank - lo) * (ordered[hi] - ordered[lo])


def source_is_variable_bitrate(bitrates: list[float], *, threshold: float = _VBR_COV_THRESHOLD) -> bool:
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
    lo = duration_s * edge_skip
    hi = duration_s * (1 - edge_skip) - window_s
    return [(start, value) for start, value in scored if lo <= start <= hi]


def select_hard_windows(
    scored: list[tuple[float, float]], *, count: int, min_gap_s: float = _HARD_WINDOW_MIN_GAP_S
) -> list[float]:
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


@dataclass(frozen=True, slots=True)
class SeedKey:
    metric: str
    knob_lo: int
    knob_hi: int
    grain: bool
    final_width: int
    final_height: int


def seed_key(vp: VideoParams, spec: TargetSpec) -> SeedKey:
    final_w, final_h = final_output_dimensions(vp)
    return SeedKey(
        metric=spec.metric,
        knob_lo=spec.knob_lo,
        knob_hi=spec.knob_hi,
        grain=vp.grain,
        final_width=final_w,
        final_height=final_h,
    )


class SeedMemory:
    """Knobs already solved this run, so a comparable source can start near the answer.

    Files only share a starting point when they land on the same rung of the
    quality curve: same metric and target, same encoder path, same output size,
    and a source squeezed to a comparable degree. Anything else searches from
    the bracket midpoint, exactly as before.
    """

    def __init__(self, *, tolerance: float = _SEED_BITRATE_TOLERANCE) -> None:
        if tolerance < 0.0:
            raise ValueError(f"seed bitrate tolerance must be non-negative, got {tolerance}")
        self._tolerance = tolerance
        self._solved: dict[SeedKey, tuple[int, int]] = {}

    def remember(self, key: SeedKey, *, source_bitrate: int, knob: int) -> None:
        if source_bitrate <= 0:
            return
        self._solved[key] = (source_bitrate, knob)

    def suggest(self, key: SeedKey, *, source_bitrate: int) -> int | None:
        if source_bitrate <= 0:
            return None
        solved = self._solved.get(key)
        if solved is None:
            return None
        previous_bitrate, knob = solved
        if abs(source_bitrate - previous_bitrate) > self._tolerance * previous_bitrate:
            return None
        return knob
