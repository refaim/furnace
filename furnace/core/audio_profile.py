"""Pure classification rules for the fake-surround detector.

This module is a port-of-entry into the core layer: it receives raw numeric
metrics produced by the FFmpeg/numpy adapter and emits a classification
verdict. No I/O, no numpy imports, no filesystem access.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import cast

from .downmix import STEREO_CHANNELS, DownmixMode

# 5.1 and 7.1 are the only multichannel layouts the detector understands.
_SURROUND_5_1_CHANNELS = 6
_SURROUND_7_1_CHANNELS = 8

# ---------------------------------------------------------------------------
# Thresholds — calibrated against the 52-track validation corpus described in
# docs/superpowers/specs/2026-04-11-fake-surround-detector-design.md.
# Changing any of these requires re-running the golden test suite.
# ---------------------------------------------------------------------------

SURROUND_SILENT_DB = -50.0  # both Ls and Rs below → 1 pt
LFE_DEAD_DB = -65.0         # rms_LFE below → 1 pt
CENTER_DOM_DB = 10.0        # rms_C - max(others) above → 1 pt
MONO_CORR = 0.98            # corr(L,R) above → fronts-mono candidate
MONO_RMS_DIFF_DB = 2.0      # |rms_L - rms_R| below → fronts-mono confirmed
SURROUNDS_COPY_CORR = 0.95  # corr(Ls,L) AND corr(Rs,R) above → 2 pts
LS_RS_IDENT_CORR = 0.85     # corr(Ls,Rs) above → 1 pt

STEREO_MONO_CORR = 0.98
STEREO_MONO_DIFF_DB = 2.0
STEREO_SUSP_CORR = 0.96
STEREO_SUSP_DIFF_DB = 3.0

FAKE_SCORE_THRESHOLD = 2
SUSPICIOUS_SCORE = 1

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class Verdict(enum.StrEnum):
    REAL = "real"
    SUSPICIOUS = "suspicious"
    FAKE = "fake"


@dataclass(frozen=True)
class AudioMetrics:
    """Raw per-channel measurements and pairwise correlations.

    For stereo tracks only ``rms_l``, ``rms_r``, and ``corr_lr`` are populated;
    multichannel-only fields are ``None``. For 5.1, ``rms_lb`` and ``rms_rb``
    are ``None``. For 7.1 every field is populated.
    """

    channels: int

    rms_l: float
    rms_r: float
    rms_c: float | None
    rms_lfe: float | None
    rms_ls: float | None
    rms_rs: float | None
    rms_lb: float | None
    rms_rb: float | None

    corr_lr: float
    corr_ls_l: float | None
    corr_rs_r: float | None
    corr_ls_rs: float | None
    corr_lb_ls: float | None
    corr_rb_rs: float | None


@dataclass(frozen=True)
class AudioProfile:
    """Classification verdict for one audio track."""

    verdict: Verdict
    score: int
    suggested: DownmixMode | None
    reasons: tuple[str, ...]
    metrics: AudioMetrics


# ---------------------------------------------------------------------------
# Classification entrypoint
#
# Task 2 lands the real stereo classifier; multichannel is still a stub and
# gets filled in by Task 3.
# ---------------------------------------------------------------------------


def classify_audio(metrics: AudioMetrics) -> AudioProfile:
    if metrics.channels == STEREO_CHANNELS:
        return _classify_stereo(metrics)
    if metrics.channels in (_SURROUND_5_1_CHANNELS, _SURROUND_7_1_CHANNELS):
        return _classify_multichannel(metrics)
    raise ValueError(f"unsupported channels: {metrics.channels}")


def _classify_stereo(metrics: AudioMetrics) -> AudioProfile:
    corr = metrics.corr_lr
    diff = abs(metrics.rms_l - metrics.rms_r)

    if corr > STEREO_MONO_CORR and diff < STEREO_MONO_DIFF_DB:
        return AudioProfile(
            verdict=Verdict.FAKE,
            score=2,
            suggested=DownmixMode.MONO,
            reasons=(
                f"left and right are identical (mono) — corr={corr:.3f}, diff={diff:.1f} dB",
            ),
            metrics=metrics,
        )

    if corr > STEREO_SUSP_CORR and diff < STEREO_SUSP_DIFF_DB:
        return AudioProfile(
            verdict=Verdict.SUSPICIOUS,
            score=1,
            suggested=DownmixMode.MONO,
            reasons=(
                f"left and right are nearly identical — corr={corr:.3f}, diff={diff:.1f} dB",
            ),
            metrics=metrics,
        )

    return AudioProfile(
        verdict=Verdict.REAL, score=0, suggested=None, reasons=(), metrics=metrics,
    )


def _classify_multichannel(metrics: AudioMetrics) -> AudioProfile:
    # All multichannel-specific fields are populated by the adapter for 5.1/7.1.
    # Cast away Optional here for type narrowing; adapter contract guarantees this.
    rms_c = cast("float", metrics.rms_c)
    rms_lfe = cast("float", metrics.rms_lfe)
    rms_ls = cast("float", metrics.rms_ls)
    rms_rs = cast("float", metrics.rms_rs)
    corr_ls_l = cast("float", metrics.corr_ls_l)
    corr_rs_r = cast("float", metrics.corr_rs_r)
    corr_ls_rs = cast("float", metrics.corr_ls_rs)

    score = 0
    reasons: list[str] = []

    sig_silent_surrounds = rms_ls < SURROUND_SILENT_DB and rms_rs < SURROUND_SILENT_DB
    if sig_silent_surrounds:
        score += 1
        reasons.append(
            f"both surrounds are silent (Ls={rms_ls:.0f}, Rs={rms_rs:.0f} dB)",
        )

    sig_lfe_dead = rms_lfe < LFE_DEAD_DB
    if sig_lfe_dead:
        score += 1
        reasons.append(f"LFE is dead ({rms_lfe:.0f} dB)")

    center_dom = rms_c - max(metrics.rms_l, metrics.rms_r, rms_ls, rms_rs)
    sig_center_dom = center_dom > CENTER_DOM_DB
    if sig_center_dom:
        score += 1
        reasons.append(
            f"center is way louder than everything else ({center_dom:.0f} dB above)",
        )

    sig_fronts_mono = (
        metrics.corr_lr > MONO_CORR
        and abs(metrics.rms_l - metrics.rms_r) < MONO_RMS_DIFF_DB
    )
    if sig_fronts_mono:
        score += 1
        reasons.append(
            f"left and right fronts are identical (mono) — corr={metrics.corr_lr:.3f}",
        )

    sig_surrounds_copy = (
        corr_ls_l > SURROUNDS_COPY_CORR and corr_rs_r > SURROUNDS_COPY_CORR
    )
    if sig_surrounds_copy:
        score += 2
        reasons.append(
            "surrounds are a copy of fronts "
            f"(corr Ls~L={corr_ls_l:.2f}, Rs~R={corr_rs_r:.2f})",
        )

    sig_ls_rs_identical = corr_ls_rs > LS_RS_IDENT_CORR
    if sig_ls_rs_identical:
        score += 1
        reasons.append(
            "left and right surrounds carry the same signal "
            f"(corr={corr_ls_rs:.2f})",
        )

    if score >= FAKE_SCORE_THRESHOLD:
        verdict = Verdict.FAKE
        suggested: DownmixMode | None = (
            DownmixMode.MONO if sig_fronts_mono else DownmixMode.STEREO
        )
    elif score == SUSPICIOUS_SCORE:
        verdict = Verdict.SUSPICIOUS
        suggested = DownmixMode.MONO if sig_fronts_mono else DownmixMode.STEREO
    else:
        verdict = Verdict.REAL
        suggested = None

    # 7.1 back-surround hint — informational only, does not change verdict.
    if metrics.channels == _SURROUND_7_1_CHANNELS:
        rms_lb = cast("float", metrics.rms_lb)
        rms_rb = cast("float", metrics.rms_rb)
        corr_lb_ls = cast("float", metrics.corr_lb_ls)
        corr_rb_rs = cast("float", metrics.corr_rb_rs)
        back_silent = rms_lb < SURROUND_SILENT_DB and rms_rb < SURROUND_SILENT_DB
        back_copy = (
            corr_lb_ls > SURROUNDS_COPY_CORR and corr_rb_rs > SURROUNDS_COPY_CORR
        )
        if back_silent:
            reasons.append(
                "7.1 back surrounds are silent "
                f"(Lb={rms_lb:.0f}, Rb={rms_rb:.0f} dB)",
            )
        elif back_copy:
            reasons.append(
                "7.1 back surrounds are a copy of sides "
                f"(corr Lb~Ls={corr_lb_ls:.2f}, Rb~Rs={corr_rb_rs:.2f})",
            )

    return AudioProfile(
        verdict=verdict,
        score=score,
        suggested=suggested,
        reasons=tuple(reasons),
        metrics=metrics,
    )
