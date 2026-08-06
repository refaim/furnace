from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TypeGuard, cast

from .downmix import STEREO_CHANNELS, SURROUND_5_0_CHANNELS, THREE_CHANNELS, DownmixMode

_SURROUND_5_1_CHANNELS = 6
_SURROUND_7_1_CHANNELS = 8

_UNAMBIGUOUS_CHANNEL_COUNTS = frozenset({STEREO_CHANNELS, _SURROUND_5_1_CHANNELS, _SURROUND_7_1_CHANNELS})

LAYOUT_2_1 = "2.1"
LAYOUT_3_0 = "3.0"
THREE_CHANNEL_LAYOUTS = frozenset({LAYOUT_2_1, LAYOUT_3_0})

LAYOUT_5_0 = "5.0"
LAYOUT_5_0_SIDE = "5.0(side)"
FIVE_CHANNEL_LAYOUTS = frozenset({LAYOUT_5_0, LAYOUT_5_0_SIDE})

LAYOUT_SENSITIVE_CHANNELS = frozenset({THREE_CHANNELS, SURROUND_5_0_CHANNELS})


SURROUND_SILENT_DB = -50.0
LFE_DEAD_DB = -65.0
CENTER_SILENT_DB = -50.0
HARD_SILENCE_DB = -90.0
CENTER_COPY_CORR = 0.95
CENTER_DOM_DB = 10.0
MONO_CORR = 0.98
MONO_RMS_DIFF_DB = 2.0
SURROUNDS_COPY_CORR = 0.95
LS_RS_IDENT_CORR = 0.85

STEREO_SUSP_CORR = 0.96
STEREO_SUSP_DIFF_DB = 3.0

FAKE_SCORE_THRESHOLD = 2
SUSPICIOUS_SCORE = 1


class Verdict(enum.StrEnum):
    REAL = "real"
    SUSPICIOUS = "suspicious"
    FAKE = "fake"


@dataclass(frozen=True)
class AudioMetrics:
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
    corr_c_lr: float | None


@dataclass(frozen=True)
class AudioProfile:
    verdict: Verdict
    score: int
    suggested: DownmixMode | None
    reasons: tuple[str, ...]
    metrics: AudioMetrics


def is_profileable(channels: int | None, channel_layout: str | None) -> TypeGuard[int]:
    if channels in _UNAMBIGUOUS_CHANNEL_COUNTS:
        return True
    if channels == THREE_CHANNELS:
        return channel_layout in THREE_CHANNEL_LAYOUTS
    return channels == SURROUND_5_0_CHANNELS and channel_layout in FIVE_CHANNEL_LAYOUTS


def classify_audio(metrics: AudioMetrics) -> AudioProfile:
    if metrics.channels == STEREO_CHANNELS:
        return _classify_stereo(metrics)
    if metrics.channels == THREE_CHANNELS:
        if metrics.rms_lfe is not None:
            return _classify_two_one(metrics)
        if metrics.rms_c is not None:
            return _classify_three_zero(metrics)
        raise ValueError("three-channel metrics carry neither LFE nor center")
    if metrics.channels in (SURROUND_5_0_CHANNELS, _SURROUND_5_1_CHANNELS, _SURROUND_7_1_CHANNELS):
        return _classify_multichannel(metrics)
    raise ValueError(f"unsupported channels: {metrics.channels}")


def _front_pair_signals(metrics: AudioMetrics) -> tuple[int, list[str], bool]:
    corr = metrics.corr_lr
    diff = abs(metrics.rms_l - metrics.rms_r)

    if corr > MONO_CORR and diff < MONO_RMS_DIFF_DB:
        return 2, [f"left and right are identical (mono) — corr={corr:.3f}, diff={diff:.1f} dB"], True
    if corr > STEREO_SUSP_CORR and diff < STEREO_SUSP_DIFF_DB:
        return 1, [f"left and right are nearly identical — corr={corr:.3f}, diff={diff:.1f} dB"], True
    return 0, [], False


def _verdict_for(score: int) -> Verdict:
    if score >= FAKE_SCORE_THRESHOLD:
        return Verdict.FAKE
    if score == SUSPICIOUS_SCORE:
        return Verdict.SUSPICIOUS
    return Verdict.REAL


def _classify_two_one(metrics: AudioMetrics) -> AudioProfile:
    rms_lfe = cast("float", metrics.rms_lfe)
    score, reasons, fronts_mono = _front_pair_signals(metrics)

    if rms_lfe <= HARD_SILENCE_DB:
        score += 2
        reasons.append(f"LFE is dead ({rms_lfe:.0f} dB in the loudest window)")
    elif rms_lfe < LFE_DEAD_DB:
        score += 1
        reasons.append(f"LFE is barely there ({rms_lfe:.0f} dB in the loudest window)")

    verdict = _verdict_for(score)
    suggested: DownmixMode | None = None
    if verdict is not Verdict.REAL:
        suggested = DownmixMode.MONO if fronts_mono else DownmixMode.STEREO
    return AudioProfile(verdict=verdict, score=score, suggested=suggested, reasons=tuple(reasons), metrics=metrics)


def _classify_three_zero(metrics: AudioMetrics) -> AudioProfile:
    rms_c = cast("float", metrics.rms_c)
    score, reasons, fronts_mono = _front_pair_signals(metrics)

    if rms_c <= HARD_SILENCE_DB:
        score += 2
        reasons.append(f"center is silent ({rms_c:.0f} dB)")
    elif rms_c < CENTER_SILENT_DB:
        score += 1
        reasons.append(f"center is barely there ({rms_c:.0f} dB)")
    else:
        corr_c_lr = cast("float", metrics.corr_c_lr)
        if corr_c_lr > CENTER_COPY_CORR:
            score += 2
            reasons.append(f"center is a mix of the fronts (corr C~L+R={corr_c_lr:.2f})")
        center_dom = rms_c - max(metrics.rms_l, metrics.rms_r)
        if center_dom > CENTER_DOM_DB:
            score += 1
            reasons.append(f"center is way louder than the fronts ({center_dom:.0f} dB above)")

    verdict = _verdict_for(score)
    suggested: DownmixMode | None = None
    if verdict is not Verdict.REAL:
        suggested = DownmixMode.MONO if fronts_mono else DownmixMode.STEREO
    return AudioProfile(verdict=verdict, score=score, suggested=suggested, reasons=tuple(reasons), metrics=metrics)


def _classify_stereo(metrics: AudioMetrics) -> AudioProfile:
    score, reasons, fronts_mono = _front_pair_signals(metrics)
    verdict = _verdict_for(score)
    suggested = DownmixMode.MONO if fronts_mono else None
    return AudioProfile(verdict=verdict, score=score, suggested=suggested, reasons=tuple(reasons), metrics=metrics)


def _classify_multichannel(metrics: AudioMetrics) -> AudioProfile:
    rms_c = cast("float", metrics.rms_c)
    rms_lfe = metrics.rms_lfe
    if metrics.channels == SURROUND_5_0_CHANNELS:
        if rms_lfe is not None:
            raise ValueError("5-channel metrics carry an LFE")
    elif rms_lfe is None:
        raise ValueError(f"{metrics.channels}-channel metrics carry no LFE")
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

    if rms_lfe is not None and rms_lfe < LFE_DEAD_DB:
        score += 1
        reasons.append(f"LFE is dead ({rms_lfe:.0f} dB in the loudest window)")

    if metrics.corr_c_lr is None:
        raise ValueError(f"{metrics.channels}-channel metrics carry no center correlation")
    corr_c_lr = metrics.corr_c_lr
    if corr_c_lr > CENTER_COPY_CORR:
        score += 2
        reasons.append(f"center is a mix of the fronts (corr C~L+R={corr_c_lr:.2f})")

    center_dom = rms_c - max(metrics.rms_l, metrics.rms_r, rms_ls, rms_rs)
    sig_center_dom = center_dom > CENTER_DOM_DB
    if sig_center_dom:
        score += 1
        reasons.append(
            f"center is way louder than everything else ({center_dom:.0f} dB above)",
        )

    sig_fronts_mono = metrics.corr_lr > MONO_CORR and abs(metrics.rms_l - metrics.rms_r) < MONO_RMS_DIFF_DB
    if sig_fronts_mono:
        score += 1
        reasons.append(
            f"left and right fronts are identical (mono) — corr={metrics.corr_lr:.3f}",
        )

    sig_surrounds_copy = corr_ls_l > SURROUNDS_COPY_CORR and corr_rs_r > SURROUNDS_COPY_CORR
    if sig_surrounds_copy:
        score += 2
        reasons.append(
            f"surrounds are a copy of fronts (corr Ls~L={corr_ls_l:.2f}, Rs~R={corr_rs_r:.2f})",
        )

    sig_ls_rs_identical = corr_ls_rs > LS_RS_IDENT_CORR
    if sig_ls_rs_identical:
        score += 1
        reasons.append(
            f"left and right surrounds carry the same signal (corr={corr_ls_rs:.2f})",
        )

    if score >= FAKE_SCORE_THRESHOLD:
        verdict = Verdict.FAKE
        suggested: DownmixMode | None = DownmixMode.MONO if sig_fronts_mono else DownmixMode.STEREO
    elif score == SUSPICIOUS_SCORE:
        verdict = Verdict.SUSPICIOUS
        suggested = DownmixMode.MONO if sig_fronts_mono else DownmixMode.STEREO
    else:
        verdict = Verdict.REAL
        suggested = None

    if metrics.channels == _SURROUND_7_1_CHANNELS:
        rms_lb = cast("float", metrics.rms_lb)
        rms_rb = cast("float", metrics.rms_rb)
        corr_lb_ls = cast("float", metrics.corr_lb_ls)
        corr_rb_rs = cast("float", metrics.corr_rb_rs)
        back_silent = rms_lb < SURROUND_SILENT_DB and rms_rb < SURROUND_SILENT_DB
        back_copy = corr_lb_ls > SURROUNDS_COPY_CORR and corr_rb_rs > SURROUNDS_COPY_CORR
        if back_silent:
            reasons.append(
                f"7.1 back surrounds are silent (Lb={rms_lb:.0f}, Rb={rms_rb:.0f} dB)",
            )
        elif back_copy:
            reasons.append(
                f"7.1 back surrounds are a copy of sides (corr Lb~Ls={corr_lb_ls:.2f}, Rb~Rs={corr_rb_rs:.2f})",
            )

    return AudioProfile(
        verdict=verdict,
        score=score,
        suggested=suggested,
        reasons=tuple(reasons),
        metrics=metrics,
    )
