from __future__ import annotations

import dataclasses
import itertools

import pytest

from furnace.core.audio_profile import (
    AudioMetrics,
    AudioProfile,
    Verdict,
    classify_audio,
    is_profileable,
)
from furnace.core.downmix import DownmixMode


def _stereo_metrics(*, corr: float = 0.5, rms_l: float = -20.0, rms_r: float = -20.0) -> AudioMetrics:
    return AudioMetrics(
        channels=2,
        rms_l=rms_l,
        rms_r=rms_r,
        rms_c=None,
        rms_lfe=None,
        rms_ls=None,
        rms_rs=None,
        rms_lb=None,
        rms_rb=None,
        corr_lr=corr,
        corr_ls_l=None,
        corr_rs_r=None,
        corr_ls_rs=None,
        corr_lb_ls=None,
        corr_rb_rs=None,
    )


def _five_one_metrics(
    *,
    rms_l: float = -25.0,
    rms_r: float = -25.0,
    rms_c: float = -22.0,
    rms_lfe: float = -30.0,
    rms_ls: float = -28.0,
    rms_rs: float = -28.0,
    corr_lr: float = 0.2,
    corr_ls_l: float = 0.1,
    corr_rs_r: float = 0.1,
    corr_ls_rs: float = 0.2,
) -> AudioMetrics:
    return AudioMetrics(
        channels=6,
        rms_l=rms_l,
        rms_r=rms_r,
        rms_c=rms_c,
        rms_lfe=rms_lfe,
        rms_ls=rms_ls,
        rms_rs=rms_rs,
        rms_lb=None,
        rms_rb=None,
        corr_lr=corr_lr,
        corr_ls_l=corr_ls_l,
        corr_rs_r=corr_rs_r,
        corr_ls_rs=corr_ls_rs,
        corr_lb_ls=None,
        corr_rb_rs=None,
    )


def _five_zero_metrics(
    *,
    rms_l: float = -25.0,
    rms_r: float = -25.0,
    rms_c: float = -22.0,
    rms_ls: float = -28.0,
    rms_rs: float = -28.0,
    corr_lr: float = 0.2,
    corr_ls_l: float = 0.1,
    corr_rs_r: float = 0.1,
    corr_ls_rs: float = 0.2,
    corr_c_lr: float = 0.2,
) -> AudioMetrics:
    return AudioMetrics(
        channels=5,
        rms_l=rms_l,
        rms_r=rms_r,
        rms_c=rms_c,
        rms_lfe=None,
        rms_ls=rms_ls,
        rms_rs=rms_rs,
        rms_lb=None,
        rms_rb=None,
        corr_lr=corr_lr,
        corr_ls_l=corr_ls_l,
        corr_rs_r=corr_rs_r,
        corr_ls_rs=corr_ls_rs,
        corr_lb_ls=None,
        corr_rb_rs=None,
        corr_c_lr=corr_c_lr,
    )


def _two_one_metrics(
    *,
    rms_l: float = -25.0,
    rms_r: float = -25.0,
    rms_lfe: float = -35.0,
    corr_lr: float = 0.5,
) -> AudioMetrics:
    return AudioMetrics(
        channels=3,
        rms_l=rms_l,
        rms_r=rms_r,
        rms_c=None,
        rms_lfe=rms_lfe,
        rms_ls=None,
        rms_rs=None,
        rms_lb=None,
        rms_rb=None,
        corr_lr=corr_lr,
        corr_ls_l=None,
        corr_rs_r=None,
        corr_ls_rs=None,
        corr_lb_ls=None,
        corr_rb_rs=None,
    )


def _three_zero_metrics(
    *,
    rms_l: float = -25.0,
    rms_r: float = -25.0,
    rms_c: float = -24.0,
    corr_lr: float = 0.5,
    corr_c_lr: float | None = 0.2,
) -> AudioMetrics:
    return AudioMetrics(
        channels=3,
        rms_l=rms_l,
        rms_r=rms_r,
        rms_c=rms_c,
        rms_lfe=None,
        rms_ls=None,
        rms_rs=None,
        rms_lb=None,
        rms_rb=None,
        corr_lr=corr_lr,
        corr_ls_l=None,
        corr_rs_r=None,
        corr_ls_rs=None,
        corr_lb_ls=None,
        corr_rb_rs=None,
        corr_c_lr=corr_c_lr,
    )


class TestProfileableLayouts:
    @pytest.mark.parametrize(
        ("channels", "layout"),
        [
            (2, "stereo"),
            (6, "5.1"),
            (6, "5.1(side)"),
            (8, "7.1"),
            (3, "2.1"),
            (3, "3.0"),
            (5, "5.0"),
            (5, "5.0(side)"),
        ],
    )
    def test_supported(self, channels: int, layout: str) -> None:
        assert is_profileable(channels, layout) is True

    @pytest.mark.parametrize(
        ("channels", "layout"),
        [
            (1, "mono"),
            (4, "quad"),
            (7, "6.1"),
            (3, "3.0(back)"),
            (3, None),
            (5, "4.1"),
            (5, None),
        ],
    )
    def test_unsupported(self, channels: int, layout: str | None) -> None:
        assert is_profileable(channels, layout) is False

    def test_channel_count_wins_for_unambiguous_layouts(self) -> None:
        assert is_profileable(2, None) is True
        assert is_profileable(6, None) is True


class TestFrontPairThresholds:
    def test_mono_boundary_is_shared_by_stereo_and_surround(self) -> None:
        assert classify_audio(_stereo_metrics(corr=0.980001)).verdict == Verdict.FAKE
        assert classify_audio(_stereo_metrics(corr=0.98)).verdict == Verdict.SUSPICIOUS
        five_one = _five_one_metrics(corr_lr=0.980001, rms_lfe=-30.0)
        assert any("fronts are identical" in r for r in classify_audio(five_one).reasons)
        five_one_below = _five_one_metrics(corr_lr=0.98, rms_lfe=-30.0)
        assert not any("fronts are identical" in r for r in classify_audio(five_one_below).reasons)

    def test_near_mono_boundary(self) -> None:
        assert classify_audio(_stereo_metrics(corr=0.960001)).verdict == Verdict.SUSPICIOUS
        assert classify_audio(_stereo_metrics(corr=0.96)).verdict == Verdict.REAL

    def test_level_difference_boundary(self) -> None:
        assert classify_audio(_stereo_metrics(corr=0.99, rms_r=-21.999)).verdict == Verdict.FAKE
        assert classify_audio(_stereo_metrics(corr=0.99, rms_r=-22.0)).verdict == Verdict.SUSPICIOUS
        assert classify_audio(_stereo_metrics(corr=0.99, rms_r=-23.0)).verdict == Verdict.REAL


class TestClassifyTwoOne:
    def test_live_lfe_and_real_stereo_is_real(self) -> None:
        p = classify_audio(_two_one_metrics())
        assert p.verdict == Verdict.REAL
        assert p.score == 0
        assert p.suggested is None

    def test_dead_lfe_alone_is_fake(self) -> None:
        p = classify_audio(_two_one_metrics(rms_lfe=-120.0))
        assert p.verdict == Verdict.FAKE
        assert p.suggested == DownmixMode.STEREO
        assert any("LFE is dead" in r for r in p.reasons)

    def test_a_quiet_but_present_lfe_is_only_suspicious(self) -> None:
        p = classify_audio(_two_one_metrics(rms_lfe=-70.0))
        assert p.verdict == Verdict.SUSPICIOUS
        assert p.score == 1
        assert any("barely there" in r for r in p.reasons)

    def test_digital_silence_boundary_is_decisive(self) -> None:
        assert classify_audio(_two_one_metrics(rms_lfe=-90.0)).verdict == Verdict.FAKE
        assert classify_audio(_two_one_metrics(rms_lfe=-89.9)).verdict == Verdict.SUSPICIOUS

    def test_dead_lfe_with_mono_fronts_suggests_mono(self) -> None:
        p = classify_audio(_two_one_metrics(rms_lfe=-120.0, corr_lr=0.999))
        assert p.verdict == Verdict.FAKE
        assert p.suggested == DownmixMode.MONO
        assert any("mono" in r for r in p.reasons)

    def test_mono_fronts_with_live_lfe_is_fake(self) -> None:
        p = classify_audio(_two_one_metrics(corr_lr=0.999))
        assert p.verdict == Verdict.FAKE
        assert p.suggested == DownmixMode.MONO

    def test_near_mono_fronts_with_live_lfe_is_suspicious(self) -> None:
        p = classify_audio(_two_one_metrics(corr_lr=0.97, rms_r=-27.0))
        assert p.verdict == Verdict.SUSPICIOUS
        assert p.suggested == DownmixMode.MONO

    def test_lfe_just_above_the_dead_threshold_is_real(self) -> None:
        p = classify_audio(_two_one_metrics(rms_lfe=-64.0))
        assert p.verdict == Verdict.REAL

    def test_the_real_dvd_case(self) -> None:
        p = classify_audio(_two_one_metrics(rms_l=-29.6, rms_r=-30.4, rms_lfe=-120.0, corr_lr=0.765))
        assert p.verdict == Verdict.FAKE
        assert p.suggested == DownmixMode.STEREO
        assert p.reasons == ("LFE is dead (-120 dB in the loudest window)",)


class TestClassifyThreeZero:
    def test_live_center_and_real_stereo_is_real(self) -> None:
        p = classify_audio(_three_zero_metrics())
        assert p.verdict == Verdict.REAL
        assert p.score == 0

    def test_silent_center_is_fake(self) -> None:
        p = classify_audio(_three_zero_metrics(rms_c=-120.0))
        assert p.verdict == Verdict.FAKE
        assert p.suggested == DownmixMode.STEREO
        assert any("center is silent" in r for r in p.reasons)

    def test_a_quiet_but_present_center_is_only_suspicious(self) -> None:
        p = classify_audio(_three_zero_metrics(rms_c=-60.0))
        assert p.verdict == Verdict.SUSPICIOUS
        assert any("barely there" in r for r in p.reasons)

    def test_a_center_derived_from_the_fronts_is_fake(self) -> None:
        p = classify_audio(_three_zero_metrics(corr_c_lr=0.99))
        assert p.verdict == Verdict.FAKE
        assert p.suggested == DownmixMode.STEREO
        assert any("mix of the fronts" in r for r in p.reasons)

    def test_a_center_correlation_below_the_threshold_is_real(self) -> None:
        assert classify_audio(_three_zero_metrics(corr_c_lr=0.95)).verdict == Verdict.REAL
        assert classify_audio(_three_zero_metrics(corr_c_lr=0.951)).verdict == Verdict.FAKE

    def test_a_missing_center_correlation_fails_loudly(self) -> None:
        with pytest.raises(TypeError):
            classify_audio(_three_zero_metrics(corr_c_lr=None))

    def test_dominant_center_is_suspicious(self) -> None:
        p = classify_audio(_three_zero_metrics(rms_c=-10.0))
        assert p.verdict == Verdict.SUSPICIOUS
        assert p.suggested == DownmixMode.STEREO
        assert any("center is way louder" in r for r in p.reasons)

    def test_mono_fronts_with_dominant_center_is_fake(self) -> None:
        p = classify_audio(_three_zero_metrics(rms_c=-10.0, corr_lr=0.999))
        assert p.verdict == Verdict.FAKE
        assert p.suggested == DownmixMode.MONO


class TestClassifyFiveZero:
    def test_a_real_5_0_mix_is_real(self) -> None:
        p = classify_audio(_five_zero_metrics())
        assert p.verdict == Verdict.REAL
        assert p.score == 0
        assert p.suggested is None
        assert p.reasons == ()

    def test_a_missing_lfe_is_never_held_against_a_5_0_track(self) -> None:
        p = classify_audio(_five_zero_metrics())
        assert not any("LFE" in r for r in p.reasons)

    def test_silent_surrounds_alone_is_suspicious(self) -> None:
        p = classify_audio(_five_zero_metrics(rms_ls=-55.0, rms_rs=-56.0))
        assert p.verdict == Verdict.SUSPICIOUS
        assert p.score == 1
        assert p.suggested == DownmixMode.STEREO
        assert any("both surrounds are silent" in r for r in p.reasons)

    def test_surrounds_copied_from_the_fronts_is_fake(self) -> None:
        p = classify_audio(_five_zero_metrics(corr_ls_l=0.99, corr_rs_r=0.97))
        assert p.verdict == Verdict.FAKE
        assert p.score == 2
        assert p.suggested == DownmixMode.STEREO
        assert any("copy of fronts" in r for r in p.reasons)

    def test_a_dominant_center_alone_is_suspicious(self) -> None:
        p = classify_audio(_five_zero_metrics(rms_c=-10.0))
        assert p.verdict == Verdict.SUSPICIOUS
        assert p.score == 1
        assert any("way louder" in r for r in p.reasons)

    def test_identical_surrounds_alone_is_suspicious(self) -> None:
        p = classify_audio(_five_zero_metrics(corr_ls_rs=0.9))
        assert p.verdict == Verdict.SUSPICIOUS
        assert p.score == 1
        assert any("same signal" in r for r in p.reasons)

    def test_mono_fronts_suggest_a_mono_downmix(self) -> None:
        p = classify_audio(_five_zero_metrics(corr_lr=0.99, rms_l=-25.0, rms_r=-25.5, rms_ls=-70.0, rms_rs=-70.0))
        assert p.verdict == Verdict.FAKE
        assert p.suggested == DownmixMode.MONO
        assert any("identical (mono)" in r for r in p.reasons)

    def test_a_center_derived_from_the_fronts_is_fake(self) -> None:
        p = classify_audio(_five_zero_metrics(corr_c_lr=0.99))
        assert p.verdict == Verdict.FAKE
        assert p.score == 2
        assert p.suggested == DownmixMode.STEREO
        assert any("mix of the fronts" in r for r in p.reasons)

    def test_a_center_correlation_below_the_threshold_is_real(self) -> None:
        p = classify_audio(_five_zero_metrics(corr_c_lr=0.94))
        assert p.verdict == Verdict.REAL
        assert p.score == 0

    def test_a_center_correlation_exactly_at_the_threshold_is_real(self) -> None:
        p = classify_audio(_five_zero_metrics(corr_c_lr=0.95))
        assert p.verdict == Verdict.REAL
        assert p.score == 0

    def test_a_center_correlation_just_above_the_threshold_is_fake(self) -> None:
        p = classify_audio(_five_zero_metrics(corr_c_lr=0.951))
        assert p.verdict == Verdict.FAKE
        assert p.score == 2

    def test_a_missing_center_correlation_is_not_held_against_the_track(self) -> None:
        p = classify_audio(dataclasses.replace(_five_zero_metrics(), corr_c_lr=None))
        assert p.verdict == Verdict.REAL
        assert p.reasons == ()

    def test_the_real_dvd_case(self) -> None:
        p = classify_audio(
            _five_zero_metrics(
                rms_l=-31.6,
                rms_r=-31.9,
                rms_c=-26.6,
                rms_ls=-29.6,
                rms_rs=-29.8,
                corr_lr=0.759,
                corr_ls_l=0.017,
                corr_rs_r=0.013,
                corr_ls_rs=-0.647,
                corr_c_lr=0.036,
            ),
        )
        assert p.verdict == Verdict.REAL
        assert p.reasons == ()

    def test_metrics_that_carry_an_lfe_are_rejected(self) -> None:
        with_lfe = dataclasses.replace(_five_zero_metrics(), rms_lfe=-30.0)
        with pytest.raises(ValueError, match="5-channel metrics carry an LFE"):
            classify_audio(with_lfe)


def test_classify_rejects_surround_metrics_without_an_lfe() -> None:
    lfe_less = dataclasses.replace(_five_one_metrics(), rms_lfe=None)
    with pytest.raises(ValueError, match="6-channel metrics carry no LFE"):
        classify_audio(lfe_less)


def test_classify_rejects_three_channels_without_lfe_or_center() -> None:
    metrics = AudioMetrics(
        channels=3,
        rms_l=-25.0,
        rms_r=-25.0,
        rms_c=None,
        rms_lfe=None,
        rms_ls=None,
        rms_rs=None,
        rms_lb=None,
        rms_rb=None,
        corr_lr=0.5,
        corr_ls_l=None,
        corr_rs_r=None,
        corr_ls_rs=None,
        corr_lb_ls=None,
        corr_rb_rs=None,
    )
    with pytest.raises(ValueError, match="three-channel metrics carry neither LFE nor center"):
        classify_audio(metrics)


def test_audio_metrics_is_frozen() -> None:
    m = _stereo_metrics()
    with pytest.raises((AttributeError, TypeError)):
        m.channels = 6  # type: ignore[misc]


def test_verdict_values() -> None:
    assert Verdict.REAL.value == "real"
    assert Verdict.SUSPICIOUS.value == "suspicious"
    assert Verdict.FAKE.value == "fake"


def test_classify_real_5_1_returns_real() -> None:
    m = _five_one_metrics(
        rms_l=-21.8,
        rms_r=-22.1,
        rms_c=-18.4,
        rms_lfe=-21.0,
        rms_ls=-25.7,
        rms_rs=-26.2,
        corr_lr=0.332,
        corr_ls_l=0.12,
        corr_rs_r=0.042,
        corr_ls_rs=0.401,
    )
    profile = classify_audio(m)
    assert profile.verdict == Verdict.REAL
    assert profile.score == 0
    assert profile.suggested is None
    assert profile.reasons == ()


def test_classify_stereo_mono_is_fake() -> None:
    m = _stereo_metrics(corr=1.0, rms_l=-22.5, rms_r=-22.5)
    p = classify_audio(m)
    assert p.verdict == Verdict.FAKE
    assert p.score == 2
    assert p.suggested == DownmixMode.MONO
    assert any("identical" in r for r in p.reasons)


def test_classify_stereo_near_mono_is_suspicious() -> None:
    m = _stereo_metrics(corr=0.97, rms_l=-25.0, rms_r=-24.0)
    p = classify_audio(m)
    assert p.verdict == Verdict.SUSPICIOUS
    assert p.score == 1
    assert p.suggested == DownmixMode.MONO
    assert any("nearly identical" in r for r in p.reasons)


def test_classify_stereo_real_stays_real() -> None:
    m = _stereo_metrics(corr=0.75, rms_l=-25.4, rms_r=-28.5)
    p = classify_audio(m)
    assert p.verdict == Verdict.REAL
    assert p.score == 0
    assert p.suggested is None
    assert p.reasons == ()


def test_classify_stereo_high_corr_large_diff_stays_real() -> None:
    m = _stereo_metrics(corr=0.99, rms_l=-20.0, rms_r=-24.0)
    p = classify_audio(m)
    assert p.verdict == Verdict.REAL
    assert p.score == 0


def test_classify_stereo_exactly_at_mono_threshold_is_not_fake() -> None:
    m = _stereo_metrics(corr=0.98, rms_l=-20.0, rms_r=-22.0)
    p = classify_audio(m)
    assert p.verdict == Verdict.SUSPICIOUS
    assert p.score == 1


def test_classify_stereo_exactly_at_susp_threshold_is_real() -> None:
    m = _stereo_metrics(corr=0.96, rms_l=-20.0, rms_r=-22.0)
    p = classify_audio(m)
    assert p.verdict == Verdict.REAL
    assert p.score == 0


def test_classify_unsupported_channels_raises() -> None:
    m = AudioMetrics(
        channels=4,
        rms_l=-20.0,
        rms_r=-20.0,
        rms_c=None,
        rms_lfe=None,
        rms_ls=None,
        rms_rs=None,
        rms_lb=None,
        rms_rb=None,
        corr_lr=0.5,
        corr_ls_l=None,
        corr_rs_r=None,
        corr_ls_rs=None,
        corr_lb_ls=None,
        corr_rb_rs=None,
    )
    with pytest.raises(ValueError, match="unsupported channels: 4"):
        classify_audio(m)


def test_audio_profile_holds_fields() -> None:
    m = _stereo_metrics()
    profile = AudioProfile(
        verdict=Verdict.FAKE,
        score=3,
        suggested=DownmixMode.STEREO,
        reasons=("test-reason",),
        metrics=m,
    )
    assert profile.verdict is Verdict.FAKE
    assert profile.score == 3
    assert profile.suggested is DownmixMode.STEREO
    assert profile.reasons == ("test-reason",)
    assert profile.metrics is m


def _seven_one_metrics(
    *,
    rms_l: float = -25.0,
    rms_r: float = -25.0,
    rms_c: float = -22.0,
    rms_lfe: float = -30.0,
    rms_ls: float = -28.0,
    rms_rs: float = -28.0,
    rms_lb: float = -28.0,
    rms_rb: float = -28.0,
    corr_lr: float = 0.2,
    corr_ls_l: float = 0.1,
    corr_rs_r: float = 0.1,
    corr_ls_rs: float = 0.2,
    corr_lb_ls: float = 0.2,
    corr_rb_rs: float = 0.2,
) -> AudioMetrics:
    return AudioMetrics(
        channels=8,
        rms_l=rms_l,
        rms_r=rms_r,
        rms_c=rms_c,
        rms_lfe=rms_lfe,
        rms_ls=rms_ls,
        rms_rs=rms_rs,
        rms_lb=rms_lb,
        rms_rb=rms_rb,
        corr_lr=corr_lr,
        corr_ls_l=corr_ls_l,
        corr_rs_r=corr_rs_r,
        corr_ls_rs=corr_ls_rs,
        corr_lb_ls=corr_lb_ls,
        corr_rb_rs=corr_rb_rs,
    )


def test_silent_surrounds_alone_is_suspicious() -> None:
    m = _five_one_metrics(rms_ls=-55.0, rms_rs=-56.0)
    p = classify_audio(m)
    assert p.verdict == Verdict.SUSPICIOUS
    assert p.score == 1
    assert any("surrounds are silent" in r for r in p.reasons)


def test_lfe_dead_alone_is_suspicious() -> None:
    m = _five_one_metrics(rms_lfe=-80.0)
    p = classify_audio(m)
    assert p.verdict == Verdict.SUSPICIOUS
    assert p.score == 1
    assert any("LFE is dead" in r for r in p.reasons)


def test_center_dominates_alone_is_suspicious() -> None:
    m = _five_one_metrics(
        rms_l=-40.0,
        rms_r=-40.0,
        rms_c=-22.0,
        rms_ls=-42.0,
        rms_rs=-42.0,
    )
    p = classify_audio(m)
    assert p.verdict == Verdict.SUSPICIOUS
    assert p.score == 1
    assert any("center is way louder" in r for r in p.reasons)


def test_ls_rs_identical_alone_is_suspicious() -> None:
    m = _five_one_metrics(corr_ls_rs=0.90)
    p = classify_audio(m)
    assert p.verdict == Verdict.SUSPICIOUS
    assert p.score == 1
    assert any("carry the same signal" in r for r in p.reasons)


def test_fronts_mono_alone_is_suspicious_and_suggests_mono() -> None:
    m = _five_one_metrics(
        rms_l=-25.0,
        rms_r=-25.0,
        corr_lr=0.99,
    )
    p = classify_audio(m)
    assert p.verdict == Verdict.SUSPICIOUS
    assert p.score == 1
    assert p.suggested == DownmixMode.MONO


def test_surrounds_copy_alone_is_fake() -> None:
    m = _five_one_metrics(corr_ls_l=0.98, corr_rs_r=0.97)
    p = classify_audio(m)
    assert p.verdict == Verdict.FAKE
    assert p.score == 2
    assert p.suggested == DownmixMode.STEREO
    assert any("copy of fronts" in r for r in p.reasons)


def test_two_signals_is_fake() -> None:
    m = _five_one_metrics(
        rms_ls=-55.0,
        rms_rs=-55.0,
        rms_lfe=-85.0,
    )
    p = classify_audio(m)
    assert p.verdict == Verdict.FAKE
    assert p.score == 2
    assert p.suggested == DownmixMode.STEREO


def test_fake_with_fronts_mono_suggests_mono() -> None:
    m = _five_one_metrics(
        rms_l=-25.0,
        rms_r=-25.0,
        corr_lr=0.99,
        rms_lfe=-80.0,
    )
    p = classify_audio(m)
    assert p.verdict == Verdict.FAKE
    assert p.score == 2
    assert p.suggested == DownmixMode.MONO


def test_pirates_1_quiet_mix_is_suspicious_not_fake() -> None:
    m = _five_one_metrics(
        rms_l=-38.2,
        rms_r=-38.0,
        rms_c=-29.0,
        rms_lfe=-59.8,
        rms_ls=-50.5,
        rms_rs=-52.2,
        corr_lr=0.708,
    )
    p = classify_audio(m)
    assert p.verdict == Verdict.SUSPICIOUS


def test_seven_one_back_surrounds_silent_adds_hint() -> None:
    m = _seven_one_metrics(rms_lb=-60.0, rms_rb=-58.0)
    p = classify_audio(m)
    assert p.verdict == Verdict.REAL
    assert p.score == 0
    assert any("7.1 back surrounds are silent" in r for r in p.reasons)


def test_seven_one_back_surrounds_copy_adds_hint() -> None:
    m = _seven_one_metrics(
        rms_lb=-28.0,
        rms_rb=-28.0,
        corr_lb_ls=0.97,
        corr_rb_rs=0.96,
    )
    p = classify_audio(m)
    assert p.verdict == Verdict.REAL
    assert p.score == 0
    assert any("7.1 back surrounds are a copy of sides" in r for r in p.reasons)


def test_seven_one_healthy_back_surrounds_add_no_hint() -> None:
    m = _seven_one_metrics()
    p = classify_audio(m)
    assert p.verdict == Verdict.REAL
    assert p.score == 0
    assert not any("7.1 back surrounds" in r for r in p.reasons)


_SIGNAL_NAMES = (
    "silent_surrounds",
    "lfe_dead",
    "center_dom",
    "fronts_mono",
    "surrounds_copy",
    "ls_rs_ident",
)
_SIGNAL_POINTS = {
    "silent_surrounds": 1,
    "lfe_dead": 1,
    "center_dom": 1,
    "fronts_mono": 1,
    "surrounds_copy": 2,
    "ls_rs_ident": 1,
    "center_derived": 2,
}


def _build_metrics_from_signals(active: frozenset[str], channels: int = 6) -> AudioMetrics:
    if channels == 5 and "lfe_dead" in active:
        raise ValueError("a 5.0 track has no LFE to kill")
    if channels != 5 and "center_derived" in active:
        raise ValueError("only 5.0 metrics carry a center correlation")
    rms_l = -25.0
    rms_r = -26.0
    rms_c = -23.0
    rms_lfe = -30.0
    rms_ls = -28.0
    rms_rs = -28.0
    corr_lr = 0.3
    corr_ls_l = 0.1
    corr_rs_r = 0.1
    corr_ls_rs = 0.2

    if "silent_surrounds" in active:
        rms_ls = -70.0
        rms_rs = -70.0
    if "lfe_dead" in active:
        rms_lfe = -80.0
    if "center_dom" in active:
        rms_c = -5.0
    if "fronts_mono" in active:
        corr_lr = 0.999
        rms_l = -20.0
        rms_r = -20.0
    if "surrounds_copy" in active:
        corr_ls_l = 0.99
        corr_rs_r = 0.99
    if "ls_rs_ident" in active:
        corr_ls_rs = 0.95

    return AudioMetrics(
        channels=channels,
        rms_l=rms_l,
        rms_r=rms_r,
        rms_c=rms_c,
        rms_lfe=rms_lfe if channels != 5 else None,
        rms_ls=rms_ls,
        rms_rs=rms_rs,
        rms_lb=None,
        rms_rb=None,
        corr_lr=corr_lr,
        corr_ls_l=corr_ls_l,
        corr_rs_r=corr_rs_r,
        corr_ls_rs=corr_ls_rs,
        corr_lb_ls=None,
        corr_rb_rs=None,
        corr_c_lr=0.99 if "center_derived" in active else None,
    )


def _expected_score(active: frozenset[str]) -> int:
    return sum(_SIGNAL_POINTS[s] for s in active)


def _expected_verdict(score: int) -> Verdict:
    if score >= 2:
        return Verdict.FAKE
    if score == 1:
        return Verdict.SUSPICIOUS
    return Verdict.REAL


_ALL_COMBINATIONS = [
    frozenset(combo) for n in range(len(_SIGNAL_NAMES) + 1) for combo in itertools.combinations(_SIGNAL_NAMES, n)
]


@pytest.mark.parametrize("active", _ALL_COMBINATIONS)
def test_classify_audio_combinatorial(active: frozenset[str]) -> None:
    m = _build_metrics_from_signals(active)
    p = classify_audio(m)

    expected_score = _expected_score(active)
    expected_verdict = _expected_verdict(expected_score)

    assert p.score == expected_score, (
        f"signals={sorted(active)}: expected score {expected_score}, got {p.score} (reasons: {list(p.reasons)})"
    )
    assert p.verdict == expected_verdict, f"signals={sorted(active)}: expected {expected_verdict}, got {p.verdict}"

    if expected_verdict == Verdict.REAL:
        assert p.suggested is None, (
            f"signals={sorted(active)}: REAL verdict must not suggest a downmix, got {p.suggested}"
        )
    elif "fronts_mono" in active:
        assert p.suggested == DownmixMode.MONO, (
            f"signals={sorted(active)}: fronts_mono active → expected MONO, got {p.suggested}"
        )
    else:
        assert p.suggested == DownmixMode.STEREO, (
            f"signals={sorted(active)}: no fronts_mono → expected STEREO, got {p.suggested}"
        )


_FIVE_ZERO_SIGNAL_NAMES = (*(s for s in _SIGNAL_NAMES if s != "lfe_dead"), "center_derived")
_FIVE_ZERO_COMBINATIONS = [
    frozenset(combo)
    for n in range(len(_FIVE_ZERO_SIGNAL_NAMES) + 1)
    for combo in itertools.combinations(_FIVE_ZERO_SIGNAL_NAMES, n)
]


@pytest.mark.parametrize("active", _FIVE_ZERO_COMBINATIONS)
def test_classify_five_zero_combinatorial(active: frozenset[str]) -> None:
    m = _build_metrics_from_signals(active, channels=5)
    p = classify_audio(m)

    expected_score = _expected_score(active)
    expected_verdict = _expected_verdict(expected_score)
    assert p.score == expected_score, (
        f"signals={sorted(active)}: expected score {expected_score}, got {p.score} (reasons: {list(p.reasons)})"
    )
    assert p.verdict == expected_verdict
    assert not any("LFE" in r for r in p.reasons), f"signals={sorted(active)}: a 5.0 track has no LFE to judge"

    if expected_verdict == Verdict.REAL:
        assert p.suggested is None, f"signals={sorted(active)}: REAL must not suggest a downmix, got {p.suggested}"
    elif "fronts_mono" in active:
        assert p.suggested == DownmixMode.MONO, f"signals={sorted(active)}: expected MONO, got {p.suggested}"
    else:
        assert p.suggested == DownmixMode.STEREO, f"signals={sorted(active)}: expected STEREO, got {p.suggested}"


def test_the_five_zero_signal_builder_rejects_a_dead_lfe() -> None:
    with pytest.raises(ValueError, match="no LFE to kill"):
        _build_metrics_from_signals(frozenset({"lfe_dead"}), channels=5)


def test_the_surround_signal_builder_rejects_a_center_correlation() -> None:
    with pytest.raises(ValueError, match=r"only 5\.0 metrics carry a center correlation"):
        _build_metrics_from_signals(frozenset({"center_derived"}))


_STEREO_GRID = [
    (1.000, 0.0, Verdict.FAKE),
    (0.998, 0.5, Verdict.FAKE),
    (0.990, 1.9, Verdict.FAKE),
    (0.980, 1.9, Verdict.SUSPICIOUS),
    (0.970, 2.5, Verdict.SUSPICIOUS),
    (0.965, 0.0, Verdict.SUSPICIOUS),
    (0.960, 2.9, Verdict.REAL),
    (0.950, 0.5, Verdict.REAL),
    (0.500, 0.0, Verdict.REAL),
    (0.999, 2.5, Verdict.SUSPICIOUS),
    (0.999, 3.5, Verdict.REAL),
]


@pytest.mark.parametrize(("corr", "diff_db", "expected"), _STEREO_GRID)
def test_stereo_grid(corr: float, diff_db: float, expected: Verdict) -> None:
    m = _stereo_metrics(corr=corr, rms_l=-20.0, rms_r=-20.0 - diff_db)
    p = classify_audio(m)
    assert p.verdict == expected, f"corr={corr}, diff={diff_db}: expected {expected.value}, got {p.verdict.value}"


def test_classify_audio_handles_digital_silence_clamp() -> None:
    m = _five_one_metrics(
        rms_l=-120.0,
        rms_r=-120.0,
        rms_c=-120.0,
        rms_lfe=-120.0,
        rms_ls=-120.0,
        rms_rs=-120.0,
        corr_lr=0.0,
        corr_ls_l=0.0,
        corr_rs_r=0.0,
        corr_ls_rs=0.0,
    )
    p = classify_audio(m)
    assert p.verdict == Verdict.FAKE
    assert p.score >= 2


GOLDEN_MULTI_CHANNEL: list[tuple[str, dict[str, float], Verdict]] = [
    (
        "Avengers AAC eng",
        {
            "rms_l": -21.8,
            "rms_r": -22.1,
            "rms_c": -18.4,
            "rms_lfe": -21.0,
            "rms_ls": -25.7,
            "rms_rs": -26.2,
            "corr_lr": 0.332,
            "corr_ls_l": 0.12,
            "corr_rs_r": 0.042,
            "corr_ls_rs": 0.401,
        },
        Verdict.REAL,
    ),
    (
        "Kingsman 2 Golden Circle AAC eng",
        {
            "rms_l": -22.3,
            "rms_r": -22.3,
            "rms_c": -20.1,
            "rms_lfe": -40.2,
            "rms_ls": -23.7,
            "rms_rs": -23.9,
            "corr_lr": 0.49,
            "corr_ls_l": 0.464,
            "corr_rs_r": 0.454,
            "corr_ls_rs": 0.53,
        },
        Verdict.REAL,
    ),
    (
        "MI 5 Rogue Nation AAC eng",
        {
            "rms_l": -29.4,
            "rms_r": -27.5,
            "rms_c": -24.6,
            "rms_lfe": -42.6,
            "rms_ls": -31.1,
            "rms_rs": -30.1,
            "corr_lr": 0.108,
            "corr_ls_l": 0.095,
            "corr_rs_r": 0.027,
            "corr_ls_rs": 0.064,
        },
        Verdict.REAL,
    ),
    (
        "Pirates 5 Dead Men AAC eng",
        {
            "rms_l": -21.2,
            "rms_r": -23.0,
            "rms_c": -18.0,
            "rms_lfe": -20.8,
            "rms_ls": -26.8,
            "rms_rs": -24.0,
            "corr_lr": 0.206,
            "corr_ls_l": 0.108,
            "corr_rs_r": 0.212,
            "corr_ls_rs": 0.271,
        },
        Verdict.REAL,
    ),
    (
        "Pirates 1 Black Pearl AAC eng",
        {
            "rms_l": -38.2,
            "rms_r": -38.0,
            "rms_c": -29.0,
            "rms_lfe": -59.8,
            "rms_ls": -50.5,
            "rms_rs": -52.2,
            "corr_lr": 0.708,
            "corr_ls_l": 0.136,
            "corr_rs_r": 0.125,
            "corr_ls_rs": 0.233,
        },
        Verdict.SUSPICIOUS,
    ),
    (
        "Pirates 1 Black Pearl AC3 rus",
        {
            "rms_l": -40.7,
            "rms_r": -40.5,
            "rms_c": -32.5,
            "rms_lfe": -62.4,
            "rms_ls": -55.1,
            "rms_rs": -55.3,
            "corr_lr": 0.644,
            "corr_ls_l": 0.029,
            "corr_rs_r": 0.002,
            "corr_ls_rs": 0.29,
        },
        Verdict.SUSPICIOUS,
    ),
    (
        "Gulyai Vasya DTS-HD MA",
        {
            "rms_l": -42.6,
            "rms_r": -42.6,
            "rms_c": -21.6,
            "rms_lfe": -85.5,
            "rms_ls": -55.3,
            "rms_rs": -55.9,
            "corr_lr": 0.402,
            "corr_ls_l": 0.203,
            "corr_rs_r": 0.332,
            "corr_ls_rs": 0.381,
        },
        Verdict.FAKE,
    ),
    (
        "Neadekvatnie TrueHD",
        {
            "rms_l": -48.3,
            "rms_r": -47.6,
            "rms_c": -32.0,
            "rms_lfe": -74.5,
            "rms_ls": -54.8,
            "rms_rs": -54.7,
            "corr_lr": 0.546,
            "corr_ls_l": 0.662,
            "corr_rs_r": 0.604,
            "corr_ls_rs": 0.619,
        },
        Verdict.FAKE,
    ),
    (
        "PRIHODI DTS",
        {
            "rms_l": -49.8,
            "rms_r": -47.5,
            "rms_c": -26.4,
            "rms_lfe": -74.9,
            "rms_ls": -46.7,
            "rms_rs": -49.0,
            "corr_lr": 0.394,
            "corr_ls_l": 0.173,
            "corr_rs_r": 0.17,
            "corr_ls_rs": 0.163,
        },
        Verdict.FAKE,
    ),
    (
        "33 (1965)",
        {
            "rms_l": -40.6,
            "rms_r": -39.0,
            "rms_c": -27.9,
            "rms_lfe": -82.5,
            "rms_ls": -45.4,
            "rms_rs": -43.9,
            "corr_lr": 0.278,
            "corr_ls_l": 0.145,
            "corr_rs_r": 0.15,
            "corr_ls_rs": 0.255,
        },
        Verdict.FAKE,
    ),
    (
        "Viy (1967)",
        {
            "rms_l": -47.6,
            "rms_r": -46.1,
            "rms_c": -28.8,
            "rms_lfe": -73.8,
            "rms_ls": -50.8,
            "rms_rs": -51.2,
            "corr_lr": 0.352,
            "corr_ls_l": -0.022,
            "corr_rs_r": 0.136,
            "corr_ls_rs": -0.165,
        },
        Verdict.FAKE,
    ),
    (
        "Ne mozhet byt (1975)",
        {
            "rms_l": -35.2,
            "rms_r": -33.5,
            "rms_c": -26.2,
            "rms_lfe": -74.9,
            "rms_ls": -38.7,
            "rms_rs": -38.9,
            "corr_lr": 0.275,
            "corr_ls_l": -0.065,
            "corr_rs_r": 0.098,
            "corr_ls_rs": 0.949,
        },
        Verdict.FAKE,
    ),
    (
        "Chapaev (1934)",
        {
            "rms_l": -33.3,
            "rms_r": -34.1,
            "rms_c": -34.5,
            "rms_lfe": -120.0,
            "rms_ls": -69.4,
            "rms_rs": -68.1,
            "corr_lr": -0.013,
            "corr_ls_l": 0.001,
            "corr_rs_r": 0.002,
            "corr_ls_rs": 0.493,
        },
        Verdict.FAKE,
    ),
    (
        "DMB 5.1",
        {
            "rms_l": -45.1,
            "rms_r": -41.5,
            "rms_c": -30.0,
            "rms_lfe": -55.3,
            "rms_ls": -50.6,
            "rms_rs": -47.9,
            "corr_lr": -0.059,
            "corr_ls_l": -0.053,
            "corr_rs_r": -0.039,
            "corr_ls_rs": 0.165,
        },
        Verdict.SUSPICIOUS,
    ),
    (
        "Osenniy marafon (1979)",
        {
            "rms_l": -44.0,
            "rms_r": -42.5,
            "rms_c": -40.6,
            "rms_lfe": -59.7,
            "rms_ls": -50.8,
            "rms_rs": -49.7,
            "corr_lr": 0.956,
            "corr_ls_l": -0.008,
            "corr_rs_r": -0.003,
            "corr_ls_rs": 0.863,
        },
        Verdict.SUSPICIOUS,
    ),
    (
        "Now You See Me 1 AAC eng",
        {
            "rms_l": -36.1,
            "rms_r": -38.9,
            "rms_c": -25.5,
            "rms_lfe": -54.3,
            "rms_ls": -42.4,
            "rms_rs": -42.6,
            "corr_lr": 0.182,
            "corr_ls_l": 0.224,
            "corr_rs_r": 0.258,
            "corr_ls_rs": 0.087,
        },
        Verdict.SUSPICIOUS,
    ),
]


@pytest.mark.parametrize(("name", "kwargs", "expected"), GOLDEN_MULTI_CHANNEL)
def test_golden_multichannel(
    name: str,
    kwargs: dict[str, float],
    expected: Verdict,
) -> None:
    m = _five_one_metrics(**kwargs)
    p = classify_audio(m)
    assert p.verdict == expected, (
        f"{name}: expected {expected.value}, got {p.verdict.value} (score={p.score}, reasons={list(p.reasons)})"
    )


GOLDEN_STEREO: list[tuple[str, float, float, float, Verdict]] = [
    ("DMB 2.0", 1.000, -31.7, -31.7, Verdict.FAKE),
    ("Operatsia 2.0", 0.998, -22.7, -23.4, Verdict.FAKE),
    ("Landysh 2.0", 0.994, -27.2, -27.5, Verdict.FAKE),
    ("Bury Me 2.0", 0.939, -27.7, -27.0, Verdict.REAL),
    ("MIB 1 2.0 rus", 0.951, -30.6, -30.6, Verdict.REAL),
    ("Neadekvatnie 2.0", 0.755, -25.4, -28.5, Verdict.REAL),
    ("Starship Troopers MVO Pervyi", 0.966, -28.0, -27.8, Verdict.SUSPICIOUS),
]


@pytest.mark.parametrize(("name", "corr", "rms_l", "rms_r", "expected"), GOLDEN_STEREO)
def test_golden_stereo(
    name: str,
    corr: float,
    rms_l: float,
    rms_r: float,
    expected: Verdict,
) -> None:
    m = _stereo_metrics(corr=corr, rms_l=rms_l, rms_r=rms_r)
    p = classify_audio(m)
    assert p.verdict == expected, (
        f"{name}: expected {expected.value}, got {p.verdict.value} (score={p.score}, reasons={list(p.reasons)})"
    )
