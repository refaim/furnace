"""Grain detection: probe gating and the static-block flicker verdict.

Grain probing covers SDR sources at ANY resolution — grainy HD/UHD film needs the
grain-aware path too — and skips HDR, which the grain path cannot score. The
classifier turns per-window static-block flicker into a boolean GRAINY verdict; a
failed probe (no samples) fails soft to GRAINY because wrongly-on merely spends
bytes while wrongly-off smears faces into wax.
"""

from __future__ import annotations

from furnace.core.detect import classify_grain, is_hdr_transfer, needs_grain_probe
from tests.conftest import make_video_info, make_video_params


class TestIsHdrTransfer:
    """One source of truth for "is this HDR?", shared by the grain probe gate, the
    planner's grain routing and the target-quality domain split — so they cannot
    drift apart and re-open the grain+HDR hole."""

    def test_pq_and_hlg_are_hdr(self) -> None:
        assert is_hdr_transfer("smpte2084") is True
        assert is_hdr_transfer("arib-std-b67") is True

    def test_sdr_curves_are_not_hdr(self) -> None:
        assert is_hdr_transfer("bt709") is False
        assert is_hdr_transfer("smpte170m") is False
        assert is_hdr_transfer("bt2020-10") is False  # SDR BT.2020, not PQ

    def test_untagged_is_not_hdr(self) -> None:
        assert is_hdr_transfer(None) is False


class TestNeedsGrainProbe:
    def test_sdr_probed_regardless_of_resolution(self) -> None:
        """Resolution does not gate: SD, HD and UHD SDR all get the probe. NVEnc at
        the non-grain target over-encodes grainy HD/UHD past a compact source."""
        assert needs_grain_probe("bt709") is True
        assert needs_grain_probe("smpte170m") is True
        assert needs_grain_probe("bt470bg") is True

    def test_untagged_transfer_probed_as_sdr(self) -> None:
        """An untagged source is assumed SDR (the common case) → probed."""
        assert needs_grain_probe(None) is True

    def test_hdr_never_probed(self) -> None:
        """The grain path scores with SSIMULACRA2, which does not score PQ/HLG
        correctly, so HDR stays on the NVEnc/CVVDP path regardless of grain."""
        assert needs_grain_probe("smpte2084") is False
        assert needs_grain_probe("arib-std-b67") is False


class TestClassifyGrain:
    def test_grainy_reference_level_detected(self) -> None:
        """Calibrated grainy DVDs measured 0.75/1.66 → GRAINY."""
        assert classify_grain([0.73, 0.74, 2.45, 0.75, 0.88]) is True

    def test_clean_control_level_not_detected(self) -> None:
        """Denoised control measured ~0.22 → CLEAN."""
        assert classify_grain([0.20, 0.22, 0.21, 0.23, 0.25]) is False

    def test_median_resists_motion_flooded_windows(self) -> None:
        """A couple of all-motion windows must not flip a clean verdict."""
        assert classify_grain([0.2, 9.3, 0.2, 8.0, 0.2]) is False

    def test_empty_sample_defaults_to_grainy(self) -> None:
        """Failed probe → GRAINY: wrongly-on costs bytes, wrongly-off costs faces."""
        assert classify_grain([]) is True

    def test_threshold_boundary(self) -> None:
        assert classify_grain([0.5]) is True   # >= threshold
        assert classify_grain([0.49]) is False

    def test_even_sample_averages_two_middle_values(self) -> None:
        """Even-length input: verdict rides the mean of the two central samples."""
        # sorted -> [0.1, 0.3, 0.7, 0.9]; median = (0.3 + 0.7) / 2 = 0.5 >= threshold
        assert classify_grain([0.9, 0.1, 0.7, 0.3]) is True
        # sorted -> [0.1, 0.3, 0.6, 0.9]; median = (0.3 + 0.6) / 2 = 0.45 < threshold
        assert classify_grain([0.9, 0.1, 0.6, 0.3]) is False


def test_video_info_grainy_defaults_false() -> None:
    vi = make_video_info()
    assert vi.grainy is False


def test_video_info_grainy_can_be_set_true() -> None:
    vi = make_video_info()
    vi.grainy = True
    assert vi.grainy is True


def test_video_params_grain_defaults_false() -> None:
    vp = make_video_params()
    assert vp.grain is False


def test_video_params_grain_can_be_set_true() -> None:
    vp = make_video_params()
    vp.grain = True
    assert vp.grain is True
