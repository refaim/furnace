"""SD grain detection: probe gating and the static-block flicker verdict.

Grain probing is confined to SD sources (HD/UHD already pass the user's blind
test on the QVBR profile). The classifier turns per-window static-block flicker
into a boolean GRAINY verdict; a failed probe (no samples) fails soft to GRAINY
because wrongly-on merely spends bytes while wrongly-off smears faces into wax.
"""

from __future__ import annotations

from furnace.core.detect import classify_grain, needs_grain_probe
from tests.conftest import make_video_info, make_video_params


class TestNeedsGrainProbe:
    def test_sd_height_probed(self) -> None:
        assert needs_grain_probe(480) is True
        assert needs_grain_probe(576) is True

    def test_hd_and_above_never_probed(self) -> None:
        assert needs_grain_probe(720) is False
        assert needs_grain_probe(1080) is False
        assert needs_grain_probe(2160) is False


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
