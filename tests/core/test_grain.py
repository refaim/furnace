from __future__ import annotations

from furnace.core.detect import classify_grain, hdr_tonemap_transfer, is_hdr_transfer
from tests.conftest import make_video_info, make_video_params


class TestIsHdrTransfer:
    def test_pq_and_hlg_are_hdr(self) -> None:
        assert is_hdr_transfer("smpte2084") is True
        assert is_hdr_transfer("arib-std-b67") is True

    def test_sdr_curves_are_not_hdr(self) -> None:
        assert is_hdr_transfer("bt709") is False
        assert is_hdr_transfer("smpte170m") is False
        assert is_hdr_transfer("bt2020-10") is False

    def test_untagged_is_not_hdr(self) -> None:
        assert is_hdr_transfer(None) is False


class TestGrainProbeCoversHdr:
    def test_sdr_probed_without_tonemap(self) -> None:
        assert hdr_tonemap_transfer("bt709") is None
        assert hdr_tonemap_transfer("smpte170m") is None
        assert hdr_tonemap_transfer("bt470bg") is None

    def test_untagged_transfer_probed_as_sdr(self) -> None:
        assert hdr_tonemap_transfer(None) is None

    def test_hdr_probed_through_a_tonemap(self) -> None:
        assert hdr_tonemap_transfer("smpte2084") == "smpte2084"
        assert hdr_tonemap_transfer("arib-std-b67") == "arib-std-b67"


class TestClassifyGrain:
    def test_grainy_reference_level_detected(self) -> None:
        assert classify_grain([0.67, 0.51, 1.21, 1.95, 0.58]) is True

    def test_clean_control_level_not_detected(self) -> None:
        assert classify_grain([0.16, 0.14, 0.15, 0.06, 0.18]) is False

    def test_median_resists_motion_flooded_windows(self) -> None:
        assert classify_grain([0.2, 9.3, 0.2, 8.0, 0.2]) is False

    def test_empty_sample_defaults_to_grainy(self) -> None:
        assert classify_grain([]) is True

    def test_threshold_boundary(self) -> None:
        assert classify_grain([0.40]) is True
        assert classify_grain([0.39]) is False

    def test_even_sample_averages_two_middle_values(self) -> None:
        assert classify_grain([0.6, 0.1, 0.5, 0.3]) is True
        assert classify_grain([0.6, 0.1, 0.4, 0.3]) is False


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
