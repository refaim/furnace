from __future__ import annotations

import pytest

from furnace.core.models import DownmixMode, DvBlCompatibility, DvMode, EncodeResult


class TestDvBlCompatibility:
    def test_values(self) -> None:
        assert int(DvBlCompatibility.NONE) == 0
        assert int(DvBlCompatibility.HDR10) == 1
        assert int(DvBlCompatibility.SDR) == 2
        assert int(DvBlCompatibility.HLG) == 4

    def test_from_int(self) -> None:
        assert DvBlCompatibility(1) == DvBlCompatibility.HDR10
        assert DvBlCompatibility(4) == DvBlCompatibility.HLG


class TestDvMode:
    def test_values(self) -> None:
        assert int(DvMode.COPY) == 0
        assert int(DvMode.TO_8_1) == 2

    def test_from_int(self) -> None:
        assert DvMode(0) == DvMode.COPY
        assert DvMode(2) == DvMode.TO_8_1


class TestEncodeResult:
    def test_basic(self) -> None:
        r = EncodeResult(return_code=0, encoder_settings="hevc_nvenc / main10")
        assert r.return_code == 0
        assert r.vmaf_score is None

    def test_with_metrics(self) -> None:
        r = EncodeResult(return_code=0, encoder_settings="test", vmaf_score=95.4, ssim_score=0.987)
        assert r.vmaf_score == 95.4
        assert r.ssim_score == 0.987


class TestDownmixMode:
    def test_values(self) -> None:
        assert DownmixMode.STEREO.value == "stereo"
        assert DownmixMode.DOWN6.value == "down6"

    def test_from_string(self) -> None:
        assert DownmixMode("stereo") == DownmixMode.STEREO
        assert DownmixMode("down6") == DownmixMode.DOWN6

    def test_invalid_string_raises(self) -> None:
        with pytest.raises(ValueError, match="foo"):
            DownmixMode("foo")
