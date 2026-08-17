from __future__ import annotations

import pytest

from furnace.core.models import DownmixMode, DvBlCompatibility, DvMode, EncodeResult
from tests.conftest import make_track, make_video_params


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
        r = EncodeResult(return_code=0, encoder_settings="av1_nvenc / main")
        assert r.return_code == 0
        assert r.encoder_settings == "av1_nvenc / main"

    def test_ssim_score_is_gone(self) -> None:
        r = EncodeResult(return_code=0, encoder_settings="x")
        assert not hasattr(r, "ssim_score")


class TestVideoParamsPassthrough:
    def test_default_is_false(self) -> None:
        vp = make_video_params()
        assert vp.passthrough is False

    def test_can_be_set_true(self) -> None:
        vp = make_video_params(passthrough=True)
        assert vp.passthrough is True


class TestTrackStreamIndex:
    def test_falls_back_to_index(self) -> None:
        assert make_track(index=3).stream_index == 3

    def test_prefers_source_index(self) -> None:
        assert make_track(index=3, source_index=0).stream_index == 0

    def test_source_index_zero_is_not_treated_as_missing(self) -> None:
        track = make_track(index=7, source_index=0)
        assert track.source_index == 0
        assert track.stream_index == 0


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


def test_analyze_status_values() -> None:
    from furnace.core.models import AnalyzeStatus

    assert AnalyzeStatus.DONE.value == "done"
    assert AnalyzeStatus.SKIPPED.value == "skipped"
    assert AnalyzeStatus.FAILED.value == "failed"


def test_analysis_outcome_done_carries_movie() -> None:
    from furnace.core.models import AnalysisOutcome, AnalyzeStatus

    movie = object()
    outcome = AnalysisOutcome(movie=movie, status=AnalyzeStatus.DONE, detail="summary")  # type: ignore[arg-type]
    assert outcome.movie is movie
    assert outcome.status is AnalyzeStatus.DONE
    assert outcome.detail == "summary"


def test_analysis_outcome_is_frozen() -> None:
    import dataclasses

    import pytest

    from furnace.core.models import AnalysisOutcome, AnalyzeStatus

    outcome = AnalysisOutcome(movie=None, status=AnalyzeStatus.FAILED, detail="boom")
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.detail = "x"  # type: ignore[misc]
