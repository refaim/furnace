"""Tests for the TargetQualityService orchestration (NVEnc QVBR search)."""
from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from furnace.core.models import EncodeResult, MetricPool, MetricScores, VideoParams
from furnace.core.target_quality import KnobSearchResult
from furnace.services.target_quality import TargetQualityService
from tests.conftest import make_video_params


class _FakeExtractor:
    """Records extract_window calls and returns a fixed return code."""

    def __init__(self, rc: int = 0) -> None:
        self.rc = rc
        self.calls: list[dict[str, object]] = []

    def extract_window(
        self,
        input_path: Path,
        output_path: Path,
        *,
        start_s: float,
        frames: int,
    ) -> int:
        self.calls.append(
            {"input": input_path, "output": output_path, "start_s": start_s, "frames": frames}
        )
        return self.rc


class _FakeProbe:
    """Returns score_fn(qvbr), recording every probe call."""

    def __init__(self, score_fn: Callable[[int], float]) -> None:
        self._score_fn = score_fn
        self.calls: list[dict[str, object]] = []

    def probe(
        self,
        input_path: Path,
        output_path: Path,  # noqa: ARG002
        video_params: object,  # noqa: ARG002
        *,
        qvbr: int,
        metric: str,
    ) -> float:
        self.calls.append({"input": input_path, "qvbr": qvbr, "metric": metric})
        return self._score_fn(qvbr)


def _service(rc: int = 0, score_fn: Callable[[int], float] | None = None) -> tuple[
    TargetQualityService, _FakeExtractor, _FakeProbe
]:
    extractor = _FakeExtractor(rc=rc)
    probe = _FakeProbe(score_fn if score_fn is not None else (lambda q: 120.0 - q))
    return TargetQualityService(extractor, probe), extractor, probe


class TestTargetQualitySearch:
    def test_extracts_three_windows(self, tmp_path: Path) -> None:
        """A long source yields three evenly-spaced extracted windows."""
        service, extractor, _probe = _service()
        vp = make_video_params(source_width=1920, source_height=1080)
        service.search(Path("movie.mkv"), vp, duration_s=7200.0, work_dir=tmp_path)
        assert len(extractor.calls) == 3
        # 18s at the default 24fps -> 432 frames per window.
        assert all(c["frames"] == 432 for c in extractor.calls)

    def test_returns_knob_search_result_in_bounds(self, tmp_path: Path) -> None:
        service, _extractor, _probe = _service()
        vp = make_video_params(source_width=1920, source_height=1080)
        result = service.search(Path("m.mkv"), vp, duration_s=7200.0, work_dir=tmp_path)
        assert isinstance(result, KnobSearchResult)
        assert 16 <= result.knob <= 44

    def test_full_pass_extracts_one_bounded_window(self, tmp_path: Path) -> None:
        """A short source extracts ONE window bounded to the reported duration
        (not the raw file) -- guards against an under-reported duration."""
        service, extractor, probe = _service()
        vp = make_video_params(source_width=1920, source_height=1080)  # 24fps
        service.search(Path("short.mkv"), vp, duration_s=40.0, work_dir=tmp_path)
        assert len(extractor.calls) == 1
        assert extractor.calls[0]["start_s"] == 0.0
        assert extractor.calls[0]["frames"] == round(40.0 * 24)
        assert probe.calls
        assert all(c["input"] == tmp_path / "tq_window_full.mkv" for c in probe.calls)

    def test_extraction_failure_raises(self, tmp_path: Path) -> None:
        service, _extractor, _probe = _service(rc=1)
        vp = make_video_params(source_width=1920, source_height=1080)
        with pytest.raises(RuntimeError, match="window extraction failed"):
            service.search(Path("m.mkv"), vp, duration_s=7200.0, work_dir=tmp_path)

    def test_full_pass_extraction_failure_raises(self, tmp_path: Path) -> None:
        service, _extractor, _probe = _service(rc=1)
        vp = make_video_params(source_width=1920, source_height=1080)
        with pytest.raises(RuntimeError, match="full-pass"):
            service.search(Path("m.mkv"), vp, duration_s=40.0, work_dir=tmp_path)

    def test_hdr_drives_cvvdp(self, tmp_path: Path) -> None:
        service, _extractor, probe = _service()
        vp = make_video_params(color_transfer="smpte2084", color_matrix="bt2020nc")
        service.search(Path("m.mkv"), vp, duration_s=7200.0, work_dir=tmp_path)
        assert probe.calls
        assert all(c["metric"] == "cvvdp" for c in probe.calls)

    def test_sdr_drives_ssimulacra2(self, tmp_path: Path) -> None:
        service, _extractor, probe = _service()
        vp = make_video_params(source_width=1920, source_height=1080)
        service.search(Path("m.mkv"), vp, duration_s=7200.0, work_dir=tmp_path)
        assert probe.calls
        assert all(c["metric"] == "ssimulacra2" for c in probe.calls)

    def test_probe_runs_once_per_window_per_knob(self, tmp_path: Path) -> None:
        """Each probed knob measures all three windows (mean pooling)."""
        service, _extractor, probe = _service()
        vp = make_video_params(source_width=1920, source_height=1080)
        service.search(Path("m.mkv"), vp, duration_s=7200.0, work_dir=tmp_path)
        first_knob = probe.calls[0]["qvbr"]
        inputs_for_first = {c["input"] for c in probe.calls if c["qvbr"] == first_knob}
        assert inputs_for_first == {
            tmp_path / "tq_window_0.mkv",
            tmp_path / "tq_window_1.mkv",
            tmp_path / "tq_window_2.mkv",
        }

    def test_frames_from_fractional_fps(self, tmp_path: Path) -> None:
        """23.976 fps (24000/1001) -> round(18 * 23.976) = 432 frames."""
        service, extractor, _probe = _service()
        vp = make_video_params(
            source_width=1920, source_height=1080, fps_num=24000, fps_den=1001,
        )
        service.search(Path("m.mkv"), vp, duration_s=7200.0, work_dir=tmp_path)
        assert all(c["frames"] == 432 for c in extractor.calls)

    def test_mean_pooling_across_windows(self, tmp_path: Path) -> None:
        """probe_fn returns the mean score across windows: with a per-window
        constant probe, the search still converges and records probes."""
        # score depends only on knob (same for every window) so the mean == score.
        service, _extractor, _probe = _service(score_fn=lambda q: 120.0 - q)
        vp = make_video_params(source_width=1920, source_height=1080)
        result = service.search(Path("m.mkv"), vp, duration_s=7200.0, work_dir=tmp_path)
        # First probe is the midpoint of [16, 44] = 30 -> score 90.
        assert result.probes[0][1] == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# Grain (SVT-AV1) path: encode window at CRF -> measure worst-case SSIMULACRA2
# ---------------------------------------------------------------------------


def _crf_from_obu(path: Path) -> int:
    """Parse the CRF encoded in a probe OBU name (tq_grain_q{crf}_w{j}.obu)."""
    m = re.search(r"_q(\d+)_", path.name)
    assert m is not None
    return int(m.group(1))


def _win_from_obu(path: Path) -> int:
    """Parse the window index in a probe OBU name (tq_grain_q{crf}_w{j}.obu)."""
    m = re.search(r"_w(\d+)\.obu$", path.name)
    assert m is not None
    return int(m.group(1))


def _grain_service(
    *, enc_rc: int = 0, score_none: bool = False,
) -> tuple[TargetQualityService, MagicMock, MagicMock]:
    extractor = _FakeExtractor()
    inline = MagicMock()  # unused on the grain path
    grain_enc = MagicMock()
    grain_enc.encode.return_value = EncodeResult(return_code=enc_rc, encoder_settings="svt")
    metrics = MagicMock()
    if score_none:
        metrics.measure.return_value = MetricScores()
    else:
        metrics.measure.side_effect = lambda reference, distorted, **kw: MetricScores(
            ssimulacra2=100.0 - _crf_from_obu(distorted)
        )
    svc = TargetQualityService(extractor, inline, grain_encoder=grain_enc, metrics=metrics)
    return svc, grain_enc, metrics


def _grain_vp() -> VideoParams:
    return make_video_params(grain=True, source_width=720, source_height=576)


class TestCanSearch:
    def test_nongrain_always_searchable(self) -> None:
        svc = TargetQualityService(_FakeExtractor(), MagicMock())
        assert svc.can_search(make_video_params()) is True

    def test_grain_needs_both_deps(self) -> None:
        vp = _grain_vp()
        assert TargetQualityService(_FakeExtractor(), MagicMock()).can_search(vp) is False
        assert (
            TargetQualityService(_FakeExtractor(), MagicMock(), grain_encoder=MagicMock()).can_search(vp)
            is False
        )
        assert (
            TargetQualityService(_FakeExtractor(), MagicMock(), metrics=MagicMock()).can_search(vp)
            is False
        )
        assert (
            TargetQualityService(
                _FakeExtractor(), MagicMock(), grain_encoder=MagicMock(), metrics=MagicMock()
            ).can_search(vp)
            is True
        )


class TestGrainSearch:
    def test_encodes_each_window_at_crf_and_measures_low_pool(self, tmp_path: Path) -> None:
        svc, grain_enc, metrics = _grain_service()
        result = svc.search(Path("grain.mkv"), _grain_vp(), duration_s=7200.0, work_dir=tmp_path)
        assert isinstance(result, KnobSearchResult)
        assert 14 <= result.knob <= 34  # CRF bounds
        # SVT probe encodes at the candidate CRF with no metrics on the encode.
        enc_kwargs = grain_enc.encode.call_args.kwargs
        assert enc_kwargs["cq_override"] == result.probes[-1][0]
        # ...and scoring uses worst-case (low-percentile) pooling.
        assert all(c.kwargs["pool"] is MetricPool.LOW for c in metrics.measure.call_args_list)

    def test_converges_toward_target(self, tmp_path: Path) -> None:
        """score = 100 - crf, grain target ~71 -> crf ~29 (within [14,34])."""
        svc, _enc, _metrics = _grain_service()
        result = svc.search(Path("grain.mkv"), _grain_vp(), duration_s=7200.0, work_dir=tmp_path)
        assert 28 <= result.knob <= 32

    def test_worst_case_min_pooled_across_windows(self, tmp_path: Path) -> None:
        """The per-window worst-cases are MIN-pooled (the worst window governs the
        CRF), not averaged: a bright reel's high p5 must not mask a dark reel's low
        p5, which would pick too-high a CRF."""
        extractor = _FakeExtractor()
        grain_enc = MagicMock()
        grain_enc.encode.return_value = EncodeResult(return_code=0, encoder_settings="svt")
        metrics = MagicMock()
        # score = (100 - crf) - 10*window_index -> window 2 is always the worst.
        metrics.measure.side_effect = lambda reference, distorted, **kw: MetricScores(
            ssimulacra2=(100.0 - _crf_from_obu(distorted)) - 10.0 * _win_from_obu(distorted)
        )
        svc = TargetQualityService(
            extractor, MagicMock(), grain_encoder=grain_enc, metrics=metrics
        )
        result = svc.search(Path("grain.mkv"), _grain_vp(), duration_s=7200.0, work_dir=tmp_path)
        # First probe: midpoint of [14,34] = 24. Windows -> 76, 66, 56.
        # min pooling records 56.0 (mean pooling would record 66.0).
        assert result.probes[0][1] == pytest.approx(56.0)

    def test_encode_failure_raises(self, tmp_path: Path) -> None:
        svc, _enc, _metrics = _grain_service(enc_rc=1)
        with pytest.raises(RuntimeError, match="grain probe encode failed"):
            svc.search(Path("grain.mkv"), _grain_vp(), duration_s=7200.0, work_dir=tmp_path)

    def test_unscorable_probe_raises(self, tmp_path: Path) -> None:
        svc, _enc, _metrics = _grain_service(score_none=True)
        with pytest.raises(RuntimeError, match="could not be scored"):
            svc.search(Path("grain.mkv"), _grain_vp(), duration_s=7200.0, work_dir=tmp_path)

    def test_search_without_grain_deps_raises(self, tmp_path: Path) -> None:
        """Defensive: search on a grain job without the SVT/metrics deps raises
        (the executor gates this via can_search, but the guard is loud)."""
        svc = TargetQualityService(_FakeExtractor(), MagicMock())
        with pytest.raises(RuntimeError, match="requires an SVT encoder"):
            svc.search(Path("grain.mkv"), _grain_vp(), duration_s=7200.0, work_dir=tmp_path)
