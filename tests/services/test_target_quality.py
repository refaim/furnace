from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

from furnace.core.models import EncodeResult, MetricPool, MetricScores, VideoParams
from furnace.core.target_quality import (
    KnobSearchResult,
    pool_grain_windows,
    probe_windows,
)
from furnace.services.target_quality import TargetQualityService, _SearchNarrator, _windows
from tests.conftest import make_video_params


class _FakeExtractor:
    def __init__(
        self,
        rc: int = 0,
        bitrates: list[tuple[float, float]] | None = None,
        *,
        ref_rc: int = 0,
    ) -> None:
        self.rc = rc
        self.ref_rc = ref_rc
        self.bitrates = bitrates if bitrates is not None else []
        self.calls: list[dict[str, object]] = []
        self.reference_calls: list[dict[str, object]] = []

    def extract_window(
        self,
        input_path: Path,
        output_path: Path,
        *,
        start_s: float,
        frames: int,
    ) -> int:
        self.calls.append({"input": input_path, "output": output_path, "start_s": start_s, "frames": frames})
        return self.rc

    def build_reference(
        self,
        input_path: Path,
        output_path: Path,
        video_params: object,
    ) -> int:
        self.reference_calls.append({"input": input_path, "output": output_path, "video_params": video_params})
        return self.ref_rc

    def window_bitrates(self, source: Path, window_s: float) -> list[tuple[float, float]]:  # noqa: ARG002
        return self.bitrates


class _FakeProbe:
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


def _service(
    rc: int = 0, score_fn: Callable[[int], float] | None = None
) -> tuple[TargetQualityService, _FakeExtractor, _FakeProbe]:
    extractor = _FakeExtractor(rc=rc)
    probe = _FakeProbe(score_fn if score_fn is not None else (lambda q: 120.0 - q))
    return TargetQualityService(extractor, probe), extractor, probe


class TestTargetQualitySearch:
    def test_extracts_three_windows(self, tmp_path: Path) -> None:
        service, extractor, _probe = _service()
        vp = make_video_params(source_width=1920, source_height=1080)
        service.search(Path("movie.mkv"), vp, duration_s=7200.0, work_dir=tmp_path)
        assert len(extractor.calls) == 3
        assert all(c["frames"] == 432 for c in extractor.calls)

    def test_returns_knob_search_result_in_bounds(self, tmp_path: Path) -> None:
        service, _extractor, _probe = _service()
        vp = make_video_params(source_width=1920, source_height=1080)
        result = service.search(Path("m.mkv"), vp, duration_s=7200.0, work_dir=tmp_path)
        assert isinstance(result, KnobSearchResult)
        assert 16 <= result.knob <= 44

    def test_full_pass_extracts_one_bounded_window(self, tmp_path: Path) -> None:
        service, extractor, probe = _service()
        vp = make_video_params(source_width=1920, source_height=1080)
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
        service, extractor, _probe = _service()
        vp = make_video_params(
            source_width=1920,
            source_height=1080,
            fps_num=24000,
            fps_den=1001,
        )
        service.search(Path("m.mkv"), vp, duration_s=7200.0, work_dir=tmp_path)
        assert all(c["frames"] == 432 for c in extractor.calls)

    def test_mean_pooling_across_windows(self, tmp_path: Path) -> None:
        service, _extractor, _probe = _service(score_fn=lambda q: 120.0 - q)
        vp = make_video_params(source_width=1920, source_height=1080)
        result = service.search(Path("m.mkv"), vp, duration_s=7200.0, work_dir=tmp_path)
        assert result.probes[0][1] == pytest.approx(90.0)


def _crf_from_obu(path: Path) -> int:
    m = re.search(r"_q(\d+)_", path.name)
    assert m is not None
    return int(m.group(1))


def _win_from_obu(path: Path) -> int:
    m = re.search(r"_w(\d+)\.obu$", path.name)
    assert m is not None
    return int(m.group(1))


def _grain_service(
    *,
    enc_rc: int = 0,
    score_none: bool = False,
) -> tuple[TargetQualityService, MagicMock, MagicMock]:
    extractor = _FakeExtractor()
    inline = MagicMock()
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
        assert TargetQualityService(_FakeExtractor(), MagicMock(), grain_encoder=MagicMock()).can_search(vp) is False
        assert TargetQualityService(_FakeExtractor(), MagicMock(), metrics=MagicMock()).can_search(vp) is False
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
        assert 26 <= result.knob <= 34
        enc_kwargs = grain_enc.encode.call_args.kwargs
        assert enc_kwargs["cq_override"] == result.probes[-1][0]
        assert all(c.kwargs["pool"] is MetricPool.LOW for c in metrics.measure.call_args_list)

    def test_converges_toward_target(self, tmp_path: Path) -> None:
        svc, _enc, _metrics = _grain_service()
        result = svc.search(Path("grain.mkv"), _grain_vp(), duration_s=7200.0, work_dir=tmp_path)
        assert 28 <= result.knob <= 32

    def test_grain_extracts_ten_windows(self, tmp_path: Path) -> None:
        svc, _enc, _metrics = _grain_service()
        svc.search(Path("grain.mkv"), _grain_vp(), duration_s=7200.0, work_dir=tmp_path)
        indices = {_win_from_obu(c.args[1]) for c in _metrics.measure.call_args_list}
        assert indices == set(range(10))

    def test_grain_vbr_selects_hardest_windows(self, tmp_path: Path) -> None:
        offsets = [500.0 + 500.0 * i for i in range(12)]
        low = {2500.0, 4000.0}
        bitrates = [(o, 1.0 if o in low else 100.0) for o in offsets]
        extractor = _FakeExtractor(bitrates=bitrates)
        grain_enc = MagicMock()
        grain_enc.encode.return_value = EncodeResult(return_code=0, encoder_settings="svt")
        metrics = MagicMock()
        metrics.measure.side_effect = lambda reference, distorted, **kw: MetricScores(
            ssimulacra2=100.0 - _crf_from_obu(distorted)
        )
        svc = TargetQualityService(extractor, MagicMock(), grain_encoder=grain_enc, metrics=metrics)
        svc.search(Path("grain.mkv"), _grain_vp(), duration_s=7200.0, work_dir=tmp_path)
        extracted = sorted(cast("float", c["start_s"]) for c in extractor.calls)
        assert extracted == sorted(o for o in offsets if o not in low)

    def test_grain_cbr_uses_even_windows(self, tmp_path: Path) -> None:
        offsets = [500.0 + 500.0 * i for i in range(12)]
        bitrates = [(o, 100.0) for o in offsets]
        extractor = _FakeExtractor(bitrates=bitrates)
        grain_enc = MagicMock()
        grain_enc.encode.return_value = EncodeResult(return_code=0, encoder_settings="svt")
        metrics = MagicMock()
        metrics.measure.side_effect = lambda reference, distorted, **kw: MetricScores(
            ssimulacra2=100.0 - _crf_from_obu(distorted)
        )
        svc = TargetQualityService(extractor, MagicMock(), grain_encoder=grain_enc, metrics=metrics)
        svc.search(Path("grain.mkv"), _grain_vp(), duration_s=7200.0, work_dir=tmp_path)
        extracted = sorted(cast("float", c["start_s"]) for c in extractor.calls)
        even = probe_windows(7200.0, count=10, window_s=18.0)
        assert even is not None
        assert extracted == sorted(even)

    def test_grain_percentile_pooled_across_windows(self, tmp_path: Path) -> None:
        extractor = _FakeExtractor()
        grain_enc = MagicMock()
        grain_enc.encode.return_value = EncodeResult(return_code=0, encoder_settings="svt")
        metrics = MagicMock()
        metrics.measure.side_effect = lambda reference, distorted, **kw: MetricScores(
            ssimulacra2=(100.0 - _crf_from_obu(distorted)) - _win_from_obu(distorted)
        )
        svc = TargetQualityService(extractor, MagicMock(), grain_encoder=grain_enc, metrics=metrics)
        result = svc.search(Path("grain.mkv"), _grain_vp(), duration_s=7200.0, work_dir=tmp_path)
        scores = [70.0 - i for i in range(10)]
        assert result.probes[0][1] == pytest.approx(pool_grain_windows(scores))
        assert result.probes[0][1] == pytest.approx(62.8)

    def test_grain_full_pass_single_window(self, tmp_path: Path) -> None:
        svc, _enc, metrics = _grain_service()
        result = svc.search(Path("grain.mkv"), _grain_vp(), duration_s=40.0, work_dir=tmp_path)
        assert all(_win_from_obu(c.args[1]) == 0 for c in metrics.measure.call_args_list)
        assert result.probes[0][1] == pytest.approx(100.0 - result.probes[0][0])

    def test_reference_built_once_per_window(self, tmp_path: Path) -> None:
        extractor = _FakeExtractor()
        grain_enc = MagicMock()
        grain_enc.encode.return_value = EncodeResult(return_code=0, encoder_settings="svt")
        metrics = MagicMock()
        metrics.measure.side_effect = lambda reference, distorted, **kw: MetricScores(
            ssimulacra2=105.0 - _crf_from_obu(distorted)
        )
        svc = TargetQualityService(extractor, MagicMock(), grain_encoder=grain_enc, metrics=metrics)
        svc.search(Path("grain.mkv"), _grain_vp(), duration_s=7200.0, work_dir=tmp_path)
        assert len(extractor.reference_calls) == 10
        assert len(metrics.measure.call_args_list) > 10
        assert {c["output"] for c in extractor.reference_calls} == {tmp_path / f"tq_ref_w{j}.mkv" for j in range(10)}
        for c in metrics.measure.call_args_list:
            win = _win_from_obu(c.args[1])
            assert c.args[0] == tmp_path / f"tq_ref_w{win}.mkv"

    def test_reference_build_failure_raises(self, tmp_path: Path) -> None:
        extractor = _FakeExtractor(ref_rc=1)
        grain_enc = MagicMock()
        grain_enc.encode.return_value = EncodeResult(return_code=0, encoder_settings="svt")
        svc = TargetQualityService(extractor, MagicMock(), grain_encoder=grain_enc, metrics=MagicMock())
        with pytest.raises(RuntimeError, match="reference build failed"):
            svc.search(Path("grain.mkv"), _grain_vp(), duration_s=7200.0, work_dir=tmp_path)

    def test_encode_failure_raises(self, tmp_path: Path) -> None:
        svc, _enc, _metrics = _grain_service(enc_rc=1)
        with pytest.raises(RuntimeError, match="grain probe encode failed"):
            svc.search(Path("grain.mkv"), _grain_vp(), duration_s=7200.0, work_dir=tmp_path)

    def test_unscorable_probe_raises(self, tmp_path: Path) -> None:
        svc, _enc, _metrics = _grain_service(score_none=True)
        with pytest.raises(RuntimeError, match="could not be scored"):
            svc.search(Path("grain.mkv"), _grain_vp(), duration_s=7200.0, work_dir=tmp_path)

    def test_search_without_grain_deps_raises(self, tmp_path: Path) -> None:
        svc = TargetQualityService(_FakeExtractor(), MagicMock())
        with pytest.raises(RuntimeError, match="requires an SVT encoder"):
            svc.search(Path("grain.mkv"), _grain_vp(), duration_s=7200.0, work_dir=tmp_path)


class TestWindowsHelper:
    def test_singular(self) -> None:
        assert _windows(1) == "1 window"

    def test_plural(self) -> None:
        assert _windows(3) == "3 windows"


class TestSearchNarrator:
    def test_opening_reports_plan(self) -> None:
        events: list[str] = []
        narrator = _SearchNarrator(
            emit=events.append,
            label="QVBR",
            metric="SSIMULACRA2",
            window_count=3,
            pool_word="mean",
        )
        narrator.opening(81.0)
        assert events == ["Probing QVBR -> SSIMULACRA2 ~81.0 (3 windows, mean-pooled)"]

    def test_window_reports_per_window_score(self) -> None:
        events: list[str] = []
        narrator = _SearchNarrator(
            emit=events.append,
            label="CRF",
            metric="SSIMULACRA2",
            window_count=10,
            pool_word="worst-case",
        )
        narrator.window(24, 3, 69.14)
        assert events == ["CRF 24: window 3/10 = 69.1"]

    def test_result_reports_pooled_score(self) -> None:
        events: list[str] = []
        narrator = _SearchNarrator(
            emit=events.append,
            label="CRF",
            metric="SSIMULACRA2",
            window_count=10,
            pool_word="worst-case",
        )
        narrator.result(24, 67.0)
        assert events == ["CRF 24 -> SSIMULACRA2 67.0"]


class TestSearchNarrationWiring:
    def test_nvenc_search_narrates_opening_windows_and_result(self, tmp_path: Path) -> None:
        service, _extractor, _probe = _service(score_fn=lambda q: 120.0 - q)
        vp = make_video_params(source_width=1920, source_height=1080)
        events: list[str] = []
        service.search(
            Path("m.mkv"),
            vp,
            duration_s=7200.0,
            work_dir=tmp_path,
            on_event=events.append,
        )
        assert "Probing QVBR -> SSIMULACRA2 ~81.0 (3 windows, mean-pooled)" in events
        assert "QVBR 30: window 1/3 = 90.0" in events
        assert "QVBR 30: window 3/3 = 90.0" in events
        assert "QVBR 30 -> SSIMULACRA2 90.0" in events

    def test_grain_search_narrates_worst_case_pooling(self, tmp_path: Path) -> None:
        svc, _enc, _metrics = _grain_service()
        events: list[str] = []
        svc.search(
            Path("grain.mkv"),
            _grain_vp(),
            duration_s=7200.0,
            work_dir=tmp_path,
            on_event=events.append,
        )
        assert "Probing CRF -> SSIMULACRA2 ~70.0 (10 windows, worst-case-pooled)" in events
        assert "CRF 30: window 10/10 = 70.0" in events
        assert "CRF 30 -> SSIMULACRA2 70.0" in events

    def test_search_without_on_event_is_silent_and_succeeds(self, tmp_path: Path) -> None:
        service, _extractor, _probe = _service()
        vp = make_video_params(source_width=1920, source_height=1080)
        result = service.search(Path("m.mkv"), vp, duration_s=7200.0, work_dir=tmp_path)
        assert isinstance(result, KnobSearchResult)
