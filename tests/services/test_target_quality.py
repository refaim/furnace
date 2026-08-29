from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from itertools import combinations, pairwise
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

from furnace.core.models import EncodeResult, MetricPool, MetricScores, VideoParams
from furnace.core.target_quality import (
    PROBE_WINDOW_MIN_GAP_SECONDS,
    KnobSearchResult,
    ProbeWindowOutcome,
    ProbeWindowSelection,
    TargetSpec,
    pool_grain_windows,
    probe_windows,
)
from furnace.services import target_quality as target_quality_service
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
        self.bitrate_window_lengths: list[float] = []

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
        self.bitrate_window_lengths.append(window_s)
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


class _WindowProbe:
    def __init__(self, score_fn: Callable[[int, int], float]) -> None:
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
        match = re.search(r"_(\d+)\.mkv$", input_path.name)
        index = int(match.group(1)) if match is not None else -1
        self.calls.append({"input": input_path, "qvbr": qvbr, "metric": metric})
        return self._score_fn(qvbr, index)


def _service(
    rc: int = 0, score_fn: Callable[[int], float] | None = None
) -> tuple[TargetQualityService, _FakeExtractor, _FakeProbe]:
    extractor = _FakeExtractor(rc=rc)
    probe = _FakeProbe(score_fn if score_fn is not None else (lambda q: 120.0 - q))
    return TargetQualityService(extractor, probe), extractor, probe


class TestTargetQualitySearch:
    def test_extracts_ten_gop_windows(self, tmp_path: Path) -> None:
        service, extractor, _probe = _service()
        vp = make_video_params(source_width=1920, source_height=1080)
        service.search(Path("movie.mkv"), vp, duration_s=7200.0, work_dir=tmp_path)
        assert len(extractor.calls) == 10
        assert all(c["frames"] == vp.gop for c in extractor.calls)

    def test_grown_windows_are_nonoverlapping_inside_the_runtime_edges(self, tmp_path: Path) -> None:
        extractor = _FakeExtractor()
        probe = _WindowProbe(lambda _knob, index: 81.0 + (-10.0 if index % 2 == 0 else 10.0))
        service = TargetQualityService(extractor, probe)
        vp = make_video_params(source_width=1920, source_height=1080)
        service.search(Path("movie.mkv"), vp, duration_s=7200.0, work_dir=tmp_path)
        starts = [cast("float", c["start_s"]) for c in extractor.calls]
        window_s = vp.gop / (vp.fps_num / vp.fps_den)
        assert len(starts) > 10
        assert all(start >= 7200.0 * 0.06 for start in starts)
        assert all(start <= 7200.0 * 0.94 - window_s for start in starts)
        assert all(abs(left - right) >= window_s for left, right in combinations(starts, 2))

    def test_returns_knob_search_result_in_bounds(self, tmp_path: Path) -> None:
        service, _extractor, _probe = _service()
        vp = make_video_params(source_width=1920, source_height=1080)
        result = service.search(Path("m.mkv"), vp, duration_s=7200.0, work_dir=tmp_path)
        assert isinstance(result, KnobSearchResult)
        assert 16 <= result.knob <= 44

    def test_full_pass_extracts_one_bounded_window(self, tmp_path: Path) -> None:
        service, extractor, probe = _service()
        vp = make_video_params(source_width=1920, source_height=1080)
        service.search(Path("short.mkv"), vp, duration_s=5.8, work_dir=tmp_path)
        assert len(extractor.calls) == 1
        assert extractor.calls[0]["start_s"] == 0.0
        assert extractor.calls[0]["frames"] == round(5.8 * 24)
        assert probe.calls
        assert all(c["input"] == tmp_path / "tq_window_full.mkv" for c in probe.calls)

    def test_smooth_short_source_stops_before_full_pass(self, tmp_path: Path) -> None:
        service, extractor, probe = _service()
        vp = make_video_params(source_width=1920, source_height=1080)
        service.search(Path("short.mkv"), vp, duration_s=125.0, work_dir=tmp_path)
        assert 0 < len(extractor.calls) < 10
        assert all(call["frames"] == vp.gop for call in extractor.calls)
        assert all(c["input"] != tmp_path / "tq_window_full.mkv" for c in probe.calls)

    def test_packing_shortfall_keeps_measured_windows(self, tmp_path: Path) -> None:
        extractor = _FakeExtractor()
        probe = _WindowProbe(lambda _knob, index: 81.0 + float(index))
        service = TargetQualityService(extractor, probe)
        events: list[str] = []
        service.search(
            Path("short.mkv"),
            make_video_params(source_width=1920, source_height=1080),
            duration_s=300.0,
            work_dir=tmp_path,
            on_event=events.append,
        )
        first_knob = cast("int", probe.calls[0]["qvbr"])
        first_knob_calls = [call for call in probe.calls if call["qvbr"] == first_knob]
        assert 10 < len(extractor.calls) < 16
        assert len(first_knob_calls) == len(extractor.calls)
        assert all(call["input"] != tmp_path / "tq_window_full.mkv" for call in probe.calls)
        assert any("packing limit" in event for event in events)
        assert not any("85%" in event for event in events)

    def test_initial_packing_limit_under_coverage_keeps_measured_windows(self, tmp_path: Path) -> None:
        extractor = _FakeExtractor()
        probe = _WindowProbe(lambda _knob, index: 81.0 + (-10.0 if index % 2 == 0 else 10.0))
        service = TargetQualityService(extractor, probe)
        events: list[str] = []
        service.search(
            Path("short.mkv"),
            make_video_params(source_width=1920, source_height=1080),
            duration_s=80.0,
            work_dir=tmp_path,
            on_event=events.append,
        )
        assert extractor.calls
        assert all(call["output"] != tmp_path / "tq_window_full.mkv" for call in extractor.calls)
        assert any("initial sample" in event and "packing limit" in event for event in events)
        assert not any("85%" in event for event in events)

    def test_short_gop_windows_keep_long_gop_start_separation(self, tmp_path: Path) -> None:
        service, extractor, _probe = _service()
        service.search(
            Path("movie.mkv"),
            make_video_params(source_width=1920, source_height=1080),
            duration_s=300.0,
            work_dir=tmp_path,
        )
        starts = sorted(cast("float", call["start_s"]) for call in extractor.calls)
        assert all(right - left >= 18.0 for left, right in pairwise(starts))

    def test_reachable_coverage_switches_to_a_bounded_full_pass(self, tmp_path: Path) -> None:
        extractor = _FakeExtractor()
        probe = _WindowProbe(lambda _knob, index: 81.0 if index == -1 else 81.0 + (-10.0 if index % 2 == 0 else 10.0))
        service = TargetQualityService(extractor, probe)
        events: list[str] = []
        service.search(
            Path("short.mkv"),
            make_video_params(source_width=1920, source_height=1080),
            duration_s=5.8,
            work_dir=tmp_path,
            on_event=events.append,
        )
        assert extractor.calls[-1]["output"] == tmp_path / "tq_window_full.mkv"
        assert extractor.calls[-1]["frames"] == round(5.8 * 24)
        assert not any("packing limit" in event for event in events)

    def test_growth_coverage_replaces_initial_windows(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        offsets = tuple(float(index * 6) for index in range(10))

        def select_windows(
            _duration_s: float,
            *,
            count: int,
            **_kwargs: float | None,
        ) -> ProbeWindowSelection:
            if count == 10:
                return ProbeWindowSelection(offsets, offsets, ProbeWindowOutcome.READY)
            return ProbeWindowSelection((), (), ProbeWindowOutcome.COVERAGE_LIMIT)

        monkeypatch.setattr(target_quality_service, "probe_windows", select_windows)
        extractor = _FakeExtractor()
        probe = _WindowProbe(lambda _knob, index: 81.0 + (-10.0 if index % 2 == 0 else 10.0))
        events: list[str] = []
        TargetQualityService(extractor, probe).search(
            Path("short.mkv"),
            make_video_params(source_width=1920, source_height=1080),
            duration_s=90.0,
            work_dir=tmp_path,
            on_event=events.append,
        )
        assert len(extractor.calls) == 11
        assert extractor.calls[-1]["output"] == tmp_path / "tq_window_full.mkv"
        assert any("switching to 1 full-pass window" in event for event in events)

    def test_extraction_failure_raises(self, tmp_path: Path) -> None:
        service, _extractor, _probe = _service(rc=1)
        vp = make_video_params(source_width=1920, source_height=1080)
        with pytest.raises(RuntimeError, match="window extraction failed"):
            service.search(Path("m.mkv"), vp, duration_s=7200.0, work_dir=tmp_path)

    def test_full_pass_extraction_failure_raises(self, tmp_path: Path) -> None:
        service, _extractor, _probe = _service(rc=1)
        vp = make_video_params(source_width=1920, source_height=1080)
        with pytest.raises(RuntimeError, match="full-pass"):
            service.search(Path("m.mkv"), vp, duration_s=5.8, work_dir=tmp_path)

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
        assert inputs_for_first == {tmp_path / f"tq_window_{i}.mkv" for i in range(10)}

    def test_frames_from_fractional_fps(self, tmp_path: Path) -> None:
        service, extractor, _probe = _service()
        vp = make_video_params(
            source_width=1920,
            source_height=1080,
            fps_num=24000,
            fps_den=1001,
        )
        service.search(Path("m.mkv"), vp, duration_s=7200.0, work_dir=tmp_path)
        assert all(c["frames"] == vp.gop for c in extractor.calls)
        expected = probe_windows(
            7200.0,
            count=10,
            window_s=vp.gop / (vp.fps_num / vp.fps_den),
        )
        assert [c["start_s"] for c in extractor.calls] == sorted(expected.offsets)

    def test_non_grain_path_mean_pools_nonuniform_windows(self, tmp_path: Path) -> None:
        extractor = _FakeExtractor()
        probe = _WindowProbe(lambda knob, index: 120.0 - knob + index * 0.2)
        service = TargetQualityService(extractor, probe)
        vp = make_video_params(source_width=1920, source_height=1080)
        result = service.search(Path("m.mkv"), vp, duration_s=7200.0, work_dir=tmp_path)
        scores = [90.0 + index * 0.2 for index in range(10)]
        assert result.probes[0][1] == pytest.approx(sum(scores) / len(scores))
        assert result.probes[0][1] != pytest.approx(pool_grain_windows(scores))

    def test_adds_one_batch_then_stops_when_standard_error_is_low(self, tmp_path: Path) -> None:
        extractor = _FakeExtractor()
        probe = _WindowProbe(lambda knob, index: 120.0 - knob + (-3.0 if index % 2 == 0 else 3.0))
        service = TargetQualityService(extractor, probe)
        vp = make_video_params(source_width=1920, source_height=1080)
        service.search(Path("m.mkv"), vp, duration_s=7200.0, work_dir=tmp_path)
        tolerance = 0.81
        assert 3.0 / math.sqrt(9) > tolerance
        assert 3.0 / math.sqrt(15) <= tolerance
        assert len(extractor.calls) == 16
        assert sum(call["qvbr"] == 30 for call in probe.calls) == 16

    def test_latches_window_count_after_the_first_knob(self, tmp_path: Path) -> None:
        extractor = _FakeExtractor()
        probe = _WindowProbe(
            lambda knob, index: (
                120.0
                - knob
                + ((-3.0 if index % 2 == 0 else 3.0) if knob == 30 else (-10.0 if index % 2 == 0 else 10.0))
            )
        )
        service = TargetQualityService(extractor, probe)
        service.search(
            Path("m.mkv"),
            make_video_params(source_width=1920, source_height=1080),
            duration_s=7200.0,
            work_dir=tmp_path,
        )
        counts = Counter(cast("int", call["qvbr"]) for call in probe.calls)
        assert len(extractor.calls) == 16
        assert len(counts) > 1
        assert set(counts.values()) == {16}

    def test_caps_ragged_sources_and_pairs_every_knob(self, tmp_path: Path) -> None:
        extractor = _FakeExtractor()
        probe = _WindowProbe(lambda knob, index: 120.0 - knob + (-10.0 if index % 2 == 0 else 10.0))
        service = TargetQualityService(extractor, probe)
        vp = make_video_params(source_width=1920, source_height=1080)
        service.search(Path("m.mkv"), vp, duration_s=7200.0, work_dir=tmp_path)
        counts = Counter(cast("int", call["qvbr"]) for call in probe.calls)
        assert len(extractor.calls) == 34
        assert len(counts) > 1
        assert set(counts.values()) == {34}
        expected_inputs = {tmp_path / f"tq_window_{i}.mkv" for i in range(34)}
        for knob in counts:
            assert {call["input"] for call in probe.calls if call["qvbr"] == knob} == expected_inputs


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

    def test_grain_vbr_keeps_hardest_windows_as_starting_set(self, tmp_path: Path) -> None:
        offsets = [500.0 + 500.0 * i for i in range(12)]
        low = {2500.0, 4000.0}
        bitrates = [(o, 1.0 if o in low else 100.0) for o in offsets]
        extractor = _FakeExtractor(bitrates=bitrates)
        grain_enc = MagicMock()
        grain_enc.encode.return_value = EncodeResult(return_code=0, encoder_settings="svt")
        metrics = MagicMock()
        metrics.measure.side_effect = lambda reference, distorted, **kw: MetricScores(
            ssimulacra2=100.0 - _crf_from_obu(distorted) + 100.0 * math.sin(_win_from_obu(distorted))
        )
        svc = TargetQualityService(extractor, MagicMock(), grain_encoder=grain_enc, metrics=metrics)
        svc.search(Path("grain.mkv"), _grain_vp(), duration_s=7200.0, work_dir=tmp_path)
        starting = sorted(cast("float", c["start_s"]) for c in extractor.calls[:10])
        assert starting == sorted(o for o in offsets if o not in low)
        assert len(extractor.calls) == 34

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
        even = probe_windows(7200.0, count=10, window_s=5.0)
        assert extracted == sorted(even.offsets)

    def test_grain_vbr_padding_shortfall_is_nonfatal_through_search(self, tmp_path: Path) -> None:
        duration_s = 300.0
        bitrates = [(61.95146326756722, 100.0), (198.6682509226761, 80.0)]
        extractor = _FakeExtractor(bitrates=bitrates)
        grain_enc = MagicMock()
        grain_enc.encode.return_value = EncodeResult(return_code=0, encoder_settings="svt")
        metrics = MagicMock()
        metrics.measure.side_effect = lambda reference, distorted, **kw: MetricScores(
            ssimulacra2=70.0 - _crf_from_obu(distorted) + 30.0 + _win_from_obu(distorted)
        )
        service = TargetQualityService(
            extractor,
            MagicMock(),
            grain_encoder=grain_enc,
            metrics=metrics,
        )
        result = service.search(
            Path("grain.mkv"),
            _grain_vp(),
            duration_s=duration_s,
            work_dir=tmp_path,
        )
        assert result.probes
        assert 10 < len(extractor.calls) < 34
        assert len(extractor.reference_calls) == len(extractor.calls)

    def test_grain_vbr_padding_keeps_eighteen_second_start_separation(self, tmp_path: Path) -> None:
        duration_s = 300.0
        vp = _grain_vp()
        window_s = vp.gop / (vp.fps_num / vp.fps_den)
        selection = probe_windows(
            duration_s,
            count=10,
            window_s=window_s,
            candidate_count=34,
        )
        hard_start = selection.candidates[0] + PROBE_WINDOW_MIN_GAP_SECONDS / 2.0
        extractor = _FakeExtractor(bitrates=[(60.0, 1.0), (hard_start, 100.0)])
        grain_enc = MagicMock()
        grain_enc.encode.return_value = EncodeResult(return_code=0, encoder_settings="svt")
        metrics = MagicMock()
        metrics.measure.side_effect = lambda reference, distorted, **kw: MetricScores(
            ssimulacra2=100.0 - _crf_from_obu(distorted)
        )
        service = TargetQualityService(
            extractor,
            MagicMock(),
            grain_encoder=grain_enc,
            metrics=metrics,
        )
        service.search(Path("grain.mkv"), vp, duration_s=duration_s, work_dir=tmp_path)
        starts = [cast("float", call["start_s"]) for call in extractor.calls]
        assert len(starts) == 10
        assert all(
            abs(left - right) >= PROBE_WINDOW_MIN_GAP_SECONDS for left, right in combinations(starts, 2)
        )

    def test_grain_vbr_classification_keeps_eighteen_second_bins(self, tmp_path: Path) -> None:
        extractor = _FakeExtractor(bitrates=[(100.0, 90.0), (200.0, 110.0)])
        grain_enc = MagicMock()
        grain_enc.encode.return_value = EncodeResult(return_code=0, encoder_settings="svt")
        metrics = MagicMock()
        metrics.measure.side_effect = lambda reference, distorted, **kw: MetricScores(
            ssimulacra2=100.0 - _crf_from_obu(distorted)
        )
        service = TargetQualityService(
            extractor,
            MagicMock(),
            grain_encoder=grain_enc,
            metrics=metrics,
        )
        service.search(Path("grain.mkv"), _grain_vp(), duration_s=7200.0, work_dir=tmp_path)
        assert extractor.bitrate_window_lengths == [18.0]

    def test_grain_percentile_pooled_across_windows(self, tmp_path: Path) -> None:
        extractor = _FakeExtractor()
        grain_enc = MagicMock()
        grain_enc.encode.return_value = EncodeResult(return_code=0, encoder_settings="svt")
        metrics = MagicMock()
        metrics.measure.side_effect = lambda reference, distorted, **kw: MetricScores(
            ssimulacra2=(100.0 - _crf_from_obu(distorted)) - 0.5 * _win_from_obu(distorted)
        )
        svc = TargetQualityService(extractor, MagicMock(), grain_encoder=grain_enc, metrics=metrics)
        result = svc.search(Path("grain.mkv"), _grain_vp(), duration_s=7200.0, work_dir=tmp_path)
        scores = [70.0 - 0.5 * i for i in range(10)]
        assert result.probes[0][1] == pytest.approx(pool_grain_windows(scores))
        assert result.probes[0][1] == pytest.approx(66.4)

    def test_grain_adapts_to_the_window_cap(self, tmp_path: Path) -> None:
        extractor = _FakeExtractor()
        grain_enc = MagicMock()
        grain_enc.encode.return_value = EncodeResult(return_code=0, encoder_settings="svt")
        metrics = MagicMock()
        metrics.measure.side_effect = lambda reference, distorted, **kw: MetricScores(
            ssimulacra2=100.0 - _crf_from_obu(distorted) + 100.0 * math.sin(_win_from_obu(distorted))
        )
        service = TargetQualityService(
            extractor,
            MagicMock(),
            grain_encoder=grain_enc,
            metrics=metrics,
        )
        service.search(Path("grain.mkv"), _grain_vp(), duration_s=7200.0, work_dir=tmp_path)
        first_knob = _crf_from_obu(metrics.measure.call_args_list[0].args[1])
        first_knob_calls = [
            call for call in metrics.measure.call_args_list if _crf_from_obu(call.args[1]) == first_knob
        ]
        assert len(extractor.calls) == 34
        assert len(extractor.reference_calls) == 34
        assert len(first_knob_calls) == 34

    def test_grain_growth_uses_percentile_dispersion(self, tmp_path: Path) -> None:
        extractor = _FakeExtractor()
        grain_enc = MagicMock()
        grain_enc.encode.return_value = EncodeResult(return_code=0, encoder_settings="svt")
        metrics = MagicMock()
        metrics.measure.side_effect = lambda reference, distorted, **kw: MetricScores(
            ssimulacra2=100.0 - _crf_from_obu(distorted) + 0.7 * (_win_from_obu(distorted) % 10)
        )
        service = TargetQualityService(
            extractor,
            MagicMock(),
            grain_encoder=grain_enc,
            metrics=metrics,
        )
        service.search(Path("grain.mkv"), _grain_vp(), duration_s=7200.0, work_dir=tmp_path)
        first_knob = _crf_from_obu(metrics.measure.call_args_list[0].args[1])
        first_knob_calls = [
            call for call in metrics.measure.call_args_list if _crf_from_obu(call.args[1]) == first_knob
        ]
        assert len(first_knob_calls) > 10

    def test_grain_reachable_coverage_switches_to_a_bounded_full_pass(self, tmp_path: Path) -> None:
        extractor = _FakeExtractor()
        grain_enc = MagicMock()
        grain_enc.encode.return_value = EncodeResult(return_code=0, encoder_settings="svt")
        metrics = MagicMock()
        metrics.measure.side_effect = lambda reference, distorted, **kw: MetricScores(
            ssimulacra2=70.0 + 10.0 * (_win_from_obu(distorted) % 10)
        )
        service = TargetQualityService(
            extractor,
            MagicMock(),
            grain_encoder=grain_enc,
            metrics=metrics,
        )
        service.search(Path("grain.mkv"), _grain_vp(), duration_s=5.8, work_dir=tmp_path)
        assert len(extractor.calls) == 1
        assert len(extractor.reference_calls) == 1
        assert extractor.calls[-1]["output"] == tmp_path / "tq_window_full.mkv"

    def test_grain_growth_coverage_replaces_initial_references(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        offsets = tuple(float(index * 6) for index in range(10))

        def select_windows(
            _duration_s: float,
            *,
            count: int,
            **_kwargs: float | None,
        ) -> ProbeWindowSelection:
            if count == 10:
                return ProbeWindowSelection(offsets, offsets, ProbeWindowOutcome.READY)
            return ProbeWindowSelection((), (), ProbeWindowOutcome.COVERAGE_LIMIT)

        monkeypatch.setattr(target_quality_service, "probe_windows", select_windows)
        extractor = _FakeExtractor()
        grain_enc = MagicMock()
        grain_enc.encode.return_value = EncodeResult(return_code=0, encoder_settings="svt")
        metrics = MagicMock()
        metrics.measure.side_effect = lambda reference, distorted, **kw: MetricScores(
            ssimulacra2=70.0 if _win_from_obu(distorted) == 0 else 70.0 + 10.0 * (_win_from_obu(distorted) % 2)
        )
        TargetQualityService(
            extractor,
            MagicMock(),
            grain_encoder=grain_enc,
            metrics=metrics,
        ).search(Path("grain.mkv"), _grain_vp(), duration_s=90.0, work_dir=tmp_path)
        assert len(extractor.calls) == 11
        assert len(extractor.reference_calls) == 11
        assert extractor.reference_calls[-1]["input"] == tmp_path / "tq_window_full.mkv"

    def test_grain_full_pass_single_window(self, tmp_path: Path) -> None:
        svc, _enc, metrics = _grain_service()
        result = svc.search(Path("grain.mkv"), _grain_vp(), duration_s=5.8, work_dir=tmp_path)
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

    def test_extension_reports_count_and_reason(self) -> None:
        events: list[str] = []
        narrator = _SearchNarrator(
            emit=events.append,
            label="QVBR",
            metric="SSIMULACRA2",
            window_count=10,
            pool_word="mean",
        )
        narrator.extending(30, 16, standard_error=1.25, tolerance=0.81)
        narrator.window(30, 11, 79.4)
        assert events == [
            "QVBR 30: spread too high (standard error 1.25 > tolerance 0.81); extending to 16 windows",
            "QVBR 30: window 11/16 = 79.4",
        ]

    def test_full_pass_reports_the_sampling_change(self) -> None:
        events: list[str] = []
        narrator = _SearchNarrator(
            emit=events.append,
            label="QVBR",
            metric="SSIMULACRA2",
            window_count=16,
            pool_word="mean",
        )
        narrator.full_pass(30)
        narrator.window(30, 1, 81.0)
        assert events == [
            "QVBR 30: probe windows would cover at least 85% of runtime; switching to 1 full-pass window",
            "QVBR 30: window 1/1 = 81.0",
        ]


class TestAdaptiveProbe:
    def test_no_progress_is_treated_as_a_packing_limit(self) -> None:
        spec = TargetSpec(
            metric="ssimulacra2",
            target_lo=80.0,
            target_hi=82.0,
            knob_lo=16,
            knob_hi=44,
            max_probes=4,
            window_count=2,
            window_batch=1,
            max_window_count=4,
            sampling_tolerance=0.1,
        )
        windows = [Path("a.mkv"), Path("b.mkv")]
        events: list[str] = []
        narrator = _SearchNarrator(
            emit=events.append,
            label="QVBR",
            metric="SSIMULACRA2",
            window_count=2,
            pool_word="mean",
        )
        growth_calls: list[int] = []
        outcomes = iter([ProbeWindowOutcome.READY])

        def grow_windows(count: int) -> ProbeWindowOutcome:
            growth_calls.append(count)
            return next(outcomes)

        probe = TargetQualityService._adaptive_probe_fn(
            spec,
            windows,
            narrator,
            grow_windows,
            score_window=lambda _knob, index, _window: float(index * 10),
            pool_scores=lambda scores: sum(scores) / len(scores),
        )
        assert probe(30) == pytest.approx(5.0)
        assert growth_calls == [3]
        assert any("packing limit" in event for event in events)


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
        assert "Probing QVBR -> SSIMULACRA2 ~81.0 (10 windows, mean-pooled)" in events
        assert "QVBR 30: window 1/10 = 90.0" in events
        assert "QVBR 30: window 10/10 = 90.0" in events
        assert "QVBR 30 -> SSIMULACRA2 90.0" in events

    def test_nvenc_search_narrates_each_extension(self, tmp_path: Path) -> None:
        extractor = _FakeExtractor()
        probe = _WindowProbe(lambda knob, index: 120.0 - knob + (-10.0 if index % 2 == 0 else 10.0))
        service = TargetQualityService(extractor, probe)
        events: list[str] = []
        service.search(
            Path("m.mkv"),
            make_video_params(source_width=1920, source_height=1080),
            duration_s=7200.0,
            work_dir=tmp_path,
            on_event=events.append,
        )
        extensions = [event for event in events if "spread too high" in event]
        assert [event.rsplit(" ", 2)[-2] for event in extensions] == ["16", "22", "28", "34"]

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


class TestSmartStart:
    def _vp(self, bitrate: int) -> VideoParams:
        return make_video_params(source_width=3840, source_height=1920, source_bitrate=bitrate)

    def test_first_search_starts_at_the_bracket_midpoint(self, tmp_path: Path) -> None:
        service, _extractor, probe = _service(score_fn=lambda q: 111.0 - q)
        service.search(Path("a.mkv"), self._vp(30_000_000), duration_s=7200.0, work_dir=tmp_path)
        assert probe.calls[0]["qvbr"] == 30

    def test_comparable_source_starts_at_the_solved_knob(self, tmp_path: Path) -> None:
        service, _extractor, probe = _service(score_fn=lambda q: 111.0 - q)
        first = service.search(Path("a.mkv"), self._vp(30_000_000), duration_s=7200.0, work_dir=tmp_path)
        assert first.hit is True
        probe.calls.clear()
        service.search(Path("b.mkv"), self._vp(29_000_000), duration_s=7200.0, work_dir=tmp_path)
        assert probe.calls[0]["qvbr"] == first.knob

    def test_distant_bitrate_starts_from_scratch(self, tmp_path: Path) -> None:
        service, _extractor, probe = _service(score_fn=lambda q: 111.0 - q)
        service.search(Path("a.mkv"), self._vp(30_000_000), duration_s=7200.0, work_dir=tmp_path)
        probe.calls.clear()
        service.search(Path("b.mkv"), self._vp(45_000_000), duration_s=7200.0, work_dir=tmp_path)
        assert probe.calls[0]["qvbr"] == 30

    def test_other_source_class_starts_from_scratch(self, tmp_path: Path) -> None:
        service, _extractor, probe = _service(score_fn=lambda q: 111.0 - q)
        service.search(Path("a.mkv"), self._vp(30_000_000), duration_s=7200.0, work_dir=tmp_path)
        probe.calls.clear()
        hdr = make_video_params(
            source_width=3840,
            source_height=1920,
            source_bitrate=30_000_000,
            color_transfer="smpte2084",
            color_matrix="bt2020nc",
        )
        service.search(Path("b.mkv"), hdr, duration_s=7200.0, work_dir=tmp_path)
        assert probe.calls[0]["qvbr"] == 30

    def test_a_missed_search_is_not_remembered(self, tmp_path: Path) -> None:
        service, _extractor, probe = _service(score_fn=lambda q: 40.0 - q * 0.01)
        missed = service.search(Path("a.mkv"), self._vp(30_000_000), duration_s=7200.0, work_dir=tmp_path)
        assert missed.hit is False
        probe.calls.clear()
        service.search(Path("b.mkv"), self._vp(30_000_000), duration_s=7200.0, work_dir=tmp_path)
        assert probe.calls[0]["qvbr"] == 30

    def test_seed_is_narrated(self, tmp_path: Path) -> None:
        service, _extractor, _probe = _service(score_fn=lambda q: 111.0 - q)
        first = service.search(Path("a.mkv"), self._vp(30_000_000), duration_s=7200.0, work_dir=tmp_path)
        events: list[str] = []
        service.search(
            Path("b.mkv"),
            self._vp(30_000_000),
            duration_s=7200.0,
            work_dir=tmp_path,
            on_event=events.append,
        )
        assert f"Starting from QVBR {first.knob} (comparable source already solved)" in events
