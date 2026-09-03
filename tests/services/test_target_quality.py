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
    resolve_target,
)
from furnace.services import target_quality as target_quality_service
from furnace.services.target_quality import (
    _MAX_REPAIR_HOLES,
    _MAX_REPAIR_ROUNDS,
    _REPAIR_GAIN_FRACTION,
    _REPAIR_STEP,
    TargetQualitySearchResult,
    TargetQualityService,
    UnverifiedHole,
    _find_holes,
    _is_responsive,
    _ProbeRunner,
    _SearchNarrator,
    _windows,
)
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
        output_path: Path,
        video_params: object,  # noqa: ARG002
        *,
        qvbr: int,
        metric: str,
    ) -> float:
        self.calls.append({"input": input_path, "output": output_path, "qvbr": qvbr, "metric": metric})
        return self._score_fn(qvbr)


class _WindowProbe:
    def __init__(self, score_fn: Callable[[int, int], float]) -> None:
        self._score_fn = score_fn
        self.calls: list[dict[str, object]] = []

    def probe(
        self,
        input_path: Path,
        output_path: Path,
        video_params: object,  # noqa: ARG002
        *,
        qvbr: int,
        metric: str,
    ) -> float:
        match = re.search(r"_(\d+)\.mkv$", input_path.name)
        index = int(match.group(1)) if match is not None else -1
        self.calls.append({"input": input_path, "output": output_path, "qvbr": qvbr, "metric": metric})
        return self._score_fn(qvbr, index)


def _service(
    rc: int = 0, score_fn: Callable[[int], float] | None = None
) -> tuple[TargetQualityService, _FakeExtractor, _FakeProbe]:
    extractor = _FakeExtractor(rc=rc)
    probe = _FakeProbe(score_fn if score_fn is not None else (lambda q: 120.0 - q))
    return TargetQualityService(extractor, probe), extractor, probe


def _window_index(path: Path) -> int:
    match = re.search(r"_(\d+)\.mkv$", path.name)
    assert match is not None
    return int(match.group(1))


def _run_guard(
    initial_scores: tuple[float, ...],
    repair_score: Callable[[int, int], float],
) -> tuple[TargetQualitySearchResult, list[tuple[int, int]], list[str]]:
    repair_calls: list[tuple[int, int]] = []

    def score(knob: int, index: int, _window: Path) -> float:
        repair_calls.append((knob, index))
        return repair_score(knob, index)

    runner = _ProbeRunner(
        probe=MagicMock(),
        repair_score_window=score,
        score_records={30: initial_scores},
    )
    events: list[str] = []
    result = TargetQualityService._apply_worst_window_guard(
        KnobSearchResult(
            knob=30,
            score=81.0,
            hit=True,
            probes=((30, 81.0),),
        ),
        resolve_target(make_video_params(source_width=1920, source_height=1080)),
        tuple(Path(f"w{index}.mkv") for index in range(len(initial_scores))),
        tuple(float(index) for index in range(len(initial_scores))),
        runner,
        _SearchNarrator(
            emit=events.append,
            label="QVBR",
            metric="SSIMULACRA2",
            window_count=len(initial_scores),
            pool_word="mean",
        ),
    )
    return result, repair_calls, events


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

    def test_grown_hole_starts_match_the_extracted_window_indices(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        order = (
            *reversed(range(10)),
            *reversed(range(10, 16)),
            *reversed(range(16, 22)),
            *reversed(range(22, 28)),
            *reversed(range(28, 34)),
        )

        def unsorted_windows(
            duration_s: float,
            *,
            count: int,
            window_s: float,
            candidate_count: int | None = None,
        ) -> ProbeWindowSelection:
            assert duration_s == 7200.0
            assert window_s > 0.0
            assert candidate_count == 34
            offsets = tuple(500.0 + index * 100.0 for index in order[:count])
            return ProbeWindowSelection(offsets, offsets, ProbeWindowOutcome.READY)

        def scores(knob: int, index: int) -> float:
            if index == 0:
                base = 60.0
            elif index == 10:
                base = 61.0
            elif index % 2 == 0:
                base = 70.0
            else:
                base = 92.0
            return base + (30 - knob) * 4.0

        monkeypatch.setattr(target_quality_service, "probe_windows", unsorted_windows)
        extractor = _FakeExtractor()
        result = TargetQualityService(extractor, _WindowProbe(scores)).search(
            Path("movie.mkv"),
            make_video_params(source_width=1920, source_height=1080),
            duration_s=7200.0,
            work_dir=tmp_path,
        )
        starts_by_index = {
            _window_index(cast("Path", call["output"])): cast("float", call["start_s"]) for call in extractor.calls
        }
        holes_by_score = {hole.score: hole for hole in result.holes}
        assert len(extractor.calls) > 10
        assert holes_by_score[60.0].start_s == starts_by_index[0]
        assert holes_by_score[61.0].start_s == starts_by_index[10]

    def test_returns_knob_search_result_in_bounds(self, tmp_path: Path) -> None:
        service, _extractor, _probe = _service()
        vp = make_video_params(source_width=1920, source_height=1080)
        result = service.search(Path("m.mkv"), vp, duration_s=7200.0, work_dir=tmp_path)
        assert isinstance(result, KnobSearchResult)
        assert isinstance(result, TargetQualitySearchResult)
        assert 16 <= result.knob <= 44

    def test_guard_does_no_extra_probes_without_holes(self, tmp_path: Path) -> None:
        service, _extractor, probe = _service(score_fn=lambda _knob: 81.0)
        result = service.search(
            Path("m.mkv"),
            make_video_params(source_width=1920, source_height=1080),
            duration_s=7200.0,
            work_dir=tmp_path,
        )
        assert len(probe.calls) == 10
        assert result.initial_knob == result.knob == 30
        assert result.holes == ()
        assert result.repaired == ()
        assert result.saturated == ()
        assert result.unverified == ()
        assert result.repair_adopted is False

    def test_hdr_holes_are_reported_without_repair(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        healthy = (9.5 * 10.0 - 8.0) / 9.0

        def scores(knob: int, index: int) -> float:
            base = 8.0 if index == 0 else healthy
            return base + (30 - knob) * 0.5

        monkeypatch.setattr(target_quality_service, "pooled_standard_error", lambda *_args, **_kwargs: 0.0)
        probe = _WindowProbe(scores)
        events: list[str] = []
        result = TargetQualityService(_FakeExtractor(), probe).search(
            Path("hdr.mkv"),
            make_video_params(
                source_width=3840,
                source_height=2160,
                color_transfer="smpte2084",
                color_matrix="bt2020nc",
            ),
            duration_s=7200.0,
            work_dir=tmp_path,
            on_event=events.append,
        )
        assert result.initial_knob == result.knob == 30
        assert len(result.holes) == 1
        assert result.repaired == ()
        assert result.saturated == ()
        assert len(result.unverified) == 1
        assert result.unverified[0].reason == "repair disabled for HDR"
        assert not any(cast("Path", call["output"]).name.startswith("tq_repair_") for call in probe.calls)
        assert events[-1] == "Repair verdict: 1 unverified hole; keeping QVBR 30"

    def test_reference_is_the_healthy_window_nearest_the_healthy_median(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_probe_windows = probe_windows

        def packing_limited_windows(
            duration_s: float,
            *,
            count: int,
            window_s: float,
            candidate_count: int | None = None,
        ) -> ProbeWindowSelection:
            selection = original_probe_windows(
                duration_s,
                count=count,
                window_s=window_s,
                candidate_count=candidate_count,
            )
            return ProbeWindowSelection(
                selection.offsets,
                selection.candidates,
                ProbeWindowOutcome.PACKING_LIMIT,
            )

        def scores(knob: int, index: int) -> float:
            at_search = (45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 132.0, 134.0, 136.0, 138.0)
            gain = 0.2 if index < 6 else 4.0
            return at_search[index] + gain * (30 - knob) / 4.0

        monkeypatch.setattr(target_quality_service, "probe_windows", packing_limited_windows)
        probe = _WindowProbe(scores)
        result = TargetQualityService(_FakeExtractor(), probe).search(
            Path("m.mkv"),
            make_video_params(source_width=1920, source_height=1080),
            duration_s=7200.0,
            work_dir=tmp_path,
        )
        repair_calls = [call for call in probe.calls if call["qvbr"] == 26]
        repair_indices = {_window_index(cast("Path", call["input"])) for call in repair_calls}
        assert repair_indices == {0, 1, 2, 3, 4, 5, 7}
        assert all(cast("Path", call["output"]).name.startswith("tq_repair_") for call in repair_calls)
        assert result.initial_knob == result.knob == 30
        assert len(result.saturated) == 6
        assert all(hole.expected_gain == pytest.approx(4.0) for hole in result.saturated)

    def test_nonpositive_measured_reference_gain_is_unverified_and_reuses_search_scores(
        self,
        tmp_path: Path,
    ) -> None:
        def scores(knob: int, index: int) -> float:
            if knob == 24:
                return 55.0 if index == 0 else 725.0 / 9.0
            if knob == 20:
                return 54.0 if index == 0 else 636.0 / 9.0
            return 63.0 if index == 0 else 50.0

        service = TargetQualityService(_FakeExtractor(), probe := _WindowProbe(scores))
        seeds = MagicMock()
        seeds.suggest.return_value = 24
        service._seeds = seeds
        result = service.search(
            Path("m.mkv"),
            make_video_params(source_width=1920, source_height=1080),
            duration_s=7200.0,
            work_dir=tmp_path,
        )
        assert result.initial_knob == result.knob == 24
        assert result.saturated == ()
        assert result.unverified == (
            UnverifiedHole(
                start_s=result.holes[0].start_s,
                score_before=55.0,
                score_after=55.0,
                reason="healthy reference did not improve",
            ),
        )
        assert not any(
            cast("Path", call["output"]).name.startswith("tq_repair_") for call in probe.calls if call["qvbr"] == 20
        )

    def test_second_round_keeps_original_reference_when_current_knob_is_cached(self) -> None:
        initial_scores = (59.0, 60.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0)
        cached_scores = (71.0, 66.0, 81.0, 82.0, 83.0, 95.0, 84.0, 85.0, 86.0, 87.0)
        repair_calls: list[int] = []

        def repair_score(knob: int, index: int, _window: Path) -> float:
            assert knob == 16
            repair_calls.append(index)
            return 71.0 if index == 1 else cached_scores[index] + 4.0

        runner = _ProbeRunner(
            probe=MagicMock(),
            repair_score_window=repair_score,
            score_records={24: initial_scores, 20: cached_scores},
        )
        narrator = _SearchNarrator(
            emit=lambda _event: None,
            label="QVBR",
            metric="SSIMULACRA2",
            window_count=10,
            pool_word="mean",
        )
        result = TargetQualityService._apply_worst_window_guard(
            KnobSearchResult(
                knob=24,
                score=73.5,
                hit=True,
                probes=((24, 73.5), (20, 82.9)),
            ),
            resolve_target(make_video_params(source_width=1920, source_height=1080)),
            tuple(Path(f"w{index}.mkv") for index in range(10)),
            tuple(float(index) for index in range(10)),
            runner,
            narrator,
        )
        assert result.knob == 16
        assert repair_calls == [1, 5]
        assert result.unverified == ()

    def test_large_hole_recovery_uses_measured_reference_gain(self, tmp_path: Path) -> None:
        healthy = (81.0 * 34.0 - 59.0) / 33.0

        def scores(knob: int, index: int) -> float:
            if index == 0:
                return 59.0 if knob == 30 else 75.0
            return healthy + (30 - knob)

        probe = _WindowProbe(scores)
        result = TargetQualityService(_FakeExtractor(), probe).search(
            Path("m.mkv"),
            make_video_params(source_width=1920, source_height=1080),
            duration_s=7200.0,
            work_dir=tmp_path,
        )
        assert result.initial_knob == 30
        assert result.knob == 26
        assert result.repaired[0].score_before == pytest.approx(59.0)
        assert result.repaired[0].score_after == pytest.approx(75.0)
        assert result.saturated == ()
        assert result.unverified == ()
        assert sum(cast("Path", call["output"]).name.startswith("tq_repair_") for call in probe.calls) == 2

    def test_second_round_keeps_the_original_never_hole_reference(self, tmp_path: Path) -> None:
        healthy = (81.0 * 34.0 - 105.0) / 32.0

        def scores(knob: int, index: int) -> float:
            if index == 0:
                return {30: 45.0, 26: 72.0, 22: 72.4}.get(knob, 72.4)
            if index == 1:
                return {30: 60.0, 26: 66.0, 22: 66.4}.get(knob, 66.4)
            return healthy + (30 - knob) * 0.5

        probe = _WindowProbe(scores)
        result = TargetQualityService(_FakeExtractor(), probe).search(
            Path("m.mkv"),
            make_video_params(source_width=1920, source_height=1080),
            duration_s=7200.0,
            work_dir=tmp_path,
        )
        round_two_indices = [
            _window_index(cast("Path", call["input"]))
            for call in probe.calls
            if call["qvbr"] == 22 and cast("Path", call["output"]).name.startswith("tq_repair_")
        ]
        assert round_two_indices == [1, 2]
        assert result.knob == 26
        assert tuple(hole.score_before for hole in result.repaired) == pytest.approx((45.0,))
        assert tuple(hole.score_after for hole in result.repaired) == pytest.approx((72.0,))
        assert tuple(hole.score_before for hole in result.saturated) == pytest.approx((60.0,))
        assert tuple(hole.score_after for hole in result.saturated) == pytest.approx((66.4,))
        assert result.unverified == ()

    def test_full_pass_hole_is_unverified_without_self_reference(self, tmp_path: Path) -> None:
        probe = _WindowProbe(lambda knob, _index: 55.0 if knob == 30 else 50.0)
        result = TargetQualityService(_FakeExtractor(), probe).search(
            Path("m.mkv"),
            make_video_params(source_width=1920, source_height=1080),
            duration_s=5.8,
            work_dir=tmp_path,
        )
        assert result.initial_knob == result.knob == 30
        assert result.repaired == ()
        assert result.saturated == ()
        assert result.unverified[0].reason == "no healthy reference window"
        assert sum(call["qvbr"] == 26 for call in probe.calls) == 0

    def test_knob_floor_marks_holes_unverified_without_duplicate_probes(self, tmp_path: Path) -> None:
        healthy = (81.0 * 34.0 - 59.0) / 33.0
        probe = _WindowProbe(lambda _knob, index: 59.0 if index == 0 else healthy)
        extractor = _FakeExtractor()
        service = TargetQualityService(extractor, probe)
        seeds = MagicMock()
        seeds.suggest.return_value = 16
        service._seeds = seeds
        result = service.search(
            Path("m.mkv"),
            make_video_params(source_width=1920, source_height=1080),
            duration_s=7200.0,
            work_dir=tmp_path,
        )
        assert result.initial_knob == result.knob == 16
        assert len(result.unverified) == 1
        assert sum(call["qvbr"] == 16 for call in probe.calls) == len(extractor.calls)
        assert not any(cast("Path", call["output"]).name.startswith("tq_repair_") for call in probe.calls)
        assert {hole.reason for hole in result.unverified} == {"already at the knob floor"}

    def test_clearing_holes_lowers_knob_twice_and_reports_last_measurements(self, tmp_path: Path) -> None:
        healthy = (81.0 * 34.0 - 110.0) / 32.0

        def scores(knob: int, index: int) -> float:
            if index == 0:
                return {30: 45.0, 26: 71.0, 22: 75.0}.get(knob, 75.0)
            if index == 1:
                return {30: 65.0, 26: 66.0, 22: 72.0}.get(knob, 72.0)
            return healthy + (30 - knob) * 0.65

        extractor = _FakeExtractor()
        probe = _WindowProbe(scores)
        events: list[str] = []
        result = TargetQualityService(extractor, probe).search(
            Path("m.mkv"),
            make_video_params(source_width=1920, source_height=1080),
            duration_s=7200.0,
            work_dir=tmp_path,
            on_event=events.append,
        )
        assert result.initial_knob == 30
        assert result.knob == 22
        assert tuple(hole.score for hole in result.holes) == pytest.approx((45.0, 65.0))
        assert tuple(hole.score_after for hole in result.repaired) == pytest.approx((71.0, 72.0))
        assert result.saturated == ()
        assert result.unverified == ()
        assert result.repair_adopted is True
        assert sum(call["qvbr"] == 26 for call in probe.calls) == 3
        assert sum(call["qvbr"] == 22 for call in probe.calls) == 2
        assert any("2 holes found" in event and "floor 70.0" in event for event in events)
        assert any("45.0 -> 71.0" in event and "65.0 -> 66.0" in event for event in events)
        assert any("repair clearing threshold met" in event.lower() and "30 to 22" in event for event in events)

    def test_saturated_hole_keeps_original_knob(self, tmp_path: Path) -> None:
        healthy = (81.0 * 34.0 - 59.0) / 33.0

        def scores(knob: int, index: int) -> float:
            if index == 0:
                return 59.0 + (30 - knob) * 0.125
            return healthy + (30 - knob) * 0.65

        probe = _WindowProbe(scores)
        result = TargetQualityService(_FakeExtractor(), probe).search(
            Path("m.mkv"),
            make_video_params(source_width=1920, source_height=1080),
            duration_s=7200.0,
            work_dir=tmp_path,
        )
        assert result.initial_knob == result.knob == 30
        assert result.repaired == ()
        assert result.saturated[0].score_before == pytest.approx(59.0)
        assert result.saturated[0].score_after == pytest.approx(59.5)
        assert result.saturated[0].expected_gain == pytest.approx(2.6)
        assert result.unverified == ()
        assert result.repair_adopted is False

    def test_repair_stops_after_two_adopted_rounds_with_one_hole_unresolved(self) -> None:
        def repair_score(knob: int, index: int) -> float:
            if knob == 26:
                return (71.0, 72.0, 60.0, 66.0, 85.0)[index]
            return (71.0, 68.0, 89.0)[index - 2]

        result, repair_calls, _events = _run_guard(
            (45.0, 50.0, 55.0, 65.0, 81.0),
            repair_score,
        )
        assert result.knob == 22
        assert repair_calls == [
            *((26, index) for index in range(5)),
            (22, 2),
            (22, 3),
            (22, 4),
        ]
        assert len(result.repaired) == 3
        assert result.saturated == ()
        assert result.unverified[0].score_before == pytest.approx(65.0)
        assert result.unverified[0].score_after == pytest.approx(68.0)
        assert result.unverified[0].reason == "still below floor after 2 rounds"

    def test_hole_can_remain_after_adoption_then_saturate_in_the_rejected_round(self) -> None:
        def repair_score(knob: int, index: int) -> float:
            if knob == 26:
                return (71.0, 62.0, 85.0)[index]
            return (62.1, 89.0)[index - 1]

        result, repair_calls, events = _run_guard(
            (45.0, 50.0, 81.0),
            repair_score,
        )
        assert result.knob == 26
        assert repair_calls == [(26, 0), (26, 1), (26, 2), (22, 1), (22, 2)]
        assert len(result.repaired) == 1
        assert result.saturated[0].score_before == pytest.approx(50.0)
        assert result.saturated[0].score_after == pytest.approx(62.1)
        assert result.unverified == ()
        assert events[-1] == (
            "Repair verdict: 1 repaired, 1 saturated hole; lowering QVBR from 30 to 26; "
            "repair clearing threshold met"
        )

    def test_repair_caps_each_round_to_the_eight_deepest_holes(self) -> None:
        initial_scores = (*tuple(float(score) for score in range(50, 61)), 81.0)
        repair_calls: list[tuple[int, int]] = []

        def repair_score(knob: int, index: int, _window: Path) -> float:
            repair_calls.append((knob, index))
            if index == 11:
                return 85.0 if knob == 26 else 89.0
            if knob == 26:
                return 71.0 + index if index < 4 else 61.0 + index
            return (71.0, 72.0, 68.0, 69.0)[index - 4]

        runner = _ProbeRunner(
            probe=MagicMock(),
            repair_score_window=repair_score,
            score_records={30: initial_scores},
        )
        narrator = _SearchNarrator(
            emit=lambda _event: None,
            label="QVBR",
            metric="SSIMULACRA2",
            window_count=12,
            pool_word="mean",
        )
        result = TargetQualityService._apply_worst_window_guard(
            KnobSearchResult(
                knob=30,
                score=81.0,
                hit=True,
                probes=((30, 81.0),),
            ),
            resolve_target(make_video_params(source_width=1920, source_height=1080)),
            tuple(Path(f"w{index}.mkv") for index in range(12)),
            tuple(float(index) for index in range(12)),
            runner,
            narrator,
        )
        assert len(result.holes) == 11
        assert [index for knob, index in repair_calls if knob == 26] == [*range(8), 11]
        assert [index for knob, index in repair_calls if knob == 22] == [4, 5, 6, 7, 11]
        assert len(repair_calls) == _MAX_REPAIR_HOLES + 6
        capped = tuple(hole for hole in result.unverified if hole.reason == "not probed (cap of 8 holes)")
        assert tuple(hole.score_before for hole in capped) == (58.0, 59.0, 60.0)

    def test_second_round_probes_only_uncleared_holes_and_keeps_last_measurements(self, tmp_path: Path) -> None:
        healthy = (81.0 * 34.0 - 105.0) / 32.0

        def scores(knob: int, index: int) -> float:
            if index == 9:
                return {30: 45.0, 26: 71.0, 22: 75.0}.get(knob, 75.0)
            if index == 10:
                return {30: 60.0, 26: 64.0, 22: 71.0}.get(knob, 71.0)
            return healthy + (30 - knob)

        probe = _WindowProbe(scores)
        result = TargetQualityService(_FakeExtractor(), probe).search(
            Path("m.mkv"),
            make_video_params(source_width=1920, source_height=1080),
            duration_s=7200.0,
            work_dir=tmp_path,
        )
        assert result.knob == 22
        assert tuple(hole.score_before for hole in result.repaired) == pytest.approx((45.0, 60.0))
        assert tuple(hole.score_after for hole in result.repaired) == pytest.approx((71.0, 71.0))
        assert result.saturated == ()
        assert sum(call["qvbr"] == 22 for call in probe.calls) == 2

    def test_saturated_expected_gain_matches_the_adopted_knob_interval(self, tmp_path: Path) -> None:
        healthy = (81.0 * 34.0 - 105.0) / 32.0

        def scores(knob: int, index: int) -> float:
            if index == 0:
                return {30: 45.0, 26: 71.0, 22: 75.0}.get(knob, 75.0)
            if index == 1:
                return {30: 60.0, 26: 66.0, 22: 66.1}.get(knob, 66.1)
            return {30: healthy, 26: healthy + 2.0, 22: healthy + 6.0}.get(knob, healthy + 6.0)

        result = TargetQualityService(_FakeExtractor(), _WindowProbe(scores)).search(
            Path("m.mkv"),
            make_video_params(source_width=1920, source_height=1080),
            duration_s=7200.0,
            work_dir=tmp_path,
        )
        assert result.knob == 26
        assert result.saturated[0].score_before == pytest.approx(60.0)
        assert result.saturated[0].score_after == pytest.approx(66.1)
        assert result.saturated[0].expected_gain == pytest.approx(4.0)
        assert result.unverified == ()

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

    def test_grain_hole_uses_the_same_repair_guard(self, tmp_path: Path) -> None:
        extractor = _FakeExtractor()
        grain_enc = MagicMock()
        grain_enc.encode.return_value = EncodeResult(return_code=0, encoder_settings="svt")
        metrics = MagicMock()

        def measured(_reference: Path, distorted: Path, **_kwargs: object) -> MetricScores:
            knob = _crf_from_obu(distorted)
            index = _win_from_obu(distorted)
            score = 50.0 + (30 - knob) * 3.0 if index == 0 else 70.0 + (30 - knob) * 0.65
            return MetricScores(ssimulacra2=score)

        metrics.measure.side_effect = measured
        result = TargetQualityService(
            extractor,
            MagicMock(),
            grain_encoder=grain_enc,
            metrics=metrics,
        ).search(Path("grain.mkv"), _grain_vp(), duration_s=7200.0, work_dir=tmp_path)
        assert result.initial_knob == 30
        assert result.knob == 26
        assert result.holes[0].score == pytest.approx(50.0)
        assert result.repaired[0].score_after == pytest.approx(62.0)
        assert sum(_crf_from_obu(call.args[1]) == 26 for call in metrics.measure.call_args_list) == 2
        repair_outputs = [call.args[1] for call in grain_enc.encode.call_args_list if _crf_from_obu(call.args[1]) == 26]
        assert all(output.name.startswith("tq_repair_") for output in repair_outputs)

    def test_grain_repair_clamps_to_the_sd_crf_floor(self, tmp_path: Path) -> None:
        extractor = _FakeExtractor()
        grain_enc = MagicMock()
        grain_enc.encode.return_value = EncodeResult(return_code=0, encoder_settings="svt")
        metrics = MagicMock()

        def measured(_reference: Path, distorted: Path, **_kwargs: object) -> MetricScores:
            knob = _crf_from_obu(distorted)
            index = _win_from_obu(distorted)
            if index == 0:
                return MetricScores(ssimulacra2=50.0 + (27 - knob) * 5.0)
            return MetricScores(ssimulacra2=70.0 + (27 - knob) * 2.0)

        metrics.measure.side_effect = measured
        service = TargetQualityService(
            extractor,
            MagicMock(),
            grain_encoder=grain_enc,
            metrics=metrics,
        )
        seeds = MagicMock()
        seeds.suggest.return_value = 27
        service._seeds = seeds
        result = service.search(Path("grain.mkv"), _grain_vp(), duration_s=7200.0, work_dir=tmp_path)
        measured_knobs = [_crf_from_obu(call.args[1]) for call in metrics.measure.call_args_list]
        assert result.initial_knob == result.knob == 27
        assert result.repaired == ()
        assert result.unverified[0].score_after == pytest.approx(55.0)
        assert result.unverified[0].reason == (
            "still below floor; repair round not adopted: 0/1 holes cleared floor; 1 required"
        )
        assert measured_knobs.count(26) == 2
        assert 23 not in measured_knobs

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

    def test_guard_narrates_saturated_verdict(self) -> None:
        events: list[str] = []
        narrator = _SearchNarrator(
            emit=events.append,
            label="QVBR",
            metric="SSIMULACRA2",
            window_count=10,
            pool_word="mean",
        )
        narrator.holes(2, worst=45.85, floor=70.0)
        narrator.repair_probe(26, ((45.85, 46.1), (59.0, 62.0)), expected_gain=2.6)
        narrator.verdict(30, 30, repaired=0, saturated=2, unverified=0)
        assert events == [
            "2 holes found: worst score 45.9, floor 70.0",
            "Repair probe QVBR 26: 45.9 -> 46.1, 59.0 -> 62.0; expected gain 2.6",
            "Repair verdict: 2 saturated holes; keeping QVBR 30",
        ]


class TestWorstWindowGuardRules:
    def test_constants_are_pinned(self) -> None:
        assert _REPAIR_STEP == 4
        assert pytest.approx(0.25) == _REPAIR_GAIN_FRACTION
        assert _MAX_REPAIR_ROUNDS == 2
        assert _MAX_REPAIR_HOLES == 8
        assert target_quality_service._MIN_REPAIR_HOLES == 3

    def test_two_shallow_holes_skip_repair(self) -> None:
        result, repair_calls, events = _run_guard(
            (65.0, 69.0, 81.0),
            lambda _knob, _index: pytest.fail("repair probe must not run"),
        )
        assert result.initial_knob == result.knob == 30
        assert repair_calls == []
        assert result.repaired == ()
        assert result.saturated == ()
        assert len(result.unverified) == 2
        assert {hole.reason for hole in result.unverified} == {
            "below repair threshold: 2 holes, none deep"
        }
        assert events[-2:] == [
            "Repair skipped: below repair threshold: 2 holes, none deep",
            "Repair verdict: 2 unverified holes; keeping QVBR 30",
        ]

    def test_deep_hole_launches_repair_and_floor_clearing_adopts(self) -> None:
        after = (71.0, 72.0, 281.0)
        result, repair_calls, events = _run_guard(
            (45.0, 65.0, 81.0),
            lambda _knob, index: after[index],
        )
        assert result.knob == 26
        assert repair_calls == [(26, 0), (26, 1), (26, 2)]
        assert len(result.repaired) == 2
        assert result.saturated == ()
        assert result.unverified == ()
        assert any("2/2 holes cleared floor" in event and "adopted" in event for event in events)

    def test_one_of_eight_clearing_does_not_adopt(self) -> None:
        after = (71.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 85.0)
        result, repair_calls, events = _run_guard(
            (50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 81.0),
            lambda _knob, index: after[index],
        )
        assert result.initial_knob == result.knob == 30
        assert repair_calls == [(26, index) for index in range(9)]
        assert result.repaired == ()
        assert result.saturated == ()
        assert len(result.unverified) == 8
        assert tuple(hole.score_after for hole in result.unverified) == after[:8]
        assert any("1/8 holes cleared floor" in event and "4 required" in event for event in events)

    def test_half_of_eight_adopts_but_one_of_four_does_not(self) -> None:
        round_one = (71.0, 72.0, 73.0, 74.0, 61.0, 62.0, 63.0, 64.0, 85.0)
        round_two = (75.0, 66.0, 67.0, 68.0, 89.0)

        def repair_score(knob: int, index: int) -> float:
            if knob == 26:
                return round_one[index]
            return round_two[index - 4] if index < 8 else round_two[-1]

        result, repair_calls, events = _run_guard(
            (50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 81.0),
            repair_score,
        )
        assert result.knob == 26
        assert repair_calls == [
            *((26, index) for index in range(9)),
            *((22, index) for index in range(4, 9)),
        ]
        assert len(result.repaired) == 4
        assert result.saturated == ()
        assert len(result.unverified) == 4
        assert tuple(hole.score_after for hole in result.unverified) == round_two[:4]
        assert any("4/8 holes cleared floor" in event and "adopted" in event for event in events)
        assert any("1/4 holes cleared floor" in event and "not adopted" in event for event in events)

    def test_three_shallow_holes_launch_repair(self) -> None:
        result, repair_calls, events = _run_guard(
            (60.0, 65.0, 69.0, 81.0),
            lambda _knob, index: (60.0, 65.0, 69.0, 85.0)[index],
        )
        assert result.knob == 30
        assert repair_calls == [(26, index) for index in range(4)]
        assert len(result.saturated) == 3
        assert result.unverified == ()
        assert any("0/3 holes cleared floor" in event and "not adopted" in event for event in events)

    def test_repair_measurements_are_reused_within_a_knob(self) -> None:
        calls: list[tuple[int, int]] = []

        def repair_score(knob: int, index: int, _window: Path) -> float:
            calls.append((knob, index))
            return 70.0 + index

        runner = _ProbeRunner(
            probe=MagicMock(),
            repair_score_window=repair_score,
            score_records={},
        )
        records: dict[int, dict[int, float]] = {}
        windows = (Path("w0.mkv"),)
        first = TargetQualityService._score_repair_indices(26, (0,), windows, runner, records)
        second = TargetQualityService._score_repair_indices(26, (0,), windows, runner, records)
        assert first == second == {0: 70.0}
        assert calls == [(26, 0)]

    def test_floor_comparison_is_strict(self) -> None:
        holes = _find_holes((12.0, 34.0), (69.9, 70.0), floor=70.0)
        assert tuple(hole.start_s for hole in holes) == (12.0,)

    def test_repairability_threshold_rejects_twenty_percent_gain(self) -> None:
        assert _is_responsive(gain=0.8, expected_gain=4.0) is False

    def test_repairability_threshold_accepts_fifty_percent_gain(self) -> None:
        assert _is_responsive(gain=2.0, expected_gain=4.0) is True

    def test_repairability_rejects_nonpositive_expected_gain(self) -> None:
        assert _is_responsive(gain=-1.0, expected_gain=-4.0) is False


class TestAdaptiveProbe:
    def test_no_progress_is_treated_as_a_packing_limit(self) -> None:
        spec = TargetSpec(
            metric="ssimulacra2",
            target_lo=80.0,
            target_hi=82.0,
            floor=70.0,
            deep_hole_threshold=60.0,
            repairs_holes=True,
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

    def test_comparable_source_is_seeded_with_the_search_knob_before_repair(self, tmp_path: Path) -> None:
        healthy = (81.0 * 34.0 - 59.0) / 33.0

        def scores(knob: int, index: int) -> float:
            if index == 0:
                return 59.0 if knob == 30 else 71.0
            return healthy + (30 - knob)

        probe = _WindowProbe(scores)
        service = TargetQualityService(_FakeExtractor(), probe)
        first = service.search(Path("a.mkv"), self._vp(30_000_000), duration_s=7200.0, work_dir=tmp_path)
        assert first.initial_knob == 30
        assert first.knob == 26
        probe.calls.clear()
        service.search(Path("b.mkv"), self._vp(29_000_000), duration_s=7200.0, work_dir=tmp_path)
        assert probe.calls[0]["qvbr"] == 30

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
