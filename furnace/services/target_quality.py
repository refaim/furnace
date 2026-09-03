from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from furnace.core.models import MetricPool, VideoParams
from furnace.core.ports import Encoder, InlineQualityProbe, PerceptualMetrics, WindowExtractor
from furnace.core.target_quality import (
    PROBE_WINDOW_MIN_GAP_SECONDS,
    KnobSearchResult,
    ProbeWindowOutcome,
    SeedMemory,
    TargetSpec,
    interior_windows,
    pool_grain_windows,
    pooled_standard_error,
    probe_windows,
    resolve_target,
    search_knob,
    seed_key,
    select_hard_window_indices,
    source_is_variable_bitrate,
)

_BITRATE_ANALYSIS_WINDOW_SECONDS = 18.0
_REPAIR_STEP = 4
_REPAIR_GAIN_FRACTION = 0.25
_MAX_REPAIR_ROUNDS = 2
_MAX_REPAIR_HOLES = 8
_MIN_REPAIR_HOLES = 3


def _noop(_message: str) -> None: ...


def _windows(n: int) -> str:
    return f"{n} window" if n == 1 else f"{n} windows"


def _holes(n: int) -> str:
    return f"{n} hole" if n == 1 else f"{n} holes"


@dataclass(frozen=True, slots=True)
class QualityHole:
    start_s: float
    score: float


@dataclass(frozen=True, slots=True)
class RepairedHole:
    start_s: float
    score_before: float
    score_after: float


@dataclass(frozen=True, slots=True)
class SaturatedHole:
    start_s: float
    score_before: float
    score_after: float
    expected_gain: float


@dataclass(frozen=True, slots=True)
class UnverifiedHole:
    start_s: float
    score_before: float
    score_after: float
    reason: str


@dataclass(frozen=True, slots=True)
class TargetQualitySearchResult(KnobSearchResult):
    initial_knob: int
    holes: tuple[QualityHole, ...]
    repaired: tuple[RepairedHole, ...]
    saturated: tuple[SaturatedHole, ...]
    unverified: tuple[UnverifiedHole, ...]
    repair_adopted: bool


@dataclass(slots=True)
class _ProbeRunner:
    probe: Callable[[int], float]
    repair_score_window: Callable[[int, int, Path], float]
    score_records: dict[int, tuple[float, ...]]


def _find_holes(
    starts: Sequence[float],
    scores: Sequence[float],
    *,
    floor: float,
) -> tuple[QualityHole, ...]:
    indices = _hole_indices(scores, floor=floor)
    return tuple(QualityHole(start_s=starts[index], score=scores[index]) for index in indices)


def _hole_indices(scores: Sequence[float], *, floor: float) -> tuple[int, ...]:
    return tuple(index for index, score in enumerate(scores) if score < floor)


def _is_responsive(*, gain: float, expected_gain: float) -> bool:
    return expected_gain > 0.0 and gain >= _REPAIR_GAIN_FRACTION * expected_gain


@dataclass(slots=True)
class _SearchNarrator:
    emit: Callable[[str], None]
    label: str
    metric: str
    window_count: int
    pool_word: str

    def opening(self, centre: float) -> None:
        self.emit(
            f"Probing {self.label} -> {self.metric} ~{centre:.1f} "
            f"({_windows(self.window_count)}, {self.pool_word}-pooled)"
        )

    def window(self, knob: int, index: int, score: float) -> None:
        self.emit(f"{self.label} {knob}: window {index}/{self.window_count} = {score:.1f}")

    def extending(self, knob: int, count: int, *, standard_error: float, tolerance: float) -> None:
        self.window_count = count
        self.emit(
            f"{self.label} {knob}: spread too high "
            f"(standard error {standard_error:.2f} > tolerance {tolerance:.2f}); "
            f"extending to {_windows(count)}"
        )

    def full_pass(self, knob: int) -> None:
        self.window_count = 1
        self.emit(
            f"{self.label} {knob}: probe windows would cover at least 85% of runtime; switching to 1 full-pass window"
        )

    def initial_packing_limit(self, requested: int, actual: int) -> None:
        self.window_count = actual
        self.emit(
            f"Probe-sequence packing limit reduced initial sample "
            f"from {_windows(requested)} to {_windows(actual)}"
        )

    def packing_limit(
        self,
        knob: int,
        count: int,
        *,
        standard_error: float,
        tolerance: float,
    ) -> None:
        self.window_count = count
        self.emit(
            f"{self.label} {knob}: spread remains high "
            f"(standard error {standard_error:.2f} > tolerance {tolerance:.2f}); "
            f"probe-sequence packing limit is {_windows(count)}"
        )

    def result(self, knob: int, pooled: float) -> None:
        self.emit(f"{self.label} {knob} -> {self.metric} {pooled:.1f}")

    def seeded(self, knob: int) -> None:
        self.emit(f"Starting from {self.label} {knob} (comparable source already solved)")

    def holes(self, count: int, *, worst: float, floor: float) -> None:
        self.emit(f"{count} {'hole' if count == 1 else 'holes'} found: worst score {worst:.1f}, floor {floor:.1f}")

    def repair_probe(
        self,
        knob: int,
        scores: Sequence[tuple[float, float]],
        *,
        expected_gain: float,
    ) -> None:
        changes = ", ".join(f"{before:.1f} -> {after:.1f}" for before, after in scores)
        self.emit(f"Repair probe {self.label} {knob}: {changes}; expected gain {expected_gain:.1f}")

    def repair_skipped(self, reason: str) -> None:
        self.emit(f"Repair skipped: {reason}")

    def repair_decision(self, knob: int, *, cleared: int, total: int, required: int, adopted: bool) -> None:
        decision = "adopted" if adopted else "not adopted"
        self.emit(
            f"Repair round {self.label} {knob} {decision}: "
            f"{cleared}/{total} holes cleared floor; {required} required"
        )

    def verdict(
        self,
        initial_knob: int,
        final_knob: int,
        *,
        repaired: int,
        saturated: int,
        unverified: int,
    ) -> None:
        details = []
        if repaired:
            details.append(f"{repaired} repaired")
        if saturated:
            details.append(f"{saturated} {'saturated hole' if saturated == 1 else 'saturated holes'}")
        if unverified:
            details.append(f"{unverified} {'unverified hole' if unverified == 1 else 'unverified holes'}")
        summary = ", ".join(details)
        if final_knob < initial_knob:
            self.emit(
                f"Repair verdict: {summary}; lowering {self.label} from {initial_knob} to {final_knob}; "
                f"repair clearing threshold met"
            )
            return
        self.emit(f"Repair verdict: {summary}; keeping {self.label} {initial_knob}")


@dataclass(frozen=True, slots=True)
class _ProbeOffset:
    family: str
    index: int
    start_s: float


@dataclass(frozen=True, slots=True)
class _OffsetSelection:
    offsets: tuple[_ProbeOffset, ...]
    outcome: ProbeWindowOutcome


class TargetQualityService:
    def __init__(
        self,
        extractor: WindowExtractor,
        inline_probe: InlineQualityProbe,
        *,
        grain_encoder: Encoder | None = None,
        metrics: PerceptualMetrics | None = None,
    ) -> None:
        self._extractor = extractor
        self._inline_probe = inline_probe
        self._grain_encoder = grain_encoder
        self._metrics = metrics
        self._seeds = SeedMemory()

    def can_search(self, vp: VideoParams) -> bool:
        if not vp.grain:
            return True
        return self._grain_encoder is not None and self._metrics is not None

    def search(
        self,
        source: Path,
        vp: VideoParams,
        duration_s: float,
        work_dir: Path,
        *,
        on_event: Callable[[str], None] | None = None,
    ) -> TargetQualitySearchResult:
        spec = resolve_target(vp)
        fps = vp.fps_num / vp.fps_den
        window_s = vp.gop / fps
        grain_candidates: list[tuple[float, float]] | None = None

        def _offsets_for(count: int) -> _OffsetSelection:
            nonlocal grain_candidates
            selection = probe_windows(
                duration_s,
                count=count,
                window_s=window_s,
                candidate_count=spec.max_window_count,
            )
            if selection.outcome is ProbeWindowOutcome.COVERAGE_LIMIT:
                return _OffsetSelection((), selection.outcome)
            even_offsets = tuple(
                _ProbeOffset("even", index, start_s) for index, start_s in enumerate(selection.offsets)
            )
            if not vp.grain:
                return _OffsetSelection(even_offsets, selection.outcome)
            if grain_candidates is None:
                grain_candidates = self._grain_window_candidates(source, duration_s)
            padding_offsets = tuple(
                _ProbeOffset("even", index, start_s) for index, start_s in enumerate(selection.candidates)
            )
            offsets = self._select_grain_offsets(
                grain_candidates,
                even_offsets,
                padding_offsets,
                count=len(even_offsets),
                max_hard_count=spec.max_window_count,
                min_gap_s=max(window_s, PROBE_WINDOW_MIN_GAP_SECONDS),
            )
            outcome = ProbeWindowOutcome.PACKING_LIMIT if len(offsets) < count else selection.outcome
            return _OffsetSelection(offsets, outcome)

        offset_selection = _offsets_for(spec.window_count)
        windows = self._prepare_windows(
            source,
            vp,
            (None if offset_selection.outcome is ProbeWindowOutcome.COVERAGE_LIMIT else offset_selection.offsets),
            duration_s,
            work_dir,
        )
        window_starts = (
            [0.0]
            if offset_selection.outcome is ProbeWindowOutcome.COVERAGE_LIMIT
            else sorted(offset.start_s for offset in offset_selection.offsets)
        )
        active_indices = {(offset.family, offset.index) for offset in offset_selection.offsets}

        def grow_windows(count: int) -> ProbeWindowOutcome:
            desired = _offsets_for(count)
            if desired.outcome is ProbeWindowOutcome.COVERAGE_LIMIT:
                windows[:] = self._prepare_windows(source, vp, None, duration_s, work_dir)
                window_starts[:] = [0.0]
                active_indices.clear()
                return desired.outcome
            new_offsets = tuple(
                offset for offset in desired.offsets if (offset.family, offset.index) not in active_indices
            )
            ordered_new_offsets = tuple(sorted(new_offsets, key=lambda offset: offset.start_s))
            windows.extend(
                self._prepare_windows(
                    source,
                    vp,
                    ordered_new_offsets,
                    duration_s,
                    work_dir,
                    start_index=len(windows),
                )
            )
            window_starts.extend(offset.start_s for offset in ordered_new_offsets)
            active_indices.update((offset.family, offset.index) for offset in new_offsets)
            return desired.outcome

        narrator = _SearchNarrator(
            emit=on_event or _noop,
            label="CRF" if vp.grain else "QVBR",
            metric=spec.metric.upper(),
            window_count=len(windows),
            pool_word="worst-case" if vp.grain else "mean",
        )
        narrator.opening((spec.target_lo + spec.target_hi) / 2.0)
        initial_packing_limited = offset_selection.outcome is ProbeWindowOutcome.PACKING_LIMIT
        if initial_packing_limited:
            narrator.initial_packing_limit(spec.window_count, len(windows))
        key = seed_key(vp, spec)
        seed = self._seeds.suggest(key, source_bitrate=vp.source_bitrate)
        if seed is not None:
            narrator.seeded(seed)
        runner = (
            self._grain_probe_fn(
                vp,
                spec,
                windows,
                work_dir,
                narrator,
                grow_windows,
                initial_packing_limited=initial_packing_limited,
            )
            if vp.grain
            else self._inline_probe_fn(
                vp,
                spec,
                windows,
                work_dir,
                narrator,
                grow_windows,
                initial_packing_limited=initial_packing_limited,
            )
        )
        search_result = search_knob(
            runner.probe,
            target_lo=spec.target_lo,
            target_hi=spec.target_hi,
            lo=spec.knob_lo,
            hi=spec.knob_hi,
            max_probes=spec.max_probes,
            seed=seed,
        )
        result = self._apply_worst_window_guard(
            search_result,
            spec,
            windows,
            window_starts,
            runner,
            narrator,
        )
        if result.hit:
            self._seeds.remember(key, source_bitrate=vp.source_bitrate, knob=search_result.knob)
        return result

    @staticmethod
    def _healthy_reference_index(scores: dict[int, float], *, floor: float) -> int | None:
        healthy = {index: score for index, score in scores.items() if score >= floor}
        if not healthy:
            return None
        median = statistics.median(healthy.values())
        return min(healthy, key=lambda index: (abs(healthy[index] - median), index))

    @staticmethod
    def _score_repair_indices(
        knob: int,
        indices: Sequence[int],
        windows: Sequence[Path],
        runner: _ProbeRunner,
        repair_records: dict[int, dict[int, float]],
    ) -> dict[int, float]:
        search_scores = runner.score_records.get(knob)
        records = repair_records.setdefault(knob, {})
        for index in indices:
            if search_scores is not None:
                records[index] = search_scores[index]
            elif index not in records:
                records[index] = runner.repair_score_window(knob, index, windows[index])
        return {index: records[index] for index in indices}

    @classmethod
    def _repair_round(
        cls,
        active: tuple[int, ...],
        current_scores: dict[int, float],
        repair_knob: int,
        reference_index: int | None,
        windows: Sequence[Path],
        runner: _ProbeRunner,
        repair_records: dict[int, dict[int, float]],
        narrator: _SearchNarrator,
    ) -> tuple[dict[int, float], float, int] | None:
        if reference_index is None:
            return None
        indices = (*active, reference_index)
        after_scores = cls._score_repair_indices(
            repair_knob,
            indices,
            windows,
            runner,
            repair_records,
        )
        cached_scores = runner.score_records.get(repair_knob)
        current_after_scores = dict(enumerate(cached_scores)) if cached_scores is not None else after_scores
        expected_gain = after_scores[reference_index] - current_scores[reference_index]
        narrator.repair_probe(
            repair_knob,
            tuple((current_scores[index], after_scores[index]) for index in active),
            expected_gain=expected_gain,
        )
        return current_after_scores, expected_gain, reference_index

    @classmethod
    def _apply_worst_window_guard(
        cls,
        search_result: KnobSearchResult,
        spec: TargetSpec,
        windows: Sequence[Path],
        window_starts: Sequence[float],
        runner: _ProbeRunner,
        narrator: _SearchNarrator,
    ) -> TargetQualitySearchResult:
        initial_knob = search_result.knob
        initial_scores = runner.score_records[initial_knob]
        hole_indices = _hole_indices(initial_scores, floor=spec.floor)
        holes = tuple(QualityHole(start_s=window_starts[index], score=initial_scores[index]) for index in hole_indices)
        if not holes:
            return TargetQualitySearchResult(
                knob=initial_knob,
                score=search_result.score,
                hit=search_result.hit,
                probes=search_result.probes,
                initial_knob=initial_knob,
                holes=(),
                repaired=(),
                saturated=(),
                unverified=(),
                repair_adopted=False,
            )

        narrator.holes(len(holes), worst=min(hole.score for hole in holes), floor=spec.floor)
        if not spec.repairs_holes:
            unverified = tuple(
                UnverifiedHole(
                    start_s=hole.start_s,
                    score_before=hole.score,
                    score_after=hole.score,
                    reason="repair disabled for HDR",
                )
                for hole in holes
            )
            narrator.verdict(
                initial_knob,
                initial_knob,
                repaired=0,
                saturated=0,
                unverified=len(unverified),
            )
            return TargetQualitySearchResult(
                knob=initial_knob,
                score=search_result.score,
                hit=search_result.hit,
                probes=search_result.probes,
                initial_knob=initial_knob,
                holes=holes,
                repaired=(),
                saturated=(),
                unverified=unverified,
                repair_adopted=False,
            )

        repair_launched = len(holes) >= _MIN_REPAIR_HOLES or any(
            hole.score < spec.deep_hole_threshold for hole in holes
        )
        if not repair_launched:
            reason = f"below repair threshold: {_holes(len(holes))}, none deep"
            unverified = tuple(
                UnverifiedHole(
                    start_s=hole.start_s,
                    score_before=hole.score,
                    score_after=hole.score,
                    reason=reason,
                )
                for hole in holes
            )
            narrator.repair_skipped(reason)
            narrator.verdict(
                initial_knob,
                initial_knob,
                repaired=0,
                saturated=0,
                unverified=len(unverified),
            )
            return TargetQualitySearchResult(
                knob=initial_knob,
                score=search_result.score,
                hit=search_result.hit,
                probes=search_result.probes,
                initial_knob=initial_knob,
                holes=holes,
                repaired=(),
                saturated=(),
                unverified=unverified,
                repair_adopted=False,
            )

        repair_indices = tuple(
            sorted(hole_indices, key=lambda index: (initial_scores[index], index))[:_MAX_REPAIR_HOLES]
        )
        repair_index_set = set(repair_indices)
        active = repair_indices
        current_scores = dict(enumerate(initial_scores))
        reported_scores = dict(current_scores)
        reference_index = cls._healthy_reference_index(current_scores, floor=spec.floor)
        repair_records: dict[int, dict[int, float]] = {}
        classification: dict[int, tuple[float, float, float]] = {}
        unverified_reasons = {
            index: f"not probed (cap of {_MAX_REPAIR_HOLES} holes)"
            for index in hole_indices
            if index not in repair_index_set
        }
        current_knob = initial_knob
        final_knob = initial_knob
        adopted_rounds = 0

        for round_index in range(_MAX_REPAIR_ROUNDS):
            repair_knob = max(spec.knob_lo, current_knob - _REPAIR_STEP)
            if repair_knob == current_knob:
                unverified_reasons.update(dict.fromkeys(active, "already at the knob floor"))
                break
            round_result = cls._repair_round(
                active,
                current_scores,
                repair_knob,
                reference_index,
                windows,
                runner,
                repair_records,
                narrator,
            )
            if round_result is None:
                unverified_reasons.update(dict.fromkeys(active, "no healthy reference window"))
                break
            after_scores, expected_gain, measured_reference_index = round_result
            if expected_gain <= 0.0:
                unverified_reasons.update(dict.fromkeys(active, "healthy reference did not improve"))
                break
            for index in active:
                reported_scores[index] = after_scores[index]
                classification[index] = (current_scores[index], after_scores[index], expected_gain)
            cleared = tuple(index for index in active if after_scores[index] >= spec.floor)
            required = (len(active) + 1) // 2
            adopted = len(cleared) >= required
            narrator.repair_decision(
                repair_knob,
                cleared=len(cleared),
                total=len(active),
                required=required,
                adopted=adopted,
            )
            if not adopted:
                reason = (
                    f"repair round not adopted: {len(cleared)}/{len(active)} holes cleared floor; "
                    f"{required} required"
                )
                for index in active:
                    before_score, after_score, reference_gain = classification[index]
                    if after_score >= spec.floor:
                        unverified_reasons[index] = reason
                    elif _is_responsive(
                        gain=after_score - before_score,
                        expected_gain=reference_gain,
                    ):
                        unverified_reasons[index] = f"still below floor; {reason}"
                break
            final_knob = repair_knob
            current_knob = repair_knob
            for index in (*active, measured_reference_index):
                current_scores[index] = after_scores[index]
            adopted_rounds = round_index + 1
            active = tuple(index for index in active if after_scores[index] < spec.floor)
            if not active:
                break

        repaired = tuple(
            RepairedHole(
                start_s=window_starts[index],
                score_before=initial_scores[index],
                score_after=reported_scores[index],
            )
            for index in hole_indices
            if current_scores[index] >= spec.floor
        )
        saturated_indices = {
            index
            for index, (before_score, after_score, expected_gain) in classification.items()
            if current_scores[index] < spec.floor
            and index not in unverified_reasons
            and not _is_responsive(
                gain=after_score - before_score,
                expected_gain=expected_gain,
            )
        }
        saturated_result = tuple(
            SaturatedHole(
                start_s=window_starts[index],
                score_before=initial_scores[index],
                score_after=reported_scores[index],
                expected_gain=classification[index][2],
            )
            for index in hole_indices
            if index in saturated_indices
        )
        unverified = tuple(
            UnverifiedHole(
                start_s=window_starts[index],
                score_before=initial_scores[index],
                score_after=reported_scores[index],
                reason=unverified_reasons.get(
                    index,
                    f"still below floor after {adopted_rounds} rounds",
                ),
            )
            for index in hole_indices
            if current_scores[index] < spec.floor and index not in saturated_indices
        )
        narrator.verdict(
            initial_knob,
            final_knob,
            repaired=len(repaired),
            saturated=len(saturated_result),
            unverified=len(unverified),
        )
        return TargetQualitySearchResult(
            knob=final_knob,
            score=search_result.score,
            hit=search_result.hit,
            probes=search_result.probes,
            initial_knob=initial_knob,
            holes=holes,
            repaired=repaired,
            saturated=saturated_result,
            unverified=unverified,
            repair_adopted=final_knob < initial_knob,
        )

    def _grain_window_candidates(
        self,
        source: Path,
        duration_s: float,
    ) -> list[tuple[float, float]]:
        return interior_windows(
            self._extractor.window_bitrates(source, _BITRATE_ANALYSIS_WINDOW_SECONDS),
            duration_s=duration_s,
            window_s=_BITRATE_ANALYSIS_WINDOW_SECONDS,
        )

    @staticmethod
    def _select_grain_offsets(
        candidates: list[tuple[float, float]],
        even_offsets: tuple[_ProbeOffset, ...],
        padding_offsets: tuple[_ProbeOffset, ...],
        *,
        count: int,
        max_hard_count: int,
        min_gap_s: float,
    ) -> tuple[_ProbeOffset, ...]:
        if not candidates:
            return even_offsets
        if source_is_variable_bitrate([kbytes for _, kbytes in candidates]):
            hard_indices = select_hard_window_indices(candidates, count=max_hard_count)
            selected_indices = sorted(
                hard_indices,
                key=lambda index: -candidates[index][1],
            )[:count]
            chosen = [_ProbeOffset("hard", index, candidates[index][0]) for index in selected_indices]
            hard_offsets = [candidates[index][0] for index in hard_indices]
            for offset in padding_offsets:
                if len(chosen) >= count:
                    break
                if all(abs(offset.start_s - existing) >= min_gap_s for existing in hard_offsets) and all(
                    abs(offset.start_s - existing.start_s) >= min_gap_s for existing in chosen
                ):
                    chosen.append(offset)
            return tuple(chosen)
        return even_offsets

    def _prepare_windows(
        self,
        source: Path,
        vp: VideoParams,
        offsets: Sequence[_ProbeOffset] | None,
        duration_s: float,
        work_dir: Path,
        *,
        start_index: int = 0,
    ) -> list[Path]:
        fps = vp.fps_num / vp.fps_den
        if offsets is None:
            frames = max(1, round(duration_s * fps))
            window = work_dir / "tq_window_full.mkv"
            rc = self._extractor.extract_window(source, window, start_s=0.0, frames=frames)
            if rc != 0:
                raise RuntimeError(f"probe window extraction failed (rc={rc}) for full-pass")
            return [window]
        windows: list[Path] = []
        for i, offset in enumerate(sorted(offsets, key=lambda item: item.start_s), start=start_index):
            start = offset.start_s
            window = work_dir / f"tq_window_{i}.mkv"
            rc = self._extractor.extract_window(source, window, start_s=start, frames=vp.gop)
            if rc != 0:
                raise RuntimeError(f"probe window extraction failed (rc={rc}) at {start:.1f}s")
            windows.append(window)
        return windows

    def _inline_probe_fn(
        self,
        vp: VideoParams,
        spec: TargetSpec,
        windows: list[Path],
        work_dir: Path,
        narrator: _SearchNarrator,
        grow_windows: Callable[[int], ProbeWindowOutcome],
        *,
        initial_packing_limited: bool,
    ) -> _ProbeRunner:
        score_records: dict[int, tuple[float, ...]] = {}

        def score_window(knob: int, index: int, window: Path) -> float:
            return self._inline_probe.probe(
                window,
                work_dir / f"tq_probe_q{knob}_w{index}.obu",
                vp,
                qvbr=knob,
                metric=spec.metric,
            )

        def repair_score_window(knob: int, index: int, window: Path) -> float:
            return self._inline_probe.probe(
                window,
                work_dir / f"tq_repair_q{knob}_w{index}.obu",
                vp,
                qvbr=knob,
                metric=spec.metric,
            )

        return _ProbeRunner(
            probe=self._adaptive_probe_fn(
                spec,
                windows,
                narrator,
                grow_windows,
                score_window=score_window,
                pool_scores=statistics.fmean,
                score_records=score_records,
                initial_packing_limited=initial_packing_limited,
            ),
            repair_score_window=repair_score_window,
            score_records=score_records,
        )

    def _grain_probe_fn(
        self,
        vp: VideoParams,
        spec: TargetSpec,
        windows: list[Path],
        work_dir: Path,
        narrator: _SearchNarrator,
        grow_windows: Callable[[int], ProbeWindowOutcome],
        *,
        initial_packing_limited: bool,
    ) -> _ProbeRunner:
        if self._grain_encoder is None or self._metrics is None:
            raise RuntimeError("grain target-quality requires an SVT encoder and a metrics adapter")
        grain_encoder = self._grain_encoder
        metrics = self._metrics
        wanted = frozenset({spec.metric})
        score_records: dict[int, tuple[float, ...]] = {}

        references: list[Path] = []
        self._build_grain_references(vp, windows, references, work_dir)

        def grow_with_references(count: int) -> ProbeWindowOutcome:
            previous_count = len(windows)
            outcome = grow_windows(count)
            if outcome is ProbeWindowOutcome.COVERAGE_LIMIT:
                references.clear()
                previous_count = 0
            self._build_grain_references(
                vp,
                windows[previous_count:],
                references,
                work_dir,
                start_index=previous_count,
            )
            return outcome

        def encode_and_score(knob: int, index: int, window: Path, obu: Path) -> float:
            result = grain_encoder.encode(window, obu, vp, cq_override=knob)
            if result.return_code != 0:
                raise RuntimeError(f"grain probe encode failed (rc={result.return_code}) at crf={knob}")
            measured = metrics.measure(
                references[index],
                obu,
                matrix=vp.color_matrix,
                fps_num=vp.fps_num,
                fps_den=vp.fps_den,
                pool=MetricPool.LOW,
                metrics=wanted,
            )
            score = getattr(measured, spec.metric)
            if score is None:
                raise RuntimeError(f"grain probe could not be scored at crf={knob}")
            return float(score)

        def score_window(knob: int, index: int, window: Path) -> float:
            return encode_and_score(knob, index, window, work_dir / f"tq_grain_q{knob}_w{index}.obu")

        def repair_score_window(knob: int, index: int, window: Path) -> float:
            return encode_and_score(
                knob,
                index,
                window,
                work_dir / f"tq_repair_grain_q{knob}_w{index}.obu",
            )

        return _ProbeRunner(
            probe=self._adaptive_probe_fn(
                spec,
                windows,
                narrator,
                grow_with_references,
                score_window=score_window,
                pool_scores=pool_grain_windows,
                score_records=score_records,
                initial_packing_limited=initial_packing_limited,
            ),
            repair_score_window=repair_score_window,
            score_records=score_records,
        )

    def _build_grain_references(
        self,
        vp: VideoParams,
        windows: list[Path],
        references: list[Path],
        work_dir: Path,
        *,
        start_index: int = 0,
    ) -> None:
        for index, window in enumerate(windows, start=start_index):
            reference = work_dir / f"tq_ref_w{index}.mkv"
            rc = self._extractor.build_reference(window, reference, vp)
            if rc != 0:
                raise RuntimeError(f"grain probe reference build failed (rc={rc}) for window {index}")
            references.append(reference)

    @staticmethod
    def _adaptive_probe_fn(
        spec: TargetSpec,
        windows: list[Path],
        narrator: _SearchNarrator,
        grow_windows: Callable[[int], ProbeWindowOutcome],
        *,
        score_window: Callable[[int, int, Path], float],
        pool_scores: Callable[[list[float]], float],
        score_records: dict[int, tuple[float, ...]] | None = None,
        initial_packing_limited: bool = False,
    ) -> Callable[[int], float]:
        sized = False
        tolerance = spec.sampling_tolerance

        def probe_fn(knob: int) -> float:
            nonlocal sized
            scores: list[float] = []
            packing_limited = initial_packing_limited
            while True:
                while len(scores) < len(windows):
                    index = len(scores)
                    score = score_window(knob, index, windows[index])
                    scores.append(score)
                    narrator.window(knob, index + 1, score)
                if sized or packing_limited or len(windows) == 1 or len(windows) >= spec.max_window_count:
                    break
                standard_error = pooled_standard_error(scores, pool_scores=pool_scores)
                if standard_error <= tolerance:
                    break
                count = min(len(windows) + spec.window_batch, spec.max_window_count)
                previous_count = len(windows)
                outcome = grow_windows(count)
                if outcome is ProbeWindowOutcome.COVERAGE_LIMIT:
                    scores.clear()
                    narrator.full_pass(knob)
                    continue
                if outcome is ProbeWindowOutcome.PACKING_LIMIT or len(windows) == previous_count:
                    packing_limited = True
                    narrator.packing_limit(
                        knob,
                        len(windows),
                        standard_error=standard_error,
                        tolerance=tolerance,
                    )
                else:
                    narrator.extending(
                        knob,
                        len(windows),
                        standard_error=standard_error,
                        tolerance=tolerance,
                    )
            sized = True
            pooled = pool_scores(scores)
            if score_records is not None:
                score_records[knob] = tuple(scores)
            narrator.result(knob, pooled)
            return pooled

        return probe_fn
