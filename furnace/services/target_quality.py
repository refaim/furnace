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


def _noop(_message: str) -> None: ...


def _windows(n: int) -> str:
    return f"{n} window" if n == 1 else f"{n} windows"


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
    ) -> KnobSearchResult:
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
        active_indices = {(offset.family, offset.index) for offset in offset_selection.offsets}

        def grow_windows(count: int) -> ProbeWindowOutcome:
            desired = _offsets_for(count)
            if desired.outcome is ProbeWindowOutcome.COVERAGE_LIMIT:
                windows[:] = self._prepare_windows(source, vp, None, duration_s, work_dir)
                active_indices.clear()
                return desired.outcome
            new_offsets = tuple(
                offset for offset in desired.offsets if (offset.family, offset.index) not in active_indices
            )
            windows.extend(
                self._prepare_windows(
                    source,
                    vp,
                    new_offsets,
                    duration_s,
                    work_dir,
                    start_index=len(windows),
                )
            )
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
        probe_fn = (
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
        result = search_knob(
            probe_fn,
            target_lo=spec.target_lo,
            target_hi=spec.target_hi,
            lo=spec.knob_lo,
            hi=spec.knob_hi,
            max_probes=spec.max_probes,
            seed=seed,
        )
        if result.hit:
            self._seeds.remember(key, source_bitrate=vp.source_bitrate, knob=result.knob)
        return result

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
    ) -> Callable[[int], float]:
        def score_window(knob: int, index: int, window: Path) -> float:
            return self._inline_probe.probe(
                window,
                work_dir / f"tq_probe_q{knob}_w{index}.obu",
                vp,
                qvbr=knob,
                metric=spec.metric,
            )

        return self._adaptive_probe_fn(
            spec,
            windows,
            narrator,
            grow_windows,
            score_window=score_window,
            pool_scores=statistics.fmean,
            initial_packing_limited=initial_packing_limited,
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
    ) -> Callable[[int], float]:
        if self._grain_encoder is None or self._metrics is None:
            raise RuntimeError("grain target-quality requires an SVT encoder and a metrics adapter")
        grain_encoder = self._grain_encoder
        metrics = self._metrics
        wanted = frozenset({spec.metric})

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

        def score_window(knob: int, index: int, window: Path) -> float:
            obu = work_dir / f"tq_grain_q{knob}_w{index}.obu"
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

        return self._adaptive_probe_fn(
            spec,
            windows,
            narrator,
            grow_with_references,
            score_window=score_window,
            pool_scores=pool_grain_windows,
            initial_packing_limited=initial_packing_limited,
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
            narrator.result(knob, pooled)
            return pooled

        return probe_fn
