from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from furnace.core.models import MetricPool, VideoParams
from furnace.core.ports import Encoder, InlineQualityProbe, PerceptualMetrics, WindowExtractor
from furnace.core.target_quality import (
    PROBE_WINDOW_SECONDS,
    KnobSearchResult,
    SeedMemory,
    TargetSpec,
    interior_windows,
    pool_grain_windows,
    probe_windows,
    resolve_target,
    search_knob,
    seed_key,
    select_hard_windows,
    source_is_variable_bitrate,
)


def _noop(_message: str) -> None: ...


def _windows(n: int) -> str:
    return f"{n} window" if n == 1 else f"{n} windows"


@dataclass(frozen=True, slots=True)
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

    def result(self, knob: int, pooled: float) -> None:
        self.emit(f"{self.label} {knob} -> {self.metric} {pooled:.1f}")

    def seeded(self, knob: int) -> None:
        self.emit(f"Starting from {self.label} {knob} (comparable source already solved)")


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
        if vp.grain:
            offsets = self._grain_window_offsets(source, spec, duration_s)
        else:
            offsets = probe_windows(duration_s, count=spec.window_count, window_s=PROBE_WINDOW_SECONDS)
        windows = self._prepare_windows(source, vp, offsets, duration_s, work_dir)
        narrator = _SearchNarrator(
            emit=on_event or _noop,
            label="CRF" if vp.grain else "QVBR",
            metric=spec.metric.upper(),
            window_count=len(windows),
            pool_word="worst-case" if vp.grain else "mean",
        )
        narrator.opening((spec.target_lo + spec.target_hi) / 2.0)
        key = seed_key(vp, spec)
        seed = self._seeds.suggest(key, source_bitrate=vp.source_bitrate)
        if seed is not None:
            narrator.seeded(seed)
        probe_fn = (
            self._grain_probe_fn(vp, spec.metric, windows, work_dir, narrator)
            if vp.grain
            else self._inline_probe_fn(vp, spec.metric, windows, work_dir, narrator)
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

    def _grain_window_offsets(self, source: Path, spec: TargetSpec, duration_s: float) -> list[float] | None:
        even = probe_windows(duration_s, count=spec.window_count, window_s=PROBE_WINDOW_SECONDS)
        if even is None:
            return None
        candidates = interior_windows(
            self._extractor.window_bitrates(source, PROBE_WINDOW_SECONDS),
            duration_s=duration_s,
            window_s=PROBE_WINDOW_SECONDS,
        )
        if not candidates:
            return even
        if source_is_variable_bitrate([kbytes for _, kbytes in candidates]):
            return select_hard_windows(candidates, count=spec.window_count)
        return even

    def _prepare_windows(
        self,
        source: Path,
        vp: VideoParams,
        offsets: list[float] | None,
        duration_s: float,
        work_dir: Path,
    ) -> list[Path]:
        fps = vp.fps_num / vp.fps_den
        if offsets is None:
            frames = max(1, round(duration_s * fps))
            window = work_dir / "tq_window_full.mkv"
            rc = self._extractor.extract_window(source, window, start_s=0.0, frames=frames)
            if rc != 0:
                raise RuntimeError(f"probe window extraction failed (rc={rc}) for full-pass")
            return [window]
        frames = max(1, round(PROBE_WINDOW_SECONDS * fps))
        windows: list[Path] = []
        for i, start in enumerate(offsets):
            window = work_dir / f"tq_window_{i}.mkv"
            rc = self._extractor.extract_window(source, window, start_s=start, frames=frames)
            if rc != 0:
                raise RuntimeError(f"probe window extraction failed (rc={rc}) at {start:.1f}s")
            windows.append(window)
        return windows

    def _inline_probe_fn(
        self,
        vp: VideoParams,
        metric: str,
        windows: list[Path],
        work_dir: Path,
        narrator: _SearchNarrator,
    ) -> Callable[[int], float]:

        def probe_fn(knob: int) -> float:
            scores: list[float] = []
            for j, window in enumerate(windows):
                score = self._inline_probe.probe(
                    window,
                    work_dir / f"tq_probe_q{knob}_w{j}.obu",
                    vp,
                    qvbr=knob,
                    metric=metric,
                )
                scores.append(score)
                narrator.window(knob, j + 1, score)
            pooled = sum(scores) / len(scores)
            narrator.result(knob, pooled)
            return pooled

        return probe_fn

    def _grain_probe_fn(
        self,
        vp: VideoParams,
        metric: str,
        windows: list[Path],
        work_dir: Path,
        narrator: _SearchNarrator,
    ) -> Callable[[int], float]:
        if self._grain_encoder is None or self._metrics is None:
            raise RuntimeError("grain target-quality requires an SVT encoder and a metrics adapter")
        grain_encoder = self._grain_encoder
        metrics = self._metrics
        wanted = frozenset({metric})

        references: list[Path] = []
        for j, window in enumerate(windows):
            reference = work_dir / f"tq_ref_w{j}.mkv"
            rc = self._extractor.build_reference(window, reference, vp)
            if rc != 0:
                raise RuntimeError(f"grain probe reference build failed (rc={rc}) for window {j}")
            references.append(reference)

        def probe_fn(knob: int) -> float:
            scores: list[float] = []
            for j, window in enumerate(windows):
                obu = work_dir / f"tq_grain_q{knob}_w{j}.obu"
                result = grain_encoder.encode(window, obu, vp, cq_override=knob)
                if result.return_code != 0:
                    raise RuntimeError(f"grain probe encode failed (rc={result.return_code}) at crf={knob}")
                measured = metrics.measure(
                    references[j],
                    obu,
                    matrix=vp.color_matrix,
                    fps_num=vp.fps_num,
                    fps_den=vp.fps_den,
                    pool=MetricPool.LOW,
                    metrics=wanted,
                )
                score = getattr(measured, metric)
                if score is None:
                    raise RuntimeError(f"grain probe could not be scored at crf={knob}")
                scores.append(float(score))
                narrator.window(knob, j + 1, float(score))
            pooled = pool_grain_windows(scores)
            narrator.result(knob, pooled)
            return pooled

        return probe_fn
