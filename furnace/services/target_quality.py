"""Target-quality search orchestration.

Before the final encode, probe short windows of the source at several knob
values, measure the domain's driver metric, and interpolation-search the knob
that lands the score in the target band. The final encode then runs at that knob
with no metrics attached.

Two probe strategies, dispatched on the content domain (``resolve_target`` /
``vp.grain``):

- **NVEnc** (non-grain): 3 windows, each probed via the inline probe -- NVEncC
  encodes at the candidate QVBR and self-measures the metric, returning one
  score per window. Mean-pooled across windows.
- **SVT-AV1** (grain): 10 windows selected by the source's rate-control regime --
  the hardest windows by bitrate on a VBR source (which marks its hard scenes with
  bits), or evenly-spaced on a CBR source (flat bitrate; hard scenes are common) --
  each encoded at the candidate CRF, scored against the source window by the
  VapourSynth+Vship metrics adapter with worst-case (low-percentile) frame pooling,
  and MIN-pooled across windows so the hardest sampled scene governs the one
  whole-movie CRF.

The windows are extracted once and reused across every probed knob. All the
numeric policy (domain -> metric/target/bounds, window layout, the search) lives
in :mod:`furnace.core.target_quality`; this service is pure orchestration around
injected adapters.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from furnace.core.models import MetricPool, VideoParams
from furnace.core.ports import Encoder, InlineQualityProbe, PerceptualMetrics, WindowExtractor
from furnace.core.target_quality import (
    PROBE_WINDOW_SECONDS,
    KnobSearchResult,
    TargetSpec,
    interior_windows,
    probe_windows,
    resolve_target,
    search_knob,
    select_hard_windows,
    source_is_variable_bitrate,
)


def _noop(_message: str) -> None:
    """Silent narration sink used when the caller passes no ``on_event``."""


def _windows(n: int) -> str:
    """Human count: ``1 window`` / ``3 windows``."""
    return f"{n} window" if n == 1 else f"{n} windows"


@dataclass(frozen=True, slots=True)
class _SearchNarrator:
    """Formats target-quality search progress into TUI log lines.

    Carries the per-search context -- the knob ``label`` (CRF/QVBR), the driver
    ``metric``, the ``window_count`` and the across-window ``pool_word`` -- so the
    probe loop can narrate each finished window and each probed knob with a terse
    call. ``emit`` is the sink; the caller routes it to a channel that stays
    visible while the raw per-probe ffmpeg/nvencc output is muted."""

    emit: Callable[[str], None]
    label: str
    metric: str
    window_count: int
    pool_word: str

    def opening(self, centre: float) -> None:
        """One line describing the search plan, before the first probe."""
        self.emit(
            f"Probing {self.label} -> {self.metric} ~{centre:.1f} "
            f"({_windows(self.window_count)}, {self.pool_word}-pooled)"
        )

    def window(self, knob: int, index: int, score: float) -> None:
        """Liveness line as each probe window finishes (``index`` is 1-based)."""
        self.emit(f"{self.label} {knob}: window {index}/{self.window_count} = {score:.1f}")

    def result(self, knob: int, pooled: float) -> None:
        """The pooled score that governs the knob, once all windows are in."""
        self.emit(f"{self.label} {knob} -> {self.metric} {pooled:.1f}")


class TargetQualityService:
    """Drive the quality-knob search for one job via injected adapters.

    ``inline_probe`` handles the NVEnc path. ``grain_encoder`` (an SVT-AV1
    encoder) and ``metrics`` (VapourSynth+Vship) handle the grain path; both must
    be present for a grain job to be searchable (see :meth:`can_search`).
    """

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

    def can_search(self, vp: VideoParams) -> bool:
        """Whether this service can search ``vp``.

        The NVEnc path is always available; the grain path needs both an SVT
        encoder and a metrics adapter (a grain job falls back to the fixed CRF
        recipe when they aren't configured). Interlaced grain needs no extra
        deinterlace plugin: the metric reference is deinterlaced by the encode's
        own ffmpeg bwdif when it is built (see ``WindowExtractor.build_reference``).
        """
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
        """Search the quality knob that hits ``vp``'s content-domain target.

        Extracts the probe windows once into ``work_dir`` (or, for a short
        source, probes the whole thing), then interpolation-searches the knob.
        Raises if a window cannot be extracted or a probe cannot be scored.

        ``on_event`` receives short human-readable progress lines (the search
        plan, each finished probe window, each probed knob's pooled score) so the
        run TUI can show what the search is doing while the raw per-probe encoder
        output is muted. Omitted -> a silent no-op sink.
        """
        spec = resolve_target(vp)
        if vp.grain:
            offsets = self._grain_window_offsets(source, spec, duration_s)
        else:
            offsets = probe_windows(
                duration_s, count=spec.window_count, window_s=PROBE_WINDOW_SECONDS
            )
        windows = self._prepare_windows(source, vp, offsets, duration_s, work_dir)
        narrator = _SearchNarrator(
            emit=on_event or _noop,
            label="CRF" if vp.grain else "QVBR",
            metric=spec.metric.upper(),
            window_count=len(windows),
            pool_word="worst-case" if vp.grain else "mean",
        )
        narrator.opening((spec.target_lo + spec.target_hi) / 2.0)
        probe_fn = (
            self._grain_probe_fn(vp, spec.metric, windows, work_dir, narrator)
            if vp.grain
            else self._inline_probe_fn(vp, spec.metric, windows, work_dir, narrator)
        )
        return search_knob(
            probe_fn,
            target_lo=spec.target_lo,
            target_hi=spec.target_hi,
            lo=spec.knob_lo,
            hi=spec.knob_hi,
            max_probes=spec.max_probes,
        )

    def _grain_window_offsets(
        self, source: Path, spec: TargetSpec, duration_s: float
    ) -> list[float] | None:
        """Pick the grain probe-window offsets by source rate-control regime.

        A short source returns ``None`` (full-pass, handled by ``_prepare_windows``).
        Otherwise read the per-window source bitrate over the interior of the
        timeline: on a VBR source the encoder concentrated bits on the hard scenes,
        so the highest-bitrate windows ARE the hard scenes -- sample those; on a
        CBR/flat source the bitrate says nothing about difficulty (it can point at
        the easy scenes), but hard scenes are common there, so fall back to the
        evenly-spaced layout. If the source bitrate can't be read at all, fall back
        to even sampling too.
        """
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
        """Extract the probe windows.

        Full-pass (``offsets is None``, short source) extracts ONE window bounded
        to the reported duration rather than handing the raw file to the probe:
        ``extract_window`` is frame-count-bounded, so even a duration that is
        under-reported (a container that lies about a multi-hour file) yields a
        bounded window instead of re-encoding hours of video per probed knob.
        """
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
        """NVEnc strategy: mean of the inline-measured metric across the windows.

        Mean pooling over evenly-spaced windows is a deliberate choice here (not
        the grain path's difficulty-aware sampling): QVBR is scene-adaptive, so it
        holds per-scene quality within one pass, and the mean over a representative
        spread generalises. The grain path can't rely on that -- CRF is one value
        for the whole movie -- so it samples more windows and pools worst-case.
        A failed probe does not silently skew the mean -- the inline probe raises
        and aborts the search. ``narrator`` reports each finished window and the
        mean per knob to the run TUI.
        """

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
        """SVT strategy: encode each window at the candidate CRF, score it against
        a geometry-matched reference with worst-case (low-percentile) frame pooling
        of the driver ``metric``, and MIN the per-window worst-cases -- the hardest
        sampled scene governs the chosen CRF. CRF is one value for the whole movie,
        so it must satisfy the hardest scene (mean would let an easy window mask a
        hard one and pick too-high a CRF -> мыло); the window SELECTION already
        targets the hard scenes (see :meth:`_grain_window_offsets`), so the min
        protects them. Only ``metric`` is computed on the GPU (the other Vship
        kernels are skipped). A failed encode or an unavailable score aborts the
        search loudly rather than skewing the result. ``narrator`` reports each
        finished window and the min per knob to the run TUI.

        The metric reference is built ONCE per window, up front (it is identical
        across every probed CRF), through the encode's OWN ffmpeg geometry
        filtergraph (``build_reference``): deinterlace/crop/scale applied by the
        same tool that encodes, so a crop can't phase-shift the reference against
        the encode -- the failure mode that used to collapse SSIMULACRA2 and rail
        the CRF search at its floor.
        """
        if self._grain_encoder is None or self._metrics is None:
            raise RuntimeError(
                "grain target-quality requires an SVT encoder and a metrics adapter"
            )
        grain_encoder = self._grain_encoder
        metrics = self._metrics
        wanted = frozenset({metric})

        references: list[Path] = []
        for j, window in enumerate(windows):
            reference = work_dir / f"tq_ref_w{j}.mkv"
            rc = self._extractor.build_reference(window, reference, vp)
            if rc != 0:
                raise RuntimeError(
                    f"grain probe reference build failed (rc={rc}) for window {j}"
                )
            references.append(reference)

        def probe_fn(knob: int) -> float:
            scores: list[float] = []
            for j, window in enumerate(windows):
                obu = work_dir / f"tq_grain_q{knob}_w{j}.obu"
                result = grain_encoder.encode(window, obu, vp, cq_override=knob)
                if result.return_code != 0:
                    raise RuntimeError(
                        f"grain probe encode failed (rc={result.return_code}) at crf={knob}"
                    )
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
            pooled = min(scores)
            narrator.result(knob, pooled)
            return pooled

        return probe_fn
