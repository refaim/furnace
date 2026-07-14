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
- **SVT-AV1** (grain): 10 windows, each encoded at the candidate CRF, then scored
  against the source window by the VapourSynth+Vship metrics adapter with
  worst-case (low-percentile) frame pooling. Pooled across windows by dropping
  the 2 hardest and targeting the worst of the rest -- the search must see the
  common hard scenes, but a couple of freak scenes must not pin the whole-movie
  CRF and bloat the file (calibrated across the grain collection).

The windows are extracted once and reused across every probed knob. All the
numeric policy (domain -> metric/target/bounds, window layout, the search) lives
in :mod:`furnace.core.target_quality`; this service is pure orchestration around
injected adapters.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from furnace.core.models import MetricPool, VideoParams
from furnace.core.ports import Encoder, InlineQualityProbe, PerceptualMetrics, WindowExtractor
from furnace.core.quality import final_output_dimensions
from furnace.core.target_quality import (
    PROBE_WINDOW_SECONDS,
    KnobSearchResult,
    probe_windows,
    resolve_target,
    search_knob,
)


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
        recipe when they aren't configured).

        This gate does not check for a bwdif plugin on an interlaced grain source
        (the metrics adapter needs one to score it): the cli's
        ``_check_interlaced_grain_metrics_ready`` preflight already refuses to
        start ``run`` in that configuration, so it never reaches a probe here.
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
    ) -> KnobSearchResult:
        """Search the quality knob that hits ``vp``'s content-domain target.

        Extracts the probe windows once into ``work_dir`` (or, for a short
        source, probes the whole thing), then interpolation-searches the knob.
        Raises if a window cannot be extracted or a probe cannot be scored.
        """
        spec = resolve_target(vp)
        offsets = probe_windows(
            duration_s, count=spec.window_count, window_s=PROBE_WINDOW_SECONDS
        )
        windows = self._prepare_windows(source, vp, offsets, duration_s, work_dir)
        probe_fn = (
            self._grain_probe_fn(vp, spec.metric, windows, work_dir, spec.pool_drop)
            if vp.grain
            else self._inline_probe_fn(vp, spec.metric, windows, work_dir)
        )
        return search_knob(
            probe_fn,
            target_lo=spec.target_lo,
            target_hi=spec.target_hi,
            lo=spec.knob_lo,
            hi=spec.knob_hi,
            max_probes=spec.max_probes,
        )

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
    ) -> Callable[[int], float]:
        """NVEnc strategy: mean of the inline-measured metric across the windows.

        Mean pooling over evenly-spaced windows is a deliberate choice here (not
        the grain path's difficulty-aware sampling): QVBR is scene-adaptive, so it
        holds per-scene quality within one pass, and the mean over a representative
        spread generalises. The grain path can't rely on that -- CRF is one value
        for the whole movie -- so it samples more windows and pools worst-case.
        A failed probe does not silently skew the mean -- the inline probe raises
        and aborts the search.
        """

        def probe_fn(knob: int) -> float:
            scores = [
                self._inline_probe.probe(
                    window,
                    work_dir / f"tq_probe_q{knob}_w{j}.obu",
                    vp,
                    qvbr=knob,
                    metric=metric,
                )
                for j, window in enumerate(windows)
            ]
            return sum(scores) / len(scores)

        return probe_fn

    def _grain_probe_fn(
        self,
        vp: VideoParams,
        metric: str,
        windows: list[Path],
        work_dir: Path,
        pool_drop: int,
    ) -> Callable[[int], float]:
        """SVT strategy: encode each window at the candidate CRF, score it against
        the source window with worst-case (low-percentile) frame pooling of the
        driver ``metric``, then pool ACROSS windows by dropping the ``pool_drop``
        hardest and targeting the worst of the rest. CRF is one value for the whole
        movie, so the pool leans worst-case (mean would let a bright reel's high p5
        mask a dark reel's low p5 and pick too-high a CRF -> мыло), but a couple of
        freak worst-case scenes must not pin the whole-movie CRF and bloat the file,
        so the very hardest ``pool_drop`` windows are dropped (calibrated across the
        grain collection). The drop is clamped so at least one window always
        governs. Only ``metric`` is computed on the GPU (the other Vship kernels are
        skipped). A failed encode or an unavailable score aborts the search loudly
        rather than skewing the result.
        """
        if self._grain_encoder is None or self._metrics is None:
            raise RuntimeError(
                "grain target-quality requires an SVT encoder and a metrics adapter"
            )
        grain_encoder = self._grain_encoder
        metrics = self._metrics
        wanted = frozenset({metric})
        final_w, final_h = final_output_dimensions(vp)

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
                    window,
                    obu,
                    crop=vp.crop,
                    deinterlace=vp.deinterlace,
                    final_width=final_w,
                    final_height=final_h,
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
            # Drop the pool_drop hardest windows (lowest scores) and target the
            # worst of the rest; clamp so a short (few-window) source still has one
            # governing window.
            keep_from = min(pool_drop, len(scores) - 1)
            return sorted(scores)[keep_from]

        return probe_fn
