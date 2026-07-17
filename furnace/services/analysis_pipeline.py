from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from furnace.core.detect import (
    classify_passthrough,
    hdr_transfer_for_cropdetect,
    is_dvd_resolution,
)
from furnace.core.models import (
    AnalysisOutcome,
    CropRect,
    Movie,
    ScanResult,
)
from furnace.core.ports import PlanReporter, Prober
from furnace.core.progress import ProgressSample
from furnace.services.analyzer import Analyzer

logger = logging.getLogger(__name__)

_ANALYZE_WEIGHT = 0.7
_CROP_WEIGHT = 0.3
_POLL_INTERVAL_S = 0.1


@dataclass(frozen=True)
class AnalysisBatchResult:
    movies: list[tuple[Movie, Path]]
    crops: dict[Path, CropRect]


class AnalysisPipeline:
    def __init__(
        self,
        analyzer: Analyzer,
        prober: Prober,
        reporter: PlanReporter,
        *,
        max_workers: int,
    ) -> None:
        self._analyzer = analyzer
        self._prober = prober
        self._reporter = reporter
        self._max_workers = max_workers

    def run(
        self,
        scan_results: list[ScanResult],
        *,
        copy_video: bool,
        dry_run: bool,
    ) -> AnalysisBatchResult:
        n = len(scan_results)
        self._reporter.analyze_batch_start(n)

        file_progress = [0.0] * n
        results: dict[int, tuple[AnalysisOutcome, CropRect | None]] = {}
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(
                    self._process,
                    i,
                    sr,
                    file_progress,
                    copy_video=copy_video,
                    dry_run=dry_run,
                ): i
                for i, sr in enumerate(scan_results)
            }
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=_POLL_INTERVAL_S)
                self._reporter.analyze_batch_progress(sum(file_progress))
                for fut in sorted(done, key=lambda f: futures[f]):
                    i, outcome, crop = fut.result()
                    results[i] = (outcome, crop)
                    self._reporter.analyze_batch_item(
                        scan_results[i].main_file.name,
                        outcome.detail,
                        status=outcome.status,
                    )

        self._reporter.analyze_batch_progress(float(n))
        self._reporter.analyze_batch_finish()

        movies: list[tuple[Movie, Path]] = []
        crops: dict[Path, CropRect] = {}
        for i, sr in enumerate(scan_results):
            outcome, crop = results[i]
            if outcome.movie is not None:
                movies.append((outcome.movie, sr.output_path))
                if crop is not None:
                    crops[outcome.movie.main_file] = crop
        return AnalysisBatchResult(movies=movies, crops=crops)

    def _process(
        self,
        index: int,
        scan_result: ScanResult,
        file_progress: list[float],
        *,
        copy_video: bool,
        dry_run: bool,
    ) -> tuple[int, AnalysisOutcome, CropRect | None]:
        def _set(frac: float) -> None:
            file_progress[index] = frac

        outcome = self._analyzer.analyze(
            scan_result,
            on_progress=lambda f: _set(f * _ANALYZE_WEIGHT),
        )
        crop: CropRect | None = None
        if outcome.movie is not None and not dry_run:
            passthrough, _reason = classify_passthrough(outcome.movie.video, copy_video=copy_video)
            if not passthrough:
                crop = self._detect_crop(
                    outcome.movie,
                    on_progress=lambda f: _set(_ANALYZE_WEIGHT + f * _CROP_WEIGHT),
                )
        file_progress[index] = 1.0
        return index, outcome, crop

    def _detect_crop(
        self,
        movie: Movie,
        *,
        on_progress: Callable[[float], None],
    ) -> CropRect | None:
        def _forward(sample: ProgressSample) -> None:
            if sample.fraction is not None:
                on_progress(sample.fraction)

        try:
            is_dvd = is_dvd_resolution(movie.video.width, movie.video.height)
            raw_crop = self._prober.detect_crop(
                movie.main_file,
                movie.video.duration_s,
                interlaced=movie.video.interlaced,
                is_dvd=is_dvd,
                hdr_transfer=hdr_transfer_for_cropdetect(movie.video.color_transfer),
                on_progress=_forward,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Crop detection failed for %s: %s", movie.main_file.name, exc)
            return None
        if raw_crop is None:
            logger.warning("%s: cropdetect unable to determine crop", movie.main_file.name)
            return None
        if raw_crop.w == movie.video.width and raw_crop.h == movie.video.height:
            logger.info(
                "%s: no black bars detected (crop equals full frame %dx%d)",
                movie.main_file.name,
                movie.video.width,
                movie.video.height,
            )
            return None
        logger.info(
            "%s: crop detected %d:%d:%d:%d (source %dx%d)",
            movie.main_file.name,
            raw_crop.w,
            raw_crop.h,
            raw_crop.x,
            raw_crop.y,
            movie.video.width,
            movie.video.height,
        )
        return raw_crop
