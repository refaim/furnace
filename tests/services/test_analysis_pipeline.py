"""Unit tests for ``AnalysisPipeline``.

The pipeline fans per-file analysis (``analyze`` + cropdetect) out across a
thread pool. Workers do I/O only; the reporter is driven solely from the main
thread. Results are reassembled in input order regardless of completion order.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from furnace.core.models import (
    AnalysisOutcome,
    AnalyzeStatus,
    CropRect,
    ScanResult,
)
from furnace.core.ports import PlanReporter, Prober
from furnace.core.progress import ProgressSample
from furnace.services.analysis_pipeline import AnalysisPipeline
from furnace.services.analyzer import Analyzer
from tests.conftest import make_movie, make_video_info
from tests.fakes.recording_reporter import Event, RecordingPlanReporter

# ---------------------------------------------------------------------------
# Fakes / factories
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CropCall:
    """One recorded ``detect_crop`` invocation."""

    path: Path
    duration_s: float
    interlaced: bool
    is_dvd: bool
    hdr_transfer: str | None
    on_progress: object


class _FakeAnalyzer:
    """Returns a canned ``AnalysisOutcome`` keyed by the scan's main file.

    When given an ``on_progress`` callback it drives it through a mid-run and a
    final value, exercising the pipeline's analyze-progress wiring.
    """

    def __init__(self, outcomes: dict[Path, AnalysisOutcome]) -> None:
        self._outcomes = outcomes

    def analyze(
        self,
        scan_result: ScanResult,
        *,
        on_progress: Callable[[float], None] | None = None,
    ) -> AnalysisOutcome:
        assert on_progress is not None
        on_progress(0.5)
        on_progress(1.0)
        return self._outcomes[scan_result.main_file]


class _FakeProber:
    """Returns a canned crop (or raises) keyed by the movie's main file.

    Every call is captured in ``calls`` so tests can assert which files had
    cropdetect run, and that the per-call arguments were threaded correctly.
    """

    def __init__(
        self,
        crops: dict[Path, CropRect | None] | None = None,
        raises: dict[Path, Exception] | None = None,
    ) -> None:
        self._crops = crops if crops is not None else {}
        self._raises = raises if raises is not None else {}
        self.calls: list[_CropCall] = []

    def detect_crop(
        self,
        path: Path,
        duration_s: float,
        *,
        interlaced: bool = False,
        is_dvd: bool = False,
        hdr_transfer: str | None = None,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> CropRect | None:
        self.calls.append(
            _CropCall(path, duration_s, interlaced, is_dvd, hdr_transfer, on_progress),
        )
        assert on_progress is not None
        on_progress(ProgressSample(fraction=0.5))  # forwarded to the file slot
        on_progress(ProgressSample(fraction=None))  # no fraction -> ignored
        exc = self._raises.get(path)
        if exc is not None:
            raise exc
        return self._crops.get(path)


def _sr(tmp_path: Path, name: str) -> ScanResult:
    """A ScanResult whose output_path is a sibling of the source."""
    return ScanResult(
        main_file=tmp_path / name,
        satellite_files=[],
        output_path=tmp_path / f"out_{name}",
    )


def _done(
    main_file: Path,
    *,
    width: int = 1920,
    height: int = 1080,
    interlaced: bool = False,
    color_transfer: str | None = "bt709",
    detail: str = "summary",
) -> AnalysisOutcome:
    """A DONE outcome whose Movie's main_file matches the scan's main_file."""
    video = make_video_info(
        width=width,
        height=height,
        interlaced=interlaced,
        color_transfer=color_transfer,
        source_file=main_file,
    )
    movie = make_movie(main_file=main_file, video=video)
    return AnalysisOutcome(movie, AnalyzeStatus.DONE, detail)


def _build_pipeline(
    analyzer: object,
    prober: object,
    reporter: PlanReporter,
    *,
    max_workers: int = 1,
) -> AnalysisPipeline:
    """Construct the pipeline, casting the structural fakes to the real ports."""
    return AnalysisPipeline(
        cast("Analyzer", analyzer),
        cast("Prober", prober),
        reporter,
        max_workers=max_workers,
    )


_FULL_FRAME = CropRect(w=1920, h=1080, x=0, y=0)


# ---------------------------------------------------------------------------
# Deterministic single-worker batch
# ---------------------------------------------------------------------------


def test_two_done_input_order_with_crop_and_fullframe(tmp_path: Path) -> None:
    """Both DONE movies appear in input order; only the cropped file gets a crop."""
    sr_a = _sr(tmp_path, "a.mkv")
    sr_b = _sr(tmp_path, "b.mkv")
    outcomes = {sr_a.main_file: _done(sr_a.main_file, detail="a summary"),
                sr_b.main_file: _done(sr_b.main_file, detail="b summary")}
    crop_a = CropRect(w=1920, h=800, x=0, y=140)
    prober = _FakeProber(crops={sr_a.main_file: crop_a, sr_b.main_file: _FULL_FRAME})
    reporter = RecordingPlanReporter()
    pipeline = _build_pipeline(_FakeAnalyzer(outcomes), prober, reporter, max_workers=1)

    result = pipeline.run([sr_a, sr_b], copy_video=False, dry_run=False)

    assert [(m.main_file, out) for m, out in result.movies] == [
        (sr_a.main_file, sr_a.output_path),
        (sr_b.main_file, sr_b.output_path),
    ]
    assert result.crops == {sr_a.main_file: crop_a}

    assert reporter.events[0] == Event("analyze_batch_start", (2,), ())
    assert reporter.events[-1] == Event("analyze_batch_finish", (), ())
    items = [e for e in reporter.events if e.method == "analyze_batch_item"]
    assert items == [
        Event("analyze_batch_item", ("a.mkv", "a summary"), (("status", AnalyzeStatus.DONE),)),
        Event("analyze_batch_item", ("b.mkv", "b summary"), (("status", AnalyzeStatus.DONE),)),
    ]


def test_dry_run_skips_crop_detection(tmp_path: Path) -> None:
    """dry_run never calls detect_crop; the movie still appears with no crop."""
    sr = _sr(tmp_path, "a.mkv")
    outcomes = {sr.main_file: _done(sr.main_file)}
    prober = _FakeProber(crops={sr.main_file: CropRect(w=1920, h=800, x=0, y=140)})
    pipeline = _build_pipeline(_FakeAnalyzer(outcomes), prober, RecordingPlanReporter())

    result = pipeline.run([sr], copy_video=False, dry_run=True)

    assert prober.calls == []
    assert result.crops == {}
    assert [m.main_file for m, _ in result.movies] == [sr.main_file]


def test_copy_video_passthrough_skips_crop(tmp_path: Path) -> None:
    """A passthrough-eligible movie under copy_video never runs cropdetect."""
    sr = _sr(tmp_path, "a.mkv")
    outcomes = {sr.main_file: _done(sr.main_file)}
    prober = _FakeProber(crops={sr.main_file: CropRect(w=1920, h=800, x=0, y=140)})
    pipeline = _build_pipeline(_FakeAnalyzer(outcomes), prober, RecordingPlanReporter())

    result = pipeline.run([sr], copy_video=True, dry_run=False)

    assert prober.calls == []
    assert result.crops == {}
    assert [m.main_file for m, _ in result.movies] == [sr.main_file]


def test_skipped_and_failed_excluded_but_reported(tmp_path: Path) -> None:
    """movie=None outcomes are dropped from movies/crops yet still reported."""
    sr_a = _sr(tmp_path, "a.mkv")
    sr_b = _sr(tmp_path, "b.mkv")
    outcomes = {
        sr_a.main_file: AnalysisOutcome(None, AnalyzeStatus.SKIPPED, "already encoded"),
        sr_b.main_file: AnalysisOutcome(None, AnalyzeStatus.FAILED, "probe failed"),
    }
    prober = _FakeProber()
    reporter = RecordingPlanReporter()
    pipeline = _build_pipeline(_FakeAnalyzer(outcomes), prober, reporter, max_workers=1)

    result = pipeline.run([sr_a, sr_b], copy_video=False, dry_run=False)

    assert result.movies == []
    assert result.crops == {}
    assert prober.calls == []
    items = [e for e in reporter.events if e.method == "analyze_batch_item"]
    assert items == [
        Event("analyze_batch_item", ("a.mkv", "already encoded"), (("status", AnalyzeStatus.SKIPPED),)),
        Event("analyze_batch_item", ("b.mkv", "probe failed"), (("status", AnalyzeStatus.FAILED),)),
    ]


def test_detect_crop_branches(tmp_path: Path) -> None:
    """All four _detect_crop outcomes: real crop, full-frame, None, raise."""
    real = _sr(tmp_path, "real.mkv")     # (a) real crop -> kept
    full = _sr(tmp_path, "full.mkv")     # (b) full-frame -> None
    none = _sr(tmp_path, "none.mkv")     # (c) detect returns None
    boom = _sr(tmp_path, "boom.mkv")     # (d) raises RuntimeError
    srs = [real, full, none, boom]
    outcomes = {s.main_file: _done(s.main_file) for s in srs}
    real_crop = CropRect(w=1920, h=816, x=0, y=132)
    prober = _FakeProber(
        crops={real.main_file: real_crop, full.main_file: _FULL_FRAME, none.main_file: None},
        raises={boom.main_file: RuntimeError("ffmpeg blew up")},
    )
    pipeline = _build_pipeline(_FakeAnalyzer(outcomes), prober, RecordingPlanReporter(), max_workers=1)

    result = pipeline.run(srs, copy_video=False, dry_run=False)

    # Every file is DONE -> all present, in input order, even the one that raised.
    assert [m.main_file for m, _ in result.movies] == [s.main_file for s in srs]
    # Only the genuine crop survives.
    assert result.crops == {real.main_file: real_crop}
    # cropdetect was attempted for the file that raised.
    assert any(call.path == boom.main_file for call in prober.calls)


def test_detect_crop_threads_arguments(tmp_path: Path) -> None:
    """SD interlaced HDR source -> is_dvd True, interlaced True, hdr_transfer set."""
    sr = _sr(tmp_path, "dvd.mkv")
    outcomes = {
        sr.main_file: _done(
            sr.main_file,
            width=720,
            height=576,
            interlaced=True,
            color_transfer="smpte2084",
        ),
    }
    prober = _FakeProber(crops={sr.main_file: CropRect(w=720, h=432, x=0, y=72)})
    pipeline = _build_pipeline(_FakeAnalyzer(outcomes), prober, RecordingPlanReporter())

    pipeline.run([sr], copy_video=False, dry_run=False)

    assert len(prober.calls) == 1
    call = prober.calls[0]
    assert call.path == sr.main_file
    assert call.is_dvd is True
    assert call.interlaced is True
    assert call.hdr_transfer == "smpte2084"
    # The pipeline forwards cropdetect progress to the file's slot.
    assert callable(call.on_progress)


def test_process_applies_analyze_and_crop_weights(tmp_path: Path) -> None:
    """``_process`` maps analyze progress into [0, 0.7] and crop into [0.7, 1.0].

    The fakes read the worker's progress slot synchronously right after each
    ``on_progress`` call, so the 0.7/0.3 weight split is asserted deterministically
    (no thread timing involved).
    """
    sr = _sr(tmp_path, "a.mkv")
    file_progress = [0.0]
    seen: list[tuple[str, float]] = []

    class _CapAnalyzer:
        def analyze(
            self,
            scan_result: ScanResult,
            *,
            on_progress: Callable[[float], None] | None = None,
        ) -> AnalysisOutcome:
            assert on_progress is not None
            on_progress(0.5)
            seen.append(("analyze", file_progress[0]))
            on_progress(1.0)
            seen.append(("analyze", file_progress[0]))
            return _done(scan_result.main_file)

    class _CapProber:
        def detect_crop(
            self,
            path: Path,
            duration_s: float,
            *,
            interlaced: bool = False,
            is_dvd: bool = False,
            hdr_transfer: str | None = None,
            on_progress: Callable[[ProgressSample], None] | None = None,
        ) -> CropRect | None:
            _ = (path, duration_s, interlaced, is_dvd, hdr_transfer)  # signature parity
            assert on_progress is not None
            on_progress(ProgressSample(fraction=0.5))
            seen.append(("crop", file_progress[0]))
            return None

    pipeline = _build_pipeline(_CapAnalyzer(), _CapProber(), RecordingPlanReporter())
    index, outcome, crop = pipeline._process(0, sr, file_progress, copy_video=False, dry_run=False)

    assert index == 0
    assert outcome.status is AnalyzeStatus.DONE
    assert crop is None
    assert [label for label, _ in seen] == ["analyze", "analyze", "crop"]
    expected = [0.35, 0.70, 0.85]  # 0.5*0.7, 1.0*0.7, 0.7 + 0.5*0.3
    assert all(abs(value - exp) < 1e-9 for (_, value), exp in zip(seen, expected, strict=True))
    assert file_progress[0] == 1.0  # forced to complete when the worker returns


# ---------------------------------------------------------------------------
# Parallel (max_workers=3) — correctness only, no reporter line-order asserts
# ---------------------------------------------------------------------------


def test_parallel_preserves_input_order_and_crop_keys(tmp_path: Path) -> None:
    """With several workers, results reassemble in input order, crops keyed right."""
    n = 5
    srs = [_sr(tmp_path, f"m{i}.mkv") for i in range(n)]
    outcomes = {s.main_file: _done(s.main_file, detail=f"summary {i}") for i, s in enumerate(srs)}
    crops: dict[Path, CropRect | None] = {}
    expected: dict[Path, CropRect] = {}
    for i, s in enumerate(srs):
        if i % 2 == 0:
            crop = CropRect(w=1920, h=800 + i, x=0, y=140)
            crops[s.main_file] = crop
            expected[s.main_file] = crop
        else:
            crops[s.main_file] = _FULL_FRAME
    reporter = RecordingPlanReporter()
    pipeline = _build_pipeline(_FakeAnalyzer(outcomes), _FakeProber(crops=crops), reporter, max_workers=3)

    result = pipeline.run(srs, copy_video=False, dry_run=False)

    assert [m.main_file for m, _ in result.movies] == [s.main_file for s in srs]
    assert [out for _, out in result.movies] == [s.output_path for s in srs]
    assert result.crops == expected
    # Reporter framing is present (order is nondeterministic under parallelism).
    assert reporter.events[0] == Event("analyze_batch_start", (n,), ())
    assert reporter.events[-1] == Event("analyze_batch_finish", (), ())
    items = [e for e in reporter.events if e.method == "analyze_batch_item"]
    assert len(items) == n
