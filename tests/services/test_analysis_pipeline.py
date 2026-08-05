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


@dataclass(frozen=True)
class _CropCall:
    path: Path
    duration_s: float
    interlaced: bool
    is_dvd: bool
    hdr_transfer: str | None
    on_progress: object


class _FakeAnalyzer:
    def __init__(self, outcomes: dict[Path, AnalysisOutcome]) -> None:
        self._outcomes = outcomes
        self.calls: list[tuple[Path, bool, bool | None]] = []

    def analyze(
        self,
        scan_result: ScanResult,
        *,
        on_progress: Callable[[float], None] | None = None,
        copy_video: bool = False,
        grain_override: bool | None = None,
    ) -> AnalysisOutcome:
        assert on_progress is not None
        self.calls.append((scan_result.main_file, copy_video, grain_override))
        on_progress(0.5)
        on_progress(1.0)
        return self._outcomes[scan_result.main_file]


class _FakeProber:
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
        on_progress(ProgressSample(fraction=0.5))
        on_progress(ProgressSample(fraction=None))
        exc = self._raises.get(path)
        if exc is not None:
            raise exc
        return self._crops.get(path)


def _sr(tmp_path: Path, name: str) -> ScanResult:
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
    return AnalysisPipeline(
        cast("Analyzer", analyzer),
        cast("Prober", prober),
        reporter,
        max_workers=max_workers,
    )


_FULL_FRAME = CropRect(w=1920, h=1080, x=0, y=0)


def test_two_done_input_order_with_crop_and_fullframe(tmp_path: Path) -> None:
    sr_a = _sr(tmp_path, "a.mkv")
    sr_b = _sr(tmp_path, "b.mkv")
    outcomes = {
        sr_a.main_file: _done(sr_a.main_file, detail="a summary"),
        sr_b.main_file: _done(sr_b.main_file, detail="b summary"),
    }
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
    sr = _sr(tmp_path, "a.mkv")
    outcomes = {sr.main_file: _done(sr.main_file)}
    prober = _FakeProber(crops={sr.main_file: CropRect(w=1920, h=800, x=0, y=140)})
    pipeline = _build_pipeline(_FakeAnalyzer(outcomes), prober, RecordingPlanReporter())

    result = pipeline.run([sr], copy_video=False, dry_run=True)

    assert prober.calls == []
    assert result.crops == {}
    assert [m.main_file for m, _ in result.movies] == [sr.main_file]


def test_copy_video_and_grain_overrides_reach_the_analyzer(tmp_path: Path) -> None:
    sr_a = _sr(tmp_path, "a.mkv")
    sr_b = _sr(tmp_path, "b.mkv")
    outcomes = {sr_a.main_file: _done(sr_a.main_file), sr_b.main_file: _done(sr_b.main_file)}
    analyzer = _FakeAnalyzer(outcomes)
    pipeline = _build_pipeline(analyzer, _FakeProber(), RecordingPlanReporter())

    pipeline.run(
        [sr_a, sr_b],
        copy_video=True,
        dry_run=False,
        grain_overrides={sr_b.main_file: False},
    )

    assert sorted(analyzer.calls) == sorted(
        [
            (sr_a.main_file, True, None),
            (sr_b.main_file, True, False),
        ]
    )


def test_grain_overrides_default_to_empty(tmp_path: Path) -> None:
    sr = _sr(tmp_path, "a.mkv")
    analyzer = _FakeAnalyzer({sr.main_file: _done(sr.main_file)})
    pipeline = _build_pipeline(analyzer, _FakeProber(), RecordingPlanReporter())

    pipeline.run([sr], copy_video=False, dry_run=False)

    assert analyzer.calls == [(sr.main_file, False, None)]


def test_copy_video_passthrough_skips_crop(tmp_path: Path) -> None:
    sr = _sr(tmp_path, "a.mkv")
    outcomes = {sr.main_file: _done(sr.main_file)}
    prober = _FakeProber(crops={sr.main_file: CropRect(w=1920, h=800, x=0, y=140)})
    pipeline = _build_pipeline(_FakeAnalyzer(outcomes), prober, RecordingPlanReporter())

    result = pipeline.run([sr], copy_video=True, dry_run=False)

    assert prober.calls == []
    assert result.crops == {}
    assert [m.main_file for m, _ in result.movies] == [sr.main_file]


def test_skipped_and_failed_excluded_but_reported(tmp_path: Path) -> None:
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
    real = _sr(tmp_path, "real.mkv")
    full = _sr(tmp_path, "full.mkv")
    none = _sr(tmp_path, "none.mkv")
    boom = _sr(tmp_path, "boom.mkv")
    srs = [real, full, none, boom]
    outcomes = {s.main_file: _done(s.main_file) for s in srs}
    real_crop = CropRect(w=1920, h=816, x=0, y=132)
    prober = _FakeProber(
        crops={real.main_file: real_crop, full.main_file: _FULL_FRAME, none.main_file: None},
        raises={boom.main_file: RuntimeError("ffmpeg blew up")},
    )
    pipeline = _build_pipeline(_FakeAnalyzer(outcomes), prober, RecordingPlanReporter(), max_workers=1)

    result = pipeline.run(srs, copy_video=False, dry_run=False)

    assert [m.main_file for m, _ in result.movies] == [s.main_file for s in srs]
    assert result.crops == {real.main_file: real_crop}
    assert any(call.path == boom.main_file for call in prober.calls)


def test_detect_crop_threads_arguments(tmp_path: Path) -> None:
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
    assert callable(call.on_progress)


def test_process_applies_analyze_and_crop_weights(tmp_path: Path) -> None:
    sr = _sr(tmp_path, "a.mkv")
    file_progress = [0.0]
    seen: list[tuple[str, float]] = []

    class _CapAnalyzer:
        def analyze(
            self,
            scan_result: ScanResult,
            *,
            on_progress: Callable[[float], None] | None = None,
            copy_video: bool = False,  # noqa: ARG002
            grain_override: bool | None = None,  # noqa: ARG002
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
            _ = (path, duration_s, interlaced, is_dvd, hdr_transfer)
            assert on_progress is not None
            on_progress(ProgressSample(fraction=0.5))
            seen.append(("crop", file_progress[0]))
            return None

    pipeline = _build_pipeline(_CapAnalyzer(), _CapProber(), RecordingPlanReporter())
    index, outcome, crop = pipeline._process(
        0,
        sr,
        file_progress,
        copy_video=False,
        dry_run=False,
        grain_overrides={},
    )

    assert index == 0
    assert outcome.status is AnalyzeStatus.DONE
    assert crop is None
    assert [label for label, _ in seen] == ["analyze", "analyze", "crop"]
    expected = [0.35, 0.70, 0.85]
    assert all(abs(value - exp) < 1e-9 for (_, value), exp in zip(seen, expected, strict=True))
    assert file_progress[0] == 1.0


def test_parallel_preserves_input_order_and_crop_keys(tmp_path: Path) -> None:
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
    assert reporter.events[0] == Event("analyze_batch_start", (n,), ())
    assert reporter.events[-1] == Event("analyze_batch_finish", (), ())
    items = [e for e in reporter.events if e.method == "analyze_batch_item"]
    assert len(items) == n
