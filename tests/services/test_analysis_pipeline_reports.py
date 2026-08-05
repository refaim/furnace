from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

from furnace.core.models import (
    AnalysisOutcome,
    AnalyzeStatus,
    CropRect,
    ScanResult,
)
from furnace.core.ports import PlanReporter, Prober
from furnace.services.analysis_pipeline import AnalysisPipeline
from furnace.services.analyzer import Analyzer
from tests.conftest import make_movie, make_video_info
from tests.fakes.recording_reporter import Event, RecordingPlanReporter


def _framing(reporter: RecordingPlanReporter) -> list[Event]:
    return [e for e in reporter.events if e.method != "analyze_batch_progress"]


class _FakeAnalyzer:
    def __init__(self, outcomes: dict[Path, AnalysisOutcome]) -> None:
        self._outcomes = outcomes

    def analyze(
        self,
        scan_result: ScanResult,
        *,
        on_progress: Callable[[float], None] | None = None,
        copy_video: bool = False,  # noqa: ARG002
        grain_override: bool | None = None,  # noqa: ARG002
    ) -> AnalysisOutcome:
        assert on_progress is not None
        on_progress(1.0)
        return self._outcomes[scan_result.main_file]


class _FakeProber:
    def __init__(self, crops: dict[Path, CropRect | None] | None = None) -> None:
        self._crops = crops if crops is not None else {}
        self.calls: list[tuple[Path, float, bool, bool, str | None, object]] = []

    def detect_crop(
        self,
        path: Path,
        duration_s: float,
        *,
        interlaced: bool = False,
        is_dvd: bool = False,
        hdr_transfer: str | None = None,
        on_progress: object = None,
    ) -> CropRect | None:
        self.calls.append((path, duration_s, interlaced, is_dvd, hdr_transfer, on_progress))
        return self._crops.get(path)


def _sr(tmp_path: Path, name: str) -> ScanResult:
    return ScanResult(
        main_file=tmp_path / name,
        satellite_files=[],
        output_path=tmp_path / f"out_{name}",
    )


def _done(main_file: Path, *, detail: str) -> AnalysisOutcome:
    movie = make_movie(main_file=main_file, video=make_video_info(source_file=main_file))
    return AnalysisOutcome(movie, AnalyzeStatus.DONE, detail)


def _build_pipeline(
    analyzer: object,
    prober: object,
    reporter: PlanReporter,
) -> AnalysisPipeline:
    return AnalysisPipeline(
        cast("Analyzer", analyzer),
        cast("Prober", prober),
        reporter,
        max_workers=1,
    )


def test_full_reporter_sequence_mixed_statuses(tmp_path: Path) -> None:
    good = _sr(tmp_path, "good.mkv")
    old = _sr(tmp_path, "old.mkv")
    bad = _sr(tmp_path, "bad.mkv")
    outcomes = {
        good.main_file: _done(good.main_file, detail="hevc 1920x1080 24fps SDR, 2 audio (eng), 1 subs"),
        old.main_file: AnalysisOutcome(None, AnalyzeStatus.SKIPPED, "output file already exists"),
        bad.main_file: AnalysisOutcome(None, AnalyzeStatus.FAILED, "HDR10+ not supported"),
    }
    reporter = RecordingPlanReporter()
    pipeline = _build_pipeline(_FakeAnalyzer(outcomes), _FakeProber(), reporter)

    pipeline.run([good, old, bad], copy_video=False, dry_run=False)

    assert _framing(reporter) == [
        Event("analyze_batch_start", (3,), ()),
        Event(
            "analyze_batch_item",
            ("good.mkv", "hevc 1920x1080 24fps SDR, 2 audio (eng), 1 subs"),
            (("status", AnalyzeStatus.DONE),),
        ),
        Event("analyze_batch_item", ("old.mkv", "output file already exists"), (("status", AnalyzeStatus.SKIPPED),)),
        Event("analyze_batch_item", ("bad.mkv", "HDR10+ not supported"), (("status", AnalyzeStatus.FAILED),)),
        Event("analyze_batch_finish", (), ()),
    ]
    progress = [e for e in reporter.events if e.method == "analyze_batch_progress"]
    assert progress[-1] == Event("analyze_batch_progress", (3.0,), ())


def test_single_done_file_full_sequence(tmp_path: Path) -> None:
    only = _sr(tmp_path, "only.mkv")
    outcomes = {only.main_file: _done(only.main_file, detail="h264 1920x1080 24fps SDR, 1 audio (rus), 0 subs")}
    reporter = RecordingPlanReporter()
    pipeline = _build_pipeline(_FakeAnalyzer(outcomes), _FakeProber(), reporter)

    pipeline.run([only], copy_video=False, dry_run=False)

    assert _framing(reporter) == [
        Event("analyze_batch_start", (1,), ()),
        Event(
            "analyze_batch_item",
            ("only.mkv", "h264 1920x1080 24fps SDR, 1 audio (rus), 0 subs"),
            (("status", AnalyzeStatus.DONE),),
        ),
        Event("analyze_batch_finish", (), ()),
    ]
    progress = [e for e in reporter.events if e.method == "analyze_batch_progress"]
    assert progress[-1] == Event("analyze_batch_progress", (1.0,), ())
