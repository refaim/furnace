"""PlannerService emits PlanReporter events for the per-movie plan loop.

Covers:
- ``plan_file_start(name)`` -> ``plan_microop("cropdetect", has_progress=True)``
  -> ``plan_progress(fraction)`` -> ``plan_file_done(summary)``
- ``dry_run=True`` skips the cropdetect microop entirely
- ``reporter=None`` keeps the planner fully silent (legacy headless behavior)
- ``_format_plan_summary`` formats with/without crop and with/without deinterlace
- ``_on_crop_progress`` forwards samples only when both reporter and
  ``sample.fraction`` are non-None
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from furnace.core.models import (
    CropRect,
    HdrMetadata,
    Movie,
    VideoInfo,
)
from furnace.core.progress import ProgressSample
from furnace.services.planner import PlannerService, _format_plan_summary
from tests.fakes.recording_reporter import RecordingPlanReporter


def _make_video_info() -> VideoInfo:
    return VideoInfo(
        index=0,
        codec_name="hevc",
        width=1920,
        height=1080,
        pixel_area=1920 * 1080,
        fps_num=24,
        fps_den=1,
        duration_s=100.0,
        interlaced=False,
        color_matrix_raw="bt709",
        color_range="tv",
        color_transfer="bt709",
        color_primaries="bt709",
        pix_fmt="yuv420p10le",
        hdr=HdrMetadata(),
        source_file=Path("/in/x.mkv"),
        bitrate=8_000_000,
        sar_num=1,
        sar_den=1,
    )


def _make_movie() -> Movie:
    return Movie(
        main_file=Path("/in/x.mkv"),
        satellite_files=[],
        video=_make_video_info(),
        audio_tracks=[],
        subtitle_tracks=[],
        attachments=[],
        has_chapters=False,
        file_size=1_000_000,
    )


def test_plan_file_emits_start_microop_done() -> None:
    prober = MagicMock()
    prober.detect_crop.return_value = None
    reporter = RecordingPlanReporter()
    planner = PlannerService(prober=prober, previewer=None, reporter=reporter)
    planner.create_plan(
        movies=[(_make_movie(), Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        vmaf_enabled=False,
        dry_run=False,
    )

    methods = [e.method for e in reporter.events]
    assert "plan_file_start" in methods
    triples = [(e.method, e.args, e.kwargs) for e in reporter.events]
    assert ("plan_microop", ("cropdetect",), (("has_progress", True),)) in triples
    # plan_file_done is the final event emitted by the planner itself
    # (plan_saved is the CLI's responsibility, per the spec).
    assert methods[-1] == "plan_file_done"


def test_plan_dry_run_skips_cropdetect_microop() -> None:
    prober = MagicMock()
    reporter = RecordingPlanReporter()
    planner = PlannerService(prober=prober, previewer=None, reporter=reporter)
    planner.create_plan(
        movies=[(_make_movie(), Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        vmaf_enabled=False,
        dry_run=True,
    )

    labels = [e.args[0] for e in reporter.events if e.method == "plan_microop"]
    assert "cropdetect" not in labels  # has_progress lives in kwargs


def test_plan_without_reporter_is_silent() -> None:
    prober = MagicMock()
    prober.detect_crop.return_value = None
    planner = PlannerService(prober=prober, previewer=None)  # no reporter
    plan = planner.create_plan(
        movies=[(_make_movie(), Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        vmaf_enabled=False,
        dry_run=False,
    )

    assert len(plan.jobs) == 1


def test_cropdetect_progress_samples_forwarded_to_reporter() -> None:
    """Samples passed to ``on_progress`` flow through to ``plan_progress``."""

    def fake_detect_crop(
        path: Path,
        duration_s: float,
        *,
        interlaced: bool = False,
        is_dvd: bool = False,
        on_progress: Any = None,
    ) -> CropRect | None:
        assert on_progress is not None
        on_progress(ProgressSample(fraction=0.25))
        on_progress(ProgressSample(fraction=0.75))
        return None

    prober = MagicMock()
    prober.detect_crop.side_effect = fake_detect_crop
    reporter = RecordingPlanReporter()
    planner = PlannerService(prober=prober, previewer=None, reporter=reporter)
    planner.create_plan(
        movies=[(_make_movie(), Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        vmaf_enabled=False,
        dry_run=False,
    )

    fractions = [e.args[0] for e in reporter.events if e.method == "plan_progress"]
    assert fractions == [0.25, 0.75]


def test_cropdetect_progress_sample_without_fraction_is_dropped() -> None:
    """Samples whose ``fraction`` is None must not produce ``plan_progress`` events."""
    reporter = RecordingPlanReporter()
    planner = PlannerService(prober=MagicMock(), previewer=None, reporter=reporter)
    # processed_s only -> fraction is None -> nothing forwarded
    planner._on_crop_progress(ProgressSample(processed_s=10.0))

    assert [e for e in reporter.events if e.method == "plan_progress"] == []


def test_on_crop_progress_no_reporter_is_noop() -> None:
    """Without a reporter, the helper must not raise even with a fraction set."""
    planner = PlannerService(prober=MagicMock(), previewer=None)  # no reporter
    # Should silently do nothing.
    planner._on_crop_progress(ProgressSample(fraction=0.5))


def test_format_plan_summary_no_crop_no_deinterlace() -> None:
    """Without a crop or deinterlace flag, summary uses source dims only."""
    prober = MagicMock()
    prober.detect_crop.return_value = None
    planner = PlannerService(prober=prober, previewer=None)
    plan = planner.create_plan(
        movies=[(_make_movie(), Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        vmaf_enabled=False,
        dry_run=False,
    )
    summary = _format_plan_summary(_make_movie(), plan.jobs[0])
    assert summary.endswith("1920x1080 to 1920x1080")
    assert summary.startswith("cq ")
    assert "deinterlace" not in summary


def test_format_plan_summary_with_crop_uses_cropped_dims() -> None:
    """When a crop is set, summary destination dims come from the crop rect."""
    prober = MagicMock()
    prober.detect_crop.return_value = CropRect(w=1920, h=800, x=0, y=140)
    planner = PlannerService(prober=prober, previewer=None)
    plan = planner.create_plan(
        movies=[(_make_movie(), Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        vmaf_enabled=False,
        dry_run=False,
    )
    summary = _format_plan_summary(_make_movie(), plan.jobs[0])
    assert "1920x1080 to 1920x800" in summary


def test_format_plan_summary_includes_deinterlace_flag() -> None:
    """When ``deinterlace`` is set on video params, summary appends it."""
    interlaced_video = replace(_make_video_info(), interlaced=True)
    movie = replace(_make_movie(), video=interlaced_video)
    prober = MagicMock()
    prober.detect_crop.return_value = None
    planner = PlannerService(prober=prober, previewer=None)
    plan = planner.create_plan(
        movies=[(movie, Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        vmaf_enabled=False,
        dry_run=False,
    )
    summary = _format_plan_summary(movie, plan.jobs[0])
    assert summary.endswith(", deinterlace")


def test_plan_file_done_summary_is_emitted_via_reporter() -> None:
    """End-to-end: the reporter receives a ``plan_file_done`` with a usable summary."""
    prober = MagicMock()
    prober.detect_crop.return_value = None
    reporter = RecordingPlanReporter()
    planner = PlannerService(prober=prober, previewer=None, reporter=reporter)
    planner.create_plan(
        movies=[(_make_movie(), Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        vmaf_enabled=False,
        dry_run=False,
    )

    done_events = [e for e in reporter.events if e.method == "plan_file_done"]
    assert len(done_events) == 1
    summary = done_events[0].args[0]
    assert isinstance(summary, str)
    assert "1920x1080" in summary
    assert "cq " in summary
