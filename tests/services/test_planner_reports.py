"""PlannerService emits PlanReporter events for the per-movie plan loop.

Covers:
- ``plan_file_start(name)`` -> ``plan_file_done(summary)`` (the planner no
  longer runs cropdetect, so it emits no ``plan_microop``/``plan_progress``)
- ``reporter=None`` keeps the planner fully silent (legacy headless behavior)
- ``_format_plan_summary`` formats with/without crop and with/without deinterlace
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from furnace.core.models import (
    CropRect,
    DvMode,
    HdrMetadata,
    Movie,
    VideoInfo,
)
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


def test_plan_file_emits_start_and_done() -> None:
    reporter = RecordingPlanReporter()
    planner = PlannerService(previewer=None, reporter=reporter)
    planner.create_plan(
        movies=[(_make_movie(), Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
    )

    methods = [e.method for e in reporter.events]
    assert "plan_file_start" in methods
    # plan_file_done is the final event emitted by the planner itself
    # (plan_saved is the CLI's responsibility, per the spec).
    assert methods[-1] == "plan_file_done"
    # cropdetect is gone, so the planner emits no microop/progress events.
    assert "plan_microop" not in methods
    assert "plan_progress" not in methods


def test_plan_without_reporter_is_silent() -> None:
    planner = PlannerService(previewer=None)  # no reporter
    plan = planner.create_plan(
        movies=[(_make_movie(), Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
    )

    assert len(plan.jobs) == 1


def test_format_plan_summary_no_crop_no_deinterlace() -> None:
    """Without a crop or deinterlace flag, summary uses source dims only."""
    movie = _make_movie()
    planner = PlannerService(previewer=None)
    plan = planner.create_plan(
        movies=[(movie, Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
    )
    summary = _format_plan_summary(movie, plan.jobs[0])
    assert summary.endswith("1920x1080 to 1920x1080")
    assert summary.startswith("cq ")
    assert "deinterlace" not in summary


def test_format_plan_summary_with_crop_uses_cropped_dims() -> None:
    """When a crop is set, summary destination dims come from the crop rect."""
    movie = _make_movie()
    planner = PlannerService(previewer=None)
    plan = planner.create_plan(
        movies=[(movie, Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        precomputed_crops={movie.main_file: CropRect(w=1920, h=800, x=0, y=140)},
    )
    summary = _format_plan_summary(movie, plan.jobs[0])
    assert "1920x1080 to 1920x800" in summary


def test_format_plan_summary_includes_deinterlace_flag() -> None:
    """When ``deinterlace`` is set on video params, summary appends it."""
    interlaced_video = replace(_make_video_info(), interlaced=True)
    movie = replace(_make_movie(), video=interlaced_video)
    planner = PlannerService(previewer=None)
    plan = planner.create_plan(
        movies=[(movie, Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
    )
    summary = _format_plan_summary(movie, plan.jobs[0])
    assert summary.endswith(", deinterlace")


def test_format_plan_summary_anamorphic_dvd_uses_real_output_dims() -> None:
    """PAL DVD anamorphic source: 720x576 SAR 16:15 + crop 704x400.
    The summary must reflect the actual encoded output 744x400, not the
    raw crop dims or the SAR-corrected-but-unaligned 751x400."""
    pal_dvd_video = replace(
        _make_video_info(),
        width=720, height=576,
        sar_num=16, sar_den=15,
    )
    movie = replace(_make_movie(), video=pal_dvd_video)
    planner = PlannerService(previewer=None)
    plan = planner.create_plan(
        movies=[(movie, Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        precomputed_crops={movie.main_file: CropRect(w=704, h=400, x=8, y=88)},
    )
    summary = _format_plan_summary(movie, plan.jobs[0])
    assert "720x576 to 744x400" in summary


def test_plan_file_done_summary_is_emitted_via_reporter() -> None:
    """End-to-end: the reporter receives a ``plan_file_done`` with a usable summary."""
    reporter = RecordingPlanReporter()
    planner = PlannerService(previewer=None, reporter=reporter)
    planner.create_plan(
        movies=[(_make_movie(), Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
    )

    done_events = [e for e in reporter.events if e.method == "plan_file_done"]
    assert len(done_events) == 1
    summary = done_events[0].args[0]
    assert isinstance(summary, str)
    assert "1920x1080" in summary
    assert "cq " in summary


def test_format_plan_summary_passthrough() -> None:
    """A passthrough job renders the copy-video label, not encode settings."""
    movie = _make_movie()
    planner = PlannerService(previewer=None)
    plan = planner.create_plan(
        movies=[(movie, Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        copy_video=True,
    )
    summary = _format_plan_summary(movie, plan.jobs[0])
    assert summary == "passthrough (copy video)"


def test_format_plan_summary_interlaced_fallback() -> None:
    """An interlaced fallback-to-encode job is labelled with its reason."""
    movie = replace(_make_movie(), video=replace(_make_video_info(), interlaced=True))
    planner = PlannerService(previewer=None)
    plan = planner.create_plan(
        movies=[(movie, Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        copy_video=True,
    )
    summary = _format_plan_summary(movie, plan.jobs[0], "interlaced")
    assert summary.startswith("encode (interlaced), ")
    assert summary.endswith(", deinterlace")


def _make_dv_p7_movie() -> Movie:
    """A genuine Dolby Vision Profile 7 FEL source that must fall back to encode."""
    dv_video = replace(
        _make_video_info(),
        hdr=HdrMetadata(is_dolby_vision=True, dv_profile=7),
    )
    return replace(_make_movie(), video=dv_video)


def test_format_plan_summary_dv_p7_fallback() -> None:
    """A genuine DV P7 FEL source falls back to encode and is labelled accordingly."""
    movie = _make_dv_p7_movie()
    planner = PlannerService(previewer=None)
    plan = planner.create_plan(
        movies=[(movie, Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        copy_video=True,
    )
    job = plan.jobs[0]
    # The DV P7 FEL source genuinely fell back to the encode path.
    assert job.video_params.passthrough is False
    assert job.video_params.dv_mode == DvMode.TO_8_1
    summary = _format_plan_summary(movie, job, "DV P7 FEL")
    assert summary.startswith("encode (DV P7 FEL), ")
    assert "cq " in summary


def test_format_plan_summary_normal_encode_has_no_reason_prefix() -> None:
    """Without a fallback reason (copy-video off), summary is the plain encode line."""
    movie = _make_movie()
    planner = PlannerService(previewer=None)
    plan = planner.create_plan(
        movies=[(movie, Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
    )
    summary = _format_plan_summary(movie, plan.jobs[0])
    assert summary.startswith("cq ")
    assert "encode (" not in summary
    assert "passthrough" not in summary


def test_passthrough_summary_emitted_via_reporter() -> None:
    """End-to-end: a passthrough job's plan_file_done summary says copy video."""
    reporter = RecordingPlanReporter()
    planner = PlannerService(previewer=None, reporter=reporter)
    planner.create_plan(
        movies=[(_make_movie(), Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        copy_video=True,
    )
    done = [e for e in reporter.events if e.method == "plan_file_done"]
    summary = done[0].args[0]
    assert summary == "passthrough (copy video)"


def test_interlaced_fallback_reason_emitted_via_reporter() -> None:
    """End-to-end: an interlaced fallback surfaces its reason in the report path."""
    movie = replace(_make_movie(), video=replace(_make_video_info(), interlaced=True))
    reporter = RecordingPlanReporter()
    planner = PlannerService(previewer=None, reporter=reporter)
    planner.create_plan(
        movies=[(movie, Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        copy_video=True,
    )
    done = [e for e in reporter.events if e.method == "plan_file_done"]
    summary = done[0].args[0]
    assert isinstance(summary, str)
    assert summary.startswith("encode (interlaced), ")


def test_dv_p7_fallback_reason_emitted_via_reporter() -> None:
    """End-to-end: a genuine DV P7 FEL fallback surfaces its reason in the report path."""
    movie = _make_dv_p7_movie()
    reporter = RecordingPlanReporter()
    planner = PlannerService(previewer=None, reporter=reporter)
    planner.create_plan(
        movies=[(movie, Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        copy_video=True,
    )
    done = [e for e in reporter.events if e.method == "plan_file_done"]
    summary = done[0].args[0]
    assert isinstance(summary, str)
    assert summary.startswith("encode (DV P7 FEL), ")
