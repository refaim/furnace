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
    assert methods[-1] == "plan_file_done"
    assert "plan_microop" not in methods
    assert "plan_progress" not in methods


def test_plan_without_reporter_is_silent() -> None:
    planner = PlannerService(previewer=None)
    plan = planner.create_plan(
        movies=[(_make_movie(), Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
    )

    assert len(plan.jobs) == 1


def test_format_plan_summary_no_crop_no_deinterlace() -> None:
    movie = _make_movie()
    planner = PlannerService(previewer=None)
    plan = planner.create_plan(
        movies=[(movie, Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
    )
    summary = _format_plan_summary(movie, plan.jobs[0])
    assert summary == "1920x1080 to 1920x1080"
    assert "cq" not in summary


def test_format_plan_summary_with_crop_uses_cropped_dims() -> None:
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
    pal_dvd_video = replace(
        _make_video_info(),
        width=720,
        height=576,
        sar_num=16,
        sar_den=15,
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
    assert "cq" not in summary


def test_format_plan_summary_passthrough() -> None:
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
    dv_video = replace(
        _make_video_info(),
        hdr=HdrMetadata(is_dolby_vision=True, dv_profile=7),
    )
    return replace(_make_movie(), video=dv_video)


def test_format_plan_summary_dv_p7_fallback() -> None:
    movie = _make_dv_p7_movie()
    planner = PlannerService(previewer=None)
    plan = planner.create_plan(
        movies=[(movie, Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        copy_video=True,
    )
    job = plan.jobs[0]
    assert job.video_params.passthrough is False
    assert job.video_params.dv_mode == DvMode.TO_8_1
    summary = _format_plan_summary(movie, job, "DV P7 FEL")
    assert summary.startswith("encode (DV P7 FEL), ")
    assert "cq" not in summary


def test_format_plan_summary_normal_encode_has_no_reason_prefix() -> None:
    movie = _make_movie()
    planner = PlannerService(previewer=None)
    plan = planner.create_plan(
        movies=[(movie, Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
    )
    summary = _format_plan_summary(movie, plan.jobs[0])
    assert summary.startswith("1920x1080 to ")
    assert "cq" not in summary
    assert "encode (" not in summary
    assert "passthrough" not in summary


def test_passthrough_summary_emitted_via_reporter() -> None:
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
