"""Tests that ``Analyzer.analyze`` emits the right ``PlanReporter`` events.

The analyzer threads an optional ``PlanReporter`` so the CLI can render
per-file lifecycle: ``analyze_file_start`` -> ``analyze_microop`` (probing,
optional ``HDR side data``, optional ``idet``, optional per-track
``audio profile track N``) -> ``analyze_file_done`` /
``analyze_file_failed`` / ``analyze_file_skipped``. Progress fractions
from the ffmpeg per-point callback are forwarded via
``analyze_progress``.

The tests use the ``RecordingPlanReporter`` test fake to capture the
event sequence and assert against it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from furnace.core.audio_profile import AudioMetrics
from furnace.core.models import ScanResult
from furnace.core.progress import ProgressSample
from furnace.services.analyzer import Analyzer
from tests.fakes.recording_reporter import Event, RecordingPlanReporter


def _arg_str(event: Event, idx: int = 0) -> str:
    """Return the str-typed positional arg of a captured event."""
    val = event.args[idx]
    assert isinstance(val, str)
    return val


def _arg_float(event: Event, idx: int = 0) -> float:
    """Return the float-typed positional arg of a captured event."""
    val = event.args[idx]
    assert isinstance(val, float)
    return val


def _make_scan(tmp_path: Path, name: str = "x.mkv") -> ScanResult:
    main = tmp_path / name
    main.write_bytes(b"\x00" * 8)
    out = tmp_path / "out" / name
    return ScanResult(
        main_file=main,
        satellite_files=[],
        output_path=out,
    )


def _sdr_probe_dict() -> dict[str, Any]:
    return {
        "streams": [
            {
                "codec_type": "video",
                "index": 0,
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "24/1",
                "r_frame_rate": "24/1",
                "duration": "100",
                "color_primaries": "bt709",
                "color_transfer": "bt709",
                "color_space": "bt709",
                "pix_fmt": "yuv420p",
                "field_order": "progressive",
                "sample_aspect_ratio": "1:1",
                "side_data_list": [],
            },
        ],
        "format": {},
        "chapters": [],
    }


def _make_prober_simple_sdr() -> MagicMock:
    prober = MagicMock()
    prober.get_encoder_tag.return_value = None
    prober.probe.return_value = _sdr_probe_dict()
    prober.run_idet.return_value = 0.0
    prober.probe_hdr_side_data.return_value = []
    return prober


def test_simple_sdr_emits_start_probing_done(tmp_path: Path) -> None:
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=_make_prober_simple_sdr(), reporter=reporter)
    movie = analyzer.analyze(_make_scan(tmp_path, "Inception.mkv"))
    assert movie is not None
    methods = [e.method for e in reporter.events]
    assert methods[0] == "analyze_file_start"
    assert reporter.events[0].args == ("Inception.mkv",)
    assert ("analyze_microop", ("probing",), (("has_progress", False),)) in [
        (e.method, e.args, e.kwargs) for e in reporter.events
    ]
    assert methods[-1] == "analyze_file_done"


def test_skip_already_encoded_emits_skipped(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    prober.get_encoder_tag.return_value = "Furnace 1.13.2"
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    movie = analyzer.analyze(_make_scan(tmp_path, "foo.mkv"))
    assert movie is None
    assert reporter.events[-1].method == "analyze_file_skipped"


def test_skip_already_encoded_silent_no_reporter(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    prober.get_encoder_tag.return_value = "Furnace 1.13.2"
    analyzer = Analyzer(prober=prober)
    movie = analyzer.analyze(_make_scan(tmp_path, "foo.mkv"))
    assert movie is None


def test_no_video_stream_emits_skipped(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    prober.probe.return_value = {
        "streams": [{"codec_type": "audio"}],
        "format": {},
        "chapters": [],
    }
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    movie = analyzer.analyze(_make_scan(tmp_path, "foo.mkv"))
    assert movie is None
    assert reporter.events[-1].method == "analyze_file_skipped"
    assert "no video stream" in _arg_str(reporter.events[-1])


def test_no_video_stream_silent_no_reporter(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    prober.probe.return_value = {
        "streams": [{"codec_type": "audio"}],
        "format": {},
        "chapters": [],
    }
    analyzer = Analyzer(prober=prober)
    movie = analyzer.analyze(_make_scan(tmp_path, "foo.mkv"))
    assert movie is None


def test_probe_failure_emits_failed(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    prober.probe.side_effect = RuntimeError("boom")
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    movie = analyzer.analyze(_make_scan(tmp_path, "foo.mkv"))
    assert movie is None
    assert reporter.events[-1].method == "analyze_file_failed"
    assert reporter.events[-1].args == ("probe failed",)


def test_probe_failure_silent_no_reporter(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    prober.probe.side_effect = RuntimeError("boom")
    analyzer = Analyzer(prober=prober)
    movie = analyzer.analyze(_make_scan(tmp_path, "foo.mkv"))
    assert movie is None


def test_parse_video_failure_emits_failed(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    # avg_frame_rate as garbage -> ValueError in parse
    bad = _sdr_probe_dict()
    bad["streams"][0]["avg_frame_rate"] = "abc/def"
    prober.probe.return_value = bad
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    movie = analyzer.analyze(_make_scan(tmp_path, "foo.mkv"))
    assert movie is None
    assert reporter.events[-1].method == "analyze_file_failed"
    assert reporter.events[-1].args == ("parse failed",)


def test_parse_video_failure_silent_no_reporter(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    bad = _sdr_probe_dict()
    bad["streams"][0]["avg_frame_rate"] = "abc/def"
    prober.probe.return_value = bad
    analyzer = Analyzer(prober=prober)
    movie = analyzer.analyze(_make_scan(tmp_path, "foo.mkv"))
    assert movie is None


def test_hdr10_emits_hdr_side_data_microop(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    probe = _sdr_probe_dict()
    probe["streams"][0]["color_transfer"] = "smpte2084"
    prober.probe.return_value = probe
    prober.probe_hdr_side_data.return_value = []
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    analyzer.analyze(_make_scan(tmp_path, "hdr.mkv"))
    labels = [_arg_str(e) for e in reporter.events if e.method == "analyze_microop"]
    assert "HDR side data" in labels


def test_hlg_emits_hdr_side_data_microop(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    probe = _sdr_probe_dict()
    probe["streams"][0]["color_transfer"] = "arib-std-b67"
    prober.probe.return_value = probe
    prober.probe_hdr_side_data.return_value = []
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    analyzer.analyze(_make_scan(tmp_path, "hlg.mkv"))
    labels = [_arg_str(e) for e in reporter.events if e.method == "analyze_microop"]
    assert "HDR side data" in labels


def test_hdr10_silent_no_reporter(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    probe = _sdr_probe_dict()
    probe["streams"][0]["color_transfer"] = "smpte2084"
    prober.probe.return_value = probe
    prober.probe_hdr_side_data.return_value = []
    analyzer = Analyzer(prober=prober)
    movie = analyzer.analyze(_make_scan(tmp_path, "hdr.mkv"))
    assert movie is not None


def test_hdr10_plus_emits_failed_then_raises(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    probe = _sdr_probe_dict()
    probe["streams"][0]["color_transfer"] = "smpte2084"
    prober.probe.return_value = probe
    prober.probe_hdr_side_data.return_value = [{"side_data_type": "HDR10+ Dynamic Metadata"}]
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    with pytest.raises(ValueError, match="HDR10\\+"):
        analyzer.analyze(_make_scan(tmp_path, "hdr10p.mkv"))
    assert any(
        e.method == "analyze_file_failed" and e.args == ("HDR10+ not supported",)
        for e in reporter.events
    )


def test_hdr10_plus_silent_no_reporter_still_raises(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    probe = _sdr_probe_dict()
    probe["streams"][0]["color_transfer"] = "smpte2084"
    prober.probe.return_value = probe
    prober.probe_hdr_side_data.return_value = [{"side_data_type": "HDR10+ Dynamic Metadata"}]
    analyzer = Analyzer(prober=prober)
    with pytest.raises(ValueError, match="HDR10\\+"):
        analyzer.analyze(_make_scan(tmp_path, "hdr10p.mkv"))


def test_idet_microop_and_progress_forwarded(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    probe = _sdr_probe_dict()
    probe["streams"][0]["field_order"] = "tt"
    probe["streams"][0]["avg_frame_rate"] = "25/1"
    probe["streams"][0]["r_frame_rate"] = "25/1"
    prober.probe.return_value = probe

    captured: list[Any] = []

    def fake_run_idet(path: Path, duration_s: float, *, on_progress: Any = None) -> float:
        captured.append((path, duration_s, on_progress))
        if on_progress is not None:
            on_progress(ProgressSample(fraction=0.25))
            on_progress(ProgressSample(fraction=0.5))
            on_progress(ProgressSample(fraction=None))
        return 0.0

    prober.run_idet.side_effect = fake_run_idet
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    analyzer.analyze(_make_scan(tmp_path, "interlaced.mkv"))

    labels = [_arg_str(e) for e in reporter.events if e.method == "analyze_microop"]
    assert "idet" in labels

    progress = [e for e in reporter.events if e.method == "analyze_progress"]
    assert [_arg_float(e) for e in progress] == [0.25, 0.5]

    assert captured
    assert captured[0][2] is not None


def test_idet_silent_no_reporter_does_not_pass_callback(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    probe = _sdr_probe_dict()
    probe["streams"][0]["field_order"] = "tt"
    probe["streams"][0]["avg_frame_rate"] = "25/1"
    probe["streams"][0]["r_frame_rate"] = "25/1"
    prober.probe.return_value = probe

    analyzer = Analyzer(prober=prober)
    analyzer.analyze(_make_scan(tmp_path, "interlaced.mkv"))

    # When no reporter, on_progress should be None (default arg).
    call = prober.run_idet.call_args
    assert call.kwargs.get("on_progress") is None


def test_idet_failure_does_not_emit_failed(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    probe = _sdr_probe_dict()
    probe["streams"][0]["field_order"] = "tt"
    probe["streams"][0]["avg_frame_rate"] = "25/1"
    probe["streams"][0]["r_frame_rate"] = "25/1"
    prober.probe.return_value = probe
    prober.run_idet.side_effect = RuntimeError("idet crashed")

    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    movie = analyzer.analyze(_make_scan(tmp_path, "interlaced.mkv"))

    # idet failure is swallowed; analysis still completes successfully.
    assert movie is not None
    assert reporter.events[-1].method == "analyze_file_done"


def _stereo_metrics() -> AudioMetrics:
    return AudioMetrics(
        channels=2,
        rms_l=-20.0, rms_r=-20.0, rms_c=None, rms_lfe=None,
        rms_ls=None, rms_rs=None, rms_lb=None, rms_rb=None,
        corr_lr=0.3, corr_ls_l=None, corr_rs_r=None, corr_ls_rs=None,
        corr_lb_ls=None, corr_rb_rs=None,
    )


def _real_5_1_metrics() -> AudioMetrics:
    return AudioMetrics(
        channels=6,
        rms_l=-22.0, rms_r=-22.0, rms_c=-18.0, rms_lfe=-20.0,
        rms_ls=-25.0, rms_rs=-25.0, rms_lb=None, rms_rb=None,
        corr_lr=0.3, corr_ls_l=0.1, corr_rs_r=0.1, corr_ls_rs=0.4,
        corr_lb_ls=None, corr_rb_rs=None,
    )


def test_audio_profile_microop_and_progress_forwarded(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    probe = _sdr_probe_dict()
    probe["streams"].append({
        "index": 1,
        "codec_type": "audio",
        "codec_name": "ac3",
        "channels": 2,
        "tags": {"language": "eng"},
        "disposition": {"default": 1, "forced": 0},
    })
    prober.probe.return_value = probe

    def fake_profile(*, path: Path, stream_index: int, channels: int,
                    duration_s: float, on_progress: Any = None) -> AudioMetrics:
        if on_progress is not None:
            on_progress(ProgressSample(fraction=0.5))
            on_progress(ProgressSample(fraction=None))
            on_progress(ProgressSample(fraction=1.0))
        return _stereo_metrics()

    prober.profile_audio_track.side_effect = fake_profile
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    analyzer.analyze(_make_scan(tmp_path, "audio.mkv"))

    labels = [_arg_str(e) for e in reporter.events if e.method == "analyze_microop"]
    assert "audio profile track 1" in labels
    progress = [e for e in reporter.events if e.method == "analyze_progress"]
    fractions = [_arg_float(e) for e in progress]
    assert 0.5 in fractions
    assert 1.0 in fractions


def test_audio_profile_failure_continues(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    probe = _sdr_probe_dict()
    probe["streams"].append({
        "index": 1,
        "codec_type": "audio",
        "codec_name": "ac3",
        "channels": 2,
        "tags": {"language": "eng"},
        "disposition": {"default": 1, "forced": 0},
    })
    prober.probe.return_value = probe
    prober.profile_audio_track.side_effect = RuntimeError("decoded zero windows")
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    movie = analyzer.analyze(_make_scan(tmp_path, "audio.mkv"))
    assert movie is not None
    # Failure of one audio track is fail-soft; analyze still finishes ``done``.
    assert reporter.events[-1].method == "analyze_file_done"


def test_analyze_without_reporter_is_silent(tmp_path: Path) -> None:
    analyzer = Analyzer(prober=_make_prober_simple_sdr())  # no reporter
    movie = analyzer.analyze(_make_scan(tmp_path, "x.mkv"))
    assert movie is not None


def test_done_summary_has_codec_resolution_audio_subs(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    probe = _sdr_probe_dict()
    probe["streams"].append({
        "index": 1,
        "codec_type": "audio",
        "codec_name": "ac3",
        "channels": 6,
        "tags": {"language": "eng"},
        "disposition": {"default": 1, "forced": 0},
    })
    probe["streams"].append({
        "index": 2,
        "codec_type": "subtitle",
        "codec_name": "subrip",
        "tags": {"language": "rus"},
        "disposition": {"default": 0, "forced": 0},
    })
    prober.probe.return_value = probe
    prober.profile_audio_track.return_value = _real_5_1_metrics()
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    movie = analyzer.analyze(_make_scan(tmp_path, "movie.mkv"))
    assert movie is not None

    done = next(e for e in reporter.events if e.method == "analyze_file_done")
    summary = _arg_str(done)
    assert isinstance(summary, str)
    assert "h264" in summary
    assert "1920x1080" in summary
    assert "24fps" in summary
    assert "SDR" in summary
    assert "1 audio" in summary
    assert "1 subs" in summary
    assert "eng" in summary


def test_summary_marks_interlaced(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    probe = _sdr_probe_dict()
    probe["streams"][0]["field_order"] = "tt"
    probe["streams"][0]["avg_frame_rate"] = "25/1"
    probe["streams"][0]["r_frame_rate"] = "25/1"
    prober.probe.return_value = probe
    prober.run_idet.return_value = 0.9
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    analyzer.analyze(_make_scan(tmp_path, "i.mkv"))
    done = next(e for e in reporter.events if e.method == "analyze_file_done")
    assert "interlaced" in _arg_str(done)


def test_summary_hdr10_class(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    probe = _sdr_probe_dict()
    probe["streams"][0]["color_transfer"] = "smpte2084"
    prober.probe.return_value = probe
    prober.probe_hdr_side_data.return_value = [
        {"side_data_type": "Mastering display metadata"},
    ]
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    analyzer.analyze(_make_scan(tmp_path, "hdr10.mkv"))
    done = next(e for e in reporter.events if e.method == "analyze_file_done")
    assert "HDR10" in _arg_str(done)


def test_summary_hlg_class(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    probe = _sdr_probe_dict()
    probe["streams"][0]["color_transfer"] = "arib-std-b67"
    prober.probe.return_value = probe
    prober.probe_hdr_side_data.return_value = []
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    analyzer.analyze(_make_scan(tmp_path, "hlg.mkv"))
    done = next(e for e in reporter.events if e.method == "analyze_file_done")
    assert "HLG" in _arg_str(done)


def test_summary_dv_class(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    probe = _sdr_probe_dict()
    probe["streams"][0]["color_transfer"] = "smpte2084"
    probe["streams"][0]["side_data_list"] = [
        {
            "side_data_type": "DOVI configuration record",
            "dv_profile": 8,
            "dv_bl_signal_compatibility_id": 1,
        },
    ]
    prober.probe.return_value = probe
    prober.probe_hdr_side_data.return_value = []
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    analyzer.analyze(_make_scan(tmp_path, "dv.mkv"))
    done = next(e for e in reporter.events if e.method == "analyze_file_done")
    summary = _arg_str(done)
    assert "DV" in summary
    assert "P8" in summary
    assert "BL=HDR10" in summary


def test_summary_dv_bl_sdr_compat(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    probe = _sdr_probe_dict()
    probe["streams"][0]["color_transfer"] = "smpte2084"
    probe["streams"][0]["side_data_list"] = [
        {
            "side_data_type": "DOVI configuration record",
            "dv_profile": 8,
            "dv_bl_signal_compatibility_id": 2,
        },
    ]
    prober.probe.return_value = probe
    prober.probe_hdr_side_data.return_value = []
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    analyzer.analyze(_make_scan(tmp_path, "dv.mkv"))
    done = next(e for e in reporter.events if e.method == "analyze_file_done")
    assert "BL=SDR" in _arg_str(done)


def test_summary_dv_bl_hlg_compat(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    probe = _sdr_probe_dict()
    probe["streams"][0]["color_transfer"] = "smpte2084"
    probe["streams"][0]["side_data_list"] = [
        {
            "side_data_type": "DOVI configuration record",
            "dv_profile": 8,
            "dv_bl_signal_compatibility_id": 4,
        },
    ]
    prober.probe.return_value = probe
    prober.probe_hdr_side_data.return_value = []
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    analyzer.analyze(_make_scan(tmp_path, "dv.mkv"))
    done = next(e for e in reporter.events if e.method == "analyze_file_done")
    assert "BL=HLG" in _arg_str(done)


def test_summary_dv_unknown_bl_compat(tmp_path: Path) -> None:
    prober = _make_prober_simple_sdr()
    probe = _sdr_probe_dict()
    probe["streams"][0]["color_transfer"] = "smpte2084"
    probe["streams"][0]["side_data_list"] = [
        {
            "side_data_type": "DOVI configuration record",
            "dv_profile": 7,
            "dv_bl_signal_compatibility_id": 0,
        },
    ]
    prober.probe.return_value = probe
    prober.probe_hdr_side_data.return_value = []
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    analyzer.analyze(_make_scan(tmp_path, "dv.mkv"))
    done = next(e for e in reporter.events if e.method == "analyze_file_done")
    assert "BL=none" in _arg_str(done)


def test_summary_no_audio_languages(tmp_path: Path) -> None:
    """``analyze_file_done`` summary handles audio with no language tags."""
    prober = _make_prober_simple_sdr()
    probe = _sdr_probe_dict()
    probe["streams"].append({
        "index": 1,
        "codec_type": "audio",
        "codec_name": "ac3",
        "channels": 2,
        "tags": {"language": ""},  # empty language
        "disposition": {"default": 1, "forced": 0},
    })
    prober.probe.return_value = probe
    prober.profile_audio_track.return_value = _stereo_metrics()
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    analyzer.analyze(_make_scan(tmp_path, "movie.mkv"))
    done = next(e for e in reporter.events if e.method == "analyze_file_done")
    summary = _arg_str(done)
    assert "1 audio" in summary
    # No language list when languages are empty
    assert "()" not in summary


def test_summary_zero_fps_den(tmp_path: Path) -> None:
    """``_format_analyze_summary`` handles fps_den=0 without ZeroDivisionError."""
    prober = _make_prober_simple_sdr()
    probe = _sdr_probe_dict()
    # avg_frame_rate=0/0 -> fps_den becomes 1 in parser. Use r_frame_rate too.
    probe["streams"][0]["avg_frame_rate"] = "0/0"
    probe["streams"][0]["r_frame_rate"] = "0/0"
    prober.probe.return_value = probe
    reporter = RecordingPlanReporter()
    analyzer = Analyzer(prober=prober, reporter=reporter)
    movie = analyzer.analyze(_make_scan(tmp_path, "movie.mkv"))
    assert movie is not None
