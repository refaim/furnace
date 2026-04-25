"""Smoke integration test: real services + recording reporter, fake adapters.

Bypasses typer; calls the reporter-aware portion of cli.plan directly
through the same wiring it uses, with all external tools stubbed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from furnace.services.analyzer import Analyzer
from furnace.services.planner import PlannerService
from furnace.services.scanner import Scanner
from tests.fakes.recording_reporter import RecordingPlanReporter


def _stub_prober() -> MagicMock:
    p = MagicMock()
    p.get_encoder_tag.return_value = None
    p.probe.return_value = {
        "streams": [{
            "codec_type": "video", "index": 0, "codec_name": "h264",
            "width": 1920, "height": 1080,
            "avg_frame_rate": "24/1", "r_frame_rate": "24/1",
            "duration": "100",
            "color_primaries": "bt709", "color_transfer": "bt709",
            "color_space": "bt709", "pix_fmt": "yuv420p",
            "field_order": "progressive", "sample_aspect_ratio": "1:1",
            "side_data_list": [],
        }],
        "format": {},
        "chapters": [],
    }
    p.detect_crop.return_value = None
    return p


def test_plan_emits_full_event_sequence(tmp_path: Path) -> None:
    # Source dir: one MKV file (no discs).
    src = tmp_path / "src"
    src.mkdir()
    (src / "Inception.mkv").touch()
    out = tmp_path / "out"

    reporter = RecordingPlanReporter()
    prober = _stub_prober()

    scanner = Scanner(prober=prober, reporter=reporter)
    scan_results = scanner.scan(src, out)
    assert len(scan_results) == 1

    analyzer = Analyzer(prober=prober, reporter=reporter)
    movies = []
    for sr in scan_results:
        m = analyzer.analyze(sr)
        if m is not None:
            movies.append((m, sr.output_path))

    planner = PlannerService(prober=prober, previewer=None, reporter=reporter)
    planner.create_plan(
        movies=movies,
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        vmaf_enabled=False,
        dry_run=False,
    )

    methods = [e.method for e in reporter.events]
    assert methods[0] == "scan_file"
    assert "analyze_file_start" in methods
    assert "analyze_file_done" in methods
    assert "plan_file_start" in methods
    assert ("plan_microop", ("cropdetect",), (("has_progress", True),)) in [
        (e.method, e.args, e.kwargs) for e in reporter.events
    ]
    assert "plan_file_done" in methods
