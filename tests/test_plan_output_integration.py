from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from furnace.services.analysis_pipeline import AnalysisPipeline
from furnace.services.analyzer import Analyzer
from furnace.services.planner import PlannerService
from furnace.services.scanner import Scanner
from tests.fakes.recording_reporter import RecordingPlanReporter


def _stub_prober() -> MagicMock:
    p = MagicMock()
    p.get_encoder_tag.return_value = None
    p.probe.return_value = {
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
            }
        ],
        "format": {},
        "chapters": [],
    }
    p.detect_crop.return_value = None
    p.sample_grain.return_value = [0.2, 0.2, 0.2, 0.2, 0.2]
    return p


def test_plan_emits_full_event_sequence(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "Inception.mkv").touch()
    out = tmp_path / "out"

    reporter = RecordingPlanReporter()
    prober = _stub_prober()

    scanner = Scanner(prober=prober, reporter=reporter)
    scan_results = scanner.scan(src, out)
    assert len(scan_results) == 1

    analyzer = Analyzer(prober=prober)
    pipeline = AnalysisPipeline(
        analyzer=analyzer,
        prober=prober,
        reporter=reporter,
        max_workers=1,
    )
    batch = pipeline.run(scan_results, copy_video=False, dry_run=False)
    assert len(batch.movies) == 1

    planner = PlannerService(previewer=None, reporter=reporter)
    planner.create_plan(
        movies=batch.movies,
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        precomputed_crops=batch.crops,
    )

    methods = [e.method for e in reporter.events]
    assert methods[0] == "scan_file"
    assert "analyze_batch_start" in methods
    assert "analyze_batch_item" in methods
    assert "analyze_batch_finish" in methods
    assert "plan_file_start" in methods
    assert "plan_file_done" in methods
