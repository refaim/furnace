from pathlib import Path

from furnace.core.models import DiscType
from furnace.core.ports import PlanReporter
from tests.fakes.recording_reporter import Event, RecordingPlanReporter


def test_records_method_name_and_args(tmp_path: Path) -> None:
    plan_path = tmp_path / "furnace-plan.json"
    r = RecordingPlanReporter()
    r.detect_disc(DiscType.BLURAY, "Matrix_BD")
    r.demux_disc_cached("OldMatrix_BD")
    r.demux_title_start(5)
    r.demux_title_substep("rip", has_progress=True)
    r.demux_title_progress(0.37)
    r.demux_title_done()
    r.scan_file("Inception.mkv")
    r.analyze_file_failed("HDR10+ not supported")
    r.plan_saved(plan_path, 7)
    r.interrupted()
    r.pause()
    r.resume()

    assert r.events == [
        Event("detect_disc", (DiscType.BLURAY, "Matrix_BD")),
        Event("demux_disc_cached", ("OldMatrix_BD",)),
        Event("demux_title_start", (5,)),
        Event("demux_title_substep", ("rip",), (("has_progress", True),)),
        Event("demux_title_progress", (0.37,)),
        Event("demux_title_done", ()),
        Event("scan_file", ("Inception.mkv",)),
        Event("analyze_file_failed", ("HDR10+ not supported",)),
        Event("plan_saved", (plan_path, 7)),
        Event("interrupted", ()),
        Event("pause", ()),
        Event("resume", ()),
    ]


def test_satisfies_protocol() -> None:
    """Lock in structural conformance with PlanReporter (runtime-checkable)."""
    assert isinstance(RecordingPlanReporter(), PlanReporter)
