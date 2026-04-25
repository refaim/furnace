from pathlib import Path
from unittest.mock import MagicMock

from furnace.services.scanner import Scanner
from tests.fakes.recording_reporter import RecordingPlanReporter


def test_scan_emits_one_scan_file_per_video(tmp_path: Path) -> None:
    (tmp_path / "Inception.mkv").touch()
    (tmp_path / "Tenet.mkv").touch()
    (tmp_path / "notes.txt").touch()  # not a video, must be skipped

    reporter = RecordingPlanReporter()
    scanner = Scanner(prober=MagicMock(), reporter=reporter)
    results = scanner.scan(tmp_path, tmp_path / "out")

    assert len(results) == 2
    file_events = [e for e in reporter.events if e.method == "scan_file"]
    names = sorted(str(e.args[0]) for e in file_events)
    assert names == ["Inception.mkv", "Tenet.mkv"]


def test_scan_without_reporter_is_silent(tmp_path: Path) -> None:
    (tmp_path / "x.mkv").touch()
    scanner = Scanner(prober=MagicMock())  # no reporter
    results = scanner.scan(tmp_path, tmp_path / "out")
    assert len(results) == 1  # still works


def test_scan_single_file_emits_scan_file(tmp_path: Path) -> None:
    video = tmp_path / "Movie.mkv"
    video.touch()

    reporter = RecordingPlanReporter()
    scanner = Scanner(prober=MagicMock(), reporter=reporter)
    results = scanner.scan(video, tmp_path / "out")

    assert len(results) == 1
    file_events = [e for e in reporter.events if e.method == "scan_file"]
    assert len(file_events) == 1
    assert file_events[0].args[0] == "Movie.mkv"


def test_scan_single_file_without_reporter_is_silent(tmp_path: Path) -> None:
    video = tmp_path / "Movie.mkv"
    video.touch()

    scanner = Scanner(prober=MagicMock())  # no reporter
    results = scanner.scan(video, tmp_path / "out")

    assert len(results) == 1
