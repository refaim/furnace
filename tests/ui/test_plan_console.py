from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from furnace.core.models import DiscType
from furnace.ui.plan_console import RichPlanReporter


def _make_reporter() -> tuple[RichPlanReporter, StringIO]:
    buf = StringIO()
    console = Console(
        file=buf,
        force_terminal=True,
        width=120,
        color_system=None,
        legacy_windows=False,
    )
    reporter = RichPlanReporter(
        source=Path("D:/Library"),
        output=Path("Z:/plans/library"),
        console=console,
        ascii_only=True,
    )
    return reporter, buf


def test_header_printed_once() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.stop()
    text = buf.getvalue()
    assert text.count("Source:") == 1
    assert "D:/Library" in text or "D:\\Library" in text
    assert "Z:/plans/library" in text or "Z:\\plans\\library" in text


def test_detect_renders_one_row_per_disc() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.detect_disc(DiscType.BLURAY, "Matrix_BD")
    reporter.detect_disc(DiscType.BLURAY, "OldMatrix_BD")
    reporter.detect_disc(DiscType.DVD, "DirtyHarry_DVD")
    reporter.stop()
    text = buf.getvalue()
    # Phase header exactly once
    assert text.count("Detect") == 1
    # All three rows present
    assert "BDMV" in text
    assert "DVD" in text
    assert "Matrix_BD" in text
    assert "OldMatrix_BD" in text
    assert "DirtyHarry_DVD" in text


def test_no_detect_block_when_no_discs() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.stop()
    text = buf.getvalue()
    assert "Detect" not in text


def test_demux_disc_cached_one_line() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.demux_disc_cached("OldMatrix_BD")
    reporter.stop()
    text = buf.getvalue()
    assert "Demux" in text
    assert "OldMatrix_BD" in text
    assert "from cache" in text


def test_demux_fresh_disc_unfolds_titles() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.demux_disc_start("Matrix_BD")
    reporter.demux_title_start(3)
    reporter.demux_title_substep("rip", has_progress=True)
    reporter.demux_title_progress(0.5)
    reporter.demux_title_substep("remux", has_progress=True)
    reporter.demux_title_progress(0.5)
    reporter.demux_title_done()
    reporter.stop()
    text = buf.getvalue()
    assert "Matrix_BD" in text
    assert "title 3" in text
    # Only the final state survives the floating bar after done
    assert "done" in text


def test_demux_title_failed_renders_FAILED() -> None:  # noqa: N802
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.demux_disc_start("DirtyHarry_DVD")
    reporter.demux_title_start(1)
    reporter.demux_title_substep("rip", has_progress=True)
    reporter.demux_title_failed("eac3to timeout")
    reporter.stop()
    text = buf.getvalue()
    assert "title 1" in text
    assert "FAILED" in text
    assert "eac3to timeout" in text


def test_demux_phase_header_appears_once() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.demux_disc_cached("A")
    reporter.demux_disc_cached("B")
    reporter.stop()
    text = buf.getvalue()
    assert text.count("Demux") == 1


def test_demux_title_substep_without_start_is_noop() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    # No demux_title_start; substep should not crash or render a title row.
    reporter.demux_title_substep("rip", has_progress=True)
    reporter.stop()
    text = buf.getvalue()
    assert "title" not in text


def test_demux_title_progress_without_substep_is_noop() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    # progress called before any substep / progress bar exists.
    reporter.demux_title_progress(0.25)
    reporter.stop()
    # Should not crash. No title row should be present.
    text = buf.getvalue()
    assert "title" not in text


def test_demux_title_done_without_start_is_noop() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.demux_title_done()
    reporter.stop()
    text = buf.getvalue()
    assert "title" not in text


def test_demux_title_failed_without_start_is_noop() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.demux_title_failed("boom")
    reporter.stop()
    text = buf.getvalue()
    assert "title" not in text
    assert "FAILED" not in text


def test_start_progress_asserts_invariant() -> None:
    """_start_progress crashes if invoked while a Progress is already alive."""
    reporter, _ = _make_reporter()
    reporter.start()
    reporter.demux_disc_start("Matrix_BD")
    reporter.demux_title_start(3)
    reporter.demux_title_substep("rip", has_progress=True)
    # _progress is now alive; force a direct call to provoke the assert
    with pytest.raises(AssertionError, match="previous progress not stopped"):
        reporter._start_progress(has_progress=True)
    reporter.stop()


def test_demux_title_substep_without_progress_bar() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.demux_disc_start("Some_BD")
    reporter.demux_title_start(2)
    # has_progress=False — exercises the indeterminate (total=None) branch.
    reporter.demux_title_substep("scan", has_progress=False)
    reporter.demux_title_done()
    reporter.stop()
    text = buf.getvalue()
    assert "title 2" in text
    assert "done" in text


def test_scan_renders_per_file_with_phase_header_once() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.scan_file("Inception.mkv")
    reporter.scan_file("Tenet.mkv")
    reporter.scan_skipped("weird.mkv", "no video stream")
    reporter.stop()
    text = buf.getvalue()
    assert text.count("Scan") == 1
    assert "Inception.mkv" in text
    assert "Tenet.mkv" in text
    assert "weird.mkv" in text
    assert "SKIPPED" in text
    assert "no video stream" in text


def test_analyze_done_renders_summary() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.analyze_file_start("Inception.mkv")
    reporter.analyze_microop("probing", has_progress=False)
    reporter.analyze_file_done(
        "hevc 3840x2076 24fps HDR10, 5 audio (rus,eng), 12 subs"
    )
    reporter.stop()
    text = buf.getvalue()
    assert "Analyze" in text
    assert "Inception.mkv" in text
    assert "hevc 3840x2076" in text
    # Status arrow between file name and summary
    assert "Inception.mkv -> hevc" in text


def test_analyze_failed_inline() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.analyze_file_start("broken.mkv")
    reporter.analyze_file_failed("HDR10+ not supported")
    reporter.stop()
    text = buf.getvalue()
    assert "broken.mkv" in text
    assert "FAILED" in text
    assert "HDR10+ not supported" in text


def test_analyze_progress_with_microop_bar() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.analyze_file_start("interlaced.mkv")
    reporter.analyze_microop("idet", has_progress=True)
    reporter.analyze_progress(0.4)
    reporter.analyze_progress(1.0)
    reporter.analyze_file_done(
        "h264 720x480 30fps SDR (interlaced), 2 audio (eng), 0 subs"
    )
    reporter.stop()
    text = buf.getvalue()
    assert "interlaced.mkv" in text
    assert "h264 720x480" in text


def test_analyze_microop_without_file_start_is_noop() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    # No analyze_file_start; microop should be a no-op (no progress bar started).
    reporter.analyze_microop("probing", has_progress=True)
    reporter.stop()
    text = buf.getvalue()
    assert "Analyze" not in text
    assert "probing" not in text


def test_analyze_progress_without_microop_is_noop() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    # progress called before any microop / progress bar exists.
    reporter.analyze_progress(0.25)
    reporter.stop()
    # Should not crash. No analyze row should be present.
    text = buf.getvalue()
    assert "Analyze" not in text


def test_analyze_file_done_without_start_is_noop() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.analyze_file_done("some summary")
    reporter.stop()
    text = buf.getvalue()
    assert "Analyze" not in text
    assert "some summary" not in text


def test_analyze_file_failed_without_start_is_noop() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.analyze_file_failed("boom")
    reporter.stop()
    text = buf.getvalue()
    assert "Analyze" not in text
    assert "FAILED" not in text


def test_analyze_file_skipped_without_start_is_noop() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.analyze_file_skipped("not a video")
    reporter.stop()
    text = buf.getvalue()
    assert "Analyze" not in text
    assert "SKIPPED" not in text


def test_analyze_file_skipped_renders_inline() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.analyze_file_start("weird.mkv")
    reporter.analyze_file_skipped("not a video")
    reporter.stop()
    text = buf.getvalue()
    assert "weird.mkv" in text
    assert "SKIPPED" in text
    assert "not a video" in text


def test_plan_renders_per_file_and_drops_plan_saved() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.plan_file_start("Inception.mkv")
    reporter.plan_microop("cropdetect", has_progress=True)
    reporter.plan_progress(1.0)
    reporter.plan_file_done("cq 22, 3840x2076 to 3840x1600")
    reporter.plan_saved(Path("Z:/plans/library/furnace-plan.json"), 7)
    reporter.stop()
    text = buf.getvalue()
    assert "Plan" in text
    assert "Inception.mkv" in text
    assert "cq 22" in text
    # plan_saved must NOT print the final summary line
    assert "furnace-plan.json" not in text
    assert "(7 jobs)" not in text


def test_interrupted_prints_final_line() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.plan_file_start("foo.mkv")
    reporter.plan_microop("cropdetect", has_progress=True)
    reporter.plan_progress(0.3)
    reporter.interrupted()
    text = buf.getvalue()
    assert "interrupted" in text


def test_pause_resume_stop_and_restart_progress() -> None:
    reporter, _buf = _make_reporter()
    reporter.start()
    reporter.plan_file_start("foo.mkv")
    reporter.plan_microop("cropdetect", has_progress=True)
    reporter.plan_progress(0.3)
    reporter.pause()
    # After pause(), no live progress object is held
    assert reporter._progress is None
    reporter.resume()
    # resume() does not auto-restart Progress — next *_microop call recreates it
    reporter.plan_progress(0.6)
    reporter.plan_file_done("cq 22, 1920x1080 to 1920x1080")
    reporter.stop()


def test_plan_microop_without_file_start_is_noop() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    # No plan_file_start; microop should be a no-op (no progress bar started).
    reporter.plan_microop("cropdetect", has_progress=True)
    reporter.stop()
    text = buf.getvalue()
    assert "Plan" not in text
    assert "cropdetect" not in text


def test_plan_progress_without_microop_is_noop() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    # progress called before any microop / progress bar exists.
    reporter.plan_progress(0.25)
    reporter.stop()
    text = buf.getvalue()
    assert "Plan" not in text


def test_plan_file_done_without_start_is_noop() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.plan_file_done("some summary")
    reporter.stop()
    text = buf.getvalue()
    assert "Plan" not in text
    assert "some summary" not in text


def test_plan_microop_without_progress_bar() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.plan_file_start("foo.mkv")
    # has_progress=False — exercises the indeterminate (total=None) branch.
    reporter.plan_microop("probing", has_progress=False)
    reporter.plan_file_done("cq 22, 1920x1080 to 1920x1080")
    reporter.stop()
    text = buf.getvalue()
    assert "Plan" in text
    assert "foo.mkv" in text


def test_plan_phase_header_appears_once_across_multiple_files() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.plan_file_start("a.mkv")
    reporter.plan_file_done("cq 22")
    reporter.plan_file_start("b.mkv")
    reporter.plan_file_done("cq 22")
    reporter.stop()
    text = buf.getvalue()
    assert text.count("Plan") == 1


def test_plan_saved_is_silent_with_no_prior_phases() -> None:
    """plan_saved must never emit visible output even when called directly."""
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.plan_saved(Path("Z:/p/furnace-plan.json"), 5)
    reporter.stop()
    text = buf.getvalue()
    assert "furnace-plan.json" not in text
    assert "5 jobs" not in text
    assert "Plan" not in text


def test_phase_headers_separated_by_blank_line() -> None:
    """Each non-first phase header is preceded by a blank line."""
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.detect_disc(DiscType.BLURAY, "Matrix_BD")
    reporter.scan_file("Matrix_BD_title_1.mkv")
    reporter.stop()
    text = buf.getvalue()
    # Both phase headers present
    assert "Detect" in text
    assert "Scan" in text
    # Between the Detect block and the Scan header there must be a blank line:
    # the line preceding the "Scan" line should be empty.
    raw_lines = text.splitlines()
    scan_line_idx = next(i for i, line in enumerate(raw_lines) if line.startswith("Scan"))
    assert raw_lines[scan_line_idx - 1] == ""


def test_first_phase_has_no_leading_blank_line() -> None:
    """The very first phase header is not preceded by an extra blank line."""
    buf = StringIO()
    console = Console(
        file=buf,
        force_terminal=False,
        width=120,
        color_system=None,
        legacy_windows=False,
    )
    reporter = RichPlanReporter(
        source=Path("D:/L"),
        output=Path("Z:/p"),
        console=console,
        ascii_only=True,
    )
    reporter.start()
    reporter.detect_disc(DiscType.BLURAY, "Matrix_BD")
    reporter.stop()
    raw_lines = buf.getvalue().splitlines()
    # After the Source/Output header (and its trailing blank line emitted by
    # start()), Detect is the first phase header — there should be exactly one
    # blank line (from start()) before it, not two.
    detect_idx = next(i for i, line in enumerate(raw_lines) if line.startswith("Detect"))
    # The blank line immediately preceding is from start(); the line before
    # that must be the "Output:" line, not another blank.
    assert raw_lines[detect_idx - 1] == ""
    assert raw_lines[detect_idx - 2].startswith("Output:")


def test_demux_title_uses_indented_prefix() -> None:
    """Demux titles are indented with 2 spaces under their disc."""
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.demux_disc_start("Matrix_BD")
    reporter.demux_title_start(3)
    reporter.demux_title_done()
    reporter.stop()
    text = buf.getvalue()
    # Disc name flush left
    assert "\nMatrix_BD\n" in text or text.startswith("Matrix_BD") or "Matrix_BD" in text.splitlines()
    # Title row indented with 2 spaces
    assert "  title 3 -> done" in text


def test_demux_disc_cached_uses_arrow() -> None:
    reporter, buf = _make_reporter()
    reporter.start()
    reporter.demux_disc_cached("OldMatrix_BD")
    reporter.stop()
    text = buf.getvalue()
    assert "OldMatrix_BD -> from cache" in text


def test_non_tty_output_has_no_ansi_escapes() -> None:
    buf = StringIO()
    console = Console(
        file=buf,
        force_terminal=False,
        width=120,
        color_system=None,
        legacy_windows=False,
    )
    reporter = RichPlanReporter(
        source=Path("D:/L"),
        output=Path("Z:/p"),
        console=console,
        ascii_only=True,
    )
    reporter.start()
    reporter.detect_disc(DiscType.BLURAY, "Matrix_BD")
    reporter.scan_file("Inception.mkv")
    reporter.analyze_file_start("Inception.mkv")
    reporter.analyze_microop("idet", has_progress=True)
    reporter.analyze_progress(0.5)
    reporter.analyze_file_done("hevc 3840x2076 24fps HDR10, 5 audio (rus,eng), 12 subs")
    reporter.plan_saved(Path("Z:/p/furnace-plan.json"), 1)
    reporter.stop()
    text = buf.getvalue()
    assert "\x1b[" not in text  # no ANSI escape sequences


def test_canonical_plan_golden_snapshot() -> None:
    buf = StringIO()
    console = Console(
        file=buf,
        force_terminal=False,
        width=120,
        color_system=None,
        legacy_windows=False,
    )
    reporter = RichPlanReporter(
        source=Path("D:/Library"),
        output=Path("Z:/plans/library"),
        console=console,
        ascii_only=True,
    )
    reporter.start()
    reporter.detect_disc(DiscType.BLURAY, "Matrix_BD")
    reporter.detect_disc(DiscType.DVD, "DirtyHarry_DVD")
    reporter.demux_disc_start("Matrix_BD")
    reporter.demux_title_start(3)
    reporter.demux_title_substep("rip", has_progress=True)
    reporter.demux_title_progress(1.0)
    reporter.demux_title_substep("remux", has_progress=True)
    reporter.demux_title_progress(1.0)
    reporter.demux_title_done()
    reporter.demux_disc_start("DirtyHarry_DVD")
    reporter.demux_title_start(1)
    reporter.demux_title_substep("rip", has_progress=True)
    reporter.demux_title_progress(1.0)
    reporter.demux_title_done()
    reporter.scan_file("Matrix_BD_title_3.mkv")
    reporter.scan_file("DirtyHarry_DVD_title_1.mkv")
    reporter.analyze_file_start("Matrix_BD_title_3.mkv")
    reporter.analyze_microop("probing", has_progress=False)
    reporter.analyze_file_done("hevc 3840x2160 24fps HDR10, 4 audio (eng), 0 subs")
    reporter.analyze_file_start("DirtyHarry_DVD_title_1.mkv")
    reporter.analyze_microop("idet", has_progress=True)
    reporter.analyze_progress(1.0)
    reporter.analyze_file_done("mpeg2video 720x480 30fps SDR (interlaced), 2 audio (eng), 1 subs")
    reporter.plan_file_start("Matrix_BD_title_3.mkv")
    reporter.plan_microop("cropdetect", has_progress=True)
    reporter.plan_progress(1.0)
    reporter.plan_file_done("cq 22, 3840x2160 to 3840x2160")
    reporter.plan_file_start("DirtyHarry_DVD_title_1.mkv")
    reporter.plan_microop("cropdetect", has_progress=True)
    reporter.plan_progress(1.0)
    reporter.plan_file_done("cq 19, 720x480 to 720x540, deinterlace")
    reporter.plan_saved(Path("Z:/plans/library/furnace-plan.json"), 2)
    reporter.stop()
    actual = buf.getvalue()
    expected = (Path(__file__).parent / "snapshots" / "canonical_plan.txt").read_text(encoding="utf-8")
    assert actual == expected
