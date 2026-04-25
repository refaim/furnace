from __future__ import annotations

import contextlib
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from furnace.cli import _setup_logging, app
from furnace.core.models import JobStatus, TrackType
from tests.conftest import make_job, make_movie, make_plan, make_track

runner = CliRunner()


# ---------------------------------------------------------------------------
# _setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def _cleanup_root_handlers(self) -> None:
        """Remove all FileHandler instances from root logger."""
        root = logging.getLogger()
        for h in list(root.handlers):
            if isinstance(h, logging.FileHandler):
                root.removeHandler(h)
                h.close()
        # Also remove any StreamHandlers we may have added
        for h in list(root.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                fmt = h.formatter
                if fmt and "[furnace]" in (fmt._fmt or ""):
                    root.removeHandler(h)
                    h.close()

    def test_creates_log_directory(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        _setup_logging(log_dir, console=False)
        assert log_dir.is_dir()
        self._cleanup_root_handlers()

    def test_creates_file_handler(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        _setup_logging(log_dir, console=False)
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) >= 1
        self._cleanup_root_handlers()

    def test_log_file_created(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        _setup_logging(log_dir, console=False)
        assert (log_dir / "furnace.log").exists()
        self._cleanup_root_handlers()

    def test_console_enabled_adds_stream_handler(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        _setup_logging(log_dir, console=True)
        root = logging.getLogger()
        stream_handlers = [
            h
            for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            and h.formatter is not None
            and "[furnace]" in (h.formatter._fmt or "")
        ]
        assert len(stream_handlers) >= 1
        self._cleanup_root_handlers()

    def test_console_disabled_no_stream_handler(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        # Count existing stream handlers before
        root = logging.getLogger()
        before = [
            h
            for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            and h.formatter is not None
            and "[furnace]" in (h.formatter._fmt or "")
        ]
        _setup_logging(log_dir, console=False)
        after = [
            h
            for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            and h.formatter is not None
            and "[furnace]" in (h.formatter._fmt or "")
        ]
        assert len(after) == len(before)
        self._cleanup_root_handlers()

    def test_root_logger_set_to_debug(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        _setup_logging(log_dir, console=False)
        root = logging.getLogger()
        assert root.level == logging.DEBUG
        self._cleanup_root_handlers()

    def test_nested_directory_created(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "a" / "b" / "c"
        _setup_logging(log_dir, console=False)
        assert log_dir.is_dir()
        self._cleanup_root_handlers()


# ---------------------------------------------------------------------------
# plan --dry-run
# ---------------------------------------------------------------------------


def _make_tool_paths(tmp_path: Path) -> MagicMock:
    """Create a mock ToolPaths with all required attributes pointing to tmp_path files."""
    cfg = MagicMock()
    cfg.ffmpeg = tmp_path / "ffmpeg"
    cfg.ffprobe = tmp_path / "ffprobe"
    cfg.mkvmerge = tmp_path / "mkvmerge"
    cfg.mkvpropedit = tmp_path / "mkvpropedit"
    cfg.mkclean = tmp_path / "mkclean"
    cfg.eac3to = tmp_path / "eac3to"
    cfg.qaac64 = tmp_path / "qaac64"
    cfg.mpv = tmp_path / "mpv"
    cfg.makemkvcon = tmp_path / "makemkvcon"
    cfg.nvencc = tmp_path / "nvencc"
    cfg.dovi_tool = None
    return cfg


class TestPlanDryRun:
    def test_dry_run_no_movies(self, tmp_path: Path) -> None:
        """--dry-run with no scan results prints zero jobs."""
        source = tmp_path / "src"
        source.mkdir()
        output = tmp_path / "out"

        cfg = _make_tool_paths(tmp_path)
        plan_obj = make_plan(
            jobs=[],
            source=str(source),
            destination=str(output),
        )

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.MpvAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.MakemkvAdapter"),
            patch("furnace.cli.DiscDemuxer") as mock_demuxer_cls,
            patch("furnace.cli.Scanner") as mock_scanner_cls,
            patch("furnace.cli.Analyzer") as mock_analyzer_cls,
            patch("furnace.cli.PlannerService") as mock_planner_cls,
        ):
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = []
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng", "--dry-run"],
            )

        assert result.exit_code == 0, result.output
        # plan_saved no longer prints a visible line, so we verify via mocks
        # that the planner was invoked and produced a zero-job plan.
        mock_planner_cls.return_value.create_plan.assert_called_once()
        # Analyzer should not have been called since scanner returned empty
        mock_analyzer_cls.return_value.analyze.assert_not_called()

    def test_dry_run_with_movies(self, tmp_path: Path) -> None:
        """--dry-run with scan results prints job count."""
        from furnace.core.models import ScanResult

        source = tmp_path / "src"
        source.mkdir()
        output = tmp_path / "out"

        cfg = _make_tool_paths(tmp_path)
        movie = MagicMock()
        scan_result = ScanResult(
            main_file=source / "movie.mkv",
            satellite_files=[],
            output_path=output / "movie" / "movie.mkv",
        )
        pending_job = make_job(job_id="j1", status=JobStatus.PENDING)
        done_job = make_job(job_id="j2", status=JobStatus.DONE)
        plan_obj = make_plan(
            jobs=[pending_job, done_job],
            source=str(source),
            destination=str(output),
        )

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.MpvAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.MakemkvAdapter"),
            patch("furnace.cli.DiscDemuxer") as mock_demuxer_cls,
            patch("furnace.cli.Scanner") as mock_scanner_cls,
            patch("furnace.cli.Analyzer") as mock_analyzer_cls,
            patch("furnace.cli.PlannerService") as mock_planner_cls,
        ):
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = [scan_result]
            mock_analyzer_cls.return_value.analyze.return_value = movie
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "jpn", "-sl", "rus,eng", "--dry-run"],
            )

        assert result.exit_code == 0, result.output
        # plan_saved no longer prints a visible line; verify the planner ran
        # and produced the expected two-job plan.
        mock_planner_cls.return_value.create_plan.assert_called_once()
        assert len(mock_planner_cls.return_value.create_plan.return_value.jobs) == 2

    def test_dry_run_passes_language_lists(self, tmp_path: Path) -> None:
        """Language lists are correctly parsed and passed to planner."""
        source = tmp_path / "src"
        source.mkdir()
        output = tmp_path / "out"

        cfg = _make_tool_paths(tmp_path)
        plan_obj = make_plan(jobs=[])

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.MpvAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.MakemkvAdapter"),
            patch("furnace.cli.DiscDemuxer") as mock_demuxer_cls,
            patch("furnace.cli.Scanner") as mock_scanner_cls,
            patch("furnace.cli.PlannerService") as mock_planner_cls,
        ):
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = []
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "rus,eng", "-sl", "rus", "--dry-run"],
            )

        assert result.exit_code == 0, result.output
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args
        assert call_kwargs.kwargs["audio_lang_filter"] == ["rus", "eng"]
        assert call_kwargs.kwargs["sub_lang_filter"] == ["rus"]

    def test_dry_run_passes_null_track_selector(self, tmp_path: Path) -> None:
        """In --dry-run mode, PlannerService receives track_selector=None."""
        source = tmp_path / "src"
        source.mkdir()
        output = tmp_path / "out"

        cfg = _make_tool_paths(tmp_path)
        plan_obj = make_plan(jobs=[])

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.MpvAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.MakemkvAdapter"),
            patch("furnace.cli.DiscDemuxer") as mock_demuxer_cls,
            patch("furnace.cli.Scanner") as mock_scanner_cls,
            patch("furnace.cli.PlannerService") as mock_planner_cls,
        ):
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = []
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng", "--dry-run"],
            )

        assert result.exit_code == 0, result.output
        planner_init_kwargs = mock_planner_cls.call_args
        assert planner_init_kwargs.kwargs["track_selector"] is None
        assert planner_init_kwargs.kwargs["und_resolver"] is None

    def test_dry_run_vmaf_flag(self, tmp_path: Path) -> None:
        """--vmaf flag is forwarded to planner.create_plan()."""
        source = tmp_path / "src"
        source.mkdir()
        output = tmp_path / "out"

        cfg = _make_tool_paths(tmp_path)
        plan_obj = make_plan(jobs=[])

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.MpvAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.MakemkvAdapter"),
            patch("furnace.cli.DiscDemuxer") as mock_demuxer_cls,
            patch("furnace.cli.Scanner") as mock_scanner_cls,
            patch("furnace.cli.PlannerService") as mock_planner_cls,
        ):
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = []
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng", "--dry-run", "--vmaf"],
            )

        assert result.exit_code == 0, result.output
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args.kwargs
        assert call_kwargs["vmaf_enabled"] is True


# ---------------------------------------------------------------------------
# plan (non-dry-run) — save_plan path
# ---------------------------------------------------------------------------


class TestPlanSave:
    def test_save_plan_writes_file(self, tmp_path: Path) -> None:
        """Non-dry-run plan command saves plan JSON and prints path."""
        source = tmp_path / "src"
        source.mkdir()
        output = tmp_path / "out"

        cfg = _make_tool_paths(tmp_path)
        job = make_job(job_id="j1", status=JobStatus.PENDING)
        plan_obj = make_plan(
            jobs=[job],
            source=str(source),
            destination=str(output),
        )

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.MpvAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.MakemkvAdapter"),
            patch("furnace.cli.DiscDemuxer") as mock_demuxer_cls,
            patch("furnace.cli.Scanner") as mock_scanner_cls,
            patch("furnace.cli.Analyzer"),
            patch("furnace.cli.PlannerService") as mock_planner_cls,
            patch("furnace.cli.save_plan") as mock_save,
        ):
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = []
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng"],
            )

        assert result.exit_code == 0, result.output
        # plan_saved no longer prints a visible line; verify the plan was
        # written to disk via save_plan and the planner was invoked.
        mock_save.assert_called_once()
        mock_planner_cls.return_value.create_plan.assert_called_once()


# ---------------------------------------------------------------------------
# plan --names
# ---------------------------------------------------------------------------


class TestPlanNames:
    def test_names_map_loaded(self, tmp_path: Path) -> None:
        """--names option loads JSON names map and passes to scanner."""
        import json

        source = tmp_path / "src"
        source.mkdir()
        output = tmp_path / "out"
        names_file = tmp_path / "names.json"
        names_file.write_text(json.dumps({"movie.mkv": "Movie Title"}), encoding="utf-8")

        cfg = _make_tool_paths(tmp_path)
        plan_obj = make_plan(jobs=[])

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.MpvAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.MakemkvAdapter"),
            patch("furnace.cli.DiscDemuxer") as mock_demuxer_cls,
            patch("furnace.cli.Scanner") as mock_scanner_cls,
            patch("furnace.cli.PlannerService") as mock_planner_cls,
        ):
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = []
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                [
                    "plan",
                    str(source),
                    "-o",
                    str(output),
                    "-al",
                    "eng",
                    "-sl",
                    "eng",
                    "--dry-run",
                    "--names",
                    str(names_file),
                ],
            )

        assert result.exit_code == 0, result.output
        # Scanner.scan should have received the names map
        call_args = mock_scanner_cls.return_value.scan.call_args
        assert call_args.args[2] == {"movie.mkv": "Movie Title"}


# ---------------------------------------------------------------------------
# plan with detected discs (dry_run — disc code skipped)
# ---------------------------------------------------------------------------


class TestPlanDiscDryRun:
    def test_detected_discs_skipped_in_dry_run(self, tmp_path: Path) -> None:
        """When discs are detected but --dry-run is set, demux phase is skipped."""
        from furnace.core.models import DiscSource, DiscType

        source = tmp_path / "src"
        source.mkdir()
        output = tmp_path / "out"

        cfg = _make_tool_paths(tmp_path)
        disc = DiscSource(path=source / "BDMV", disc_type=DiscType.BLURAY)
        plan_obj = make_plan(jobs=[])

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.MpvAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.MakemkvAdapter"),
            patch("furnace.cli.DiscDemuxer") as mock_demuxer_cls,
            patch("furnace.cli.Scanner") as mock_scanner_cls,
            patch("furnace.cli.PlannerService") as mock_planner_cls,
        ):
            mock_demuxer_cls.return_value.detect.return_value = [disc]
            mock_scanner_cls.return_value.scan.return_value = []
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng", "--dry-run"],
            )

        assert result.exit_code == 0, result.output
        # list_titles should NOT be called in dry-run
        mock_demuxer_cls.return_value.list_titles.assert_not_called()


# ---------------------------------------------------------------------------
# plan demux_dir assignment
# ---------------------------------------------------------------------------


class TestPlanDemuxDirAssignment:
    def test_demux_dir_not_set_when_no_discs(self, tmp_path: Path) -> None:
        """demux_dir stays None when no disc demux happened."""
        source = tmp_path / "src"
        source.mkdir()
        output = tmp_path / "out"

        cfg = _make_tool_paths(tmp_path)
        plan_obj = make_plan(jobs=[], demux_dir=None)

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.MpvAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.MakemkvAdapter"),
            patch("furnace.cli.DiscDemuxer") as mock_demuxer_cls,
            patch("furnace.cli.Scanner") as mock_scanner_cls,
            patch("furnace.cli.PlannerService") as mock_planner_cls,
        ):
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = []
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng", "--dry-run"],
            )

        assert result.exit_code == 0, result.output
        assert plan_obj.demux_dir is None


# ---------------------------------------------------------------------------
# plan — analyzer returns None (skip)
# ---------------------------------------------------------------------------


class TestPlanAnalyzerNone:
    def test_analyzer_none_skips_movie(self, tmp_path: Path) -> None:
        """When analyzer.analyze returns None, movie is skipped."""
        from furnace.core.models import ScanResult

        source = tmp_path / "src"
        source.mkdir()
        output = tmp_path / "out"

        cfg = _make_tool_paths(tmp_path)
        scan_result = ScanResult(
            main_file=source / "movie.mkv",
            satellite_files=[],
            output_path=output / "movie" / "movie.mkv",
        )
        plan_obj = make_plan(jobs=[])

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.MpvAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.MakemkvAdapter"),
            patch("furnace.cli.DiscDemuxer") as mock_demuxer_cls,
            patch("furnace.cli.Scanner") as mock_scanner_cls,
            patch("furnace.cli.Analyzer") as mock_analyzer_cls,
            patch("furnace.cli.PlannerService") as mock_planner_cls,
        ):
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = [scan_result]
            mock_analyzer_cls.return_value.analyze.return_value = None
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng", "--dry-run"],
            )

        assert result.exit_code == 0, result.output
        # Planner should have been called with an empty movies list
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args
        assert call_kwargs.kwargs["movies"] == []


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------


class TestRunCommand:
    def test_run_all_done_no_pending(self, tmp_path: Path) -> None:
        """run command with all-done jobs: launches TUI, prints report."""
        plan_file = tmp_path / "plan.json"
        plan_obj = make_plan(
            jobs=[make_job(job_id="j1", status=JobStatus.DONE)],
            destination=str(tmp_path / "out"),
        )

        cfg = _make_tool_paths(tmp_path)

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.load_plan", return_value=plan_obj),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.RunApp") as mock_run_app_cls,
            patch("furnace.cli.ReportPrinter"),
        ):
            mock_run_app_cls.return_value.run.return_value = None

            result = runner.invoke(app, ["run", str(plan_file)])

        assert result.exit_code == 0, result.output
        mock_run_app_cls.assert_called_once()
        # Verify RunApp was constructed with total_jobs=0 (done jobs aren't pending)
        init_kwargs = mock_run_app_cls.call_args.kwargs
        assert init_kwargs["total_jobs"] == 0

    def test_run_with_pending_jobs(self, tmp_path: Path) -> None:
        """run command counts pending+error jobs for TUI."""
        plan_file = tmp_path / "plan.json"
        plan_obj = make_plan(
            jobs=[
                make_job(job_id="j1", status=JobStatus.PENDING),
                make_job(job_id="j2", status=JobStatus.ERROR),
                make_job(job_id="j3", status=JobStatus.DONE),
            ],
            destination=str(tmp_path / "out"),
        )

        cfg = _make_tool_paths(tmp_path)

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.load_plan", return_value=plan_obj),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.RunApp") as mock_run_app_cls,
            patch("furnace.cli.ReportPrinter"),
        ):
            mock_run_app_cls.return_value.run.return_value = None

            result = runner.invoke(app, ["run", str(plan_file)])

        assert result.exit_code == 0, result.output
        init_kwargs = mock_run_app_cls.call_args.kwargs
        assert init_kwargs["total_jobs"] == 2  # 1 pending + 1 error

    def test_run_calls_report_printer(self, tmp_path: Path) -> None:
        """After TUI exits (no shutdown), ReportPrinter is called."""
        plan_file = tmp_path / "plan.json"
        plan_obj = make_plan(
            jobs=[make_job(job_id="j1", status=JobStatus.DONE)],
            destination=str(tmp_path / "out"),
        )

        cfg = _make_tool_paths(tmp_path)

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.load_plan", return_value=plan_obj),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.RunApp") as mock_run_app_cls,
            patch("furnace.cli.ReportPrinter") as mock_printer_cls,
        ):
            mock_run_app_cls.return_value.run.return_value = None

            result = runner.invoke(app, ["run", str(plan_file)])

        assert result.exit_code == 0, result.output
        mock_printer_cls.return_value.print_report.assert_called_once()

    def test_run_shutdown_event_calls_os_exit(self, tmp_path: Path) -> None:
        """When shutdown_event is set (ESC), os._exit(0) is called."""
        plan_file = tmp_path / "plan.json"
        plan_obj = make_plan(
            jobs=[make_job(job_id="j1", status=JobStatus.PENDING)],
            destination=str(tmp_path / "out"),
        )

        cfg = _make_tool_paths(tmp_path)

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.load_plan", return_value=plan_obj),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.RunApp") as mock_run_app_cls,
            patch("furnace.cli.os._exit") as mock_exit,
        ):
            # Make the RunApp.run() set the shutdown_event
            def _run_sets_shutdown() -> None:
                # The shutdown_event is passed as kwarg to RunApp
                shutdown_evt = mock_run_app_cls.call_args.kwargs["shutdown_event"]
                shutdown_evt.set()

            mock_run_app_cls.return_value.run.side_effect = _run_sets_shutdown

            runner.invoke(app, ["run", str(plan_file)])

        mock_exit.assert_called_once_with(0)

    def test_run_cleanup_demux_dir_all_done(self, tmp_path: Path) -> None:
        """Demux directory is removed when all jobs are done."""
        plan_file = tmp_path / "plan.json"
        demux_dir = tmp_path / "demux"
        demux_dir.mkdir()
        (demux_dir / "dummy.mkv").touch()

        plan_obj = make_plan(
            jobs=[make_job(job_id="j1", status=JobStatus.DONE)],
            destination=str(tmp_path / "out"),
            demux_dir=str(demux_dir),
        )

        cfg = _make_tool_paths(tmp_path)

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.load_plan", return_value=plan_obj),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.RunApp") as mock_run_app_cls,
            patch("furnace.cli.ReportPrinter"),
        ):
            mock_run_app_cls.return_value.run.return_value = None

            result = runner.invoke(app, ["run", str(plan_file)])

        assert result.exit_code == 0, result.output
        assert not demux_dir.exists()

    def test_run_no_cleanup_demux_dir_not_all_done(self, tmp_path: Path) -> None:
        """Demux directory is NOT removed when some jobs are pending."""
        plan_file = tmp_path / "plan.json"
        demux_dir = tmp_path / "demux"
        demux_dir.mkdir()

        plan_obj = make_plan(
            jobs=[
                make_job(job_id="j1", status=JobStatus.DONE),
                make_job(job_id="j2", status=JobStatus.PENDING),
            ],
            destination=str(tmp_path / "out"),
            demux_dir=str(demux_dir),
        )

        cfg = _make_tool_paths(tmp_path)

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.load_plan", return_value=plan_obj),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.RunApp") as mock_run_app_cls,
            patch("furnace.cli.ReportPrinter"),
        ):
            mock_run_app_cls.return_value.run.return_value = None

            result = runner.invoke(app, ["run", str(plan_file)])

        assert result.exit_code == 0, result.output
        assert demux_dir.exists()

    def test_run_no_cleanup_when_demux_dir_none(self, tmp_path: Path) -> None:
        """No cleanup attempted when demux_dir is None."""
        plan_file = tmp_path / "plan.json"
        plan_obj = make_plan(
            jobs=[make_job(job_id="j1", status=JobStatus.DONE)],
            destination=str(tmp_path / "out"),
            demux_dir=None,
        )

        cfg = _make_tool_paths(tmp_path)

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.load_plan", return_value=plan_obj),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.RunApp") as mock_run_app_cls,
            patch("furnace.cli.ReportPrinter"),
            patch.object(shutil, "rmtree") as mock_rmtree,
        ):
            mock_run_app_cls.return_value.run.return_value = None

            result = runner.invoke(app, ["run", str(plan_file)])

        assert result.exit_code == 0, result.output
        mock_rmtree.assert_not_called()

    def test_run_passes_vmaf_enabled(self, tmp_path: Path) -> None:
        """vmaf_enabled from plan is passed to RunApp."""
        plan_file = tmp_path / "plan.json"
        plan_obj = make_plan(
            jobs=[],
            destination=str(tmp_path / "out"),
            vmaf_enabled=True,
        )

        cfg = _make_tool_paths(tmp_path)

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.load_plan", return_value=plan_obj),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.RunApp") as mock_run_app_cls,
            patch("furnace.cli.ReportPrinter"),
        ):
            mock_run_app_cls.return_value.run.return_value = None

            result = runner.invoke(app, ["run", str(plan_file)])

        assert result.exit_code == 0, result.output
        assert mock_run_app_cls.call_args.kwargs["vmaf_enabled"] is True

    def test_run_config_option(self, tmp_path: Path) -> None:
        """--config option is forwarded to load_config."""
        plan_file = tmp_path / "plan.json"
        config_file = tmp_path / "custom.toml"
        plan_obj = make_plan(
            jobs=[],
            destination=str(tmp_path / "out"),
        )

        cfg = _make_tool_paths(tmp_path)

        with (
            patch("furnace.cli.load_config", return_value=cfg) as mock_load_cfg,
            patch("furnace.cli.load_plan", return_value=plan_obj),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.RunApp") as mock_run_app_cls,
            patch("furnace.cli.ReportPrinter"),
        ):
            mock_run_app_cls.return_value.run.return_value = None

            result = runner.invoke(app, ["run", str(plan_file), "--config", str(config_file)])

        assert result.exit_code == 0, result.output
        mock_load_cfg.assert_called_once_with(config_file)


# ---------------------------------------------------------------------------
# plan — config option
# ---------------------------------------------------------------------------


class TestPlanConfigOption:
    def test_config_option_forwarded(self, tmp_path: Path) -> None:
        """--config is forwarded to load_config in plan command."""
        source = tmp_path / "src"
        source.mkdir()
        output = tmp_path / "out"
        config_file = tmp_path / "my.toml"

        cfg = _make_tool_paths(tmp_path)
        plan_obj = make_plan(jobs=[])

        with (
            patch("furnace.cli.load_config", return_value=cfg) as mock_load_cfg,
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.MpvAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.MakemkvAdapter"),
            patch("furnace.cli.DiscDemuxer") as mock_demuxer_cls,
            patch("furnace.cli.Scanner") as mock_scanner_cls,
            patch("furnace.cli.PlannerService") as mock_planner_cls,
        ):
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = []
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                [
                    "plan",
                    str(source),
                    "-o",
                    str(output),
                    "-al",
                    "eng",
                    "-sl",
                    "eng",
                    "--dry-run",
                    "--config",
                    str(config_file),
                ],
            )

        assert result.exit_code == 0, result.output
        mock_load_cfg.assert_called_once_with(config_file)


# ---------------------------------------------------------------------------
# run — _run_executor closure
# ---------------------------------------------------------------------------


class TestRunExecutorClosure:
    def test_executor_fn_creates_adapters_and_runs(self, tmp_path: Path) -> None:
        """The executor_fn closure creates adapters and calls executor.run()."""
        plan_file = tmp_path / "plan.json"
        plan_obj = make_plan(
            jobs=[make_job(job_id="j1", status=JobStatus.PENDING)],
            destination=str(tmp_path / "out"),
        )

        cfg = _make_tool_paths(tmp_path)

        captured_executor_fn: list[Any] = []

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.load_plan", return_value=plan_obj),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.RunApp") as mock_run_app_cls,
            patch("furnace.cli.ReportPrinter"),
        ):
            # Capture the executor_fn instead of running it
            def _capture_and_noop() -> None:
                captured_executor_fn.append(mock_run_app_cls.call_args.kwargs["executor_fn"])

            mock_run_app_cls.return_value.run.side_effect = _capture_and_noop

            result = runner.invoke(app, ["run", str(plan_file)])

        assert result.exit_code == 0, result.output
        assert len(captured_executor_fn) == 1

        # Now call the captured executor_fn with full adapter mocking
        executor_fn = captured_executor_fn[0]
        mock_progress = MagicMock()
        mock_progress.add_tool_line = MagicMock()

        with (
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.QaacAdapter"),
            patch("furnace.cli.MkvmergeAdapter"),
            patch("furnace.cli.MkvpropeditAdapter"),
            patch("furnace.cli.MkcleanAdapter"),
            patch("furnace.cli.NVEncCAdapter"),
            patch("furnace.cli.Executor") as mock_executor_cls,
        ):
            executor_fn(mock_progress)

        mock_executor_cls.return_value.run.assert_called_once()
        mock_progress.stop.assert_called_once()

    def test_executor_fn_with_dovi_tool(self, tmp_path: Path) -> None:
        """When dovi_tool is set, DoviToolAdapter is created."""
        plan_file = tmp_path / "plan.json"
        plan_obj = make_plan(
            jobs=[make_job(job_id="j1", status=JobStatus.PENDING)],
            destination=str(tmp_path / "out"),
        )

        cfg = _make_tool_paths(tmp_path)
        cfg.dovi_tool = tmp_path / "dovi_tool"

        captured_executor_fn: list[Any] = []

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.load_plan", return_value=plan_obj),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.RunApp") as mock_run_app_cls,
            patch("furnace.cli.ReportPrinter"),
        ):
            def _capture() -> None:
                captured_executor_fn.append(mock_run_app_cls.call_args.kwargs["executor_fn"])

            mock_run_app_cls.return_value.run.side_effect = _capture

            result = runner.invoke(app, ["run", str(plan_file)])

        assert result.exit_code == 0, result.output
        executor_fn = captured_executor_fn[0]
        mock_progress = MagicMock()

        with (
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.QaacAdapter"),
            patch("furnace.cli.MkvmergeAdapter"),
            patch("furnace.cli.MkvpropeditAdapter"),
            patch("furnace.cli.MkcleanAdapter"),
            patch("furnace.cli.NVEncCAdapter"),
            patch("furnace.cli.DoviToolAdapter") as mock_dovi,
            patch("furnace.cli.Executor") as mock_executor_cls,
        ):
            executor_fn(mock_progress)

        mock_dovi.assert_called_once()
        mock_executor_cls.return_value.run.assert_called_once()

    def test_executor_fn_stops_progress_on_error(self, tmp_path: Path) -> None:
        """progress.stop() is called even when executor.run() raises."""
        plan_file = tmp_path / "plan.json"
        plan_obj = make_plan(
            jobs=[make_job(job_id="j1", status=JobStatus.PENDING)],
            destination=str(tmp_path / "out"),
        )

        cfg = _make_tool_paths(tmp_path)

        captured_executor_fn: list[Any] = []

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.load_plan", return_value=plan_obj),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.RunApp") as mock_run_app_cls,
            patch("furnace.cli.ReportPrinter"),
        ):
            def _capture() -> None:
                captured_executor_fn.append(mock_run_app_cls.call_args.kwargs["executor_fn"])

            mock_run_app_cls.return_value.run.side_effect = _capture
            runner.invoke(app, ["run", str(plan_file)])

        executor_fn = captured_executor_fn[0]
        mock_progress = MagicMock()

        with (
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.QaacAdapter"),
            patch("furnace.cli.MkvmergeAdapter"),
            patch("furnace.cli.MkvpropeditAdapter"),
            patch("furnace.cli.MkcleanAdapter"),
            patch("furnace.cli.NVEncCAdapter"),
            patch("furnace.cli.Executor") as mock_executor_cls,
        ):
            mock_executor_cls.return_value.run.side_effect = RuntimeError("boom")
            with contextlib.suppress(RuntimeError):
                executor_fn(mock_progress)

        # progress.stop() must be called even on error (finally block)
        mock_progress.stop.assert_called_once()


# ---------------------------------------------------------------------------
# run — demux_dir exists but path not on disk
# ---------------------------------------------------------------------------


class TestRunDemuxDirEdgeCases:
    def test_demux_dir_set_but_not_on_disk(self, tmp_path: Path) -> None:
        """When demux_dir is set in plan but the path doesn't exist, no error."""
        plan_file = tmp_path / "plan.json"
        plan_obj = make_plan(
            jobs=[make_job(job_id="j1", status=JobStatus.DONE)],
            destination=str(tmp_path / "out"),
            demux_dir=str(tmp_path / "nonexistent_demux"),
        )

        cfg = _make_tool_paths(tmp_path)

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.load_plan", return_value=plan_obj),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.RunApp") as mock_run_app_cls,
            patch("furnace.cli.ReportPrinter"),
        ):
            mock_run_app_cls.return_value.run.return_value = None

            result = runner.invoke(app, ["run", str(plan_file)])

        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------


class TestMainModule:
    def test_main_calls_app(self) -> None:
        """Running furnace as `python -m furnace` calls app()."""
        import runpy

        with patch("furnace.cli.app") as mock_app:
            with contextlib.suppress(SystemExit):
                runpy.run_module("furnace", run_name="__main__")
        mock_app.assert_called_once()


# ---------------------------------------------------------------------------
# _make_preview_track_cb
# ---------------------------------------------------------------------------


class TestMakePreviewTrackCb:
    def test_audio_track_calls_preview_audio(self, tmp_path: Path) -> None:
        """Preview callback for an audio track calls mpv.preview_audio."""
        from furnace.cli import _make_preview_track_cb

        movie = make_movie(main_file=tmp_path / "m.mkv")
        mpv = MagicMock()
        track = make_track(
            index=1,
            track_type=TrackType.AUDIO,
            source_file=tmp_path / "audio.mka",
        )

        cb = _make_preview_track_cb(movie, mpv)
        cb(track)

        mpv.preview_audio.assert_called_once_with(movie.main_file, track.source_file, track.index)
        mpv.preview_subtitle.assert_not_called()

    def test_subtitle_track_calls_preview_subtitle(self, tmp_path: Path) -> None:
        """Preview callback for a subtitle track calls mpv.preview_subtitle."""
        from furnace.cli import _make_preview_track_cb

        movie = make_movie(main_file=tmp_path / "m.mkv")
        mpv = MagicMock()
        track = make_track(
            index=2,
            track_type=TrackType.SUBTITLE,
            codec_name="subrip",
            source_file=tmp_path / "subs.srt",
        )

        cb = _make_preview_track_cb(movie, mpv)
        cb(track)

        mpv.preview_subtitle.assert_called_once_with(movie.main_file, track.source_file, track.index)
        mpv.preview_audio.assert_not_called()


# ---------------------------------------------------------------------------
# _select_tracks_tui
# ---------------------------------------------------------------------------


class TestSelectTracksTui:
    def test_returns_app_runner_result(self, tmp_path: Path) -> None:
        """_select_tracks_tui returns whatever the app_runner returns."""
        from furnace.cli import _select_tracks_tui
        from furnace.ui.tui import TrackSelection

        movie = make_movie(main_file=tmp_path / "m.mkv")
        track = make_track(index=1, track_type=TrackType.AUDIO)
        expected = TrackSelection(tracks=[track], downmix={})

        fake_runner = MagicMock(return_value=expected)
        result = _select_tracks_tui(
            movie,
            [track],
            TrackType.AUDIO,
            MagicMock(),
            app_runner=fake_runner,
        )

        assert result is expected
        fake_runner.assert_called_once()

    def test_none_result_falls_back_to_empty(self, tmp_path: Path) -> None:
        """If the app_runner returns None, _select_tracks_tui returns an empty TrackSelection."""
        from furnace.cli import _select_tracks_tui
        from furnace.ui.tui import TrackSelection

        movie = make_movie(main_file=tmp_path / "m.mkv")
        fake_runner = MagicMock(return_value=None)
        result = _select_tracks_tui(
            movie,
            [],
            TrackType.SUBTITLE,
            MagicMock(),
            app_runner=fake_runner,
        )

        assert isinstance(result, TrackSelection)
        assert result.tracks == []
        assert result.downmix == {}

    def test_factory_instantiates_track_selector_screen(self, tmp_path: Path) -> None:
        """The internal screen-factory closure returns a TrackSelectorScreen for the given movie."""
        from furnace.cli import _select_tracks_tui
        from furnace.ui.tui import TrackSelection, TrackSelectorScreen

        movie = make_movie(main_file=tmp_path / "m.mkv")
        track = make_track(index=1, track_type=TrackType.AUDIO)

        captured: list[Any] = []

        def runner(factory: Callable[[], Any]) -> TrackSelection:
            captured.append(factory())
            return TrackSelection(tracks=[], downmix={})

        _select_tracks_tui(
            movie,
            [track],
            TrackType.AUDIO,
            MagicMock(),
            app_runner=runner,
        )

        assert len(captured) == 1
        assert isinstance(captured[0], TrackSelectorScreen)


# ---------------------------------------------------------------------------
# _select_tracks_tui_for_planner
# ---------------------------------------------------------------------------


class TestSelectTracksTuiForPlanner:
    def test_audio_updates_downmix_overrides(self, tmp_path: Path) -> None:
        """For audio, the planner wrapper updates the shared downmix_overrides dict."""
        from furnace.cli import _select_tracks_tui_for_planner
        from furnace.core.models import DownmixMode
        from furnace.ui.tui import TrackSelection

        movie = make_movie(main_file=tmp_path / "m.mkv")
        track = make_track(index=1, track_type=TrackType.AUDIO, source_file=tmp_path / "a.mka")
        downmix_key = (track.source_file, track.index)
        downmix_overrides: dict[Any, Any] = {}

        def fake_runner(_app: Any) -> TrackSelection:
            return TrackSelection(tracks=[track], downmix={downmix_key: DownmixMode.STEREO})

        selected = _select_tracks_tui_for_planner(
            movie,
            [track],
            TrackType.AUDIO,
            MagicMock(),
            downmix_overrides,
            app_runner=fake_runner,
        )

        assert selected == [track]
        assert downmix_overrides == {downmix_key: DownmixMode.STEREO}

    def test_subtitle_does_not_update_downmix_overrides(self, tmp_path: Path) -> None:
        """For subtitles, downmix overrides are left alone."""
        from furnace.cli import _select_tracks_tui_for_planner
        from furnace.ui.tui import TrackSelection

        movie = make_movie(main_file=tmp_path / "m.mkv")
        track = make_track(index=2, track_type=TrackType.SUBTITLE, codec_name="subrip")
        downmix_overrides: dict[Any, Any] = {}

        def fake_runner(_app: Any) -> TrackSelection:
            return TrackSelection(tracks=[track], downmix={})

        selected = _select_tracks_tui_for_planner(
            movie,
            [track],
            TrackType.SUBTITLE,
            MagicMock(),
            downmix_overrides,
            app_runner=fake_runner,
        )

        assert selected == [track]
        assert downmix_overrides == {}


# ---------------------------------------------------------------------------
# _resolve_und_language_tui
# ---------------------------------------------------------------------------


class TestResolveUndLanguageTui:
    def test_returns_app_runner_result(self, tmp_path: Path) -> None:
        """Runner-returned language is surfaced back to caller."""
        from furnace.cli import _resolve_und_language_tui

        movie = make_movie(main_file=tmp_path / "m.mkv")
        track = make_track(index=1, track_type=TrackType.AUDIO, language="und")

        result = _resolve_und_language_tui(
            movie,
            track,
            ["rus", "eng"],
            MagicMock(),
            app_runner=lambda _a: "rus",
        )

        assert result == "rus"

    def test_none_falls_back_to_first_lang(self, tmp_path: Path) -> None:
        """If the runner returns None, fall back to the first language in the list."""
        from furnace.cli import _resolve_und_language_tui

        movie = make_movie(main_file=tmp_path / "m.mkv")
        track = make_track(index=1, track_type=TrackType.AUDIO, language="und")

        result = _resolve_und_language_tui(
            movie,
            track,
            ["jpn", "eng"],
            MagicMock(),
            app_runner=lambda _a: None,
        )

        assert result == "jpn"

    def test_factory_instantiates_language_screen(self, tmp_path: Path) -> None:
        """The internal factory returns a LanguageSelectorScreen for the supplied track."""
        from furnace.cli import _resolve_und_language_tui
        from furnace.ui.tui import LanguageSelectorScreen

        movie = make_movie(main_file=tmp_path / "m.mkv")
        track = make_track(index=1, track_type=TrackType.AUDIO, language="und")

        captured: list[Any] = []

        def runner(factory: Callable[[], Any]) -> str:
            captured.append(factory())
            return "rus"

        _resolve_und_language_tui(
            movie,
            track,
            ["rus", "eng"],
            MagicMock(),
            app_runner=runner,
        )

        assert isinstance(captured[0], LanguageSelectorScreen)


# ---------------------------------------------------------------------------
# _append_demuxed_scan_results
# ---------------------------------------------------------------------------


class TestAppendDemuxedScanResults:
    def test_appends_one_scan_result_per_demuxed_path(self, tmp_path: Path) -> None:
        """Each demuxed path becomes a ScanResult with expected output_path layout."""
        from furnace.cli import _append_demuxed_scan_results

        output = tmp_path / "out"
        demuxed = [tmp_path / "disc_title_1.mkv", tmp_path / "disc_title_2.mkv"]
        scan_results: list[Any] = []

        _append_demuxed_scan_results(scan_results, demuxed, output)

        assert len(scan_results) == 2
        assert scan_results[0].main_file == demuxed[0]
        assert scan_results[0].satellite_files == []
        assert scan_results[0].output_path == output / "disc_title_1" / "disc_title_1.mkv"
        assert scan_results[1].output_path == output / "disc_title_2" / "disc_title_2.mkv"

    def test_empty_demuxed_does_nothing(self, tmp_path: Path) -> None:
        """No demuxed paths => nothing appended."""
        from furnace.cli import _append_demuxed_scan_results

        scan_results: list[Any] = [MagicMock()]
        _append_demuxed_scan_results(scan_results, [], tmp_path / "out")
        assert len(scan_results) == 1  # unchanged


# ---------------------------------------------------------------------------
# _apply_demux_dir_to_plan
# ---------------------------------------------------------------------------


class TestApplyDemuxDirToPlan:
    def test_sets_demux_dir_on_plan(self, tmp_path: Path) -> None:
        """When demux_dir is provided, its str form is assigned to plan.demux_dir."""
        from furnace.cli import _apply_demux_dir_to_plan

        plan_obj = make_plan(jobs=[])
        _apply_demux_dir_to_plan(plan_obj, tmp_path / "demux")
        assert plan_obj.demux_dir == str(tmp_path / "demux")

    def test_none_leaves_plan_unchanged(self) -> None:
        """When demux_dir is None, plan.demux_dir stays at its current value."""
        from furnace.cli import _apply_demux_dir_to_plan

        plan_obj = make_plan(jobs=[], demux_dir=None)
        _apply_demux_dir_to_plan(plan_obj, None)
        assert plan_obj.demux_dir is None


# ---------------------------------------------------------------------------
# _run_disc_demux_interactive
# ---------------------------------------------------------------------------


class TestDvdDemuxedPaths:
    def test_matches_only_paths_prefixed_with_disc_label(self, tmp_path: Path) -> None:
        """Only paths whose name starts with the disc label are marked as DVD demuxed."""
        from furnace.cli import _dvd_demuxed_paths
        from furnace.core.models import DiscSource, DiscType

        disc = DiscSource(path=tmp_path / "mydvd" / "VIDEO_TS", disc_type=DiscType.DVD)
        mine = tmp_path / "mydvd_title_1.mkv"
        other = tmp_path / "other_title_1.mkv"
        result = _dvd_demuxed_paths([disc], {disc: [MagicMock()]}, [mine, other])
        assert result == {mine}

    def test_non_dvd_disc_ignored(self, tmp_path: Path) -> None:
        """Bluray discs are never flagged even when the filename matches."""
        from furnace.cli import _dvd_demuxed_paths
        from furnace.core.models import DiscSource, DiscType

        disc = DiscSource(path=tmp_path / "mybd" / "BDMV", disc_type=DiscType.BLURAY)
        mkv = tmp_path / "mybd_title_1.mkv"
        assert _dvd_demuxed_paths([disc], {disc: [MagicMock()]}, [mkv]) == set()

    def test_disc_not_in_selected_titles_ignored(self, tmp_path: Path) -> None:
        """DVD that wasn't selected isn't considered."""
        from furnace.cli import _dvd_demuxed_paths
        from furnace.core.models import DiscSource, DiscType

        disc = DiscSource(path=tmp_path / "mydvd" / "VIDEO_TS", disc_type=DiscType.DVD)
        mkv = tmp_path / "mydvd_title_1.mkv"
        assert _dvd_demuxed_paths([disc], {}, [mkv]) == set()


class TestRunDiscDemuxInteractive:
    def _adapters(self) -> tuple[MagicMock, MagicMock]:
        return MagicMock(), MagicMock()

    def test_no_discs_returns_empty(self, tmp_path: Path) -> None:
        """No discs detected => returns (None, [], set())."""
        from furnace.cli import _run_disc_demux_interactive

        ffmpeg, mpv = self._adapters()
        demuxer = MagicMock()

        demux_dir, paths, sar = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[],
            disc_demuxer=demuxer,
            ffmpeg_adapter=ffmpeg,
            mpv_adapter=mpv,
            playlist_app_runner=MagicMock(),
            file_app_runner=MagicMock(),
        )

        assert demux_dir is None
        assert paths == []
        assert sar == set()
        demuxer.list_titles.assert_not_called()

    def test_single_playlist_auto_selected_and_demuxed(self, tmp_path: Path) -> None:
        """One playlist -> no TUI, just demux (single file, no file-selector)."""
        from furnace.cli import _run_disc_demux_interactive
        from furnace.core.models import DiscSource, DiscTitle, DiscType

        disc_root = tmp_path / "disc_folder" / "BDMV"
        disc = DiscSource(path=disc_root, disc_type=DiscType.BLURAY)
        title = DiscTitle(number=1, duration_s=5400.0, raw_label="1: ...")

        demuxer = MagicMock()
        demuxer.list_titles.return_value = [title]
        demuxed_mkv = tmp_path / "disc_folder_title_1.mkv"
        demuxer.demux.return_value = [demuxed_mkv]

        ffmpeg, mpv = self._adapters()

        demux_dir, paths, sar = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_demuxer=demuxer,
            ffmpeg_adapter=ffmpeg,
            mpv_adapter=mpv,
            playlist_app_runner=MagicMock(),  # not called
            file_app_runner=MagicMock(),  # not called (single file, no DVD)
        )

        assert demux_dir == tmp_path / ".furnace_demux"
        assert paths == [demuxed_mkv]
        assert sar == set()
        demuxer.demux.assert_called_once()

    def test_empty_playlist_list_skips_disc(self, tmp_path: Path) -> None:
        """When list_titles returns [], that disc is skipped entirely."""
        from furnace.cli import _run_disc_demux_interactive
        from furnace.core.models import DiscSource, DiscType

        disc = DiscSource(path=tmp_path / "BDMV", disc_type=DiscType.BLURAY)

        demuxer = MagicMock()
        demuxer.list_titles.return_value = []

        ffmpeg, mpv = self._adapters()
        demux_dir, paths, sar = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_demuxer=demuxer,
            ffmpeg_adapter=ffmpeg,
            mpv_adapter=mpv,
            playlist_app_runner=MagicMock(),
            file_app_runner=MagicMock(),
        )

        assert demux_dir is None
        assert paths == []
        assert sar == set()
        demuxer.demux.assert_not_called()

    def test_multiple_playlists_uses_runner(self, tmp_path: Path) -> None:
        """Multi-playlist disc: runner picks a subset; only picked titles are demuxed.

        The runner is also driven here to invoke its factory, exercising the
        PlaylistSelectorScreen construction path.
        """
        from furnace.cli import _run_disc_demux_interactive
        from furnace.core.models import DiscSource, DiscTitle, DiscType
        from furnace.ui.tui import PlaylistSelectorScreen

        disc_root = tmp_path / "disc" / "BDMV"
        disc = DiscSource(path=disc_root, disc_type=DiscType.BLURAY)
        t1 = DiscTitle(number=1, duration_s=100, raw_label="1")
        t2 = DiscTitle(number=2, duration_s=200, raw_label="2")

        demuxer = MagicMock()
        demuxer.list_titles.return_value = [t1, t2]
        demuxed_mkv = tmp_path / "disc_title_2.mkv"
        demuxer.demux.return_value = [demuxed_mkv]

        ffmpeg, mpv = self._adapters()

        screens_built: list[Any] = []

        def playlist_runner(factory: Callable[[], Any]) -> list[DiscTitle]:
            screens_built.append(factory())
            return [t2]

        demux_dir, paths, sar = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_demuxer=demuxer,
            ffmpeg_adapter=ffmpeg,
            mpv_adapter=mpv,
            playlist_app_runner=playlist_runner,
            file_app_runner=MagicMock(),
        )

        assert demux_dir == tmp_path / ".furnace_demux"
        assert paths == [demuxed_mkv]
        assert sar == set()
        assert len(screens_built) == 1
        assert isinstance(screens_built[0], PlaylistSelectorScreen)
        # Only t2 should have been passed to demuxer.demux
        call_kwargs = demuxer.demux.call_args.kwargs
        assert call_kwargs["selected_titles"] == {disc: [t2]}

    def test_multiple_playlists_runner_returns_none_skips_disc(self, tmp_path: Path) -> None:
        """If the playlist runner returns None, that disc is skipped."""
        from furnace.cli import _run_disc_demux_interactive
        from furnace.core.models import DiscSource, DiscTitle, DiscType

        disc_root = tmp_path / "disc" / "BDMV"
        disc = DiscSource(path=disc_root, disc_type=DiscType.BLURAY)
        t1 = DiscTitle(number=1, duration_s=100, raw_label="1")
        t2 = DiscTitle(number=2, duration_s=200, raw_label="2")

        demuxer = MagicMock()
        demuxer.list_titles.return_value = [t1, t2]

        ffmpeg, mpv = self._adapters()

        demux_dir, paths, sar = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_demuxer=demuxer,
            ffmpeg_adapter=ffmpeg,
            mpv_adapter=mpv,
            playlist_app_runner=MagicMock(return_value=None),
            file_app_runner=MagicMock(),
        )

        assert demux_dir is None
        assert paths == []
        assert sar == set()
        demuxer.demux.assert_not_called()

    def test_dvd_demuxed_file_triggers_file_selector_and_sar(self, tmp_path: Path) -> None:
        """DVD-demuxed files run the file-selector, surface SAR overrides, and
        the file runner builds a FileSelectorScreen via the factory."""
        from furnace.cli import _run_disc_demux_interactive
        from furnace.core.models import DiscSource, DiscTitle, DiscType
        from furnace.ui.tui import FileSelection, FileSelectorScreen

        disc_root = tmp_path / "dvdroot" / "VIDEO_TS"
        disc = DiscSource(path=disc_root, disc_type=DiscType.DVD)
        title = DiscTitle(number=1, duration_s=100, raw_label="t")

        demuxer = MagicMock()
        demuxer.list_titles.return_value = [title]
        dvd_mkv = tmp_path / ".furnace_demux" / "dvdroot_title_1.mkv"
        demuxer.demux.return_value = [dvd_mkv]

        ffmpeg = MagicMock()
        ffmpeg.probe.return_value = {"format": {"duration": "100.0", "size": "1000"}}
        mpv = MagicMock()

        screens_built: list[Any] = []

        def file_runner(factory: Callable[[], Any]) -> FileSelection:
            screens_built.append(factory())
            return FileSelection(selected=[dvd_mkv], sar_override={dvd_mkv})

        demux_dir, paths, sar = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_demuxer=demuxer,
            ffmpeg_adapter=ffmpeg,
            mpv_adapter=mpv,
            playlist_app_runner=MagicMock(),
            file_app_runner=file_runner,
        )

        assert demux_dir == tmp_path / ".furnace_demux"
        assert paths == [dvd_mkv]
        assert sar == {dvd_mkv}
        assert len(screens_built) == 1
        assert isinstance(screens_built[0], FileSelectorScreen)
        # ffmpeg.probe was called for the demuxed file
        ffmpeg.probe.assert_called_once_with(dvd_mkv)

    def test_multiple_demuxed_files_trigger_file_selector(self, tmp_path: Path) -> None:
        """Non-DVD but >1 demuxed file => file selector runs."""
        from furnace.cli import _run_disc_demux_interactive
        from furnace.core.models import DiscSource, DiscTitle, DiscType
        from furnace.ui.tui import FileSelection

        disc_root = tmp_path / "bdroot" / "BDMV"
        disc = DiscSource(path=disc_root, disc_type=DiscType.BLURAY)
        t1 = DiscTitle(number=1, duration_s=1.0, raw_label="1")
        t2 = DiscTitle(number=2, duration_s=2.0, raw_label="2")
        demuxer = MagicMock()
        demuxer.list_titles.return_value = [t1, t2]
        mkv1 = tmp_path / ".furnace_demux" / "bdroot_title_1.mkv"
        mkv2 = tmp_path / ".furnace_demux" / "bdroot_title_2.mkv"
        demuxer.demux.return_value = [mkv1, mkv2]

        ffmpeg = MagicMock()
        ffmpeg.probe.return_value = {"format": {"duration": "1", "size": "2"}}
        mpv = MagicMock()
        playlist_runner = MagicMock(return_value=[t1, t2])
        file_runner = MagicMock(return_value=FileSelection(selected=[mkv1], sar_override=set()))

        demux_dir, paths, sar = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_demuxer=demuxer,
            ffmpeg_adapter=ffmpeg,
            mpv_adapter=mpv,
            playlist_app_runner=playlist_runner,
            file_app_runner=file_runner,
        )

        assert demux_dir == tmp_path / ".furnace_demux"
        assert paths == [mkv1]
        assert sar == set()
        assert ffmpeg.probe.call_count == 2
        file_runner.assert_called_once()

    def test_file_selector_returning_none_keeps_demuxed_paths(self, tmp_path: Path) -> None:
        """If the file-selector runner returns None, original demuxed paths are kept."""
        from furnace.cli import _run_disc_demux_interactive
        from furnace.core.models import DiscSource, DiscTitle, DiscType

        disc_root = tmp_path / "bdroot" / "BDMV"
        disc = DiscSource(path=disc_root, disc_type=DiscType.BLURAY)
        t1 = DiscTitle(number=1, duration_s=1.0, raw_label="1")
        t2 = DiscTitle(number=2, duration_s=2.0, raw_label="2")
        demuxer = MagicMock()
        demuxer.list_titles.return_value = [t1, t2]
        mkv1 = tmp_path / ".furnace_demux" / "bdroot_title_1.mkv"
        mkv2 = tmp_path / ".furnace_demux" / "bdroot_title_2.mkv"
        demuxer.demux.return_value = [mkv1, mkv2]

        ffmpeg = MagicMock()
        ffmpeg.probe.return_value = {"format": {"duration": "1", "size": "2"}}
        mpv = MagicMock()
        playlist_runner = MagicMock(return_value=[t1, t2])
        file_runner = MagicMock(return_value=None)

        _demux_dir, paths, sar = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_demuxer=demuxer,
            ffmpeg_adapter=ffmpeg,
            mpv_adapter=mpv,
            playlist_app_runner=playlist_runner,
            file_app_runner=file_runner,
        )

        assert paths == [mkv1, mkv2]
        assert sar == set()

    def test_probe_missing_format_defaults(self, tmp_path: Path) -> None:
        """Probing a file without a format dict falls back to zero values without crashing."""
        from furnace.cli import _run_disc_demux_interactive
        from furnace.core.models import DiscSource, DiscTitle, DiscType
        from furnace.ui.tui import FileSelection

        disc_root = tmp_path / "dvd" / "VIDEO_TS"
        disc = DiscSource(path=disc_root, disc_type=DiscType.DVD)
        title = DiscTitle(number=1, duration_s=0.0, raw_label="t")

        demuxer = MagicMock()
        demuxer.list_titles.return_value = [title]
        mkv = tmp_path / ".furnace_demux" / "dvd_title_1.mkv"
        demuxer.demux.return_value = [mkv]

        ffmpeg = MagicMock()
        ffmpeg.probe.return_value = {}  # no 'format' key
        mpv = MagicMock()
        file_runner = MagicMock(return_value=FileSelection(selected=[mkv], sar_override=set()))

        _, paths, _sar = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_demuxer=demuxer,
            ffmpeg_adapter=ffmpeg,
            mpv_adapter=mpv,
            playlist_app_runner=MagicMock(),
            file_app_runner=file_runner,
        )

        assert paths == [mkv]


# ---------------------------------------------------------------------------
# plan — full integration with interactive disc-demux path
# ---------------------------------------------------------------------------


class TestPlanSelectorClosures:
    def test_track_selector_closure_routes_through_helper(self, tmp_path: Path) -> None:
        """The track_selector closure passed to PlannerService routes to _select_tracks_tui_for_planner."""
        source = tmp_path / "src"
        source.mkdir()
        output = tmp_path / "out"

        cfg = _make_tool_paths(tmp_path)
        plan_obj = make_plan(jobs=[])

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.MpvAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.MakemkvAdapter"),
            patch("furnace.cli.DiscDemuxer") as mock_demuxer_cls,
            patch("furnace.cli.Scanner") as mock_scanner_cls,
            patch("furnace.cli.PlannerService") as mock_planner_cls,
            patch("furnace.cli.save_plan"),
            patch("furnace.cli._select_tracks_tui_for_planner") as mock_sel,
            patch("furnace.cli._resolve_und_language_tui") as mock_res,
        ):
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = []
            mock_planner_cls.return_value.create_plan.return_value = plan_obj
            mock_sel.return_value = []
            mock_res.return_value = "eng"

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng"],
            )

            assert result.exit_code == 0, result.output

            # Pull the closures the planner was instantiated with
            planner_kwargs = mock_planner_cls.call_args.kwargs
            selector = planner_kwargs["track_selector"]
            resolver = planner_kwargs["und_resolver"]

            # Invoke them — this exercises the closure bodies at lines 454/457.
            movie = make_movie(main_file=source / "m.mkv")
            track = make_track(index=1, track_type=TrackType.AUDIO)
            selector(movie, [track], TrackType.AUDIO)
            resolver(movie, track, ["eng", "rus"])

            mock_sel.assert_called_once()
            mock_res.assert_called_once()


class TestPlanDiscInteractive:
    def test_plan_calls_disc_interactive_when_discs_detected(self, tmp_path: Path) -> None:
        """plan() delegates to _run_disc_demux_interactive when discs are found and not dry_run."""
        from furnace.core.models import DiscSource, DiscType

        source = tmp_path / "src"
        source.mkdir()
        output = tmp_path / "out"

        cfg = _make_tool_paths(tmp_path)
        disc = DiscSource(path=source / "BDMV", disc_type=DiscType.BLURAY)
        demuxed = source / ".furnace_demux" / "X.mkv"
        plan_obj = make_plan(jobs=[])

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.MpvAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.MakemkvAdapter"),
            patch("furnace.cli.DiscDemuxer") as mock_demuxer_cls,
            patch("furnace.cli._run_disc_demux_interactive") as mock_interactive,
            patch("furnace.cli.Scanner") as mock_scanner_cls,
            patch("furnace.cli.Analyzer"),
            patch("furnace.cli.PlannerService") as mock_planner_cls,
            patch("furnace.cli.save_plan"),
        ):
            mock_demuxer_cls.return_value.detect.return_value = [disc]
            mock_interactive.return_value = (source / ".furnace_demux", [demuxed], {demuxed})
            mock_scanner_cls.return_value.scan.return_value = []
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng"],
            )

        assert result.exit_code == 0, result.output
        mock_interactive.assert_called_once()
        # Plan.demux_dir should be set from the returned demux_dir
        assert plan_obj.demux_dir == str(source / ".furnace_demux")
        # sar_override_paths forwarded to planner
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args.kwargs
        assert call_kwargs["sar_overrides"] == {demuxed}


# ---------------------------------------------------------------------------
# disc-demux interactive: reporter pause/resume coverage
# ---------------------------------------------------------------------------


class TestPlanDiscInteractiveReporter:
    """Cover the `reporter is not None` branches in `_run_disc_demux_interactive`."""

    def test_reporter_pause_resume_around_screens(self, tmp_path: Path) -> None:
        """When a reporter is supplied, pause/resume bracket every interactive screen."""
        from unittest.mock import call

        from furnace.cli import _run_disc_demux_interactive
        from furnace.core.models import DiscSource, DiscTitle, DiscType
        from furnace.ui.tui import FileSelection

        disc_root = tmp_path / "bdroot" / "BDMV"
        disc = DiscSource(path=disc_root, disc_type=DiscType.BLURAY)
        t1 = DiscTitle(number=1, duration_s=1.0, raw_label="1")
        t2 = DiscTitle(number=2, duration_s=2.0, raw_label="2")

        demuxer = MagicMock()
        demuxer.list_titles.return_value = [t1, t2]
        mkv1 = tmp_path / ".furnace_demux" / "bdroot_title_1.mkv"
        mkv2 = tmp_path / ".furnace_demux" / "bdroot_title_2.mkv"
        demuxer.demux.return_value = [mkv1, mkv2]

        ffmpeg = MagicMock()
        ffmpeg.probe.return_value = {"format": {"duration": "1", "size": "2"}}
        mpv = MagicMock()

        reporter = MagicMock()
        manager = MagicMock()
        manager.attach_mock(reporter.pause, "pause")
        manager.attach_mock(reporter.resume, "resume")

        playlist_runner = MagicMock(return_value=[t1, t2])
        file_runner = MagicMock(return_value=FileSelection(selected=[mkv1], sar_override=set()))

        _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_demuxer=demuxer,
            ffmpeg_adapter=ffmpeg,
            mpv_adapter=mpv,
            reporter=reporter,
            playlist_app_runner=playlist_runner,
            file_app_runner=file_runner,
        )

        # pause/resume called once around the playlist runner and once around the file runner.
        assert manager.mock_calls == [call.pause(), call.resume(), call.pause(), call.resume()]
        # demuxer.demux receives the reporter (not on_output) under the new wiring.
        assert demuxer.demux.call_args.kwargs["reporter"] is reporter


# ---------------------------------------------------------------------------
# plan: detect_disc rel_path fallback (ValueError branch)
# ---------------------------------------------------------------------------


class TestPlanDetectRelPathFallback:
    """The `rel_str = disc.path.parent.name` fallback fires when relative_to() raises."""

    def test_disc_outside_source_falls_back_to_basename(self, tmp_path: Path) -> None:
        """A disc whose parent dir is not under `source` triggers the ValueError branch."""
        from furnace.core.models import DiscSource, DiscType

        source = tmp_path / "src"
        source.mkdir()
        output = tmp_path / "out"

        # Disc lives outside `source`, so .relative_to(source) raises ValueError.
        outside = tmp_path / "elsewhere" / "BDMV"
        disc = DiscSource(path=outside, disc_type=DiscType.BLURAY)

        cfg = _make_tool_paths(tmp_path)
        plan_obj = make_plan(jobs=[])

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.MpvAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.MakemkvAdapter"),
            patch("furnace.cli.DiscDemuxer") as mock_demuxer_cls,
            patch("furnace.cli._run_disc_demux_interactive") as mock_interactive,
            patch("furnace.cli.Scanner") as mock_scanner_cls,
            patch("furnace.cli.Analyzer"),
            patch("furnace.cli.PlannerService") as mock_planner_cls,
            patch("furnace.cli.save_plan"),
            patch("furnace.cli.RichPlanReporter") as mock_reporter_cls,
        ):
            reporter_inst = mock_reporter_cls.return_value
            mock_demuxer_cls.return_value.detect.return_value = [disc]
            mock_interactive.return_value = (None, [], set())
            mock_scanner_cls.return_value.scan.return_value = []
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng"],
            )

        assert result.exit_code == 0, result.output
        # detect_disc was called with the parent-name fallback.
        reporter_inst.detect_disc.assert_called_once_with(DiscType.BLURAY, "elsewhere")


# ---------------------------------------------------------------------------
# plan: analyzer raises ValueError (HDR10+ branch)
# ---------------------------------------------------------------------------


class TestPlanAnalyzerValueError:
    """Analyzer.analyze() raising ValueError is logged and skipped, not propagated."""

    def test_analyzer_value_error_skipped(self, tmp_path: Path) -> None:
        from furnace.core.models import ScanResult

        source = tmp_path / "src"
        source.mkdir()
        output = tmp_path / "out"

        cfg = _make_tool_paths(tmp_path)
        scan_result = ScanResult(
            main_file=source / "movie.mkv",
            satellite_files=[],
            output_path=output / "movie" / "movie.mkv",
        )
        plan_obj = make_plan(jobs=[])

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.MpvAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.MakemkvAdapter"),
            patch("furnace.cli.DiscDemuxer") as mock_demuxer_cls,
            patch("furnace.cli.Scanner") as mock_scanner_cls,
            patch("furnace.cli.Analyzer") as mock_analyzer_cls,
            patch("furnace.cli.PlannerService") as mock_planner_cls,
        ):
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = [scan_result]
            mock_analyzer_cls.return_value.analyze.side_effect = ValueError("HDR10+ not supported")
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng", "--dry-run"],
            )

        assert result.exit_code == 0, result.output
        # Planner sees no movies because analyze raised.
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args.kwargs
        assert call_kwargs["movies"] == []


# ---------------------------------------------------------------------------
# plan: KeyboardInterrupt -> reporter.interrupted() + Exit(130)
# ---------------------------------------------------------------------------


class TestPlanKeyboardInterrupt:
    """Ctrl+C anywhere in the plan body exits 130 and notifies the reporter."""

    def test_keyboard_interrupt_during_detect(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        source.mkdir()
        output = tmp_path / "out"

        cfg = _make_tool_paths(tmp_path)

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.MpvAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.MakemkvAdapter"),
            patch("furnace.cli.DiscDemuxer") as mock_demuxer_cls,
            patch("furnace.cli.RichPlanReporter") as mock_reporter_cls,
        ):
            reporter_inst = mock_reporter_cls.return_value
            mock_demuxer_cls.return_value.detect.side_effect = KeyboardInterrupt

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng", "--dry-run"],
            )

        assert result.exit_code == 130
        reporter_inst.interrupted.assert_called_once()
        reporter_inst.stop.assert_called_once()


# ---------------------------------------------------------------------------
# default app-runner helpers construct & call App/screen correctly
# ---------------------------------------------------------------------------


class TestDefaultAppRunner:
    def test_run_screen_app_captures_dismiss_result(self) -> None:
        """_run_screen_app runs an App, and on_mount pushes the screen factory's screen.

        We drive the inner class by intercepting `run()` to simulate Textual
        calling compose+on_mount: we call them ourselves, then check the
        dismiss callback captures the value returned to _run_screen_app.
        """
        from furnace.cli import _run_screen_app

        sentinel = "dismiss-result"
        factory_calls: list[int] = []
        fake_screen = MagicMock()

        def _factory() -> Any:
            factory_calls.append(1)
            return fake_screen

        # Patch App.run so on_mount is invoked and the dismiss callback is triggered.
        composed: list[Any] = []

        def fake_run(self: Any) -> None:
            # Drive compose() so the Header-yielding line is covered.
            composed.extend(self.compose())
            # Stub push_screen: call the on_dismiss callback immediately with sentinel.
            pushed: list[Any] = []

            def _push_screen(screen: Any, on_dismiss: Callable[[Any], None]) -> None:
                pushed.append(screen)
                on_dismiss(sentinel)

            self.push_screen = _push_screen
            # Also stub exit so it's a no-op.
            self.exit = lambda _result: None
            self.on_mount()
            assert pushed == [fake_screen]

        with patch("textual.app.App.run", fake_run):
            result = _run_screen_app(_factory)

        assert result == sentinel
        assert factory_calls == [1]
        assert len(composed) == 1  # Header yielded

    def test_run_screen_app_handles_none_dismiss(self) -> None:
        """Dismiss callback receiving None makes _run_screen_app return None."""
        from collections.abc import Callable as _Callable

        from furnace.cli import _run_screen_app

        fake_screen = MagicMock()

        def fake_run(self: Any) -> None:
            def _push_screen(_screen: Any, on_dismiss: _Callable[[Any], None]) -> None:
                on_dismiss(None)

            self.push_screen = _push_screen
            self.exit = lambda _r: None
            self.on_mount()

        with patch("textual.app.App.run", fake_run):
            result = _run_screen_app(lambda: fake_screen)

        assert result is None
