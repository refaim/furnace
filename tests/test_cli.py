from __future__ import annotations

import contextlib
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from furnace.cli import _check_interlaced_grain_metrics_ready, _setup_logging, app
from furnace.core.models import AnalysisOutcome, AnalyzeStatus, JobStatus, TrackType
from tests.conftest import make_job, make_movie, make_plan, make_track, make_video_params

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
    cfg.bestsource = None
    cfg.vship = None
    cfg.bwdif = None
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
            mock_analyzer_cls.return_value.analyze.return_value = AnalysisOutcome(
                movie, AnalyzeStatus.DONE, "h264 1920x1080 24fps SDR, 1 audio (jpn), 1 subs"
            )
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

    def test_metrics_flag_is_rejected(self, tmp_path: Path) -> None:
        """--metrics is no longer a valid option."""
        source = tmp_path / "src"
        source.mkdir()
        with patch("furnace.cli.load_config", return_value=_make_tool_paths(tmp_path)):
            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(tmp_path / "out"),
                 "-al", "eng", "-sl", "eng", "--dry-run", "--metrics"],
            )
        assert result.exit_code != 0

    def test_copy_video_flag_forwarded(self, tmp_path: Path) -> None:
        """--copy-video flag is forwarded to planner.create_plan()."""
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
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng", "--dry-run", "--copy-video"],
            )

        assert result.exit_code == 0, result.output
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args.kwargs
        assert call_kwargs["copy_video"] is True

    def test_copy_video_short_flag_forwarded(self, tmp_path: Path) -> None:
        """-cv short flag is forwarded to planner.create_plan()."""
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
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng", "--dry-run", "-cv"],
            )

        assert result.exit_code == 0, result.output
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args.kwargs
        assert call_kwargs["copy_video"] is True

    def test_copy_video_defaults_false(self, tmp_path: Path) -> None:
        """Without the flag, copy_video defaults to False in create_plan()."""
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
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args.kwargs
        assert call_kwargs["copy_video"] is False

    def test_force_flag_forwarded(self, tmp_path: Path) -> None:
        """--force flag is forwarded to the Analyzer."""
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
            patch("furnace.cli.Analyzer") as mock_analyzer_cls,
            patch("furnace.cli.PlannerService") as mock_planner_cls,
        ):
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = []
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng", "--dry-run", "--force"],
            )

        assert result.exit_code == 0, result.output
        assert mock_analyzer_cls.call_args.kwargs["force"] is True

    def test_force_short_flag_forwarded(self, tmp_path: Path) -> None:
        """-f short flag is forwarded to the Analyzer."""
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
            patch("furnace.cli.Analyzer") as mock_analyzer_cls,
            patch("furnace.cli.PlannerService") as mock_planner_cls,
        ):
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = []
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng", "--dry-run", "-f"],
            )

        assert result.exit_code == 0, result.output
        assert mock_analyzer_cls.call_args.kwargs["force"] is True

    def test_force_defaults_false(self, tmp_path: Path) -> None:
        """Without the flag, force defaults to False on the Analyzer."""
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
        assert mock_analyzer_cls.call_args.kwargs["force"] is False


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
        """When discs are detected but --dry-run is set, demux phase is skipped.

        list_titles still runs once per disc — it now feeds the Detect phase
        rendering (``... -> N titles``). Only the actual demux is skipped.
        """
        from furnace.core.models import DiscSource, DiscTitle, DiscType

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
            mock_demuxer_cls.return_value.list_titles.return_value = [
                DiscTitle(number=1, duration_s=1.0, raw_label="1"),
            ]
            mock_scanner_cls.return_value.scan.return_value = []
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng", "--dry-run"],
            )

        assert result.exit_code == 0, result.output
        # list_titles is now called from the Detect loop (not from demux).
        mock_demuxer_cls.return_value.list_titles.assert_called_once_with(disc)
        # But the actual demux is still skipped under --dry-run.
        mock_demuxer_cls.return_value.demux.assert_not_called()


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
        """When analyzer.analyze yields a SKIPPED outcome (no movie), it is skipped."""
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
            mock_analyzer_cls.return_value.analyze.return_value = AnalysisOutcome(
                None, AnalyzeStatus.SKIPPED, "already encoded"
            )
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
        # The executor must be wired with a video_copier for passthrough jobs.
        exec_kwargs = mock_executor_cls.call_args.kwargs
        assert exec_kwargs["video_copier"] is not None
        # ...and with an always-on target-quality service (NVEnc QVBR search).
        assert exec_kwargs["target_quality"] is not None

    def test_executor_fn_wires_svt_grain_encoder(self, tmp_path: Path) -> None:
        """The executor_fn builds an SvtAv1Adapter from cfg.ffmpeg as grain_encoder."""
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
            patch("furnace.cli.SvtAv1Adapter") as mock_svt,
            patch("furnace.cli.Executor") as mock_executor_cls,
        ):
            executor_fn(mock_progress)

        # SVT adapter is built from the bundled ffmpeg path (no new config key).
        mock_svt.assert_called_once()
        svt_args = mock_svt.call_args.args
        assert svt_args[0] == cfg.ffmpeg
        # ...and wired into the executor as the grain encoder.
        exec_kwargs = mock_executor_cls.call_args.kwargs
        assert exec_kwargs["grain_encoder"] is mock_svt.return_value

    def _run_executor_fn(self, tmp_path: Path, cfg: Any) -> tuple[Any, Any]:
        """Drive ``furnace run`` far enough to capture and invoke the executor
        factory with all adapters mocked. Returns (mock_vship, mock_svt)."""
        plan_file = tmp_path / "plan.json"
        plan_obj = make_plan(
            jobs=[make_job(job_id="j1", status=JobStatus.PENDING)],
            destination=str(tmp_path / "out"),
        )
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

        with (
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.QaacAdapter"),
            patch("furnace.cli.MkvmergeAdapter"),
            patch("furnace.cli.MkvpropeditAdapter"),
            patch("furnace.cli.MkcleanAdapter"),
            patch("furnace.cli.NVEncCAdapter"),
            patch("furnace.cli.VshipMetricsAdapter") as mock_vship,
            patch("furnace.cli.SvtAv1Adapter") as mock_svt,
            patch("furnace.cli.Executor"),
        ):
            executor_fn(MagicMock())

        return mock_vship, mock_svt

    def test_executor_fn_wires_vship_metrics(self, tmp_path: Path) -> None:
        """bestsource+vship configured (bwdif absent) -> VshipMetricsAdapter built
        with bwdif=None (for the grain target-quality search)."""
        cfg = _make_tool_paths(tmp_path)
        cfg.bestsource = tmp_path / "BestSource.dll"
        cfg.vship = tmp_path / "libvship.dll"

        mock_vship, _ = self._run_executor_fn(tmp_path, cfg)

        mock_vship.assert_called_once_with(cfg.bestsource, cfg.vship, None)

    def test_executor_fn_wires_bwdif_into_vship_metrics(self, tmp_path: Path) -> None:
        """bwdif configured alongside bestsource+vship -> threaded into the adapter."""
        cfg = _make_tool_paths(tmp_path)
        cfg.bestsource = tmp_path / "BestSource.dll"
        cfg.vship = tmp_path / "libvship.dll"
        cfg.bwdif = tmp_path / "Bwdif.dll"

        mock_vship, _ = self._run_executor_fn(tmp_path, cfg)

        mock_vship.assert_called_once_with(cfg.bestsource, cfg.vship, cfg.bwdif)

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
        dovi_args = mock_dovi.call_args.args
        assert dovi_args[0] == cfg.dovi_tool
        assert dovi_args[1] == cfg.ffmpeg
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

    def test_factory_forwards_relabel_options(self, tmp_path: Path) -> None:
        """allow_relabel and lang_list are forwarded into the TrackSelectorScreen."""
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
            allow_relabel=True,
            lang_list=["jpn", "rus"],
            app_runner=runner,
        )

        screen = captured[0]
        assert isinstance(screen, TrackSelectorScreen)
        assert screen._allow_relabel is True
        assert screen._lang_list == ["jpn", "rus"]


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
        lang_overrides: dict[Any, Any] = {}

        def fake_runner(_app: Any) -> TrackSelection:
            return TrackSelection(tracks=[track], downmix={downmix_key: DownmixMode.STEREO})

        selected = _select_tracks_tui_for_planner(
            movie,
            [track],
            TrackType.AUDIO,
            MagicMock(),
            downmix_overrides,
            lang_overrides,
            app_runner=fake_runner,
        )

        assert selected == [track]
        assert downmix_overrides == {downmix_key: DownmixMode.STEREO}
        assert lang_overrides == {}

    def test_subtitle_does_not_update_downmix_overrides(self, tmp_path: Path) -> None:
        """For subtitles, downmix overrides are left alone."""
        from furnace.cli import _select_tracks_tui_for_planner
        from furnace.ui.tui import TrackSelection

        movie = make_movie(main_file=tmp_path / "m.mkv")
        track = make_track(index=2, track_type=TrackType.SUBTITLE, codec_name="subrip")
        downmix_overrides: dict[Any, Any] = {}
        lang_overrides: dict[Any, Any] = {}

        def fake_runner(_app: Any) -> TrackSelection:
            return TrackSelection(tracks=[track], downmix={})

        selected = _select_tracks_tui_for_planner(
            movie,
            [track],
            TrackType.SUBTITLE,
            MagicMock(),
            downmix_overrides,
            lang_overrides,
            app_runner=fake_runner,
        )

        assert selected == [track]
        assert downmix_overrides == {}
        assert lang_overrides == {}

    def test_audio_merges_languages_and_downmix(self, tmp_path: Path) -> None:
        """For audio, both the relabel languages and downmix are merged into the shared dicts."""
        from furnace.cli import _select_tracks_tui_for_planner
        from furnace.core.models import DownmixMode
        from furnace.ui.tui import TrackSelection

        movie = make_movie(main_file=tmp_path / "m.mkv")
        track = make_track(index=1, track_type=TrackType.AUDIO, source_file=tmp_path / "a.mka")
        key = (track.source_file, track.index)
        downmix_overrides: dict[Any, Any] = {}
        lang_overrides: dict[Any, Any] = {}

        def fake_runner(_app: Any) -> TrackSelection:
            return TrackSelection(
                tracks=[track],
                downmix={key: DownmixMode.STEREO},
                languages={key: "rus"},
            )

        selected = _select_tracks_tui_for_planner(
            movie,
            [track],
            TrackType.AUDIO,
            MagicMock(),
            downmix_overrides,
            lang_overrides,
            app_runner=fake_runner,
        )

        assert selected == [track]
        assert downmix_overrides == {key: DownmixMode.STEREO}
        assert lang_overrides == {key: "rus"}

    def test_subtitle_merges_languages_only(self, tmp_path: Path) -> None:
        """For subtitles, languages merge into the shared dict but downmix is untouched."""
        from furnace.cli import _select_tracks_tui_for_planner
        from furnace.ui.tui import TrackSelection

        movie = make_movie(main_file=tmp_path / "m.mkv")
        track = make_track(index=2, track_type=TrackType.SUBTITLE, codec_name="subrip", source_file=tmp_path / "s.srt")
        key = (track.source_file, track.index)
        downmix_overrides: dict[Any, Any] = {}
        lang_overrides: dict[Any, Any] = {}

        def fake_runner(_app: Any) -> TrackSelection:
            return TrackSelection(tracks=[track], downmix={}, languages={key: "eng"})

        selected = _select_tracks_tui_for_planner(
            movie,
            [track],
            TrackType.SUBTITLE,
            MagicMock(),
            downmix_overrides,
            lang_overrides,
            app_runner=fake_runner,
        )

        assert selected == [track]
        assert lang_overrides == {key: "eng"}
        assert downmix_overrides == {}

    def test_forwards_allow_relabel_and_lang_list(self, tmp_path: Path) -> None:
        """allow_relabel and lang_list are forwarded down into _select_tracks_tui."""
        from furnace.cli import _select_tracks_tui_for_planner
        from furnace.ui.tui import TrackSelection

        movie = make_movie(main_file=tmp_path / "m.mkv")
        track = make_track(index=1, track_type=TrackType.AUDIO, source_file=tmp_path / "a.mka")

        with patch("furnace.cli._select_tracks_tui") as mock_sel:
            mock_sel.return_value = TrackSelection(tracks=[track], downmix={}, languages={})
            _select_tracks_tui_for_planner(
                movie,
                [track],
                TrackType.AUDIO,
                MagicMock(),
                {},
                {},
                allow_relabel=True,
                lang_list=["jpn", "rus"],
            )

        kwargs = mock_sel.call_args.kwargs
        assert kwargs["allow_relabel"] is True
        assert kwargs["lang_list"] == ["jpn", "rus"]


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

        demux_dir, paths, sar, _grain = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[],
            disc_titles={},
            disc_demuxer=demuxer,
            ffmpeg_adapter=ffmpeg,
            mpv_adapter=mpv,
            playlist_app_runner=MagicMock(),
            file_app_runner=MagicMock(),
        )

        assert demux_dir is None
        assert paths == []
        assert sar == set()
        # list_titles is no longer called inside _run_disc_demux_interactive —
        # it's invoked by `cli.plan`'s Detect loop and passed in as `disc_titles`.
        demuxer.list_titles.assert_not_called()

    def test_single_playlist_auto_selected_and_demuxed(self, tmp_path: Path) -> None:
        """One playlist -> no TUI, just demux (single HD file, no file-selector)."""
        from furnace.cli import _run_disc_demux_interactive
        from furnace.core.models import DiscSource, DiscTitle, DiscType

        disc_root = tmp_path / "disc_folder" / "BDMV"
        disc = DiscSource(path=disc_root, disc_type=DiscType.BLURAY)
        title = DiscTitle(number=1, duration_s=5400.0, raw_label="1: ...")

        demuxer = MagicMock()
        demuxed_mkv = tmp_path / "disc_folder_title_1.mkv"
        demuxer.demux.return_value = [demuxed_mkv]

        ffmpeg, mpv = self._adapters()
        # HD single file: probed for the SD grain gate, stays non-SD, no screen.
        ffmpeg.probe.return_value = {
            "format": {"duration": "5400.0", "size": "1000"},
            "streams": [{"codec_type": "video", "height": 1080}],
        }

        demux_dir, paths, sar, _grain = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_titles={disc: [title]},
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
        demuxer.list_titles.assert_not_called()

    def test_empty_playlist_list_skips_disc(self, tmp_path: Path) -> None:
        """When list_titles returned [] (now passed in via disc_titles), that
        disc is skipped entirely."""
        from furnace.cli import _run_disc_demux_interactive
        from furnace.core.models import DiscSource, DiscType

        disc = DiscSource(path=tmp_path / "BDMV", disc_type=DiscType.BLURAY)

        demuxer = MagicMock()

        ffmpeg, mpv = self._adapters()
        demux_dir, paths, sar, _grain = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_titles={disc: []},
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
        demuxed_mkv = tmp_path / "disc_title_2.mkv"
        demuxer.demux.return_value = [demuxed_mkv]

        ffmpeg, mpv = self._adapters()

        screens_built: list[Any] = []

        def playlist_runner(factory: Callable[[], Any]) -> list[DiscTitle]:
            screens_built.append(factory())
            return [t2]

        demux_dir, paths, sar, _grain = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_titles={disc: [t1, t2]},
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

        ffmpeg, mpv = self._adapters()

        demux_dir, paths, sar, _grain = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_titles={disc: [t1, t2]},
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
        dvd_mkv = tmp_path / ".furnace_demux" / "dvdroot_title_1.mkv"
        demuxer.demux.return_value = [dvd_mkv]

        ffmpeg = MagicMock()
        ffmpeg.probe.return_value = {"format": {"duration": "100.0", "size": "1000"}}
        mpv = MagicMock()

        screens_built: list[Any] = []

        def file_runner(factory: Callable[[], Any]) -> FileSelection:
            screens_built.append(factory())
            return FileSelection(selected=[dvd_mkv], sar_override={dvd_mkv})

        demux_dir, paths, sar, _grain = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_titles={disc: [title]},
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
        mkv1 = tmp_path / ".furnace_demux" / "bdroot_title_1.mkv"
        mkv2 = tmp_path / ".furnace_demux" / "bdroot_title_2.mkv"
        demuxer.demux.return_value = [mkv1, mkv2]

        ffmpeg = MagicMock()
        ffmpeg.probe.return_value = {"format": {"duration": "1", "size": "2"}}
        mpv = MagicMock()
        playlist_runner = MagicMock(return_value=[t1, t2])
        file_runner = MagicMock(return_value=FileSelection(selected=[mkv1], sar_override=set()))

        demux_dir, paths, sar, _grain = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_titles={disc: [t1, t2]},
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
        mkv1 = tmp_path / ".furnace_demux" / "bdroot_title_1.mkv"
        mkv2 = tmp_path / ".furnace_demux" / "bdroot_title_2.mkv"
        demuxer.demux.return_value = [mkv1, mkv2]

        ffmpeg = MagicMock()
        ffmpeg.probe.return_value = {"format": {"duration": "1", "size": "2"}}
        mpv = MagicMock()
        playlist_runner = MagicMock(return_value=[t1, t2])
        file_runner = MagicMock(return_value=None)

        _demux_dir, paths, sar, _grain = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_titles={disc: [t1, t2]},
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
        mkv = tmp_path / ".furnace_demux" / "dvd_title_1.mkv"
        demuxer.demux.return_value = [mkv]

        ffmpeg = MagicMock()
        ffmpeg.probe.return_value = {}  # no 'format' key
        mpv = MagicMock()
        file_runner = MagicMock(return_value=FileSelection(selected=[mkv], sar_override=set()))

        _, paths, _sar, _grain = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_titles={disc: [title]},
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

    def test_track_selector_forwards_lang_list_and_relabel_under_ignore_langs(self, tmp_path: Path) -> None:
        """Under -il, the track_selector closure picks audio/sub lang_list by type and forwards allow_relabel=True."""
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
            patch("furnace.cli._resolve_und_language_tui"),
        ):
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = []
            mock_planner_cls.return_value.create_plan.return_value = plan_obj
            mock_sel.return_value = []

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "rus,eng", "-sl", "jpn", "-il"],
            )
            assert result.exit_code == 0, result.output

            selector = mock_planner_cls.call_args.kwargs["track_selector"]
            movie = make_movie(main_file=source / "m.mkv")
            audio = make_track(index=1, track_type=TrackType.AUDIO)
            sub = make_track(index=2, track_type=TrackType.SUBTITLE, codec_name="subrip")

            selector(movie, [audio], TrackType.AUDIO)
            audio_call = mock_sel.call_args
            assert audio_call.kwargs["allow_relabel"] is True
            assert audio_call.kwargs["lang_list"] == ["rus", "eng"]

            selector(movie, [sub], TrackType.SUBTITLE)
            sub_call = mock_sel.call_args
            assert sub_call.kwargs["allow_relabel"] is True
            assert sub_call.kwargs["lang_list"] == ["jpn"]

    def test_track_selector_relabel_false_without_flag(self, tmp_path: Path) -> None:
        """Without -il, the track_selector closure forwards allow_relabel=False."""
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
            patch("furnace.cli._resolve_und_language_tui"),
        ):
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = []
            mock_planner_cls.return_value.create_plan.return_value = plan_obj
            mock_sel.return_value = []

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "rus", "-sl", "eng"],
            )
            assert result.exit_code == 0, result.output

            selector = mock_planner_cls.call_args.kwargs["track_selector"]
            movie = make_movie(main_file=source / "m.mkv")
            audio = make_track(index=1, track_type=TrackType.AUDIO)
            selector(movie, [audio], TrackType.AUDIO)

            assert mock_sel.call_args.kwargs["allow_relabel"] is False
            assert mock_sel.call_args.kwargs["lang_list"] == ["rus"]


class TestPlanIgnoreLangs:
    def _run_plan(self, tmp_path: Path, extra_args: list[str]) -> MagicMock:
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
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng", "--dry-run", *extra_args],
            )
            assert result.exit_code == 0, result.output
        return mock_planner_cls

    def test_ignore_langs_long_flag_sets_planner_and_lang_overrides(self, tmp_path: Path) -> None:
        """--ignore-langs sets PlannerService(ignore_langs=True) and passes lang_overrides to create_plan."""
        mock_planner_cls = self._run_plan(tmp_path, ["--ignore-langs"])
        assert mock_planner_cls.call_args.kwargs["ignore_langs"] is True
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args.kwargs
        assert call_kwargs["lang_overrides"] == {}

    def test_ignore_langs_short_flag(self, tmp_path: Path) -> None:
        """The -il short flag also enables ignore_langs."""
        mock_planner_cls = self._run_plan(tmp_path, ["-il"])
        assert mock_planner_cls.call_args.kwargs["ignore_langs"] is True

    def test_ignore_langs_defaults_false(self, tmp_path: Path) -> None:
        """Without the flag, ignore_langs is False and lang_overrides is an empty dict."""
        mock_planner_cls = self._run_plan(tmp_path, [])
        assert mock_planner_cls.call_args.kwargs["ignore_langs"] is False
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args.kwargs
        assert call_kwargs["lang_overrides"] == {}


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
            patch("furnace.cli.AnalysisPipeline") as mock_pipeline_cls,
            patch("furnace.cli.PlannerService") as mock_planner_cls,
            patch("furnace.cli.save_plan"),
        ):
            mock_demuxer_cls.return_value.detect.return_value = [disc]
            mock_demuxer_cls.return_value.list_titles.return_value = []
            mock_interactive.return_value = (source / ".furnace_demux", [demuxed], {demuxed}, {demuxed: True})
            mock_scanner_cls.return_value.scan.return_value = []
            mock_pipeline_cls.return_value.run.return_value.movies = []
            mock_pipeline_cls.return_value.run.return_value.crops = {}
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng"],
            )

        assert result.exit_code == 0, result.output
        mock_interactive.assert_called_once()
        # _run_disc_demux_interactive received the disc_titles dict from the
        # Detect loop.
        assert "disc_titles" in mock_interactive.call_args.kwargs
        # Plan.demux_dir should be set from the returned demux_dir
        assert plan_obj.demux_dir == str(source / ".furnace_demux")
        # sar_override_paths forwarded to planner
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args.kwargs
        assert call_kwargs["sar_overrides"] == {demuxed}
        # grain decisions from the disc file-selector reach the planner
        assert call_kwargs["grain_overrides"] == {demuxed: True}


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
            disc_titles={disc: [t1, t2]},
            disc_demuxer=demuxer,
            ffmpeg_adapter=ffmpeg,
            mpv_adapter=mpv,
            reporter=reporter,
            playlist_app_runner=playlist_runner,
            file_app_runner=file_runner,
        )

        # pause/resume called once around the playlist runner (now inside
        # _collect_selected_titles) and once around the file runner.
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
            mock_demuxer_cls.return_value.list_titles.return_value = []
            mock_interactive.return_value = (None, [], set(), {})
            mock_scanner_cls.return_value.scan.return_value = []
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng"],
            )

        assert result.exit_code == 0, result.output
        # detect_disc was called with the parent-name fallback.
        reporter_inst.detect_disc.assert_called_once_with(DiscType.BLURAY, "elsewhere")
        # detect_disc_titles_done was called too with the empty title count.
        reporter_inst.detect_disc_titles_done.assert_called_once_with(0)


# ---------------------------------------------------------------------------
# plan: HDR10+ source (FAILED batch outcome, no exception)
# ---------------------------------------------------------------------------


class TestPlanHdr10Plus:
    """An HDR10+ source yields a FAILED batch outcome: no job, no exception."""

    def test_hdr10_plus_surfaces_failed_line_and_no_job(self, tmp_path: Path) -> None:
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
            patch("furnace.cli.RichPlanReporter") as mock_reporter_cls,
        ):
            reporter_inst = mock_reporter_cls.return_value
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = [scan_result]
            # analyze() no longer raises for HDR10+; it returns a FAILED outcome.
            mock_analyzer_cls.return_value.analyze.return_value = AnalysisOutcome(
                None, AnalyzeStatus.FAILED, "HDR10+ not supported"
            )
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng", "--dry-run"],
            )

        # No exception escapes the command (the old try/except ValueError is gone).
        assert result.exit_code == 0, result.output
        # The FAILED outcome is surfaced as a batch line for that file.
        reporter_inst.analyze_batch_item.assert_called_once_with(
            "movie.mkv", "HDR10+ not supported", status=AnalyzeStatus.FAILED
        )
        # Planner sees no movies because the HDR10+ file produced no job.
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args.kwargs
        assert call_kwargs["movies"] == []


# ---------------------------------------------------------------------------
# plan: --jobs / parallel analysis worker count
# ---------------------------------------------------------------------------


class TestPlanJobs:
    """The --jobs flag controls how many AnalysisPipeline workers are used."""

    @staticmethod
    def _invoke_capturing_pipeline(
        tmp_path: Path,
        extra_args: list[str],
        *,
        cpu_count: int | None = 8,
    ) -> Any:
        """Run ``plan --dry-run`` with AnalysisPipeline patched; return its mock class."""
        from furnace.services.analysis_pipeline import AnalysisBatchResult

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
            patch("furnace.cli.AnalysisPipeline") as mock_pipeline_cls,
            patch("furnace.cli.PlannerService") as mock_planner_cls,
            patch("furnace.cli.os.cpu_count", return_value=cpu_count),
        ):
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = []
            mock_pipeline_cls.return_value.run.return_value = AnalysisBatchResult(movies=[], crops={})
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                [
                    "plan", str(source), "-o", str(output),
                    "-al", "eng", "-sl", "eng", "--dry-run", *extra_args,
                ],
            )

        assert result.exit_code == 0, result.output
        return mock_pipeline_cls

    def test_jobs_flag_forwards_max_workers(self, tmp_path: Path) -> None:
        """``--jobs 4`` reaches AnalysisPipeline as max_workers=4."""
        pipeline_cls = self._invoke_capturing_pipeline(tmp_path, ["--jobs", "4"])
        assert pipeline_cls.call_args.kwargs["max_workers"] == 4

    def test_jobs_short_flag_forwards_max_workers(self, tmp_path: Path) -> None:
        """``-j 3`` reaches AnalysisPipeline as max_workers=3."""
        pipeline_cls = self._invoke_capturing_pipeline(tmp_path, ["-j", "3"])
        assert pipeline_cls.call_args.kwargs["max_workers"] == 3

    def test_jobs_flag_floored_at_one(self, tmp_path: Path) -> None:
        """``--jobs 0`` is floored up to a single worker."""
        pipeline_cls = self._invoke_capturing_pipeline(tmp_path, ["--jobs", "0"])
        assert pipeline_cls.call_args.kwargs["max_workers"] == 1

    def test_default_workers_is_cpu_count_minus_two(self, tmp_path: Path) -> None:
        """Without --jobs, workers default to max(1, os.cpu_count() - 2)."""
        pipeline_cls = self._invoke_capturing_pipeline(tmp_path, [], cpu_count=8)
        assert pipeline_cls.call_args.kwargs["max_workers"] == 6

    def test_default_workers_floored_when_few_cpus(self, tmp_path: Path) -> None:
        """A one-core machine still yields at least one worker."""
        pipeline_cls = self._invoke_capturing_pipeline(tmp_path, [], cpu_count=1)
        assert pipeline_cls.call_args.kwargs["max_workers"] == 1

    def test_default_workers_when_cpu_count_none(self, tmp_path: Path) -> None:
        """os.cpu_count() returning None falls back to a single worker."""
        pipeline_cls = self._invoke_capturing_pipeline(tmp_path, [], cpu_count=None)
        assert pipeline_cls.call_args.kwargs["max_workers"] == 1

    def test_jobs_one_yields_same_plan_as_default(self, tmp_path: Path) -> None:
        """``--jobs 1`` is accepted and feeds the planner the same movies as the default."""
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
        movie = make_movie(main_file=source / "movie.mkv")
        outcome = AnalysisOutcome(movie, AnalyzeStatus.DONE, "summary")
        plan_obj = make_plan(jobs=[])

        def _run(extra_args: list[str]) -> Any:
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
                mock_analyzer_cls.return_value.analyze.return_value = outcome
                mock_planner_cls.return_value.create_plan.return_value = plan_obj

                result = runner.invoke(
                    app,
                    [
                        "plan", str(source), "-o", str(output),
                        "-al", "eng", "-sl", "eng", "--dry-run", *extra_args,
                    ],
                )
            assert result.exit_code == 0, result.output
            return mock_planner_cls.return_value.create_plan.call_args.kwargs["movies"]

        expected = [(movie, output / "movie" / "movie.mkv")]
        default_movies = _run([])
        jobs1_movies = _run(["--jobs", "1"])

        assert default_movies == expected
        assert jobs1_movies == expected


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


# ---------------------------------------------------------------------------
# scan command
# ---------------------------------------------------------------------------


class TestScanCommand:
    def test_no_filters_calls_service_with_defaults(self, tmp_path: Path) -> None:
        """`furnace scan SRC` runs the service with all filters off."""
        src = tmp_path / "movies"
        src.mkdir()
        cfg = _make_tool_paths(tmp_path)

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.ScanService") as mock_service_cls,
            patch("furnace.cli.render_scan_table"),
        ):
            mock_service_cls.return_value.scan.return_value = ([], 0)

            result = runner.invoke(app, ["scan", str(src)])

        assert result.exit_code == 0, result.output
        call = mock_service_cls.return_value.scan.call_args
        assert call.args[0] == src
        assert call.kwargs["not_encoded"] is False
        assert call.kwargs["encoded"] is False
        assert call.kwargs["max_version"] is None

    def test_not_encoded_flag_forwarded(self, tmp_path: Path) -> None:
        src = tmp_path / "movies"
        src.mkdir()
        cfg = _make_tool_paths(tmp_path)

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.ScanService") as mock_service_cls,
            patch("furnace.cli.render_scan_table"),
        ):
            mock_service_cls.return_value.scan.return_value = ([], 0)

            result = runner.invoke(app, ["scan", str(src), "--not-encoded"])

        assert result.exit_code == 0, result.output
        assert mock_service_cls.return_value.scan.call_args.kwargs["not_encoded"] is True

    def test_encoded_flag_forwarded(self, tmp_path: Path) -> None:
        src = tmp_path / "movies"
        src.mkdir()
        cfg = _make_tool_paths(tmp_path)

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.ScanService") as mock_service_cls,
            patch("furnace.cli.render_scan_table"),
        ):
            mock_service_cls.return_value.scan.return_value = ([], 0)

            result = runner.invoke(app, ["scan", str(src), "--encoded"])

        assert result.exit_code == 0, result.output
        assert mock_service_cls.return_value.scan.call_args.kwargs["encoded"] is True

    def test_max_version_parsed_to_tuple(self, tmp_path: Path) -> None:
        src = tmp_path / "movies"
        src.mkdir()
        cfg = _make_tool_paths(tmp_path)

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.ScanService") as mock_service_cls,
            patch("furnace.cli.render_scan_table"),
        ):
            mock_service_cls.return_value.scan.return_value = ([], 0)

            result = runner.invoke(app, ["scan", str(src), "--max-version", "1.19.3"])

        assert result.exit_code == 0, result.output
        assert mock_service_cls.return_value.scan.call_args.kwargs["max_version"] == (1, 19, 3)

    def test_union_of_flags_forwarded(self, tmp_path: Path) -> None:
        src = tmp_path / "movies"
        src.mkdir()
        cfg = _make_tool_paths(tmp_path)

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.ScanService") as mock_service_cls,
            patch("furnace.cli.render_scan_table"),
        ):
            mock_service_cls.return_value.scan.return_value = ([], 0)

            result = runner.invoke(
                app, ["scan", str(src), "--not-encoded", "--max-version", "1.19.3"]
            )

        assert result.exit_code == 0, result.output
        kwargs = mock_service_cls.return_value.scan.call_args.kwargs
        assert kwargs["not_encoded"] is True
        assert kwargs["max_version"] == (1, 19, 3)

    def test_bad_max_version_is_cli_error(self, tmp_path: Path) -> None:
        """A non-X.Y.Z --max-version yields a typer BadParameter (exit != 0)."""
        src = tmp_path / "movies"
        src.mkdir()
        cfg = _make_tool_paths(tmp_path)

        with (
            patch("furnace.cli.load_config", return_value=cfg) as mock_load_cfg,
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.ScanService") as mock_service_cls,
            patch("furnace.cli.render_scan_table"),
        ):
            result = runner.invoke(app, ["scan", str(src), "--max-version", "1.2"])

        assert result.exit_code != 0
        assert "max-version" in result.output
        # The error is raised before any work happens.
        mock_load_cfg.assert_not_called()
        mock_service_cls.return_value.scan.assert_not_called()

    def test_config_forwarded_to_load_config(self, tmp_path: Path) -> None:
        src = tmp_path / "movies"
        src.mkdir()
        config_file = tmp_path / "my.toml"
        cfg = _make_tool_paths(tmp_path)

        with (
            patch("furnace.cli.load_config", return_value=cfg) as mock_load_cfg,
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.ScanService") as mock_service_cls,
            patch("furnace.cli.render_scan_table"),
        ):
            mock_service_cls.return_value.scan.return_value = ([], 0)

            result = runner.invoke(app, ["scan", str(src), "--config", str(config_file)])

        assert result.exit_code == 0, result.output
        mock_load_cfg.assert_called_once_with(config_file)

    def test_ffmpeg_adapter_built_from_config(self, tmp_path: Path) -> None:
        src = tmp_path / "movies"
        src.mkdir()
        cfg = _make_tool_paths(tmp_path)

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.FFmpegAdapter") as mock_ffmpeg_cls,
            patch("furnace.cli.ScanService") as mock_service_cls,
            patch("furnace.cli.render_scan_table"),
        ):
            mock_service_cls.return_value.scan.return_value = ([], 0)

            result = runner.invoke(app, ["scan", str(src)])

        assert result.exit_code == 0, result.output
        ffmpeg_args = mock_ffmpeg_cls.call_args.args
        assert ffmpeg_args[0] == cfg.ffmpeg
        assert ffmpeg_args[1] == cfg.ffprobe
        # The service is built with the ffprobe-backed adapter.
        assert mock_service_cls.call_args.kwargs["prober"] is mock_ffmpeg_cls.return_value

    def test_total_taken_from_service(self, tmp_path: Path) -> None:
        """The summary total (M) is the count the service reports alongside rows."""
        src = tmp_path / "movies"
        src.mkdir()
        cfg = _make_tool_paths(tmp_path)

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.ScanService") as mock_service_cls,
            patch("furnace.cli.render_scan_table") as mock_render,
        ):
            mock_service_cls.return_value.scan.return_value = ([], 3)

            result = runner.invoke(app, ["scan", str(src)])

        assert result.exit_code == 0, result.output
        render_kwargs = mock_render.call_args.kwargs
        assert render_kwargs["root"] == src
        assert render_kwargs["total"] == 3

    def test_unreadable_rows_become_warnings(self, tmp_path: Path) -> None:
        """Each unreadable row is surfaced as a stderr warning; readable rows are not."""
        from furnace.core.scan import ScanRow, VideoSummary

        src = tmp_path / "movies"
        src.mkdir()
        cfg = _make_tool_paths(tmp_path)
        good = ScanRow(
            path=src / "good.mkv",
            furnace_version=(1, 0, 0),
            video=VideoSummary(codec="hevc", bit_depth=10, hdr="SDR"),
            audio=(),
            subtitles=(),
        )
        bad = ScanRow(
            path=src / "bad.mkv",
            furnace_version=None,
            video=VideoSummary(None, None, None),
            audio=(),
            subtitles=(),
            unreadable=True,
        )

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.ScanService") as mock_service_cls,
            patch("furnace.cli.render_scan_table") as mock_render,
        ):
            mock_service_cls.return_value.scan.return_value = ([good, bad], 2)

            result = runner.invoke(app, ["scan", str(src)])

        assert result.exit_code == 0, result.output
        warnings = mock_render.call_args.kwargs["warnings"]
        assert any("bad.mkv" in w for w in warnings)
        assert all("good.mkv" not in w for w in warnings)

    def test_rows_forwarded_to_renderer(self, tmp_path: Path) -> None:
        from furnace.core.scan import ScanRow, VideoSummary

        src = tmp_path / "movies"
        src.mkdir()
        cfg = _make_tool_paths(tmp_path)
        row = ScanRow(
            path=src / "a.mkv",
            furnace_version=(1, 0, 0),
            video=VideoSummary(codec="hevc", bit_depth=10, hdr="SDR"),
            audio=(),
            subtitles=(),
        )

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.ScanService") as mock_service_cls,
            patch("furnace.cli.render_scan_table") as mock_render,
        ):
            mock_service_cls.return_value.scan.return_value = ([row], 1)

            result = runner.invoke(app, ["scan", str(src)])

        assert result.exit_code == 0, result.output
        assert mock_render.call_args.args[0] == [row]

    def test_integration_real_service_and_renderer(self, tmp_path: Path) -> None:
        """End-to-end wiring: real ScanService + real renderer, only the prober stubbed.

        Catches drift between the CLI, the service, the ``ScanRow`` model and the
        renderer that the fully-mocked tests above cannot see.
        """
        src = tmp_path / "movies"
        src.mkdir()
        encoded = src / "encoded.mkv"
        plain = src / "plain.mkv"
        encoded.touch()
        plain.touch()
        cfg = _make_tool_paths(tmp_path)

        probe_map: dict[Path, dict[str, Any]] = {
            encoded: {
                "streams": [
                    {"codec_type": "video", "codec_name": "hevc"},
                    {
                        "codec_type": "audio",
                        "codec_name": "eac3",
                        "channels": 6,
                        "tags": {"language": "rus"},
                    },
                ],
                "format": {"tags": {"ENCODER": "Furnace v1.19.3"}},
            },
            plain: {
                "streams": [{"codec_type": "video", "codec_name": "h264"}],
                "format": {"tags": {"ENCODER": "Lavf60"}},
            },
        }
        fake_prober = MagicMock()
        fake_prober.probe.side_effect = lambda p: probe_map[p]

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.FFmpegAdapter", return_value=fake_prober),
        ):
            result = runner.invoke(app, ["scan", str(src)])

        assert result.exit_code == 0, result.output
        out = result.output
        assert "encoded.mkv" in out
        assert "Furnace v1.19.3" in out
        assert "rus eac3 6ch" in out
        assert "plain.mkv" in out
        assert "h264" in out
        assert "not encoded" in out
        assert "2 of 2 shown" in out

    def test_integration_filter_reduces_rows_below_total(self, tmp_path: Path) -> None:
        """A filter trims rendered rows (N) below the discovered total (M), end-to-end."""
        src = tmp_path / "movies"
        src.mkdir()
        encoded = src / "encoded.mkv"
        plain = src / "plain.mkv"
        encoded.touch()
        plain.touch()
        cfg = _make_tool_paths(tmp_path)

        probe_map: dict[Path, dict[str, Any]] = {
            encoded: {
                "streams": [{"codec_type": "video", "codec_name": "hevc"}],
                "format": {"tags": {"ENCODER": "Furnace v1.19.3"}},
            },
            plain: {
                "streams": [{"codec_type": "video", "codec_name": "h264"}],
                "format": {"tags": {"ENCODER": "Lavf60"}},
            },
        }
        fake_prober = MagicMock()
        fake_prober.probe.side_effect = lambda p: probe_map[p]

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.FFmpegAdapter", return_value=fake_prober),
        ):
            result = runner.invoke(app, ["scan", str(src), "--not-encoded"])

        assert result.exit_code == 0, result.output
        out = result.output
        assert "plain.mkv" in out
        assert "encoded.mkv" not in out
        assert "1 of 2 shown" in out

    def test_outdated_flag_forwarded(self, tmp_path: Path) -> None:
        src = tmp_path / "movies"
        src.mkdir()
        cfg = _make_tool_paths(tmp_path)

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.ScanService") as mock_service_cls,
            patch("furnace.cli.render_scan_table") as mock_render,
        ):
            mock_service_cls.return_value.scan.return_value = ([], 0)

            result = runner.invoke(app, ["scan", str(src), "--outdated"])

        assert result.exit_code == 0, result.output
        assert mock_service_cls.return_value.scan.call_args.kwargs["outdated"] is True
        assert mock_render.call_args.kwargs["outdated"] is True

    def test_outdated_defaults_false(self, tmp_path: Path) -> None:
        src = tmp_path / "movies"
        src.mkdir()
        cfg = _make_tool_paths(tmp_path)

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.ScanService") as mock_service_cls,
            patch("furnace.cli.render_scan_table") as mock_render,
        ):
            mock_service_cls.return_value.scan.return_value = ([], 0)

            result = runner.invoke(app, ["scan", str(src)])

        assert result.exit_code == 0, result.output
        assert mock_service_cls.return_value.scan.call_args.kwargs["outdated"] is False
        assert mock_render.call_args.kwargs["outdated"] is False

    @pytest.mark.parametrize(
        "clashing",
        [["--not-encoded"], ["--encoded"], ["--max-version", "2.0.0"]],
    )
    def test_outdated_is_standalone(self, tmp_path: Path, clashing: list[str]) -> None:
        """--outdated combined with any status filter is a usage error, raised early."""
        src = tmp_path / "movies"
        src.mkdir()

        with (
            patch("furnace.cli.load_config") as mock_load_cfg,
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.ScanService") as mock_service_cls,
            patch("furnace.cli.render_scan_table"),
        ):
            result = runner.invoke(app, ["scan", str(src), "--outdated", *clashing])

        assert result.exit_code != 0
        assert "outdated" in result.output
        mock_load_cfg.assert_not_called()
        mock_service_cls.return_value.scan.assert_not_called()

    def test_integration_outdated_flags_and_drops(self, tmp_path: Path) -> None:
        """End-to-end --outdated: foreign + defective kept, clean current dropped."""
        src = tmp_path / "movies"
        src.mkdir()
        clean = src / "a_clean.mkv"
        foreign = src / "b_foreign.mkv"
        old = src / "c_old.mkv"
        for p in (clean, foreign, old):
            p.touch()
        cfg = _make_tool_paths(tmp_path)

        probe_map: dict[Path, dict[str, Any]] = {
            clean: {
                "streams": [
                    {"codec_type": "video", "codec_name": "av1", "height": 1080, "color_space": "bt709"},
                    {"codec_type": "audio", "codec_name": "aac", "channels": 2, "tags": {"language": "eng"}},
                ],
                "format": {"tags": {"ENCODER": "Furnace v2.9.0", "ENCODER_SETTINGS": "av1_nvenc / main"}},
            },
            foreign: {
                "streams": [{"codec_type": "video", "codec_name": "h264", "height": 1080}],
                "format": {"tags": {"ENCODER": "Lavf60"}},
            },
            old: {
                "streams": [{"codec_type": "video", "codec_name": "av1", "height": 1080, "color_space": "bt709"}],
                "format": {"tags": {"ENCODER": "Furnace v2.1.0", "ENCODER_SETTINGS": "av1_nvenc / main"}},
            },
        }
        fake_prober = MagicMock()
        fake_prober.probe.side_effect = lambda p: probe_map[p]

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.FFmpegAdapter", return_value=fake_prober),
        ):
            result = runner.invoke(app, ["scan", str(src), "--outdated"])

        assert result.exit_code == 0, result.output
        out = result.output
        assert "Severity" in out
        assert "b_foreign.mkv" in out
        assert "FOREIGN" in out
        assert "c_old.mkv" in out
        assert "a_clean.mkv" not in out
        assert "2 of 3 shown" in result.output

    def test_integration_outdated_keeps_unreadable_and_exits_zero(self, tmp_path: Path) -> None:
        """--outdated still exits 0 and surfaces an unreadable file as its own row."""
        src = tmp_path / "movies"
        src.mkdir()
        good = src / "a_good.mkv"
        bad = src / "b_bad.mkv"
        good.touch()
        bad.touch()
        cfg = _make_tool_paths(tmp_path)

        def probe(p: Path) -> dict[str, Any]:
            if p == bad:
                raise OSError("boom")
            # Clean current file → dropped from the outdated work-list.
            return {
                "streams": [
                    {"codec_type": "video", "codec_name": "av1", "height": 1080, "color_space": "bt709"},
                    {"codec_type": "audio", "codec_name": "aac", "channels": 2, "tags": {"language": "eng"}},
                ],
                "format": {"tags": {"ENCODER": "Furnace v2.9.0", "ENCODER_SETTINGS": "av1_nvenc / main"}},
            }

        fake_prober = MagicMock()
        fake_prober.probe.side_effect = probe

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli.FFmpegAdapter", return_value=fake_prober),
        ):
            result = runner.invoke(app, ["scan", str(src), "--outdated"])

        assert result.exit_code == 0, result.output
        out = result.output
        assert "b_bad.mkv" in out
        assert "UNREADABLE" in out
        assert "a_good.mkv" not in out
        assert "1 of 2 shown" in out


# ---------------------------------------------------------------------------
# _probe_file_infos — height (SD grain gate) is captured
# ---------------------------------------------------------------------------


class TestProbeFileInfosHeight:
    def test_includes_first_video_stream_height(self, tmp_path: Path) -> None:
        """_probe_file_infos returns (path, duration, size, height) 4-tuples."""
        from furnace.cli import _probe_file_infos

        p = tmp_path / "a.mkv"
        ffmpeg = MagicMock()
        ffmpeg.probe.return_value = {
            "format": {"duration": "10.0", "size": "20"},
            "streams": [
                {"codec_type": "audio"},
                {"codec_type": "video", "height": 576},
            ],
        }

        infos = _probe_file_infos([p], ffmpeg)

        assert infos == [(p, 10.0, 20, 576)]

    def test_no_video_stream_height_zero(self, tmp_path: Path) -> None:
        """A file with no video stream falls back to height 0 (treated non-SD)."""
        from furnace.cli import _probe_file_infos

        p = tmp_path / "a.mkv"
        ffmpeg = MagicMock()
        ffmpeg.probe.return_value = {"format": {"duration": "0", "size": "0"}}

        infos = _probe_file_infos([p], ffmpeg)

        assert infos == [(p, 0.0, 0, 0)]


# ---------------------------------------------------------------------------
# _run_disc_demux_interactive — SD grain pre-probe + threading
# ---------------------------------------------------------------------------


class TestDiscDemuxGrain:
    def _sd_probe(self, height: int = 480) -> dict[str, Any]:
        return {
            "format": {"duration": "100.0", "size": "1000"},
            "streams": [{"codec_type": "video", "height": height}],
        }

    def test_grainy_sd_file_pre_lit_and_threaded(self, tmp_path: Path) -> None:
        """A DVD SD file whose sample_grain classifies GRAINY starts pre-lit and
        its grain decision is threaded out of _run_disc_demux_interactive."""
        from furnace.cli import _run_disc_demux_interactive
        from furnace.core.models import DiscSource, DiscTitle, DiscType
        from furnace.ui.tui import FileSelection, FileSelectorScreen

        disc_root = tmp_path / "dvdroot" / "VIDEO_TS"
        disc = DiscSource(path=disc_root, disc_type=DiscType.DVD)
        title = DiscTitle(number=1, duration_s=100, raw_label="t")

        demuxer = MagicMock()
        dvd_mkv = tmp_path / ".furnace_demux" / "dvdroot_title_1.mkv"
        demuxer.demux.return_value = [dvd_mkv]

        ffmpeg = MagicMock()
        ffmpeg.probe.return_value = self._sd_probe(480)
        ffmpeg.sample_grain.return_value = [1.0, 1.2]  # median >= 0.5 -> GRAINY

        screens_built: list[Any] = []

        def file_runner(factory: Callable[[], Any]) -> FileSelection:
            screens_built.append(factory())
            return FileSelection(selected=[dvd_mkv], sar_override=set(), grain={dvd_mkv: True})

        _dir, paths, _sar, grain = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_titles={disc: [title]},
            disc_demuxer=demuxer,
            ffmpeg_adapter=ffmpeg,
            mpv_adapter=MagicMock(),
            playlist_app_runner=MagicMock(),
            file_app_runner=file_runner,
        )

        assert paths == [dvd_mkv]
        assert grain == {dvd_mkv: True}
        # Grain was pre-probed once, on the SD file, with its duration.
        ffmpeg.sample_grain.assert_called_once_with(dvd_mkv, 100.0)
        # The screen was seeded with sd_files and a pre-lit grain default.
        screen = screens_built[0]
        assert isinstance(screen, FileSelectorScreen)
        assert screen._sd_files == {dvd_mkv}
        assert screen._grain_defaults == {dvd_mkv}

    def test_clean_sd_file_not_pre_lit(self, tmp_path: Path) -> None:
        """A CLEAN sample_grain verdict leaves the SD file's grain default off."""
        from furnace.cli import _run_disc_demux_interactive
        from furnace.core.models import DiscSource, DiscTitle, DiscType

        disc_root = tmp_path / "dvdroot" / "VIDEO_TS"
        disc = DiscSource(path=disc_root, disc_type=DiscType.DVD)
        title = DiscTitle(number=1, duration_s=100, raw_label="t")

        demuxer = MagicMock()
        dvd_mkv = tmp_path / ".furnace_demux" / "dvdroot_title_1.mkv"
        demuxer.demux.return_value = [dvd_mkv]

        ffmpeg = MagicMock()
        ffmpeg.probe.return_value = self._sd_probe(480)
        ffmpeg.sample_grain.return_value = [0.1, 0.2]  # median < 0.5 -> CLEAN

        screens_built: list[Any] = []

        def file_runner(factory: Callable[[], Any]) -> Any:
            screens_built.append(factory())
            from furnace.ui.tui import FileSelection

            return FileSelection(selected=[dvd_mkv], sar_override=set(), grain={dvd_mkv: False})

        _dir, _paths, _sar, grain = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_titles={disc: [title]},
            disc_demuxer=demuxer,
            ffmpeg_adapter=ffmpeg,
            mpv_adapter=MagicMock(),
            playlist_app_runner=MagicMock(),
            file_app_runner=file_runner,
        )

        assert grain == {dvd_mkv: False}
        assert screens_built[0]._sd_files == {dvd_mkv}
        assert screens_built[0]._grain_defaults == set()

    def test_single_sd_file_triggers_screen(self, tmp_path: Path) -> None:
        """A single non-DVD SD file (len==1, no DVD) still opens the file-selector
        because it is grain-eligible."""
        from furnace.cli import _run_disc_demux_interactive
        from furnace.core.models import DiscSource, DiscTitle, DiscType
        from furnace.ui.tui import FileSelection

        disc_root = tmp_path / "bdroot" / "BDMV"
        disc = DiscSource(path=disc_root, disc_type=DiscType.BLURAY)
        title = DiscTitle(number=1, duration_s=100, raw_label="t")

        demuxer = MagicMock()
        mkv = tmp_path / ".furnace_demux" / "bdroot_title_1.mkv"
        demuxer.demux.return_value = [mkv]

        ffmpeg = MagicMock()
        ffmpeg.probe.return_value = self._sd_probe(480)
        ffmpeg.sample_grain.return_value = [1.0]

        file_runner = MagicMock(
            return_value=FileSelection(selected=[mkv], sar_override=set(), grain={mkv: True})
        )

        _dir, paths, _sar, grain = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_titles={disc: [title]},
            disc_demuxer=demuxer,
            ffmpeg_adapter=ffmpeg,
            mpv_adapter=MagicMock(),
            playlist_app_runner=MagicMock(),
            file_app_runner=file_runner,
        )

        file_runner.assert_called_once()
        assert paths == [mkv]
        assert grain == {mkv: True}

    def test_grain_defaults_never_include_non_sd(self, tmp_path: Path) -> None:
        """INVARIANT: even a GRAINY HD file is never pre-lit — grain_defaults ⊆ sd_files."""
        from furnace.cli import _run_disc_demux_interactive
        from furnace.core.models import DiscSource, DiscTitle, DiscType
        from furnace.ui.tui import FileSelection

        disc_root = tmp_path / "bdroot" / "BDMV"
        disc = DiscSource(path=disc_root, disc_type=DiscType.BLURAY)
        t1 = DiscTitle(number=1, duration_s=100, raw_label="1")
        t2 = DiscTitle(number=2, duration_s=200, raw_label="2")

        demuxer = MagicMock()
        hd_mkv = tmp_path / ".furnace_demux" / "bdroot_title_1.mkv"
        sd_mkv = tmp_path / ".furnace_demux" / "bdroot_title_2.mkv"
        demuxer.demux.return_value = [hd_mkv, sd_mkv]

        ffmpeg = MagicMock()

        def _probe(path: Path) -> dict[str, Any]:
            height = 1080 if path == hd_mkv else 480
            return {
                "format": {"duration": "100.0", "size": "1000"},
                "streams": [{"codec_type": "video", "height": height}],
            }

        ffmpeg.probe.side_effect = _probe
        ffmpeg.sample_grain.return_value = [9.0]  # everything reads GRAINY

        screens_built: list[Any] = []

        def file_runner(factory: Callable[[], Any]) -> FileSelection:
            screens_built.append(factory())
            return FileSelection(selected=[hd_mkv, sd_mkv], sar_override=set(), grain={sd_mkv: True})

        _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_titles={disc: [t1, t2]},
            disc_demuxer=demuxer,
            ffmpeg_adapter=ffmpeg,
            mpv_adapter=MagicMock(),
            playlist_app_runner=MagicMock(return_value=[t1, t2]),
            file_app_runner=file_runner,
        )

        screen = screens_built[0]
        assert screen._sd_files == {sd_mkv}
        assert screen._grain_defaults == {sd_mkv}
        # sample_grain only ran for the SD file, never for the HD one.
        ffmpeg.sample_grain.assert_called_once_with(sd_mkv, 100.0)

    def test_grain_probe_uses_stream_duration(self, tmp_path: Path) -> None:
        """Finding 1: the pre-probe seeks with the video STREAM's duration (not
        format.duration) so it matches the analyzer's stream-first precedence."""
        from furnace.cli import _run_disc_demux_interactive
        from furnace.core.models import DiscSource, DiscTitle, DiscType
        from furnace.ui.tui import FileSelection

        disc_root = tmp_path / "dvdroot" / "VIDEO_TS"
        disc = DiscSource(path=disc_root, disc_type=DiscType.DVD)
        title = DiscTitle(number=1, duration_s=100, raw_label="t")

        demuxer = MagicMock()
        dvd_mkv = tmp_path / ".furnace_demux" / "dvdroot_title_1.mkv"
        demuxer.demux.return_value = [dvd_mkv]

        ffmpeg = MagicMock()
        # Stream duration (55.0) differs from format.duration (100.0).
        ffmpeg.probe.return_value = {
            "format": {"duration": "100.0", "size": "1000"},
            "streams": [{"codec_type": "video", "height": 480, "duration": "55.0"}],
        }
        ffmpeg.sample_grain.return_value = [1.0]

        file_runner = MagicMock(
            return_value=FileSelection(selected=[dvd_mkv], sar_override=set(), grain={dvd_mkv: True})
        )

        _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_titles={disc: [title]},
            disc_demuxer=demuxer,
            ffmpeg_adapter=ffmpeg,
            mpv_adapter=MagicMock(),
            playlist_app_runner=MagicMock(),
            file_app_runner=file_runner,
        )

        # Seeks with the stream duration, not format's 100.0.
        ffmpeg.sample_grain.assert_called_once_with(dvd_mkv, 55.0)

    def test_grain_probe_falls_back_to_format_duration(self, tmp_path: Path) -> None:
        """Finding 1 fallback: with no video-stream duration, the pre-probe uses
        format.duration (matching the analyzer's fallback)."""
        from furnace.cli import _run_disc_demux_interactive
        from furnace.core.models import DiscSource, DiscTitle, DiscType
        from furnace.ui.tui import FileSelection

        disc_root = tmp_path / "dvdroot" / "VIDEO_TS"
        disc = DiscSource(path=disc_root, disc_type=DiscType.DVD)
        title = DiscTitle(number=1, duration_s=100, raw_label="t")

        demuxer = MagicMock()
        dvd_mkv = tmp_path / ".furnace_demux" / "dvdroot_title_1.mkv"
        demuxer.demux.return_value = [dvd_mkv]

        ffmpeg = MagicMock()
        # No stream duration -> fall back to format.duration (77.0).
        ffmpeg.probe.return_value = {
            "format": {"duration": "77.0", "size": "1000"},
            "streams": [{"codec_type": "video", "height": 480}],
        }
        ffmpeg.sample_grain.return_value = [1.0]

        file_runner = MagicMock(
            return_value=FileSelection(selected=[dvd_mkv], sar_override=set(), grain={dvd_mkv: True})
        )

        _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_titles={disc: [title]},
            disc_demuxer=demuxer,
            ffmpeg_adapter=ffmpeg,
            mpv_adapter=MagicMock(),
            playlist_app_runner=MagicMock(),
            file_app_runner=file_runner,
        )

        ffmpeg.sample_grain.assert_called_once_with(dvd_mkv, 77.0)

    def test_pre_probe_raise_defaults_grainy_and_survives(self, tmp_path: Path) -> None:
        """Finding 2: a hard sample_grain failure (broken ffmpeg) on one SD file is
        caught and defaulted to GRAINY, without crashing the run or affecting a
        sibling SD file whose probe succeeds CLEAN."""
        from furnace.cli import _run_disc_demux_interactive
        from furnace.core.models import DiscSource, DiscTitle, DiscType
        from furnace.ui.tui import FileSelection, FileSelectorScreen

        disc_root = tmp_path / "bdroot" / "BDMV"
        disc = DiscSource(path=disc_root, disc_type=DiscType.BLURAY)
        t1 = DiscTitle(number=1, duration_s=100, raw_label="1")
        t2 = DiscTitle(number=2, duration_s=100, raw_label="2")

        demuxer = MagicMock()
        raising_mkv = tmp_path / ".furnace_demux" / "bdroot_title_1.mkv"
        clean_mkv = tmp_path / ".furnace_demux" / "bdroot_title_2.mkv"
        demuxer.demux.return_value = [raising_mkv, clean_mkv]

        ffmpeg = MagicMock()
        ffmpeg.probe.return_value = self._sd_probe(480)

        def _grain(path: Path, dur: float) -> list[float]:
            if path == raising_mkv:
                raise OSError("ffmpeg exploded")
            return [0.1]  # CLEAN

        ffmpeg.sample_grain.side_effect = _grain

        screens_built: list[Any] = []

        def file_runner(factory: Callable[[], Any]) -> FileSelection:
            screens_built.append(factory())
            return FileSelection(
                selected=[raising_mkv, clean_mkv], sar_override=set(), grain={}
            )

        _dir, paths, _sar, _grain_out = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_titles={disc: [t1, t2]},
            disc_demuxer=demuxer,
            ffmpeg_adapter=ffmpeg,
            mpv_adapter=MagicMock(),
            playlist_app_runner=MagicMock(return_value=[t1, t2]),
            file_app_runner=file_runner,
        )

        # The run survived (no propagated OSError) and reached the selector.
        assert paths == [raising_mkv, clean_mkv]
        screen = screens_built[0]
        assert isinstance(screen, FileSelectorScreen)
        # The raising file defaulted GRAINY; the CLEAN sibling is unaffected.
        assert screen._grain_defaults == {raising_mkv}

    def test_pre_probe_runtimeerror_defaults_grainy(self, tmp_path: Path) -> None:
        """Finding 2 (RuntimeError variant): a RuntimeError from sample_grain is
        caught the same way and defaults the SD file to GRAINY."""
        from furnace.cli import _run_disc_demux_interactive
        from furnace.core.models import DiscSource, DiscTitle, DiscType
        from furnace.ui.tui import FileSelection

        disc_root = tmp_path / "dvdroot" / "VIDEO_TS"
        disc = DiscSource(path=disc_root, disc_type=DiscType.DVD)
        title = DiscTitle(number=1, duration_s=100, raw_label="t")

        demuxer = MagicMock()
        dvd_mkv = tmp_path / ".furnace_demux" / "dvdroot_title_1.mkv"
        demuxer.demux.return_value = [dvd_mkv]

        ffmpeg = MagicMock()
        ffmpeg.probe.return_value = self._sd_probe(480)
        ffmpeg.sample_grain.side_effect = RuntimeError("probe blew up")

        screens_built: list[Any] = []

        def file_runner(factory: Callable[[], Any]) -> FileSelection:
            screens_built.append(factory())
            return FileSelection(selected=[dvd_mkv], sar_override=set(), grain={dvd_mkv: True})

        _dir, paths, _sar, _grain_out = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_titles={disc: [title]},
            disc_demuxer=demuxer,
            ffmpeg_adapter=ffmpeg,
            mpv_adapter=MagicMock(),
            playlist_app_runner=MagicMock(),
            file_app_runner=file_runner,
        )

        assert paths == [dvd_mkv]
        assert screens_built[0]._grain_defaults == {dvd_mkv}


# ---------------------------------------------------------------------------
# plan — plain-files (no discs) SD grain flow
# ---------------------------------------------------------------------------


class TestPlanPlainFilesGrain:
    def _sd_streams(self, height: int) -> dict[str, Any]:
        return {
            "format": {"duration": "100.0", "size": "1000"},
            "streams": [{"codec_type": "video", "height": height}],
        }

    def test_sd_source_shows_screen_and_threads_grain(self, tmp_path: Path) -> None:
        """A plain SD source (no discs) opens the file-selector and its grain
        decision reaches create_plan as grain_overrides."""
        from furnace.core.models import ScanResult
        from furnace.services.analysis_pipeline import AnalysisBatchResult
        from furnace.ui.tui import FileSelection

        source = tmp_path / "src"
        source.mkdir()
        output = tmp_path / "out"
        main_file = source / "movie.mkv"

        cfg = _make_tool_paths(tmp_path)
        scan_result = ScanResult(
            main_file=main_file,
            satellite_files=[],
            output_path=output / "movie" / "movie.mkv",
        )
        plan_obj = make_plan(jobs=[])
        selection = FileSelection(selected=[main_file], sar_override=set(), grain={main_file: True})

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.FFmpegAdapter") as mock_ffmpeg_cls,
            patch("furnace.cli.MpvAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.MakemkvAdapter"),
            patch("furnace.cli.DiscDemuxer") as mock_demuxer_cls,
            patch("furnace.cli.Scanner") as mock_scanner_cls,
            patch("furnace.cli.AnalysisPipeline") as mock_pipeline_cls,
            patch("furnace.cli.PlannerService") as mock_planner_cls,
            patch("furnace.cli.save_plan"),
            patch("furnace.cli._run_screen_app", return_value=selection) as mock_runner,
        ):
            mock_ffmpeg_cls.return_value.probe.return_value = self._sd_streams(480)
            mock_ffmpeg_cls.return_value.sample_grain.return_value = [1.0]  # GRAINY
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = [scan_result]
            mock_pipeline_cls.return_value.run.return_value = AnalysisBatchResult(movies=[], crops={})
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng"],
            )

        assert result.exit_code == 0, result.output
        mock_runner.assert_called_once()
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args.kwargs
        assert call_kwargs["grain_overrides"] == {main_file: True}

    def test_hd_only_source_no_screen_and_grain_none(self, tmp_path: Path) -> None:
        """A plain HD-only source shows no screen and passes grain_overrides=None."""
        from furnace.core.models import ScanResult
        from furnace.services.analysis_pipeline import AnalysisBatchResult

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
            patch("furnace.cli.FFmpegAdapter") as mock_ffmpeg_cls,
            patch("furnace.cli.MpvAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.MakemkvAdapter"),
            patch("furnace.cli.DiscDemuxer") as mock_demuxer_cls,
            patch("furnace.cli.Scanner") as mock_scanner_cls,
            patch("furnace.cli.AnalysisPipeline") as mock_pipeline_cls,
            patch("furnace.cli.PlannerService") as mock_planner_cls,
            patch("furnace.cli.save_plan"),
            patch("furnace.cli._run_screen_app") as mock_runner,
        ):
            mock_ffmpeg_cls.return_value.probe.return_value = self._sd_streams(1080)
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = [scan_result]
            mock_pipeline_cls.return_value.run.return_value = AnalysisBatchResult(movies=[], crops={})
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng"],
            )

        assert result.exit_code == 0, result.output
        mock_runner.assert_not_called()
        mock_ffmpeg_cls.return_value.sample_grain.assert_not_called()
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args.kwargs
        assert call_kwargs["grain_overrides"] is None

    def test_sd_source_screen_dismissed_keeps_grain_none(self, tmp_path: Path) -> None:
        """If the plain-files grain screen is dismissed (None), no overrides are
        threaded and grain_overrides stays None."""
        from furnace.core.models import ScanResult
        from furnace.services.analysis_pipeline import AnalysisBatchResult

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
            patch("furnace.cli.FFmpegAdapter") as mock_ffmpeg_cls,
            patch("furnace.cli.MpvAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.MakemkvAdapter"),
            patch("furnace.cli.DiscDemuxer") as mock_demuxer_cls,
            patch("furnace.cli.Scanner") as mock_scanner_cls,
            patch("furnace.cli.AnalysisPipeline") as mock_pipeline_cls,
            patch("furnace.cli.PlannerService") as mock_planner_cls,
            patch("furnace.cli.save_plan"),
            patch("furnace.cli._run_screen_app", return_value=None) as mock_runner,
        ):
            mock_ffmpeg_cls.return_value.probe.return_value = self._sd_streams(480)
            mock_ffmpeg_cls.return_value.sample_grain.return_value = [1.0]
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = [scan_result]
            mock_pipeline_cls.return_value.run.return_value = AnalysisBatchResult(movies=[], crops={})
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng"],
            )

        assert result.exit_code == 0, result.output
        mock_runner.assert_called_once()
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args.kwargs
        assert call_kwargs["grain_overrides"] is None

    def test_dry_run_no_screen_grain_none(self, tmp_path: Path) -> None:
        """--dry-run never opens the grain screen and passes grain_overrides=None."""
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
        movie = MagicMock()
        plan_obj = make_plan(jobs=[])

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.FFmpegAdapter") as mock_ffmpeg_cls,
            patch("furnace.cli.MpvAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.MakemkvAdapter"),
            patch("furnace.cli.DiscDemuxer") as mock_demuxer_cls,
            patch("furnace.cli.Scanner") as mock_scanner_cls,
            patch("furnace.cli.Analyzer") as mock_analyzer_cls,
            patch("furnace.cli.PlannerService") as mock_planner_cls,
            patch("furnace.cli._run_screen_app") as mock_runner,
        ):
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = [scan_result]
            mock_analyzer_cls.return_value.analyze.return_value = AnalysisOutcome(
                movie, AnalyzeStatus.DONE, "summary"
            )
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng", "--dry-run"],
            )

        assert result.exit_code == 0, result.output
        mock_runner.assert_not_called()
        mock_ffmpeg_cls.return_value.sample_grain.assert_not_called()
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args.kwargs
        assert call_kwargs["grain_overrides"] is None

    def test_deselected_plain_file_does_not_reach_pipeline(self, tmp_path: Path) -> None:
        """Un-checking a plain file in the selector filters it out: the deselected
        file's scan result never reaches the analysis pipeline, while the still-
        selected file (and its grain decision) does."""
        from furnace.core.models import ScanResult
        from furnace.services.analysis_pipeline import AnalysisBatchResult
        from furnace.ui.tui import FileSelection

        source = tmp_path / "src"
        source.mkdir()
        output = tmp_path / "out"
        kept_file = source / "keep.mkv"
        dropped_file = source / "drop.mkv"

        cfg = _make_tool_paths(tmp_path)
        kept_result = ScanResult(
            main_file=kept_file,
            satellite_files=[],
            output_path=output / "keep" / "keep.mkv",
        )
        dropped_result = ScanResult(
            main_file=dropped_file,
            satellite_files=[],
            output_path=output / "drop" / "drop.mkv",
        )
        plan_obj = make_plan(jobs=[])
        # The selector returns only the kept file (dropped file un-checked); its
        # grain decision is likewise only present for the kept file.
        selection = FileSelection(selected=[kept_file], sar_override=set(), grain={kept_file: True})

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.FFmpegAdapter") as mock_ffmpeg_cls,
            patch("furnace.cli.MpvAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.MakemkvAdapter"),
            patch("furnace.cli.DiscDemuxer") as mock_demuxer_cls,
            patch("furnace.cli.Scanner") as mock_scanner_cls,
            patch("furnace.cli.AnalysisPipeline") as mock_pipeline_cls,
            patch("furnace.cli.PlannerService") as mock_planner_cls,
            patch("furnace.cli.save_plan"),
            patch("furnace.cli._run_screen_app", return_value=selection) as mock_runner,
        ):
            mock_ffmpeg_cls.return_value.probe.return_value = self._sd_streams(480)
            mock_ffmpeg_cls.return_value.sample_grain.return_value = [1.0]  # GRAINY
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = [kept_result, dropped_result]
            mock_pipeline_cls.return_value.run.return_value = AnalysisBatchResult(movies=[], crops={})
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng"],
            )

        assert result.exit_code == 0, result.output
        mock_runner.assert_called_once()
        # The pipeline must receive only the still-selected scan result.
        pipeline_results = mock_pipeline_cls.return_value.run.call_args.args[0]
        pipeline_files = {sr.main_file for sr in pipeline_results}
        assert pipeline_files == {kept_file}
        assert dropped_file not in pipeline_files
        # The kept file's grain decision is still threaded through.
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args.kwargs
        assert call_kwargs["grain_overrides"] == {kept_file: True}

    def test_all_plain_files_deselected_does_not_crash(self, tmp_path: Path) -> None:
        """De-selecting every plain file yields an empty scan-result set that flows
        through without crashing: the pipeline gets no files and create_plan is
        still called (with empty movies)."""
        from furnace.core.models import ScanResult
        from furnace.services.analysis_pipeline import AnalysisBatchResult
        from furnace.ui.tui import FileSelection

        source = tmp_path / "src"
        source.mkdir()
        output = tmp_path / "out"
        main_file = source / "movie.mkv"

        cfg = _make_tool_paths(tmp_path)
        scan_result = ScanResult(
            main_file=main_file,
            satellite_files=[],
            output_path=output / "movie" / "movie.mkv",
        )
        plan_obj = make_plan(jobs=[])
        selection = FileSelection(selected=[], sar_override=set(), grain={})

        with (
            patch("furnace.cli.load_config", return_value=cfg),
            patch("furnace.cli._setup_logging"),
            patch("furnace.cli.FFmpegAdapter") as mock_ffmpeg_cls,
            patch("furnace.cli.MpvAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.MakemkvAdapter"),
            patch("furnace.cli.DiscDemuxer") as mock_demuxer_cls,
            patch("furnace.cli.Scanner") as mock_scanner_cls,
            patch("furnace.cli.AnalysisPipeline") as mock_pipeline_cls,
            patch("furnace.cli.PlannerService") as mock_planner_cls,
            patch("furnace.cli.save_plan"),
            patch("furnace.cli._run_screen_app", return_value=selection) as mock_runner,
        ):
            mock_ffmpeg_cls.return_value.probe.return_value = self._sd_streams(480)
            mock_ffmpeg_cls.return_value.sample_grain.return_value = [1.0]
            mock_demuxer_cls.return_value.detect.return_value = []
            mock_scanner_cls.return_value.scan.return_value = [scan_result]
            mock_pipeline_cls.return_value.run.return_value = AnalysisBatchResult(movies=[], crops={})
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng"],
            )

        assert result.exit_code == 0, result.output
        mock_runner.assert_called_once()
        pipeline_results = mock_pipeline_cls.return_value.run.call_args.args[0]
        assert list(pipeline_results) == []
        mock_planner_cls.return_value.create_plan.assert_called_once()


class TestInterlacedGrainMetricsPreflight:
    """`_check_interlaced_grain_metrics_ready` fails fast (before any encode) when
    an interlaced grain job would be scored but no bwdif plugin is configured.

    Grain jobs always target-quality-search when bestsource+vship are configured,
    so the check gates on those tool paths, NOT on the (retired-for-NVEnc)
    ``--metrics`` flag."""

    @staticmethod
    def _cfg(tmp_path: Path, *, bestsource: Path | None, vship: Path | None, bwdif: Path | None) -> Any:
        cfg = _make_tool_paths(tmp_path)
        cfg.bestsource = bestsource
        cfg.vship = vship
        cfg.bwdif = bwdif
        return cfg

    @staticmethod
    def _metrics_cfg(tmp_path: Path, *, bwdif: Path | None) -> Any:
        return TestInterlacedGrainMetricsPreflight._cfg(
            tmp_path, bestsource=tmp_path / "bs.dll", vship=tmp_path / "vs.dll", bwdif=bwdif,
        )

    @staticmethod
    def _plan(*, grain: bool = True, deinterlace: bool = True,
              status: JobStatus = JobStatus.PENDING) -> Any:
        vp = make_video_params(grain=grain, deinterlace=deinterlace)
        return make_plan(
            jobs=[make_job(job_id="j1", status=status, video_params=vp)],
        )

    def test_raises_when_bwdif_missing(self, tmp_path: Path) -> None:
        cfg = self._metrics_cfg(tmp_path, bwdif=None)
        plan = self._plan()
        with pytest.raises(typer.Exit) as exc:
            _check_interlaced_grain_metrics_ready(plan, cfg)
        assert exc.value.exit_code == 1

    def test_ok_when_bwdif_present(self, tmp_path: Path) -> None:
        cfg = self._metrics_cfg(tmp_path, bwdif=tmp_path / "bwdif.dll")
        _check_interlaced_grain_metrics_ready(self._plan(), cfg)

    def test_raises_even_when_offender_is_pending(self, tmp_path: Path) -> None:
        # The grain search runs (and needs bwdif) whenever vship is configured,
        # so a pending interlaced grain job is still a fail-fast offender.
        cfg = self._metrics_cfg(tmp_path, bwdif=None)
        with pytest.raises(typer.Exit) as exc:
            _check_interlaced_grain_metrics_ready(self._plan(), cfg)
        assert exc.value.exit_code == 1

    def test_ok_when_vship_not_configured(self, tmp_path: Path) -> None:
        # No vship adapter -> grain falls back to fixed CRF (no search, no measure) -> no bwdif need.
        cfg = self._cfg(tmp_path, bestsource=None, vship=None, bwdif=None)
        _check_interlaced_grain_metrics_ready(self._plan(), cfg)

    def test_ok_when_no_interlaced_grain_offender(self, tmp_path: Path) -> None:
        # Mixed plan, no offender: a DONE interlaced-grain job (not pending), a
        # progressive grain job (deinterlace False), and a non-grain interlaced
        # job (grain False) all fall outside the offender filter.
        cfg = self._metrics_cfg(tmp_path, bwdif=None)
        plan = make_plan(
            jobs=[
                make_job(job_id="done", status=JobStatus.DONE,
                         video_params=make_video_params(grain=True, deinterlace=True)),
                make_job(job_id="prog", status=JobStatus.PENDING,
                         video_params=make_video_params(grain=True, deinterlace=False)),
                make_job(job_id="nvenc", status=JobStatus.PENDING,
                         video_params=make_video_params(grain=False, deinterlace=True)),
            ],
        )
        _check_interlaced_grain_metrics_ready(plan, cfg)
