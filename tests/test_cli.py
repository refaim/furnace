from __future__ import annotations

import contextlib
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from furnace.cli import _setup_logging, app
from furnace.core.models import AnalysisOutcome, AnalyzeStatus, JobStatus, TrackType
from tests.conftest import make_job, make_movie, make_plan, make_track

runner = CliRunner()


class TestSetupLogging:
    def _cleanup_root_handlers(self) -> None:
        root = logging.getLogger()
        for h in list(root.handlers):
            if isinstance(h, logging.FileHandler):
                root.removeHandler(h)
                h.close()
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


def _make_tool_paths(tmp_path: Path) -> MagicMock:
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
    cfg.hdr10plus_tool = None
    cfg.bestsource = None
    cfg.vship = None
    return cfg


class TestPlanDryRun:
    def test_dry_run_no_movies(self, tmp_path: Path) -> None:
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
        mock_planner_cls.return_value.create_plan.assert_called_once()
        mock_analyzer_cls.return_value.analyze.assert_not_called()

    def test_dry_run_with_movies(self, tmp_path: Path) -> None:
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
        mock_planner_cls.return_value.create_plan.assert_called_once()
        assert len(mock_planner_cls.return_value.create_plan.return_value.jobs) == 2

    def test_dry_run_passes_language_lists(self, tmp_path: Path) -> None:
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
        source = tmp_path / "src"
        source.mkdir()
        with patch("furnace.cli.load_config", return_value=_make_tool_paths(tmp_path)):
            result = runner.invoke(
                app,
                [
                    "plan",
                    str(source),
                    "-o",
                    str(tmp_path / "out"),
                    "-al",
                    "eng",
                    "-sl",
                    "eng",
                    "--dry-run",
                    "--metrics",
                ],
            )
        assert result.exit_code != 0

    def test_copy_video_flag_forwarded(self, tmp_path: Path) -> None:
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


class TestPlanSave:
    def test_save_plan_writes_file(self, tmp_path: Path) -> None:
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
        mock_save.assert_called_once()
        mock_planner_cls.return_value.create_plan.assert_called_once()


class TestPlanNames:
    def test_names_map_loaded(self, tmp_path: Path) -> None:
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
        call_args = mock_scanner_cls.return_value.scan.call_args
        assert call_args.args[2] == {"movie.mkv": "Movie Title"}


class TestPlanDiscDryRun:
    def test_detected_discs_skipped_in_dry_run(self, tmp_path: Path) -> None:
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
        mock_demuxer_cls.return_value.list_titles.assert_called_once_with(disc)
        mock_demuxer_cls.return_value.demux.assert_not_called()


class TestPlanDemuxDirAssignment:
    def test_demux_dir_not_set_when_no_discs(self, tmp_path: Path) -> None:
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


class TestPlanAnalyzerNone:
    def test_analyzer_none_skips_movie(self, tmp_path: Path) -> None:
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
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args
        assert call_kwargs.kwargs["movies"] == []


class TestRunCommand:
    def test_run_all_done_no_pending(self, tmp_path: Path) -> None:
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
        init_kwargs = mock_run_app_cls.call_args.kwargs
        assert init_kwargs["total_jobs"] == 0

    def test_run_with_pending_jobs(self, tmp_path: Path) -> None:
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
        assert init_kwargs["total_jobs"] == 2

    def test_run_calls_report_printer(self, tmp_path: Path) -> None:
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

            def _run_sets_shutdown() -> None:
                shutdown_evt = mock_run_app_cls.call_args.kwargs["shutdown_event"]
                shutdown_evt.set()

            mock_run_app_cls.return_value.run.side_effect = _run_sets_shutdown

            runner.invoke(app, ["run", str(plan_file)])

        mock_exit.assert_called_once_with(0)

    def test_run_cleanup_demux_dir_all_done(self, tmp_path: Path) -> None:
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


class TestPlanConfigOption:
    def test_config_option_forwarded(self, tmp_path: Path) -> None:
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


class TestRunExecutorClosure:
    def test_executor_fn_creates_adapters_and_runs(self, tmp_path: Path) -> None:
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

            def _capture_and_noop() -> None:
                captured_executor_fn.append(mock_run_app_cls.call_args.kwargs["executor_fn"])

            mock_run_app_cls.return_value.run.side_effect = _capture_and_noop

            result = runner.invoke(app, ["run", str(plan_file)])

        assert result.exit_code == 0, result.output
        assert len(captured_executor_fn) == 1

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
        exec_kwargs = mock_executor_cls.call_args.kwargs
        assert exec_kwargs["video_copier"] is not None
        assert exec_kwargs["target_quality"] is not None

    def test_executor_fn_wires_svt_grain_encoder(self, tmp_path: Path) -> None:
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

        mock_svt.assert_called_once()
        svt_args = mock_svt.call_args.args
        assert svt_args[0] == cfg.ffmpeg
        exec_kwargs = mock_executor_cls.call_args.kwargs
        assert exec_kwargs["grain_encoder"] is mock_svt.return_value

    def _run_executor_fn(self, tmp_path: Path, cfg: Any) -> tuple[Any, Any]:
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
        cfg = _make_tool_paths(tmp_path)
        cfg.bestsource = tmp_path / "BestSource.dll"
        cfg.vship = tmp_path / "libvship.dll"

        mock_vship, _ = self._run_executor_fn(tmp_path, cfg)

        mock_vship.assert_called_once_with(cfg.bestsource, cfg.vship)

    def test_executor_fn_with_dovi_tool(self, tmp_path: Path) -> None:
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

    def test_executor_fn_with_hdr10plus_tool(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.json"
        plan_obj = make_plan(
            jobs=[make_job(job_id="j1", status=JobStatus.PENDING)],
            destination=str(tmp_path / "out"),
        )

        cfg = _make_tool_paths(tmp_path)
        cfg.hdr10plus_tool = tmp_path / "hdr10plus_tool"

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
            patch("furnace.cli.Hdr10PlusToolAdapter") as mock_hdr10plus,
            patch("furnace.cli.Executor") as mock_executor_cls,
        ):
            executor_fn(mock_progress)

        mock_hdr10plus.assert_called_once()
        assert mock_hdr10plus.call_args.args[0] == cfg.hdr10plus_tool
        assert mock_hdr10plus.call_args.args[1] == cfg.ffmpeg
        executor_kwargs = mock_executor_cls.call_args.kwargs
        assert executor_kwargs["hdr10plus_processor"] is mock_hdr10plus.return_value

    def test_executor_fn_without_hdr10plus_tool(self, tmp_path: Path) -> None:
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

        with (
            patch("furnace.cli.FFmpegAdapter"),
            patch("furnace.cli.Eac3toAdapter"),
            patch("furnace.cli.QaacAdapter"),
            patch("furnace.cli.MkvmergeAdapter"),
            patch("furnace.cli.MkvpropeditAdapter"),
            patch("furnace.cli.MkcleanAdapter"),
            patch("furnace.cli.NVEncCAdapter"),
            patch("furnace.cli.Hdr10PlusToolAdapter") as mock_hdr10plus,
            patch("furnace.cli.Executor") as mock_executor_cls,
        ):
            executor_fn(MagicMock())

        mock_hdr10plus.assert_not_called()
        assert mock_executor_cls.call_args.kwargs["hdr10plus_processor"] is None

    def test_executor_fn_stops_progress_on_error(self, tmp_path: Path) -> None:
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

        mock_progress.stop.assert_called_once()


class TestRunDemuxDirEdgeCases:
    def test_demux_dir_set_but_not_on_disk(self, tmp_path: Path) -> None:
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


class TestMainModule:
    def test_main_calls_app(self) -> None:
        import runpy

        with patch("furnace.cli.app") as mock_app:
            with contextlib.suppress(SystemExit):
                runpy.run_module("furnace", run_name="__main__")
        mock_app.assert_called_once()


class TestMakePreviewTrackCb:
    def test_audio_track_calls_preview_audio(self, tmp_path: Path) -> None:
        from furnace.cli import _make_preview_track_cb

        main = tmp_path / "m.mkv"
        track = make_track(index=1, track_type=TrackType.AUDIO, source_file=main)
        movie = make_movie(main_file=main, audio_tracks=[track])
        mpv = MagicMock()

        cb = _make_preview_track_cb(movie, mpv)
        cb(track)

        mpv.preview_audio.assert_called_once_with(main, None, 1)
        mpv.preview_subtitle.assert_not_called()

    def test_subtitle_track_calls_preview_subtitle(self, tmp_path: Path) -> None:
        from furnace.cli import _make_preview_track_cb

        main = tmp_path / "m.mkv"
        track = make_track(
            index=4,
            track_type=TrackType.SUBTITLE,
            codec_name="subrip",
            source_file=main,
        )
        movie = make_movie(main_file=main, subtitle_tracks=[track])
        mpv = MagicMock()

        cb = _make_preview_track_cb(movie, mpv)
        cb(track)

        mpv.preview_subtitle.assert_called_once_with(main, None, 1)
        mpv.preview_audio.assert_not_called()

    def test_audio_track_id_counts_from_one_inside_the_file(self, tmp_path: Path) -> None:
        from furnace.cli import _make_preview_track_cb

        main = tmp_path / "m.mkv"
        tracks = [make_track(index=i, track_type=TrackType.AUDIO, source_file=main) for i in (1, 2, 3)]
        movie = make_movie(main_file=main, audio_tracks=tracks)
        mpv = MagicMock()

        cb = _make_preview_track_cb(movie, mpv)
        cb(tracks[2])

        mpv.preview_audio.assert_called_once_with(main, None, 3)

    def test_subtitle_track_id_ignores_the_global_stream_index(self, tmp_path: Path) -> None:
        from furnace.cli import _make_preview_track_cb

        main = tmp_path / "m.mkv"
        subs = [
            make_track(index=i, track_type=TrackType.SUBTITLE, codec_name="subrip", source_file=main)
            for i in (4, 5, 6)
        ]
        movie = make_movie(main_file=main, subtitle_tracks=subs)
        mpv = MagicMock()

        cb = _make_preview_track_cb(movie, mpv)
        cb(subs[1])

        mpv.preview_subtitle.assert_called_once_with(main, None, 2)

    def test_external_audio_track_id_follows_the_internal_ones(self, tmp_path: Path) -> None:
        from furnace.cli import _make_preview_track_cb

        main = tmp_path / "m.mkv"
        external = tmp_path / "audio.mka"
        internal = [make_track(index=i, track_type=TrackType.AUDIO, source_file=main) for i in (1, 2)]
        outside = make_track(index=100, track_type=TrackType.AUDIO, source_file=external)
        movie = make_movie(main_file=main, audio_tracks=[*internal, outside])
        mpv = MagicMock()

        cb = _make_preview_track_cb(movie, mpv)
        cb(outside)

        mpv.preview_audio.assert_called_once_with(main, external, 3)

    def test_external_subtitle_track_id_follows_the_internal_ones(self, tmp_path: Path) -> None:
        from furnace.cli import _make_preview_track_cb

        main = tmp_path / "m.mkv"
        external = tmp_path / "subs.srt"
        internal = [
            make_track(index=i, track_type=TrackType.SUBTITLE, codec_name="subrip", source_file=main) for i in (4, 5)
        ]
        outside = make_track(index=100, track_type=TrackType.SUBTITLE, codec_name="subrip", source_file=external)
        movie = make_movie(main_file=main, subtitle_tracks=[*internal, outside])
        mpv = MagicMock()

        cb = _make_preview_track_cb(movie, mpv)
        cb(outside)

        mpv.preview_subtitle.assert_called_once_with(main, external, 3)

    def test_second_track_of_the_same_external_file(self, tmp_path: Path) -> None:
        from furnace.cli import _make_preview_track_cb

        main = tmp_path / "m.mkv"
        external = tmp_path / "audio.mka"
        internal = [make_track(index=1, track_type=TrackType.AUDIO, source_file=main)]
        outside = [make_track(index=i, track_type=TrackType.AUDIO, source_file=external) for i in (100, 101)]
        movie = make_movie(main_file=main, audio_tracks=[*internal, *outside])
        mpv = MagicMock()

        cb = _make_preview_track_cb(movie, mpv)
        cb(outside[1])

        mpv.preview_audio.assert_called_once_with(main, external, 3)


class TestSelectTracksTui:
    def test_returns_app_runner_result(self, tmp_path: Path) -> None:
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


class TestSelectTracksTuiForPlanner:
    def test_audio_updates_downmix_overrides(self, tmp_path: Path) -> None:
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


class TestResolveUndLanguageTui:
    def test_returns_app_runner_result(self, tmp_path: Path) -> None:
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


class TestAppendDemuxedScanResults:
    def test_appends_one_scan_result_per_demuxed_path(self, tmp_path: Path) -> None:
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
        from furnace.cli import _append_demuxed_scan_results

        scan_results: list[Any] = [MagicMock()]
        _append_demuxed_scan_results(scan_results, [], tmp_path / "out")
        assert len(scan_results) == 1


class TestApplyDemuxDirToPlan:
    def test_sets_demux_dir_on_plan(self, tmp_path: Path) -> None:
        from furnace.cli import _apply_demux_dir_to_plan

        plan_obj = make_plan(jobs=[])
        _apply_demux_dir_to_plan(plan_obj, tmp_path / "demux")
        assert plan_obj.demux_dir == str(tmp_path / "demux")

    def test_none_leaves_plan_unchanged(self) -> None:
        from furnace.cli import _apply_demux_dir_to_plan

        plan_obj = make_plan(jobs=[], demux_dir=None)
        _apply_demux_dir_to_plan(plan_obj, None)
        assert plan_obj.demux_dir is None


class TestSarToggleFiles:
    def test_selects_only_sub_hd_files(self, tmp_path: Path) -> None:
        from furnace.cli import _sar_toggle_files

        pal = tmp_path / "pal.mkv"
        ntsc = tmp_path / "ntsc.mkv"
        hd720 = tmp_path / "hd720.mkv"
        hd1080 = tmp_path / "hd1080.mkv"
        infos = [
            (pal, 60.0, 100, 576, None),
            (ntsc, 60.0, 100, 480, "bt709"),
            (hd720, 60.0, 100, 720, None),
            (hd1080, 60.0, 100, 1080, None),
        ]
        assert _sar_toggle_files(infos) == {pal, ntsc}

    def test_zero_height_excluded(self, tmp_path: Path) -> None:
        from furnace.cli import _sar_toggle_files

        mkv = tmp_path / "unknown.mkv"
        assert _sar_toggle_files([(mkv, 60.0, 100, 0, None)]) == set()


class TestRunDiscDemuxInteractive:
    def _adapters(self) -> tuple[MagicMock, MagicMock]:
        ffmpeg = MagicMock()
        ffmpeg.sample_grain.return_value = [0.1]
        return ffmpeg, MagicMock()

    def test_no_discs_returns_empty(self, tmp_path: Path) -> None:
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
        demuxer.list_titles.assert_not_called()

    def test_single_playlist_auto_selected_and_demuxed(self, tmp_path: Path) -> None:
        from furnace.cli import _run_disc_demux_interactive
        from furnace.core.models import DiscSource, DiscTitle, DiscType

        disc_root = tmp_path / "disc_folder" / "BDMV"
        disc = DiscSource(path=disc_root, disc_type=DiscType.BLURAY)
        title = DiscTitle(number=1, duration_s=5400.0, raw_label="1: ...")

        demuxer = MagicMock()
        demuxed_mkv = tmp_path / "disc_folder_title_1.mkv"
        demuxer.demux.return_value = [demuxed_mkv]

        ffmpeg, mpv = self._adapters()
        ffmpeg.probe.return_value = {
            "format": {"duration": "5400.0", "size": "1000"},
            "streams": [
                {"codec_type": "video", "height": 2160, "color_transfer": "smpte2084"},
            ],
        }

        from furnace.ui.tui import FileSelection

        demux_dir, paths, sar, _grain = _run_disc_demux_interactive(
            source=tmp_path,
            detected_discs=[disc],
            disc_titles={disc: [title]},
            disc_demuxer=demuxer,
            ffmpeg_adapter=ffmpeg,
            mpv_adapter=mpv,
            playlist_app_runner=MagicMock(),
            file_app_runner=MagicMock(
                return_value=FileSelection(selected=[demuxed_mkv], sar_override=set(), grain={}),
            ),
        )

        assert demux_dir == tmp_path / ".furnace_demux"
        assert paths == [demuxed_mkv]
        assert sar == set()
        demuxer.demux.assert_called_once()
        demuxer.list_titles.assert_not_called()

    def test_empty_playlist_list_skips_disc(self, tmp_path: Path) -> None:
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
        call_kwargs = demuxer.demux.call_args.kwargs
        assert call_kwargs["selected_titles"] == {disc: [t2]}

    def test_multiple_playlists_runner_returns_none_skips_disc(self, tmp_path: Path) -> None:
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

    def test_sd_demuxed_file_triggers_file_selector_and_sar(self, tmp_path: Path) -> None:
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
        ffmpeg.probe.return_value = {
            "format": {"duration": "100.0", "size": "1000"},
            "streams": [{"codec_type": "video", "height": 576}],
        }
        ffmpeg.sample_grain.return_value = [1.0]
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
        assert screens_built[0]._sar_files == {dvd_mkv}
        ffmpeg.probe.assert_called_once_with(dvd_mkv)

    def test_multiple_demuxed_files_trigger_file_selector(self, tmp_path: Path) -> None:
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
        ffmpeg.probe.return_value = {}
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


class TestPlanSelectorClosures:
    def test_track_selector_closure_routes_through_helper(self, tmp_path: Path) -> None:
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

            planner_kwargs = mock_planner_cls.call_args.kwargs
            selector = planner_kwargs["track_selector"]
            resolver = planner_kwargs["und_resolver"]

            movie = make_movie(main_file=source / "m.mkv")
            track = make_track(index=1, track_type=TrackType.AUDIO)
            selector(movie, [track], TrackType.AUDIO)
            resolver(movie, track, ["eng", "rus"])

            mock_sel.assert_called_once()
            mock_res.assert_called_once()

    def test_track_selector_forwards_lang_list_and_relabel_under_ignore_langs(self, tmp_path: Path) -> None:
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
        mock_planner_cls = self._run_plan(tmp_path, ["--ignore-langs"])
        assert mock_planner_cls.call_args.kwargs["ignore_langs"] is True
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args.kwargs
        assert call_kwargs["lang_overrides"] == {}

    def test_ignore_langs_short_flag(self, tmp_path: Path) -> None:
        mock_planner_cls = self._run_plan(tmp_path, ["-il"])
        assert mock_planner_cls.call_args.kwargs["ignore_langs"] is True

    def test_ignore_langs_defaults_false(self, tmp_path: Path) -> None:
        mock_planner_cls = self._run_plan(tmp_path, [])
        assert mock_planner_cls.call_args.kwargs["ignore_langs"] is False
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args.kwargs
        assert call_kwargs["lang_overrides"] == {}


class TestPlanDiscInteractive:
    def test_plan_calls_disc_interactive_when_discs_detected(self, tmp_path: Path) -> None:
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
        assert "disc_titles" in mock_interactive.call_args.kwargs
        assert plan_obj.demux_dir == str(source / ".furnace_demux")
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args.kwargs
        assert call_kwargs["sar_overrides"] == {demuxed}
        assert call_kwargs["grain_overrides"] == {demuxed: True}


class TestPlanDiscInteractiveReporter:
    def test_reporter_pause_resume_around_screens(self, tmp_path: Path) -> None:
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

        assert manager.mock_calls == [call.pause(), call.resume(), call.pause(), call.resume()]
        assert demuxer.demux.call_args.kwargs["reporter"] is reporter


class TestPlanDetectRelPathFallback:
    def test_disc_outside_source_falls_back_to_basename(self, tmp_path: Path) -> None:
        from furnace.core.models import DiscSource, DiscType

        source = tmp_path / "src"
        source.mkdir()
        output = tmp_path / "out"

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
        reporter_inst.detect_disc.assert_called_once_with(DiscType.BLURAY, "elsewhere")
        reporter_inst.detect_disc_titles_done.assert_called_once_with(0)


class TestPlanAnalyzeFailure:
    def test_failed_analysis_surfaces_a_line_and_no_job(self, tmp_path: Path) -> None:
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
            mock_analyzer_cls.return_value.analyze.return_value = AnalysisOutcome(
                None, AnalyzeStatus.FAILED, "probe failed: fixture detail"
            )
            mock_planner_cls.return_value.create_plan.return_value = plan_obj

            result = runner.invoke(
                app,
                ["plan", str(source), "-o", str(output), "-al", "eng", "-sl", "eng", "--dry-run"],
            )

        assert result.exit_code == 0, result.output
        reporter_inst.analyze_batch_item.assert_called_once_with(
            "movie.mkv", "probe failed: fixture detail", status=AnalyzeStatus.FAILED
        )
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args.kwargs
        assert call_kwargs["movies"] == []


class TestPlanJobs:
    @staticmethod
    def _invoke_capturing_pipeline(
        tmp_path: Path,
        extra_args: list[str],
        *,
        cpu_count: int | None = 8,
    ) -> Any:
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
                    "plan",
                    str(source),
                    "-o",
                    str(output),
                    "-al",
                    "eng",
                    "-sl",
                    "eng",
                    "--dry-run",
                    *extra_args,
                ],
            )

        assert result.exit_code == 0, result.output
        return mock_pipeline_cls

    def test_jobs_flag_forwards_max_workers(self, tmp_path: Path) -> None:
        pipeline_cls = self._invoke_capturing_pipeline(tmp_path, ["--jobs", "4"])
        assert pipeline_cls.call_args.kwargs["max_workers"] == 4

    def test_jobs_short_flag_forwards_max_workers(self, tmp_path: Path) -> None:
        pipeline_cls = self._invoke_capturing_pipeline(tmp_path, ["-j", "3"])
        assert pipeline_cls.call_args.kwargs["max_workers"] == 3

    def test_jobs_flag_floored_at_one(self, tmp_path: Path) -> None:
        pipeline_cls = self._invoke_capturing_pipeline(tmp_path, ["--jobs", "0"])
        assert pipeline_cls.call_args.kwargs["max_workers"] == 1

    def test_default_workers_is_cpu_count_minus_two(self, tmp_path: Path) -> None:
        pipeline_cls = self._invoke_capturing_pipeline(tmp_path, [], cpu_count=8)
        assert pipeline_cls.call_args.kwargs["max_workers"] == 6

    def test_default_workers_floored_when_few_cpus(self, tmp_path: Path) -> None:
        pipeline_cls = self._invoke_capturing_pipeline(tmp_path, [], cpu_count=1)
        assert pipeline_cls.call_args.kwargs["max_workers"] == 1

    def test_default_workers_when_cpu_count_none(self, tmp_path: Path) -> None:
        pipeline_cls = self._invoke_capturing_pipeline(tmp_path, [], cpu_count=None)
        assert pipeline_cls.call_args.kwargs["max_workers"] == 1

    def test_jobs_one_yields_same_plan_as_default(self, tmp_path: Path) -> None:
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
                        "plan",
                        str(source),
                        "-o",
                        str(output),
                        "-al",
                        "eng",
                        "-sl",
                        "eng",
                        "--dry-run",
                        *extra_args,
                    ],
                )
            assert result.exit_code == 0, result.output
            return mock_planner_cls.return_value.create_plan.call_args.kwargs["movies"]

        expected = [(movie, output / "movie" / "movie.mkv")]
        default_movies = _run([])
        jobs1_movies = _run(["--jobs", "1"])

        assert default_movies == expected
        assert jobs1_movies == expected


class TestPlanKeyboardInterrupt:
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


class TestDefaultAppRunner:
    def test_run_screen_app_captures_dismiss_result(self) -> None:
        from furnace.cli import _run_screen_app

        sentinel = "dismiss-result"
        factory_calls: list[int] = []
        fake_screen = MagicMock()

        def _factory() -> Any:
            factory_calls.append(1)
            return fake_screen

        composed: list[Any] = []

        def fake_run(self: Any) -> None:
            composed.extend(self.compose())
            pushed: list[Any] = []

            def _push_screen(screen: Any, on_dismiss: Callable[[Any], None]) -> None:
                pushed.append(screen)
                on_dismiss(sentinel)

            self.push_screen = _push_screen
            self.exit = lambda _result: None
            self.on_mount()
            assert pushed == [fake_screen]

        with patch("textual.app.App.run", fake_run):
            result = _run_screen_app(_factory)

        assert result == sentinel
        assert factory_calls == [1]
        assert len(composed) == 1

    def test_run_screen_app_handles_none_dismiss(self) -> None:
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


class TestScanCommand:
    def test_no_filters_calls_service_with_defaults(self, tmp_path: Path) -> None:
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

            result = runner.invoke(app, ["scan", str(src), "--not-encoded", "--max-version", "1.19.3"])

        assert result.exit_code == 0, result.output
        kwargs = mock_service_cls.return_value.scan.call_args.kwargs
        assert kwargs["not_encoded"] is True
        assert kwargs["max_version"] == (1, 19, 3)

    def test_bad_max_version_is_cli_error(self, tmp_path: Path) -> None:
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
        assert mock_service_cls.call_args.kwargs["prober"] is mock_ffmpeg_cls.return_value

    def test_total_taken_from_service(self, tmp_path: Path) -> None:
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


class TestProbeFileInfosHeight:
    def test_includes_first_video_stream_height_and_transfer(self, tmp_path: Path) -> None:
        from furnace.cli import _probe_file_infos

        p = tmp_path / "a.mkv"
        ffmpeg = MagicMock()
        ffmpeg.probe.return_value = {
            "format": {"duration": "10.0", "size": "20"},
            "streams": [
                {"codec_type": "audio"},
                {"codec_type": "video", "height": 576, "color_transfer": "bt709"},
            ],
        }

        infos = _probe_file_infos([p], ffmpeg)

        assert infos == [(p, 10.0, 20, 576, "bt709")]

    def test_no_video_stream_height_zero_transfer_none(self, tmp_path: Path) -> None:
        from furnace.cli import _probe_file_infos

        p = tmp_path / "a.mkv"
        ffmpeg = MagicMock()
        ffmpeg.probe.return_value = {"format": {"duration": "0", "size": "0"}}

        infos = _probe_file_infos([p], ffmpeg)

        assert infos == [(p, 0.0, 0, 0, None)]

    def test_untagged_transfer_is_none(self, tmp_path: Path) -> None:
        from furnace.cli import _probe_file_infos

        p = tmp_path / "a.mkv"
        ffmpeg = MagicMock()
        ffmpeg.probe.return_value = {
            "format": {"duration": "5.0", "size": "10"},
            "streams": [{"codec_type": "video", "height": 1080}],
        }

        infos = _probe_file_infos([p], ffmpeg)

        assert infos == [(p, 5.0, 10, 1080, None)]


class TestDiscDemuxGrain:
    def _sd_probe(self, height: int = 480) -> dict[str, Any]:
        return {
            "format": {"duration": "100.0", "size": "1000"},
            "streams": [{"codec_type": "video", "height": height}],
        }

    def test_grainy_sd_file_pre_lit_and_threaded(self, tmp_path: Path) -> None:
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
        ffmpeg.sample_grain.return_value = [1.0, 1.2]

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
        ffmpeg.sample_grain.assert_called_once_with(dvd_mkv, 100.0, hdr_transfer=None)
        screen = screens_built[0]
        assert isinstance(screen, FileSelectorScreen)
        assert screen._grain_files == {dvd_mkv}
        assert screen._grain_defaults == {dvd_mkv}

    def test_clean_sd_file_not_pre_lit(self, tmp_path: Path) -> None:
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
        ffmpeg.sample_grain.return_value = [0.1, 0.2]

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
        assert screens_built[0]._grain_files == {dvd_mkv}
        assert screens_built[0]._grain_defaults == set()

    def test_single_sd_file_triggers_screen(self, tmp_path: Path) -> None:
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

        file_runner = MagicMock(return_value=FileSelection(selected=[mkv], sar_override=set(), grain={mkv: True}))

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

    def test_grain_defaults_cover_hdr_through_a_tonemap(self, tmp_path: Path) -> None:
        from furnace.cli import _run_disc_demux_interactive
        from furnace.core.models import DiscSource, DiscTitle, DiscType
        from furnace.ui.tui import FileSelection

        disc_root = tmp_path / "bdroot" / "BDMV"
        disc = DiscSource(path=disc_root, disc_type=DiscType.BLURAY)
        t1 = DiscTitle(number=1, duration_s=100, raw_label="1")
        t2 = DiscTitle(number=2, duration_s=200, raw_label="2")

        demuxer = MagicMock()
        hdr_mkv = tmp_path / ".furnace_demux" / "bdroot_title_1.mkv"
        sdr_mkv = tmp_path / ".furnace_demux" / "bdroot_title_2.mkv"
        demuxer.demux.return_value = [hdr_mkv, sdr_mkv]

        ffmpeg = MagicMock()

        def _probe(path: Path) -> dict[str, Any]:
            stream: dict[str, Any] = (
                {"codec_type": "video", "height": 2160, "color_transfer": "smpte2084"}
                if path == hdr_mkv
                else {"codec_type": "video", "height": 1080}
            )
            return {"format": {"duration": "100.0", "size": "1000"}, "streams": [stream]}

        ffmpeg.probe.side_effect = _probe
        ffmpeg.sample_grain.return_value = [9.0]

        screens_built: list[Any] = []

        def file_runner(factory: Callable[[], Any]) -> FileSelection:
            screens_built.append(factory())
            return FileSelection(selected=[hdr_mkv, sdr_mkv], sar_override=set(), grain={sdr_mkv: True})

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

        from unittest.mock import call

        screen = screens_built[0]
        assert screen._grain_files == {hdr_mkv, sdr_mkv}
        assert screen._grain_defaults == {hdr_mkv, sdr_mkv}
        assert ffmpeg.sample_grain.call_args_list == [
            call(hdr_mkv, 100.0, hdr_transfer="smpte2084"),
            call(sdr_mkv, 100.0, hdr_transfer=None),
        ]

    def test_grain_probe_uses_stream_duration(self, tmp_path: Path) -> None:
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

        ffmpeg.sample_grain.assert_called_once_with(dvd_mkv, 55.0, hdr_transfer=None)

    def test_grain_probe_falls_back_to_format_duration(self, tmp_path: Path) -> None:
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

        ffmpeg.sample_grain.assert_called_once_with(dvd_mkv, 77.0, hdr_transfer=None)

    def test_pre_probe_raise_defaults_grainy_and_survives(self, tmp_path: Path) -> None:
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

        def _grain(path: Path, dur: float, *, hdr_transfer: str | None = None) -> list[float]:
            if path == raising_mkv:
                raise OSError("ffmpeg exploded")
            return [0.1]

        ffmpeg.sample_grain.side_effect = _grain

        screens_built: list[Any] = []

        def file_runner(factory: Callable[[], Any]) -> FileSelection:
            screens_built.append(factory())
            return FileSelection(selected=[raising_mkv, clean_mkv], sar_override=set(), grain={})

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

        assert paths == [raising_mkv, clean_mkv]
        screen = screens_built[0]
        assert isinstance(screen, FileSelectorScreen)
        assert screen._grain_defaults == {raising_mkv}

    def test_pre_probe_runtimeerror_defaults_grainy(self, tmp_path: Path) -> None:
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


class TestPlanPlainFilesGrain:
    def _sd_streams(self, height: int, transfer: str | None = None) -> dict[str, Any]:
        stream: dict[str, Any] = {"codec_type": "video", "height": height}
        if transfer is not None:
            stream["color_transfer"] = transfer
        return {
            "format": {"duration": "100.0", "size": "1000"},
            "streams": [stream],
        }

    def test_audio_only_source_shows_no_screen_and_grain_none(self, tmp_path: Path) -> None:
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
            mock_ffmpeg_cls.return_value.probe.return_value = {
                "format": {"duration": "100.0", "size": "1000"},
                "streams": [{"codec_type": "audio"}],
            }
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

    def test_sd_source_shows_screen_and_threads_grain(self, tmp_path: Path) -> None:
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
        assert call_kwargs["grain_overrides"] == {main_file: True}

    def test_sd_source_shows_screen_via_sar_and_threads_override(self, tmp_path: Path) -> None:
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
        selection = FileSelection(selected=[main_file], sar_override={main_file}, grain={})
        screens_built: list[Any] = []

        def _runner(factory: Callable[[], Any]) -> FileSelection:
            screens_built.append(factory())
            return selection

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
            patch("furnace.cli._run_screen_app", side_effect=_runner),
        ):
            mock_ffmpeg_cls.return_value.probe.return_value = self._sd_streams(480, transfer="smpte2084")
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
        assert len(screens_built) == 1
        assert screens_built[0]._sar_files == {main_file}
        mock_ffmpeg_cls.return_value.sample_grain.assert_called_once_with(
            main_file,
            100.0,
            hdr_transfer="smpte2084",
        )
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args.kwargs
        assert call_kwargs["sar_overrides"] == {main_file}

    def test_hdr_only_source_shows_the_grain_screen(self, tmp_path: Path) -> None:
        from furnace.core.models import ScanResult
        from furnace.services.analysis_pipeline import AnalysisBatchResult
        from furnace.ui.tui import FileSelection

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
        main_file = scan_result.main_file
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
            mock_ffmpeg_cls.return_value.probe.return_value = self._sd_streams(
                2160,
                transfer="smpte2084",
            )
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
        mock_ffmpeg_cls.return_value.sample_grain.assert_called_once_with(
            main_file,
            100.0,
            hdr_transfer="smpte2084",
        )
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args.kwargs
        assert call_kwargs["grain_overrides"] == {main_file: True}

    def test_sd_source_screen_dismissed_keeps_grain_none(self, tmp_path: Path) -> None:
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
            mock_analyzer_cls.return_value.analyze.return_value = AnalysisOutcome(movie, AnalyzeStatus.DONE, "summary")
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
            mock_ffmpeg_cls.return_value.sample_grain.return_value = [1.0]
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
        pipeline_results = mock_pipeline_cls.return_value.run.call_args.args[0]
        pipeline_files = {sr.main_file for sr in pipeline_results}
        assert pipeline_files == {kept_file}
        assert dropped_file not in pipeline_files
        call_kwargs = mock_planner_cls.return_value.create_plan.call_args.kwargs
        assert call_kwargs["grain_overrides"] == {kept_file: True}

    def test_all_plain_files_deselected_does_not_crash(self, tmp_path: Path) -> None:
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
