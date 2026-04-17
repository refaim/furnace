from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from furnace.adapters.eac3to import (
    Eac3toAdapter,
    _ext_for_track,
    _is_eac3to_progress_line,
    _parse_duration,
    _parse_eac3to_progress_line,
)
from furnace.core.progress import ProgressSample


class TestParseEac3toProgressLine:
    def test_process_zero(self) -> None:
        assert _parse_eac3to_progress_line("process: 0%") == ProgressSample(fraction=0.0)

    def test_process_fifty(self) -> None:
        assert _parse_eac3to_progress_line("process: 50%") == ProgressSample(fraction=0.5)

    def test_process_hundred(self) -> None:
        assert _parse_eac3to_progress_line("process: 100%") == ProgressSample(fraction=1.0)

    def test_missing_percent(self) -> None:
        assert _parse_eac3to_progress_line("process: 50") is None

    def test_analyze_not_captured(self) -> None:
        # Only `process:` is captured; executor handles phase transitions explicitly
        assert _parse_eac3to_progress_line("analyze: 25%") is None

    def test_plain_text(self) -> None:
        assert _parse_eac3to_progress_line("Reading file...") is None

    def test_trailing_whitespace(self) -> None:
        assert _parse_eac3to_progress_line("process: 42%  ") == ProgressSample(fraction=0.42)


class TestIsEac3toProgressLine:
    def test_process_line(self) -> None:
        assert _is_eac3to_progress_line("process: 50%") is True

    def test_analyze_line(self) -> None:
        assert _is_eac3to_progress_line("analyze: 25%") is True

    def test_plain_text(self) -> None:
        assert _is_eac3to_progress_line("plain text") is False


class TestExtForTrack:
    def test_truehd_ac3(self) -> None:
        assert _ext_for_track("TrueHD/AC3") == ".thd"

    def test_unknown_codec(self) -> None:
        assert _ext_for_track("unknown") == ".bin"

    def test_pgs_subtitles(self) -> None:
        assert _ext_for_track("PGS subtitles") == ".sup"

    def test_dts_hd_master(self) -> None:
        # "dts-hd" starts with "dts" in the ordered map, so matches ".dts" first
        assert _ext_for_track("DTS-HD Master Audio, [eng], 5.1 channels") == ".dts"

    def test_chapters(self) -> None:
        assert _ext_for_track("Chapters, 5 chapters") == ".txt"

    def test_pgs_not_at_start(self) -> None:
        """PGS in middle of description (not matching map key at start)."""
        assert _ext_for_track("Subtitle PGS, [eng]") == ".sup"

    def test_chapters_not_at_start(self) -> None:
        """Chapters in middle of description."""
        assert _ext_for_track("Some chapters data") == ".txt"

    def test_ac3(self) -> None:
        assert _ext_for_track("AC3, [rus], 5.1 channels") == ".ac3"

    def test_h264(self) -> None:
        assert _ext_for_track("h264/AVC, 1080p24") == ".mkv"


class TestParseTrackListing:
    def test_multi_line_parse(self) -> None:
        output = (
            "1: h264/AVC, 1080p24 /1.001 (16:9)\n"
            "2: DTS-HD Master Audio, [eng], 5.1 channels, 48kHz\n"
            "3: AC3, [rus], 2.0 channels, 48kHz\n"
            "4: PGS subtitles, [eng]\n"
            "5: Chapters, 12 chapters\n"
        )
        tracks = Eac3toAdapter._parse_track_listing(output)
        assert len(tracks) == 5
        assert tracks[0].number == 1
        assert tracks[0].extension == ".mkv"
        assert tracks[0].language is None
        assert tracks[1].number == 2
        assert tracks[1].extension == ".dts"  # "dts" matches first in ordered map
        assert tracks[1].language == "eng"
        assert tracks[2].number == 3
        assert tracks[2].extension == ".ac3"
        assert tracks[2].language == "rus"
        assert tracks[3].number == 4
        assert tracks[3].extension == ".sup"
        assert tracks[3].language == "eng"
        assert tracks[4].number == 5
        assert tracks[4].extension == ".txt"
        assert tracks[4].language is None

    def test_empty_output(self) -> None:
        assert Eac3toAdapter._parse_track_listing("") == []

    def test_non_matching_lines_skipped(self) -> None:
        output = (
            "M2TS, 1 video track\n"
            "  duration: 1:23:45\n"
            "3: AC3, [eng], 5.1 channels\n"
        )
        tracks = Eac3toAdapter._parse_track_listing(output)
        assert len(tracks) == 1
        assert tracks[0].number == 3


class TestParseDurationEac3to:
    def test_garbage_returns_zero(self) -> None:
        assert _parse_duration("garbage") == 0.0

    def test_hms(self) -> None:
        assert _parse_duration("1:23:45") == pytest.approx(5025.0)

    def test_ms(self) -> None:
        assert _parse_duration("5:30") == pytest.approx(330.0)


class TestDenormalize:
    def test_denormalize_cmd(self) -> None:
        captured: list[str] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            captured.extend(str(c) for c in cmd)
            return 0, ""

        adapter = Eac3toAdapter(Path("eac3to.exe"))
        with patch("furnace.adapters.eac3to.run_tool", side_effect=fake_run_tool):
            rc = adapter.denormalize(Path("/src/audio.ac3"), Path("/out/audio.ac3"), delay_ms=0)
        assert rc == 0
        assert "-removeDialnorm" in captured
        assert "-progressnumbers" in captured

    def test_denormalize_with_delay(self) -> None:
        captured: list[str] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            captured.extend(str(c) for c in cmd)
            return 0, ""

        adapter = Eac3toAdapter(Path("eac3to.exe"))
        with patch("furnace.adapters.eac3to.run_tool", side_effect=fake_run_tool):
            rc = adapter.denormalize(Path("/src/audio.ac3"), Path("/out/audio.ac3"), delay_ms=50)
        assert rc == 0
        assert "+50ms" in captured

    def test_denormalize_progress(self) -> None:
        """Progress callback is wired through _run."""
        samples: list[ProgressSample] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                on_progress_line("process: 50%")
            return 0, ""

        adapter = Eac3toAdapter(Path("eac3to.exe"))
        with patch("furnace.adapters.eac3to.run_tool", side_effect=fake_run_tool):
            adapter.denormalize(Path("/src/a.ac3"), Path("/out/a.ac3"), delay_ms=0, on_progress=samples.append)
        assert len(samples) == 1

    def test_denormalize_analyze_line_suppressed(self) -> None:
        """analyze: lines are suppressed (return True) but don't emit samples."""
        samples: list[ProgressSample] = []
        progress_returns: list[bool] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                progress_returns.append(on_progress_line("analyze: 30%"))
            return 0, ""

        adapter = Eac3toAdapter(Path("eac3to.exe"))
        with patch("furnace.adapters.eac3to.run_tool", side_effect=fake_run_tool):
            adapter.denormalize(Path("/src/a.ac3"), Path("/out/a.ac3"), delay_ms=0, on_progress=samples.append)
        # analyze lines are consumed (return True) but no sample emitted
        assert progress_returns == [True]
        assert len(samples) == 0

    def test_non_progress_line_not_consumed(self) -> None:
        """Plain text lines are not consumed by the progress closure."""
        progress_returns: list[bool] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                progress_returns.append(on_progress_line("Reading file..."))
            return 0, ""

        adapter = Eac3toAdapter(Path("eac3to.exe"))
        with patch("furnace.adapters.eac3to.run_tool", side_effect=fake_run_tool):
            adapter.denormalize(Path("/src/a.ac3"), Path("/out/a.ac3"), delay_ms=0)
        assert progress_returns == [False]


class TestListTitlesError:
    def test_list_titles_rc_nonzero_raises(self) -> None:
        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            return 1, "error output"

        adapter = Eac3toAdapter(Path("eac3to.exe"))
        with patch("furnace.adapters.eac3to.run_tool", side_effect=fake_run_tool):
            with pytest.raises(RuntimeError, match="eac3to listing failed"):
                adapter.list_titles(Path("/disc/BDMV"))

    def test_list_titles_success(self) -> None:
        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            return 0, "1) 00800.mpls, 1:45:23\n"

        adapter = Eac3toAdapter(Path("eac3to.exe"))
        with patch("furnace.adapters.eac3to.run_tool", side_effect=fake_run_tool):
            titles = adapter.list_titles(Path("/disc/BDMV"))
        assert len(titles) == 1
        assert titles[0].number == 1


class TestDemuxTitleError:
    def test_demux_title_rc_nonzero_raises(self, tmp_path: Path) -> None:
        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            return 1, "error"

        adapter = Eac3toAdapter(Path("eac3to.exe"))
        output_dir = tmp_path / "out"
        with patch("furnace.adapters.eac3to.run_tool", side_effect=fake_run_tool):
            with pytest.raises(RuntimeError, match="eac3to demux failed"):
                adapter.demux_title(Path("/disc/BDMV"), title_num=1, output_dir=output_dir)


class TestEac3toSetLogDir:
    def test_set_log_dir(self, tmp_path: Path) -> None:
        adapter = Eac3toAdapter(Path("eac3to.exe"))
        adapter.set_log_dir(tmp_path)
        assert adapter._log_dir == tmp_path

    def test_log_path_helper(self, tmp_path: Path) -> None:
        adapter = Eac3toAdapter(Path("eac3to.exe"), log_dir=tmp_path)
        assert adapter._log_path("test") == tmp_path / "eac3to_test.log"

    def test_log_path_none(self) -> None:
        adapter = Eac3toAdapter(Path("eac3to.exe"))
        assert adapter._log_path("test") is None
