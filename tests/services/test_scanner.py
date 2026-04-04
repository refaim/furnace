from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from furnace.services.scanner import Scanner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_scanner() -> Scanner:
    prober = MagicMock()
    return Scanner(prober=prober)


# ---------------------------------------------------------------------------
# test_scanner_finds_satellites
# ---------------------------------------------------------------------------

class TestScannerFindsSatellites:
    def test_scanner_finds_satellites(self, tmp_path: Path) -> None:
        """movie.mkv -> finds movie.rus.srt and movie.eng.ac3 via startswith."""
        movie = tmp_path / "movie.mkv"
        rus_srt = tmp_path / "movie.rus.srt"
        eng_ac3 = tmp_path / "movie.eng.ac3"
        other_txt = tmp_path / "movie.txt"
        other_mkv = tmp_path / "other.mkv"

        movie.touch()
        rus_srt.touch()
        eng_ac3.touch()
        other_txt.touch()
        other_mkv.touch()

        scanner = make_scanner()
        satellites = scanner.find_satellites(movie)

        assert rus_srt in satellites
        assert eng_ac3 in satellites
        # .txt is not a satellite extension
        assert other_txt not in satellites
        # other.mkv doesn't start with "movie"
        assert other_mkv not in satellites

    def test_scanner_finds_dotted_satellites(self, tmp_path: Path) -> None:
        """movie.mkv -> finds movie.forced.rus.srt via startswith."""
        movie = tmp_path / "movie.mkv"
        forced_srt = tmp_path / "movie.forced.rus.srt"
        unrelated = tmp_path / "other.srt"

        movie.touch()
        forced_srt.touch()
        unrelated.touch()

        scanner = make_scanner()
        satellites = scanner.find_satellites(movie)

        assert forced_srt in satellites
        assert unrelated not in satellites

    def test_scanner_ignores_non_media(self, tmp_path: Path) -> None:
        """.txt and .nfo files are not included as satellites."""
        movie = tmp_path / "movie.mkv"
        txt_file = tmp_path / "movie.txt"
        nfo_file = tmp_path / "movie.nfo"
        srt_file = tmp_path / "movie.rus.srt"

        movie.touch()
        txt_file.touch()
        nfo_file.touch()
        srt_file.touch()

        scanner = make_scanner()
        satellites = scanner.find_satellites(movie)

        assert txt_file not in satellites
        assert nfo_file not in satellites
        assert srt_file in satellites


# ---------------------------------------------------------------------------
# test_scanner_names_map
# ---------------------------------------------------------------------------

class TestScannerNamesMap:
    def test_scanner_names_map(self, tmp_path: Path) -> None:
        """Rename via names_map: old filename stem -> new stem."""
        source = tmp_path / "Movie.Title.2020.mkv"
        source.touch()
        dest = tmp_path / "output"
        dest.mkdir()

        names_map = {"Movie.Title.2020.mkv": "Movie Title (2020)"}

        scanner = make_scanner()
        output_path = scanner.build_output_path(source, tmp_path, dest, names_map)

        assert output_path.name == "Movie Title (2020).mkv"
        assert output_path.parent == dest

    def test_scanner_names_map_no_match(self, tmp_path: Path) -> None:
        """Without matching names_map entry, original stem is used (cleaned)."""
        source = tmp_path / "movie.mkv"
        source.touch()
        dest = tmp_path / "output"
        dest.mkdir()

        names_map = {"other.mkv": "Other Movie"}

        scanner = make_scanner()
        output_path = scanner.build_output_path(source, tmp_path, dest, names_map)

        assert output_path.name == "movie.mkv"


# ---------------------------------------------------------------------------
# test_clean_filename
# ---------------------------------------------------------------------------

class TestCleanFilename:
    def test_clean_filename_removes_forbidden_chars(self) -> None:
        """Windows forbidden chars are removed from filename."""
        # < > / : | ? * are removed
        result = Scanner.clean_filename("Movie<Title>2020")
        assert "<" not in result
        assert ">" not in result

    def test_clean_filename_converts_double_quotes(self) -> None:
        """Double quotes are converted to single quotes."""
        result = Scanner.clean_filename('Movie "Title" 2020')
        assert '"' not in result
        assert "'" in result

    def test_clean_filename_removes_trailing_dot(self) -> None:
        """Trailing dots are removed."""
        result = Scanner.clean_filename("Movie Title.")
        assert not result.endswith(".")
        assert result == "Movie Title"

    def test_clean_filename_removes_colon(self) -> None:
        """Colon (forbidden on Windows) is removed."""
        result = Scanner.clean_filename("Movie: The Return")
        assert ":" not in result

    def test_clean_filename_plain_name_unchanged(self) -> None:
        """Plain ASCII name without forbidden chars is unchanged."""
        result = Scanner.clean_filename("The Dark Knight")
        assert result == "The Dark Knight"


# ---------------------------------------------------------------------------
# test_build_output_path
# ---------------------------------------------------------------------------

class TestBuildOutputPath:
    def test_build_output_path_mirrors_structure(self, tmp_path: Path) -> None:
        """Mirror directory structure into destination."""
        source_root = tmp_path / "src"
        subdir = source_root / "Action"
        subdir.mkdir(parents=True)
        source = subdir / "movie.mkv"
        source.touch()
        dest = tmp_path / "out"

        scanner = make_scanner()
        output = scanner.build_output_path(source, source_root, dest, None)

        assert output.parent == dest / "Action"
        assert output.name == "movie.mkv"

    def test_build_output_path_forces_mkv_extension(self, tmp_path: Path) -> None:
        """Output path always has .mkv extension."""
        source_root = tmp_path
        source = tmp_path / "movie.mp4"
        source.touch()
        dest = tmp_path / "out"

        scanner = make_scanner()
        output = scanner.build_output_path(source, source_root, dest, None)

        assert output.suffix == ".mkv"

    def test_build_output_path_rename_and_clean(self, tmp_path: Path) -> None:
        """names_map rename + forbidden char removal applied together."""
        source_root = tmp_path / "src"
        source_root.mkdir()
        source = source_root / "movie.mkv"
        source.touch()
        dest = tmp_path / "out"

        # names_map value with forbidden chars
        names_map = {"movie.mkv": 'Movie: "The" Return'}

        scanner = make_scanner()
        output = scanner.build_output_path(source, source_root, dest, names_map)

        # Colon and double quotes should be cleaned
        assert ":" not in output.name
        assert '"' not in output.name
        assert output.suffix == ".mkv"

    def test_build_output_path_flat_source(self, tmp_path: Path) -> None:
        """When source == source_root (single file scan), output goes directly in dest."""
        source = tmp_path / "movie.mkv"
        source.touch()
        dest = tmp_path / "out"
        dest.mkdir()

        scanner = make_scanner()
        output = scanner.build_output_path(source, tmp_path, dest, None)

        assert output.parent == dest


# ---------------------------------------------------------------------------
# test_scanner_ignores_demux_dir
# ---------------------------------------------------------------------------

class TestScannerIgnoresDemuxDir:
    def test_scanner_ignores_furnace_demux(self, tmp_path: Path) -> None:
        """Scanner skips .furnace_demux directory."""
        movie = tmp_path / "movie.mkv"
        movie.touch()
        demux_dir = tmp_path / ".furnace_demux"
        demux_dir.mkdir()
        demuxed = demux_dir / "demuxed.mkv"
        demuxed.touch()

        dest = tmp_path / "output"
        dest.mkdir()

        scanner = make_scanner()
        results = scanner.scan(tmp_path, dest)

        found_files = [r.main_file for r in results]
        assert movie in found_files
        assert demuxed not in found_files
