from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

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


# ---------------------------------------------------------------------------
# test_scan_single_file
# ---------------------------------------------------------------------------

class TestScanSingleFile:
    def test_scan_single_video_file(self, tmp_path: Path) -> None:
        """scan() with a single video file path -> one ScanResult."""
        movie = tmp_path / "movie.mkv"
        movie.touch()
        dest = tmp_path / "output"
        dest.mkdir()

        scanner = make_scanner()
        results = scanner.scan(movie, dest)

        assert len(results) == 1
        assert results[0].main_file == movie

    def test_scan_single_non_video_file_returns_empty(self, tmp_path: Path) -> None:
        """scan() with a non-video file -> empty result."""
        txt_file = tmp_path / "readme.txt"
        txt_file.touch()
        dest = tmp_path / "output"
        dest.mkdir()

        scanner = make_scanner()
        results = scanner.scan(txt_file, dest)

        assert results == []

    def test_scan_single_mp4_file(self, tmp_path: Path) -> None:
        """scan() with a single .mp4 file -> one ScanResult."""
        movie = tmp_path / "movie.mp4"
        movie.touch()
        dest = tmp_path / "output"
        dest.mkdir()

        scanner = make_scanner()
        results = scanner.scan(movie, dest)

        assert len(results) == 1
        assert results[0].main_file == movie

    def test_scan_single_file_finds_satellites(self, tmp_path: Path) -> None:
        """scan() with a single video file also finds satellites."""
        movie = tmp_path / "movie.mkv"
        movie.touch()
        srt = tmp_path / "movie.eng.srt"
        srt.touch()
        dest = tmp_path / "output"
        dest.mkdir()

        scanner = make_scanner()
        results = scanner.scan(movie, dest)

        assert len(results) == 1
        assert srt in results[0].satellite_files


# ---------------------------------------------------------------------------
# test_scan_directory_skips_non_video
# ---------------------------------------------------------------------------

class TestScanDirectorySkipsNonVideo:
    def test_non_video_file_skipped_in_walk(self, tmp_path: Path) -> None:
        """Non-video files in directory walk are skipped (line 72)."""
        movie = tmp_path / "movie.mkv"
        movie.touch()
        txt = tmp_path / "readme.txt"
        txt.touch()
        nfo = tmp_path / "movie.nfo"
        nfo.touch()
        dest = tmp_path / "output"
        dest.mkdir()

        scanner = make_scanner()
        results = scanner.scan(tmp_path, dest)

        found = [r.main_file for r in results]
        assert movie in found
        assert txt not in found
        assert nfo not in found
        assert len(results) == 1


# ---------------------------------------------------------------------------
# test_build_output_path_relative_to_valueerror
# ---------------------------------------------------------------------------

class TestBuildOutputPathRelativeToFallback:
    def test_relative_to_valueerror_fallback(self, tmp_path: Path) -> None:
        """When source is not relative to source_root -> fallback to source.name."""
        source = Path("/completely/different/path/movie.mkv")
        source_root = tmp_path / "src"
        source_root.mkdir()
        dest = tmp_path / "out"

        scanner = make_scanner()
        output = scanner.build_output_path(source, source_root, dest, None)

        # Should fallback to source.name as relative path
        assert output.name == "movie.mkv"
        assert output.parent == dest


# ---------------------------------------------------------------------------
# test_load_names_map
# ---------------------------------------------------------------------------

class TestLoadNamesMap:
    def test_parse_entries(self, tmp_path: Path) -> None:
        """Parse rename file with valid entries."""
        names_file = tmp_path / "names.txt"
        names_file.write_text(
            "movie1.mkv = Movie One\nmovie2.mkv = Movie Two\n",
            encoding="utf-8",
        )
        result = Scanner.load_names_map(names_file)
        assert result == {"movie1.mkv": "Movie One", "movie2.mkv": "Movie Two"}

    def test_parse_comments_ignored(self, tmp_path: Path) -> None:
        """Lines starting with # are comments and ignored."""
        names_file = tmp_path / "names.txt"
        names_file.write_text(
            "# This is a comment\nmovie1.mkv = Movie One\n",
            encoding="utf-8",
        )
        result = Scanner.load_names_map(names_file)
        assert result == {"movie1.mkv": "Movie One"}

    def test_parse_blank_lines_ignored(self, tmp_path: Path) -> None:
        """Blank lines are ignored."""
        names_file = tmp_path / "names.txt"
        names_file.write_text(
            "movie1.mkv = Movie One\n\n\nmovie2.mkv = Movie Two\n",
            encoding="utf-8",
        )
        result = Scanner.load_names_map(names_file)
        assert result == {"movie1.mkv": "Movie One", "movie2.mkv": "Movie Two"}

    def test_parse_lines_without_equals_ignored(self, tmp_path: Path) -> None:
        """Lines without '=' are ignored."""
        names_file = tmp_path / "names.txt"
        names_file.write_text(
            "movie1.mkv = Movie One\nthis has no equals sign\nmovie2.mkv = Movie Two\n",
            encoding="utf-8",
        )
        result = Scanner.load_names_map(names_file)
        assert result == {"movie1.mkv": "Movie One", "movie2.mkv": "Movie Two"}

    def test_parse_value_with_equals_sign(self, tmp_path: Path) -> None:
        """Value containing '=' is handled correctly (split on first '=' only)."""
        names_file = tmp_path / "names.txt"
        names_file.write_text(
            "movie.mkv = Title = Subtitle\n",
            encoding="utf-8",
        )
        result = Scanner.load_names_map(names_file)
        assert result == {"movie.mkv": "Title = Subtitle"}

    def test_parse_mixed_content(self, tmp_path: Path) -> None:
        """Mix of entries, comments, blank lines, and lines without '='."""
        names_file = tmp_path / "names.txt"
        content = (
            "# Rename file\n"
            "\n"
            "movie1.mkv = Movie One\n"
            "no-equals-here\n"
            "\n"
            "# Another comment\n"
            "movie2.mkv = Movie Two\n"
            "   \n"
        )
        names_file.write_text(content, encoding="utf-8")
        result = Scanner.load_names_map(names_file)
        assert result == {"movie1.mkv": "Movie One", "movie2.mkv": "Movie Two"}

    def test_empty_key_or_value_ignored(self, tmp_path: Path) -> None:
        """Lines with empty key or empty value after strip are ignored."""
        names_file = tmp_path / "names.txt"
        names_file.write_text(
            " = Movie One\nmovie.mkv = \n good.mkv = Good Movie\n",
            encoding="utf-8",
        )
        result = Scanner.load_names_map(names_file)
        assert result == {"good.mkv": "Good Movie"}
