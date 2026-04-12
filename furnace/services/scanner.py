from __future__ import annotations

import logging
from pathlib import Path

from furnace.core.models import ScanResult
from furnace.core.ports import Prober

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS: set[str] = {
    ".mkv",
    ".avi",
    ".mp4",
    ".m4v",
    ".mov",
    ".wmv",
    ".flv",
    ".ts",
    ".mpg",
    ".mpeg",
}
SATELLITE_EXTENSIONS: set[str] = {
    ".srt",
    ".ass",
    ".ssa",
    ".ac3",
    ".dts",
    ".eac3",
    ".flac",
    ".m4a",
    ".mp3",
    ".wav",
    ".sup",
}
WINDOWS_FORBIDDEN_CHARS: str = '<>/:"|?*'


class Scanner:
    def __init__(self, prober: Prober) -> None:
        self._prober = prober

    def scan(
        self,
        source: Path,
        dest: Path,
        names_map: dict[str, str] | None = None,
    ) -> list[ScanResult]:
        """Recursive walk. For each video file: find satellites, build output path, create ScanResult."""
        results: list[ScanResult] = []

        if source.is_file():
            if source.suffix.lower() in VIDEO_EXTENSIONS:
                satellites = self.find_satellites(source)
                output_path = self.build_output_path(source, source.parent, dest, names_map)
                results.append(
                    ScanResult(
                        main_file=source,
                        satellite_files=satellites,
                        output_path=output_path,
                    )
                )
            return results

        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            # Skip .furnace_demux directory
            if ".furnace_demux" in path.parts:
                continue
            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            satellites = self.find_satellites(path)
            output_path = self.build_output_path(path, source, dest, names_map)
            results.append(
                ScanResult(
                    main_file=path,
                    satellite_files=satellites,
                    output_path=output_path,
                )
            )
            logger.debug("Scanned %s -> %s (%d satellites)", path, output_path, len(satellites))

        logger.debug("Scan complete: %d video files found in %s", len(results), source)
        return results

    def find_satellites(self, video_path: Path) -> list[Path]:
        """Same directory, filename startswith(video_stem), extension in SATELLITE_EXTENSIONS."""
        stem = video_path.stem
        directory = video_path.parent
        satellites: list[Path] = []
        for candidate in sorted(directory.iterdir()):
            if not candidate.is_file():
                continue
            if candidate == video_path:
                continue
            if candidate.suffix.lower() not in SATELLITE_EXTENSIONS:
                continue
            if candidate.name.startswith(stem):
                satellites.append(candidate)
        return satellites

    @staticmethod
    def clean_filename(name: str) -> str:
        """Remove Windows forbidden chars, double quotes -> single, trailing dot removed."""
        result = []
        for ch in name:
            if ch in WINDOWS_FORBIDDEN_CHARS:
                if ch == '"':
                    result.append("'")
                # skip other forbidden chars
            else:
                result.append(ch)
        return "".join(result).rstrip(".")

    @staticmethod
    def build_output_path(
        source: Path,
        source_root: Path,
        dest_root: Path,
        names_map: dict[str, str] | None,
    ) -> Path:
        """Mirror directory structure + rename + clean."""
        # Get relative path from source root
        try:
            relative = source.relative_to(source_root)
        except ValueError:
            relative = Path(source.name)

        # Apply names_map rename to the filename stem (without extension)
        original_name = source.name
        new_stem: str | None = None
        if names_map:
            new_stem = names_map.get(original_name)

        if new_stem is not None:
            # names_map provides the full new name (without extension implied)
            # The map value is the new stem (no extension)
            clean_stem = Scanner.clean_filename(new_stem)
            new_filename = clean_stem + ".mkv"
        else:
            # Keep original stem, force .mkv extension, clean the name
            clean_stem = Scanner.clean_filename(source.stem)
            new_filename = clean_stem + ".mkv"

        # Mirror directory structure
        relative_dir = relative.parent
        return dest_root / relative_dir / new_filename

    @staticmethod
    def load_names_map(path: Path) -> dict[str, str]:
        """Parse rename file: 'old.mkv = New Name' format."""
        names_map: dict[str, str] = {}
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                parts = line.split("=", 1)
                old_name = parts[0].strip()
                new_name = parts[1].strip()
                if old_name and new_name:
                    names_map[old_name] = new_name
        return names_map
