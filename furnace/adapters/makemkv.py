from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

from furnace.core.models import DiscTitle
from furnace.core.progress import ProgressSample

from ._subprocess import OutputCallback, run_tool

logger = logging.getLogger(__name__)


# "Title #4 was added (13 cell(s), 1:12:32)"
_TITLE_ADDED_RE = re.compile(r"Title #(\d+) was added \(\d+ cell\(s\), (\d+:\d{2}:\d{2})\)")


def _parse_duration(s: str) -> float:
    """Parse 'H:MM:SS' into total seconds."""
    parts = s.split(":")
    if len(parts) == len(["H", "MM", "SS"]):
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    if len(parts) == len(["M", "SS"]):
        return int(parts[0]) * 60 + int(parts[1])
    return 0.0


class MakemkvAdapter:
    """Implements DiscDemuxerPort for DVD via makemkvcon."""

    def __init__(
        self,
        makemkvcon_path: Path,
        on_output: OutputCallback = None,
        log_dir: Path | None = None,
    ) -> None:
        self._makemkvcon = makemkvcon_path
        self._on_output = on_output
        self._log_dir = log_dir

    def set_log_dir(self, log_dir: Path | None) -> None:
        self._log_dir = log_dir

    def _log_path(self, label: str) -> Path | None:
        if self._log_dir is None:
            return None
        return self._log_dir / f"makemkv_{label}.log"

    def list_titles(self, disc_path: Path) -> list[DiscTitle]:
        """Run makemkvcon info to list DVD titles."""
        cmd = [
            str(self._makemkvcon),
            "--noscan",
            "info",
            f"file:{disc_path}",
        ]
        rc, output = run_tool(cmd, on_output=self._on_output, log_path=self._log_path("list_titles"))
        if rc != 0:
            raise RuntimeError(f"makemkvcon info failed for {disc_path} (rc={rc})")
        return self._parse_info_output(output)

    def demux_title(
        self,
        disc_path: Path,
        title_num: int,
        output_dir: Path,
        _on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> list[Path]:
        """Demux one DVD title to MKV.

        makemkvcon uses 0-based index from the "added" titles list.
        The title_num we receive is the original DVD title number;
        we need to map it to the 0-based index.

        Progress reporting is not implemented for makemkvcon yet: lines
        flow to `on_output` for the log widget, but no structured samples
        are emitted. `_on_progress` is accepted for Protocol conformance
        with DiscDemuxerPort and silently ignored.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # First, list titles to build the mapping from title number to 0-based index
        titles = self.list_titles(disc_path)
        index = None
        for i, t in enumerate(titles):
            if t.number == title_num:
                index = i
                break
        if index is None:
            raise RuntimeError(f"Title {title_num} not found in makemkvcon listing for {disc_path}")

        # Snapshot files before demux
        before = set(output_dir.iterdir())

        cmd = [
            str(self._makemkvcon),
            "--noscan",
            "mkv",
            f"file:{disc_path}",
            str(index),
            str(output_dir),
        ]

        rc, _output = run_tool(
            cmd,
            on_output=self._on_output,
            log_path=self._log_path(f"demux_t{title_num}"),
        )
        if rc != 0:
            raise RuntimeError(f"makemkvcon demux failed for {disc_path} title {title_num} (rc={rc})")

        # Find newly created MKV files
        after = set(output_dir.iterdir())
        new_files = sorted(p for p in (after - before) if p.is_file() and p.suffix.lower() == ".mkv")
        if not new_files:
            raise RuntimeError(f"makemkvcon produced no MKV files for {disc_path} title {title_num}")
        return new_files

    @staticmethod
    def _parse_info_output(output: str) -> list[DiscTitle]:
        """Parse makemkvcon info output.

        Looks for lines like: "Title #4 was added (13 cell(s), 1:12:32)"
        """
        results: list[DiscTitle] = []
        for line in output.splitlines():
            m = _TITLE_ADDED_RE.search(line)
            if not m:
                continue
            number = int(m.group(1))
            duration_str = m.group(2)
            duration_s = _parse_duration(duration_str)
            results.append(
                DiscTitle(
                    number=number,
                    duration_s=duration_s,
                    raw_label=line.strip(),
                )
            )
        return results
