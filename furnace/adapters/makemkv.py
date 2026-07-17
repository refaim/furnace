from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

from furnace.core.models import DiscTitle
from furnace.core.progress import ProgressSample

from ._subprocess import OutputCallback, run_tool

logger = logging.getLogger(__name__)


_TITLE_ADDED_RE = re.compile(r"Title #(\d+) was added \(\d+ cell\(s\), (\d+:\d{2}:\d{2})\)")


def _parse_duration(s: str) -> float:
    parts = s.split(":")
    if len(parts) == len(["H", "MM", "SS"]):
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    if len(parts) == len(["M", "SS"]):
        return int(parts[0]) * 60 + int(parts[1])
    return 0.0


_PRGV_RE = re.compile(r"^PRGV:(\d+),(\d+),(\d+)\s*$")
_PRGC_RE = re.compile(r'^PRGC:\d+,\d+,"(.*)"\s*$')

_SAVING_LABEL_TOKEN = "Saving to MKV file"  # noqa: S105 — not a password


def _parse_makemkv_progress_line(line: str) -> ProgressSample | None:
    m = _PRGV_RE.match(line.strip())
    if not m:
        return None
    current = int(m.group(1))
    max_val = int(m.group(3))
    if max_val == 0:
        return None
    return ProgressSample(fraction=current / max_val)


def _is_saving_prgc(line: str) -> bool:
    m = _PRGC_RE.match(line.strip())
    if not m:
        return False
    return _SAVING_LABEL_TOKEN in m.group(1)


class MakemkvAdapter:
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
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)

        titles = self.list_titles(disc_path)
        index = None
        for i, t in enumerate(titles):
            if t.number == title_num:
                index = i
                break
        if index is None:
            raise RuntimeError(f"Title {title_num} not found in makemkvcon listing for {disc_path}")

        before = set(output_dir.iterdir())

        cmd = [
            str(self._makemkvcon),
            "-r",
            "--progress=-same",
            "--noscan",
            "mkv",
            f"file:{disc_path}",
            str(index),
            str(output_dir),
        ]

        saving_phase = [False]

        def _on_progress_line(line: str) -> bool:
            if _is_saving_prgc(line):
                saving_phase[0] = True
                return False
            sample = _parse_makemkv_progress_line(line)
            if sample is None:
                return False
            if saving_phase[0] and on_progress is not None:
                on_progress(sample)
            return True

        rc, _output = run_tool(
            cmd,
            on_output=self._on_output,
            on_progress_line=_on_progress_line,
            log_path=self._log_path(f"demux_t{title_num}"),
        )
        if rc != 0:
            raise RuntimeError(f"makemkvcon demux failed for {disc_path} title {title_num} (rc={rc})")

        after = set(output_dir.iterdir())
        new_files = sorted(p for p in (after - before) if p.is_file() and p.suffix.lower() == ".mkv")
        if not new_files:
            raise RuntimeError(f"makemkvcon produced no MKV files for {disc_path} title {title_num}")
        return new_files

    @staticmethod
    def _parse_info_output(output: str) -> list[DiscTitle]:
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
