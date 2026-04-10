from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

from furnace.core.progress import ProgressSample

from ._subprocess import OutputCallback, run_tool

logger = logging.getLogger(__name__)

_MKCLEAN_PROGRESS_RE = re.compile(r"^Progress\s+(\d)/3:\s*(\d+)%\s*$")


def _parse_mkclean_progress_line(line: str) -> ProgressSample | None:
    """Parse an mkclean staged progress line into a sample.

    Format (confirmed by existing UI regex at run_tui.py:49):

        Progress 1/3:   42%
        Progress 2/3:   85%
        Progress 3/3:  100%

    Three phases, each 0-100%. Maps to a single continuous 0-100% fraction
    (same mapping the UI hack used at run_tui.py:535).
    """
    m = _MKCLEAN_PROGRESS_RE.match(line.strip())
    if not m:
        return None
    try:
        stage = int(m.group(1))
        stage_pct = int(m.group(2))
    except ValueError:
        return None
    if not 1 <= stage <= 3:
        return None
    fraction = ((stage - 1) + stage_pct / 100.0) / 3.0
    return ProgressSample(fraction=max(0.0, min(1.0, fraction)))


class MkcleanAdapter:
    """Implements Cleaner."""

    def __init__(self, mkclean_path: Path, on_output: OutputCallback = None, log_dir: Path | None = None) -> None:
        self._mkclean = mkclean_path
        self._on_output = on_output
        self._log_dir = log_dir

    def set_log_dir(self, log_dir: Path | None) -> None:
        self._log_dir = log_dir

    def clean(
        self,
        input_path: Path,
        output_path: Path,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int:
        """mkclean input.mkv output.mkv"""
        cmd = [str(self._mkclean), str(input_path), str(output_path)]
        log_path = self._log_dir / "mkclean.log" if self._log_dir else None

        def _on_progress_line(line: str) -> bool:
            sample = _parse_mkclean_progress_line(line)
            if sample is None:
                return False
            if on_progress is not None:
                on_progress(sample)
            return True

        rc, _out = run_tool(
            cmd,
            on_output=self._on_output,
            log_path=log_path,
            on_progress_line=_on_progress_line,
        )
        return rc
