from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

from furnace.core.progress import ProgressSample

from ._subprocess import OutputCallback, run_tool

logger = logging.getLogger(__name__)


_QAAC_PROGRESS_RE = re.compile(r"^\[(\d+(?:\.\d+)?)%\]")
_QAAC_SPEED_RE = re.compile(r"\((\d+(?:\.\d+)?)x\)")


def _parse_qaac_progress_line(line: str) -> ProgressSample | None:
    """Parse a qaac stderr progress line into a sample.

    Format (confirmed by existing UI regex at run_tui.py:50):

        [42.5%] 0:30/1:43:01.600 (30.5x), ETA 3:20

    Captures: leading percent in brackets, optional `(NNx)` speed multiplier.
    """
    m_pct = _QAAC_PROGRESS_RE.match(line.strip())
    if not m_pct:
        return None
    try:
        fraction = float(m_pct.group(1)) / 100.0
    except ValueError:
        return None
    speed: float | None = None
    m_speed = _QAAC_SPEED_RE.search(line)
    if m_speed:
        try:
            speed = float(m_speed.group(1))
        except ValueError:
            speed = None
    return ProgressSample(fraction=fraction, speed=speed)


class QaacAdapter:
    """Implements AacEncoder."""

    def __init__(self, qaac_path: Path, on_output: OutputCallback = None, log_dir: Path | None = None) -> None:
        self._qaac = qaac_path
        self._on_output = on_output
        self._log_dir = log_dir

    def set_log_dir(self, log_dir: Path | None) -> None:
        self._log_dir = log_dir

    def encode_aac(
        self,
        input_wav: Path,
        output_m4a: Path,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int:
        """qaac64 --tvbr 91 --quality 2 --rate keep --no-delay --threading input -o output"""
        cmd = [
            str(self._qaac),
            "--tvbr", "91",
            "--quality", "2",
            "--rate", "keep",
            "--no-delay",
            "--threading",
            str(input_wav),
            "-o", str(output_m4a),
        ]

        def _on_progress_line(line: str) -> bool:
            sample = _parse_qaac_progress_line(line)
            if sample is None:
                return False
            if on_progress is not None:
                on_progress(sample)
            return True

        log_path = self._log_dir / "qaac.log" if self._log_dir else None
        rc, _out = run_tool(
            cmd,
            on_output=self._on_output,
            on_progress_line=_on_progress_line,
            log_path=log_path,
        )
        return rc
