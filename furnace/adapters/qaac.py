from __future__ import annotations

import logging
from pathlib import Path

from ._subprocess import OutputCallback, run_tool

logger = logging.getLogger(__name__)


class QaacAdapter:
    """Implements AacEncoder."""

    def __init__(self, qaac_path: Path, on_output: OutputCallback = None, log_dir: Path | None = None) -> None:
        self._qaac = qaac_path
        self._on_output = on_output
        self._log_dir = log_dir

    def set_log_dir(self, log_dir: Path | None) -> None:
        self._log_dir = log_dir

    def encode_aac(self, input_wav: Path, output_m4a: Path) -> int:
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
        log_path = self._log_dir / "qaac.log" if self._log_dir else None
        rc, _out = run_tool(cmd, on_output=self._on_output, log_path=log_path)
        return rc
