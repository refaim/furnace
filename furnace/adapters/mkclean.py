from __future__ import annotations

import logging
from pathlib import Path

from ._subprocess import OutputCallback, run_tool

logger = logging.getLogger(__name__)


class MkcleanAdapter:
    """Implements Cleaner."""

    def __init__(self, mkclean_path: Path, on_output: OutputCallback = None, log_dir: Path | None = None) -> None:
        self._mkclean = mkclean_path
        self._on_output = on_output
        self._log_dir = log_dir

    def set_log_dir(self, log_dir: Path | None) -> None:
        self._log_dir = log_dir

    def clean(self, input_path: Path, output_path: Path) -> int:
        """mkclean input.mkv output.mkv"""
        cmd = [str(self._mkclean), str(input_path), str(output_path)]
        log_path = self._log_dir / "mkclean.log" if self._log_dir else None
        rc, _out = run_tool(cmd, on_output=self._on_output, log_path=log_path)
        return rc
