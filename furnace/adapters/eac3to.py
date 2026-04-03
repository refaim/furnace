from __future__ import annotations

import logging
from pathlib import Path

from ._subprocess import OutputCallback, run_tool

logger = logging.getLogger(__name__)


class Eac3toAdapter:
    """Implements AudioDecoder (denormalize, decode_lossless)."""

    def __init__(self, eac3to_path: Path, on_output: OutputCallback = None, log_dir: Path | None = None) -> None:
        self._eac3to = eac3to_path
        self._on_output = on_output
        self._log_dir = log_dir

    def set_log_dir(self, log_dir: Path | None) -> None:
        self._log_dir = log_dir

    def _log_path(self, label: str) -> Path | None:
        if self._log_dir is None:
            return None
        return self._log_dir / f"eac3to_{label}.log"

    def _delay_arg(self, delay_ms: int) -> list[str]:
        """Return eac3to delay argument as list, or empty list if delay is 0."""
        if delay_ms == 0:
            return []
        if delay_ms > 0:
            return [f"+{delay_ms}ms"]
        return [f"{delay_ms}ms"]

    def denormalize(self, input_path: Path, output_path: Path, delay_ms: int) -> int:
        """eac3to input output -removeDialnorm -progressnumbers -nolog [+Xms/-Xms]"""
        cmd = [
            str(self._eac3to),
            str(input_path),
            str(output_path),
            "-removeDialnorm",
            "-progressnumbers",
            *self._delay_arg(delay_ms),
        ]
        rc, _out = run_tool(cmd, on_output=self._on_output, log_path=self._log_path("denorm"))
        return rc

    def decode_lossless(self, input_path: Path, output_path: Path, delay_ms: int) -> int:
        """eac3to input output.wav -removeDialnorm -progressnumbers -nolog [+Xms/-Xms]"""
        cmd = [
            str(self._eac3to),
            str(input_path),
            str(output_path),
            "-removeDialnorm",
            "-progressnumbers",
            *self._delay_arg(delay_ms),
        ]
        rc, _out = run_tool(cmd, on_output=self._on_output, log_path=self._log_path("decode"))
        return rc
