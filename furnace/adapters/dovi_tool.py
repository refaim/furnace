from __future__ import annotations

import logging
from pathlib import Path

from furnace.core.models import DvMode

from ._subprocess import OutputCallback, run_tool

logger = logging.getLogger(__name__)


class DoviToolAdapter:
    """Implements DoviProcessor port via dovi_tool CLI."""

    def __init__(
        self,
        dovi_tool_path: Path,
        on_output: OutputCallback = None,
        log_dir: Path | None = None,
    ) -> None:
        self._dovi_tool = dovi_tool_path
        self._on_output = on_output
        self._log_dir = log_dir

    def set_log_dir(self, log_dir: Path | None) -> None:
        self._log_dir = log_dir

    def _build_extract_cmd(
        self,
        input_path: Path,
        output_rpu: Path,
        mode: DvMode,
    ) -> list[str | Path]:
        cmd: list[str | Path] = [self._dovi_tool]
        if mode == DvMode.TO_8_1:
            cmd += ["-m", "2"]
        cmd += ["extract-rpu", input_path, "-o", output_rpu]
        return cmd

    def extract_rpu(
        self,
        input_path: Path,
        output_rpu: Path,
        mode: DvMode,
    ) -> int:
        """Extract RPU from HEVC stream."""
        cmd = self._build_extract_cmd(input_path, output_rpu, mode)
        logger.debug("dovi_tool cmd: %s", " ".join(str(c) for c in cmd))
        log_path = self._log_dir / "dovi_tool_extract.log" if self._log_dir else None
        rc, _out = run_tool(cmd, on_output=self._on_output, log_path=log_path)
        return rc
