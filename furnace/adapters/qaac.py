from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class QaacAdapter:
    """Implements AacEncoder."""

    def __init__(self, qaac_path: Path) -> None:
        self._qaac = qaac_path

    def encode_aac(self, input_wav: Path, output_m4a: Path) -> int:
        """qaac64 --tvbr 91 --quality 2 --rate keep --no-delay --threading input.wav -o output.m4a"""
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
        logger.info("qaac encode_aac cmd: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(
                "qaac encode_aac failed (rc=%d): %s", result.returncode, result.stderr
            )
        return result.returncode
