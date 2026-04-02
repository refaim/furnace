from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class MkcleanAdapter:
    """Implements Cleaner."""

    def __init__(self, mkclean_path: Path) -> None:
        self._mkclean = mkclean_path

    def clean(self, input_path: Path, output_path: Path) -> int:
        """mkclean input.mkv output.mkv"""
        cmd = [
            str(self._mkclean),
            str(input_path),
            str(output_path),
        ]
        logger.info("mkclean clean cmd: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(
                "mkclean failed (rc=%d): %s", result.returncode, result.stderr
            )
        return result.returncode
