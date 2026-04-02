from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class Eac3toAdapter:
    """Implements AudioDecoder (denormalize, decode_lossless)."""

    def __init__(self, eac3to_path: Path) -> None:
        self._eac3to = eac3to_path

    def _delay_arg(self, delay_ms: int) -> list[str]:
        """Return eac3to delay argument as list, or empty list if delay is 0."""
        if delay_ms == 0:
            return []
        if delay_ms > 0:
            return [f"+{delay_ms}ms"]
        return [f"{delay_ms}ms"]  # negative: already has minus sign

    def denormalize(self, input_path: Path, output_path: Path, delay_ms: int) -> int:
        """eac3to input output -removeDialnorm [+Xms/-Xms]
        For AC3/EAC3/DTS core -- denormalization.
        Explicit -removeDialnorm overrides eac3to.ini which may have -keepDialnorm."""
        cmd = [
            str(self._eac3to),
            str(input_path),
            str(output_path),
            "-removeDialnorm",
            *self._delay_arg(delay_ms),
        ]
        logger.info("eac3to denormalize cmd: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(
                "eac3to denormalize failed (rc=%d): %s", result.returncode, result.stderr
            )
        return result.returncode

    def decode_lossless(self, input_path: Path, output_path: Path, delay_ms: int) -> int:
        """eac3to input output.wav -removeDialnorm [+Xms/-Xms]
        For DTS-HD MA, TrueHD, FLAC, PCM -- decode to WAV with dialnorm removal.
        Explicit -removeDialnorm overrides eac3to.ini which may have -keepDialnorm."""
        cmd = [
            str(self._eac3to),
            str(input_path),
            str(output_path),
            "-removeDialnorm",
            *self._delay_arg(delay_ms),
        ]
        logger.info("eac3to decode_lossless cmd: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(
                "eac3to decode_lossless failed (rc=%d): %s", result.returncode, result.stderr
            )
        return result.returncode
