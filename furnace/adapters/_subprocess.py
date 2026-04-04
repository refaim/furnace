"""Shared subprocess runner with real-time stdout+stderr streaming and per-tool log files."""
from __future__ import annotations

import logging
import subprocess
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO

logger = logging.getLogger(__name__)

# Type for output callback: receives one line at a time
OutputCallback = Callable[[str], None] | None


def run_tool(
    cmd: Sequence[str | Path],
    on_output: OutputCallback = None,
    log_path: Path | None = None,
    cwd: Path | None = None,
) -> tuple[int, str]:
    """Run a subprocess, streaming both stdout and stderr to callback in real-time.

    Args:
        cmd: Command and arguments.
        on_output: Callback receiving each output line for live display.
        log_path: If provided, write full command + all output to this file.

    Returns (return_code, combined_output_text).
    """
    str_cmd = [str(c) for c in cmd]
    logger.debug("run_tool cmd: %s", " ".join(str_cmd))

    # Open log file if requested
    log_file = log_path.open("w", encoding="utf-8") if log_path else None
    if log_file is not None:
        log_file.write(f"$ {' '.join(str_cmd)}\n\n")

    process = subprocess.Popen(
        str_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        cwd=str(cwd) if cwd else None,
    )

    all_lines: list[str] = []

    def _read_stream(stream: IO[str]) -> None:
        for line in stream:
            line = line.rstrip()
            if line:
                all_lines.append(line)
                if on_output is not None:
                    on_output(line)
                if log_file is not None:
                    log_file.write(line + "\n")
                    log_file.flush()

    stdout_thread = threading.Thread(target=_read_stream, args=(process.stdout,), daemon=True)
    stderr_thread = threading.Thread(target=_read_stream, args=(process.stderr,), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    process.wait()
    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    if log_file is not None:
        log_file.write(f"\n--- exit code: {process.returncode} ---\n")
        log_file.close()

    output_text = "\n".join(all_lines)
    if process.returncode != 0:
        logger.error("run_tool failed (rc=%d): %s", process.returncode, output_text[-500:])

    return process.returncode, output_text
