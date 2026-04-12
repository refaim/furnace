"""Shared subprocess runner with real-time stdout+stderr streaming, per-tool
log files, and optional structured progress parsing.

Byte-level reader that splits on both `\r` and `\n`, so tools that report
progress only with carriage returns (nvencc) and tools that use newlines
(ffmpeg, eac3to, qaac, mkvmerge, mkclean) are both handled through the same
path. Adapters bind their progress parsers via `on_progress_line`.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO

logger = logging.getLogger(__name__)

OutputCallback = Callable[[str], None] | None


def run_tool(
    cmd: Sequence[str | Path],
    on_output: OutputCallback = None,
    on_progress_line: Callable[[str], bool] | None = None,
    log_path: Path | None = None,
    cwd: Path | None = None,
) -> tuple[int, str]:
    """Run a subprocess, streaming both stdout and stderr to callbacks.

    Args:
        cmd: Command and arguments.
        on_output: Called with each decoded line for live display / log.
        on_progress_line: Called with each decoded line for progress parsing.
            Adapters bind their `_parse_X_progress_line` closure here. The
            closure returns True when the line was consumed as structured
            progress — in that case the line is suppressed from `on_output`
            and the log file. Non-progress lines must return False so they
            flow normally to log / output.
        log_path: If provided, write full command + all output to this file.
        cwd: Optional working directory for the subprocess.

    Returns:
        `(return_code, combined_output_text)` — the text includes only the
        non-consumed lines.

    Behavior:
        - stdin is wired to DEVNULL so tools never block on prompts.
        - stdout and stderr are read as bytes on two threads and split on
          either `\\r` or `\\n`. Empty chunks between `\\r\\n` are skipped.
    """
    str_cmd = [str(c) for c in cmd]
    logger.debug("run_tool cmd: %s", " ".join(str_cmd))

    log_file = log_path.open("w", encoding="utf-8") if log_path else None
    if log_file is not None:
        log_file.write(f"$ {' '.join(str_cmd)}\n\n")
        log_file.flush()

    try:
        process = subprocess.Popen(
            str_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            cwd=str(cwd) if cwd else None,
        )

        all_lines: list[str] = []
        lock = threading.Lock()

        def _emit(line: str) -> None:
            if not line:
                return
            # Give the progress parser first dibs. If it consumes the line
            # (returns True), the line does NOT go to log / on_output —
            # progress spam stays out of the diagnostic channel.
            if on_progress_line is not None and on_progress_line(line):
                return
            with lock:
                all_lines.append(line)
                if log_file is not None:
                    log_file.write(line + "\n")
                    log_file.flush()
            if on_output is not None:
                on_output(line)

        def _read_stream(stream: IO[bytes]) -> None:
            """Read `stream` byte-by-byte, split on `\\r` or `\\n`, decode, emit."""
            buf = bytearray()
            try:
                while True:
                    byte = stream.read(1)
                    if not byte:
                        if buf:
                            _emit(buf.decode("utf-8", errors="replace"))
                            buf.clear()
                        return
                    if byte in (b"\r", b"\n"):
                        if buf:
                            _emit(buf.decode("utf-8", errors="replace"))
                            buf.clear()
                    else:
                        buf += byte
            except OSError as exc:
                logger.warning("run_tool reader died with %s", exc)

        if process.stdout is None or process.stderr is None:
            raise RuntimeError("subprocess.Popen did not create stdout/stderr pipes")
        stdout_thread = threading.Thread(
            target=_read_stream,
            args=(process.stdout,),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_read_stream,
            args=(process.stderr,),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        process.wait()
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            logger.warning(
                "run_tool: reader thread did not finish in 5s for: %s",
                str_cmd[0],
            )

        if log_file is not None:
            log_file.write(f"\n--- exit code: {process.returncode} ---\n")

        output_text = "\n".join(all_lines)
        if process.returncode != 0:
            logger.error("run_tool failed (rc=%d): %s", process.returncode, output_text[-500:])

        return process.returncode, output_text
    finally:
        if log_file is not None:
            log_file.close()
