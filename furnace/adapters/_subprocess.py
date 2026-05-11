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
    cancel_event: threading.Event | None = None,
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
        cancel_event: If set during execution, the running child is killed
            and `run_tool` returns. Default `None` preserves blocking-wait
            behavior for all callers that don't need cancellation.

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

        # Polled wait so we can react to cancel_event.
        if cancel_event is not None:
            while process.poll() is None:
                if cancel_event.is_set():
                    # SIGKILL — plan-phase probes (eac3to/makemkvcon/mkvmerge -i)
                    # have nothing to flush; cancel must be instant on Ctrl+C.
                    process.kill()
                    break
                cancel_event.wait(timeout=0.1)
        # process.wait() reaps the zombie and produces returncode (also runs
        # for the no-cancel-event path where the polled block was skipped).
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


def run_pipeline(
    producer_cmd: Sequence[str | Path],
    consumer_cmd: Sequence[str | Path],
    on_output: OutputCallback = None,
    log_path: Path | None = None,
    cwd: Path | None = None,
) -> tuple[int, str]:
    """Run ``producer_cmd | consumer_cmd`` as a Unix-style pipeline.

    Producer's stdout is wired straight into consumer's stdin (raw bytes,
    no decoding). Producer stderr, consumer stdout, and consumer stderr
    are read on three threads and split on either ``\\r`` or ``\\n`` —
    same byte reader as ``run_tool`` so behaviour is consistent across
    tools (nvencc-style ``\\r`` progress and ffmpeg/eac3to-style ``\\n``
    log lines both work).

    Args:
        producer_cmd: First command. Its stdout becomes the consumer's stdin.
        consumer_cmd: Second command. Reads stdin from the producer.
        on_output: Called with each decoded line (producer-stderr,
            consumer-stdout, consumer-stderr — in interleaved arrival order).
        log_path: If provided, write ``$ <prod> | <cons>`` header, then
            interleaved lines, then both exit codes.
        cwd: Forwarded to both subprocesses.

    Returns:
        ``(rc, combined_text)`` where ``rc = consumer.returncode`` if the
        consumer failed (non-zero), else ``producer.returncode``. Either
        side failing surfaces as a non-zero rc to the caller.

    Behaviour:
        - Producer ``stdin`` is wired to ``DEVNULL`` so it never blocks.
        - Parent closes its copy of ``producer.stdout`` once the consumer
          has been spawned with it as ``stdin``. This is the standard
          CPython pipeline pattern; without it the kernel cannot deliver
          ``SIGPIPE`` / EOF when the consumer exits early.
        - Consumer is waited on first (its rc is the one that matters
          end-to-end), then the producer.
    """
    str_producer = [str(c) for c in producer_cmd]
    str_consumer = [str(c) for c in consumer_cmd]
    logger.debug(
        "run_pipeline cmd: %s | %s",
        " ".join(str_producer),
        " ".join(str_consumer),
    )

    log_file = log_path.open("w", encoding="utf-8") if log_path else None
    if log_file is not None:
        log_file.write(
            f"$ {' '.join(str_producer)} | {' '.join(str_consumer)}\n\n",
        )
        log_file.flush()

    cwd_str = str(cwd) if cwd else None
    producer = subprocess.Popen(
        str_producer,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        cwd=cwd_str,
    )
    if producer.stdout is None or producer.stderr is None:
        producer.kill()
        producer.wait()
        if log_file is not None:
            log_file.close()
        raise RuntimeError(
            "subprocess.Popen did not create producer stdout/stderr pipes",
        )

    consumer = subprocess.Popen(
        str_consumer,
        stdin=producer.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        cwd=cwd_str,
    )
    if consumer.stdout is None or consumer.stderr is None:
        consumer.kill()
        producer.kill()
        consumer.wait()
        producer.wait()
        if log_file is not None:
            log_file.close()
        raise RuntimeError(
            "subprocess.Popen did not create consumer stdout/stderr pipes",
        )

    # Standard CPython pipeline pattern: parent must close its copy of
    # the producer's stdout so the consumer is the only reader.
    producer.stdout.close()

    try:
        all_lines: list[str] = []
        lock = threading.Lock()

        def _emit(line: str) -> None:
            with lock:
                all_lines.append(line)
                if log_file is not None:
                    log_file.write(line + "\n")
                    log_file.flush()
            if on_output is not None:
                on_output(line)

        def _read_stream(stream: IO[bytes]) -> None:
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
                logger.warning("run_pipeline reader died with %s", exc)

        threads = [
            threading.Thread(
                target=_read_stream, args=(producer.stderr,), daemon=True,
            ),
            threading.Thread(
                target=_read_stream, args=(consumer.stdout,), daemon=True,
            ),
            threading.Thread(
                target=_read_stream, args=(consumer.stderr,), daemon=True,
            ),
        ]
        for t in threads:
            t.start()

        consumer.wait()
        # The consumer may have exited before reading the producer to EOF
        # (e.g. dovi_tool found enough input). Closing the parent's copy
        # of producer.stdout above let SIGPIPE reach the producer on its
        # next write, but some tools (notably ffmpeg) treat SIGPIPE as a
        # recoverable error and keep running. Bound the wait so a stuck
        # producer can't hang the whole pipeline.
        try:
            producer.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning(
                "run_pipeline: producer %s did not exit 10s after consumer; killing",
                str_producer[0],
            )
            producer.kill()
            producer.wait()
        thread_names = ["producer.stderr", "consumer.stdout", "consumer.stderr"]
        for t in threads:
            t.join(timeout=5)
        stuck = [name for name, t in zip(thread_names, threads, strict=True) if t.is_alive()]
        if stuck:
            logger.warning(
                "run_pipeline: reader thread(s) did not finish in 5s for %s: %s",
                str_consumer[0],
                ", ".join(stuck),
            )

        if log_file is not None:
            log_file.write(
                f"\n--- producer exit code: {producer.returncode} ---\n",
            )
            log_file.write(
                f"--- consumer exit code: {consumer.returncode} ---\n",
            )

        rc = (
            consumer.returncode
            if consumer.returncode != 0
            else producer.returncode
        )
        output_text = "\n".join(all_lines)
        if rc != 0:
            logger.error(
                "run_pipeline failed (producer rc=%d, consumer rc=%d): %s",
                producer.returncode,
                consumer.returncode,
                output_text[-500:],
            )
        return rc, output_text
    finally:
        if log_file is not None:
            log_file.close()
