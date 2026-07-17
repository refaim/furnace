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

        if cancel_event is not None:
            while process.poll() is None:
                if cancel_event.is_set():
                    process.kill()
                    break
                cancel_event.wait(timeout=0.1)
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
                target=_read_stream,
                args=(producer.stderr,),
                daemon=True,
            ),
            threading.Thread(
                target=_read_stream,
                args=(consumer.stdout,),
                daemon=True,
            ),
            threading.Thread(
                target=_read_stream,
                args=(consumer.stderr,),
                daemon=True,
            ),
        ]
        for t in threads:
            t.start()

        consumer.wait()
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

        rc = consumer.returncode if consumer.returncode != 0 else producer.returncode
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
