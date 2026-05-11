# dovi_tool extract-rpu via ffmpeg pipe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `DoviToolAdapter.extract_rpu` so DV-content jobs no longer hang indefinitely on UHD MKV remuxes. Replace the broken "pass MKV path to dovi_tool" call with an `ffmpeg ... | dovi_tool extract-rpu -` pipeline backed by a new generic `_subprocess.run_pipeline` helper.

**Architecture:** New `run_pipeline(producer_cmd, consumer_cmd, ...)` helper in `furnace/adapters/_subprocess.py` runs a Unix-style two-process pipeline with the same byte-level stdout/stderr splitter and log-file format as the existing `run_tool`. `DoviToolAdapter` gets an `ffmpeg_path` constructor argument; its `extract_rpu` builds a producer ffmpeg command (HEVC Annex B → stdout) and a consumer dovi_tool command (`extract-rpu - -o RPU.bin`) and dispatches them through `run_pipeline`. Public `extract_rpu(input_path, output_rpu, mode) -> int` signature is unchanged so the executor and `DoviProcessor` port stay untouched. CLI wiring passes `cfg.ffmpeg` alongside `cfg.dovi_tool` when constructing the adapter.

**Tech Stack:** Python 3.13, `unittest.mock` for adapter mocking, real subprocess (via `sys.executable -c "..."`) for pipeline runner tests — same pattern as `tests/adapters/test_subprocess_runner.py`. Pytest with 100 % line + branch coverage enforced by `make check`. TDD strict: failing tests come first. All test/lint/type runs go through the Makefile (`make test`, `make check`), never `uv run pytest` directly. The Makefile only runs the full suite (no per-file selection); intermediate tasks therefore expect a known-red state, and full green is verified at the end.

**Design spec:** `docs/superpowers/specs/2026-05-10-dovi-tool-mkv-pipe-design.md`

**Commit policy:** NO intermediate commits. The user commits when explicitly told to at the end (per `CLAUDE.md` and standing preference). No commit step appears inside individual tasks; commits are out of scope for this plan.

---

## File Structure

**Created:**

- `tests/adapters/test_subprocess_pipeline.py` — new file. Tests `run_pipeline` end-to-end with real Python subprocesses plus four pipe-validation cases that mock `subprocess.Popen`.

**Modified:**

- `furnace/adapters/_subprocess.py` — append the new `run_pipeline` function below `run_tool`. No change to `run_tool` itself.
- `furnace/adapters/dovi_tool.py` — full rewrite of the adapter (kept tiny). Constructor takes `ffmpeg_path`. `_build_extract_cmd` loses `input_path` and gains `-` (stdin). New `_build_ffmpeg_pipe_cmd`. `extract_rpu` calls `run_pipeline`.
- `furnace/cli.py` — line 561-562: pass `cfg.ffmpeg` to `DoviToolAdapter(...)`.
- `furnace/__init__.py` — bump `VERSION` to `1.14.5`.
- `pyproject.toml` — bump `version` to `1.14.5`.
- `tests/adapters/test_dovi_tool.py` — full rewrite. The old `_build_extract_cmd(Path("input.mkv"), ...)` cases no longer compile (signature change), and the patch target shifts from `run_tool` to `run_pipeline`. Easier to overwrite than edit.
- `tests/test_cli.py` — extend `test_executor_fn_with_dovi_tool` (around line 938) to assert the adapter is constructed with both `cfg.dovi_tool` and `cfg.ffmpeg`.

**No changes to:**

- `furnace/core/ports.py` — `DoviProcessor.extract_rpu` signature is unchanged.
- `furnace/services/executor.py` — already calls `extract_rpu(input_path=..., output_rpu=..., mode=...)`; nothing to update.
- `tests/services/test_executor.py` — every existing DoviProcessor test mocks the port and asserts the same kwargs. Stays green untouched.

---

## Task 1: Write all failing tests

**Files:**
- Create: `tests/adapters/test_subprocess_pipeline.py`
- Test: `tests/adapters/test_dovi_tool.py` (full rewrite)
- Test: `tests/test_cli.py:938-983` (extend one test)

Rationale: TDD up-front. After this task the tests describe the new pipeline runner and the new adapter shape completely; production code in Task 2 makes them pass. Running `make test` at the end of this task confirms the suite is red for the *expected* reasons — so we know the tests exercise the right paths.

- [ ] **Step 1: Create `tests/adapters/test_subprocess_pipeline.py` in full**

Create the file with this exact content:

```python
"""Tests for run_pipeline — two-process Unix pipeline runner.

Real subprocesses (sys.executable) for behaviour tests; same style as
tests/adapters/test_subprocess_runner.py. Popen-mocking only for the
four pipe-validation guards, mirroring TestRunToolPipeValidation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from furnace.adapters._subprocess import run_pipeline


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


class TestRunPipelineRealProcess:
    def test_happy_path_pipes_data(self) -> None:
        producer = _py("print('hello')")
        consumer = _py(
            "import sys; data = sys.stdin.read(); "
            "sys.exit(0 if 'hello' in data else 1)",
        )
        rc, _ = run_pipeline(producer, consumer)
        assert rc == 0

    def test_consumer_nonzero_rc_returned(self) -> None:
        producer = _py("print('x')")
        consumer = _py("import sys; sys.exit(7)")
        rc, _ = run_pipeline(producer, consumer)
        assert rc == 7

    def test_producer_nonzero_rc_returned_when_consumer_ok(self) -> None:
        producer = _py(
            "import sys; sys.stdout.write('x'); sys.exit(13)",
        )
        consumer = _py("import sys; sys.stdin.read(); sys.exit(0)")
        rc, _ = run_pipeline(producer, consumer)
        assert rc == 13

    def test_consumer_rc_takes_priority_over_producer(self) -> None:
        producer = _py(
            "import sys; sys.stdout.write('x'); sys.exit(13)",
        )
        consumer = _py("import sys; sys.stdin.read(); sys.exit(7)")
        rc, _ = run_pipeline(producer, consumer)
        assert rc == 7

    def test_both_rc_zero_returns_zero(self) -> None:
        producer = _py("import sys; sys.stdout.write('x')")
        consumer = _py("import sys; sys.stdin.read()")
        rc, _ = run_pipeline(producer, consumer)
        assert rc == 0

    def test_producer_stderr_captured(self) -> None:
        received: list[str] = []
        producer = _py(
            "import sys; sys.stderr.write('PRODERR\\n'); "
            "sys.stdout.write('x')",
        )
        consumer = _py("import sys; sys.stdin.read()")
        rc, _ = run_pipeline(producer, consumer, on_output=received.append)
        assert rc == 0
        assert any("PRODERR" in line for line in received)

    def test_consumer_stderr_captured(self) -> None:
        received: list[str] = []
        producer = _py("print('x')")
        consumer = _py(
            "import sys; sys.stdin.read(); "
            "sys.stderr.write('CONERR\\n')",
        )
        rc, _ = run_pipeline(producer, consumer, on_output=received.append)
        assert rc == 0
        assert any("CONERR" in line for line in received)

    def test_consumer_stdout_captured(self) -> None:
        received: list[str] = []
        producer = _py("print('x')")
        consumer = _py(
            "import sys; sys.stdin.read(); print('CONOUT')",
        )
        rc, _ = run_pipeline(producer, consumer, on_output=received.append)
        assert rc == 0
        assert any("CONOUT" in line for line in received)

    def test_binary_safe_byte_passthrough(self, tmp_path: Path) -> None:
        """Producer's stdout bytes reach the consumer byte-exact —
        critical because we will pipe binary HEVC streams.
        """
        marker = tmp_path / "marker.bin"
        producer = _py(
            "import sys; sys.stdout.buffer.write(b'AB\\x00CD\\xff')",
        )
        consumer = _py(
            "import sys; "
            f"open({str(marker)!r}, 'wb').write(sys.stdin.buffer.read())",
        )
        rc, _ = run_pipeline(producer, consumer)
        assert rc == 0
        assert marker.read_bytes() == b"AB\x00CD\xff"

    def test_log_file_contains_both_commands_and_exit_codes(
        self, tmp_path: Path,
    ) -> None:
        log = tmp_path / "pipe.log"
        producer = _py("import sys; sys.stderr.write('p_err\\n')")
        consumer = _py("import sys; sys.stdin.read(); sys.exit(3)")
        rc, _ = run_pipeline(producer, consumer, log_path=log)
        assert rc == 3
        text = log.read_text(encoding="utf-8")
        first_line = text.splitlines()[0]
        assert first_line.startswith("$ ")
        assert " | " in first_line
        assert "--- producer exit code: 0 ---" in text
        assert "--- consumer exit code: 3 ---" in text
        assert "p_err" in text

    def test_no_log_path_no_file(self, tmp_path: Path) -> None:
        producer = _py("print('x')")
        consumer = _py("import sys; sys.stdin.read()")
        rc, _ = run_pipeline(producer, consumer, log_path=None)
        assert rc == 0
        assert list(tmp_path.iterdir()) == []

    def test_no_on_output_smoke(self) -> None:
        producer = _py("import sys; sys.stderr.write('x\\n')")
        consumer = _py("import sys; sys.stdin.read()")
        rc, _ = run_pipeline(producer, consumer, on_output=None)
        assert rc == 0

    def test_cwd_forwarded_to_producer(self, tmp_path: Path) -> None:
        producer = _py(
            "import os; "
            "open('prod.txt', 'w').write(os.getcwd()); "
            "print('done')",
        )
        consumer = _py("import sys; sys.stdin.read()")
        rc, _ = run_pipeline(producer, consumer, cwd=tmp_path)
        assert rc == 0
        out = (tmp_path / "prod.txt").read_text(encoding="utf-8")
        assert Path(out).resolve() == tmp_path.resolve()

    def test_cwd_forwarded_to_consumer(self, tmp_path: Path) -> None:
        producer = _py("print('x')")
        consumer = _py(
            "import os, sys; sys.stdin.read(); "
            "open('cons.txt', 'w').write(os.getcwd())",
        )
        rc, _ = run_pipeline(producer, consumer, cwd=tmp_path)
        assert rc == 0
        out = (tmp_path / "cons.txt").read_text(encoding="utf-8")
        assert Path(out).resolve() == tmp_path.resolve()


class TestRunPipelinePipeValidation:
    """Mirrors TestRunToolPipeValidation: defensive guards on Popen pipes."""

    def _mock_proc(self, *, stdout: object, stderr: object) -> MagicMock:
        m = MagicMock()
        m.stdout = stdout
        m.stderr = stderr
        return m

    def test_raises_when_producer_stdout_missing(self) -> None:
        prod = self._mock_proc(stdout=None, stderr=MagicMock())
        cons = self._mock_proc(stdout=MagicMock(), stderr=MagicMock())
        with patch(
            "furnace.adapters._subprocess.subprocess.Popen",
            side_effect=[prod, cons],
        ), pytest.raises(RuntimeError, match="pipes"):
            run_pipeline(["x"], ["y"])

    def test_raises_when_producer_stderr_missing(self) -> None:
        prod = self._mock_proc(stdout=MagicMock(), stderr=None)
        cons = self._mock_proc(stdout=MagicMock(), stderr=MagicMock())
        with patch(
            "furnace.adapters._subprocess.subprocess.Popen",
            side_effect=[prod, cons],
        ), pytest.raises(RuntimeError, match="pipes"):
            run_pipeline(["x"], ["y"])

    def test_raises_when_consumer_stdout_missing(self) -> None:
        prod = self._mock_proc(stdout=MagicMock(), stderr=MagicMock())
        cons = self._mock_proc(stdout=None, stderr=MagicMock())
        with patch(
            "furnace.adapters._subprocess.subprocess.Popen",
            side_effect=[prod, cons],
        ), pytest.raises(RuntimeError, match="pipes"):
            run_pipeline(["x"], ["y"])

    def test_raises_when_consumer_stderr_missing(self) -> None:
        prod = self._mock_proc(stdout=MagicMock(), stderr=MagicMock())
        cons = self._mock_proc(stdout=MagicMock(), stderr=None)
        with patch(
            "furnace.adapters._subprocess.subprocess.Popen",
            side_effect=[prod, cons],
        ), pytest.raises(RuntimeError, match="pipes"):
            run_pipeline(["x"], ["y"])
```

- [ ] **Step 2: Replace `tests/adapters/test_dovi_tool.py` in full**

Overwrite the entire file with this content:

```python
"""Tests for DoviToolAdapter — ffmpeg pipe + dovi_tool consumer.

run_pipeline is patched in every command-execution test (no real
subprocess). Builders are tested directly. The adapter signature now
requires both dovi_tool_path and ffmpeg_path, since the bug fix routes
the source MKV through ffmpeg before dovi_tool sees it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from furnace.adapters.dovi_tool import DoviToolAdapter
from furnace.core.models import DvMode

DOVI = Path("dovi_tool.exe")
FFMPEG = Path("ffmpeg.exe")


class TestFfmpegPipeCmd:
    def test_input_flag_points_to_source(self) -> None:
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        cmd = adapter._build_ffmpeg_pipe_cmd(Path("input.mkv"))
        str_cmd = [str(c) for c in cmd]
        assert str_cmd[0] == str(FFMPEG)
        i_idx = str_cmd.index("-i")
        assert str_cmd[i_idx + 1] == "input.mkv"

    def test_maps_first_video_stream(self) -> None:
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        cmd = adapter._build_ffmpeg_pipe_cmd(Path("input.mkv"))
        str_cmd = [str(c) for c in cmd]
        m_idx = str_cmd.index("-map")
        assert str_cmd[m_idx + 1] == "0:v:0"

    def test_copies_codec_no_reencode(self) -> None:
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        cmd = adapter._build_ffmpeg_pipe_cmd(Path("input.mkv"))
        str_cmd = [str(c) for c in cmd]
        c_idx = str_cmd.index("-c")
        assert str_cmd[c_idx + 1] == "copy"

    def test_applies_annexb_bitstream_filter(self) -> None:
        """Without hevc_mp4toannexb, MP4-style length-prefixed NALs
        reach dovi_tool which only understands Annex B start codes."""
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        cmd = adapter._build_ffmpeg_pipe_cmd(Path("input.mkv"))
        str_cmd = [str(c) for c in cmd]
        bsf_idx = str_cmd.index("-bsf:v")
        assert str_cmd[bsf_idx + 1] == "hevc_mp4toannexb"

    def test_emits_raw_hevc_to_stdout(self) -> None:
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        cmd = adapter._build_ffmpeg_pipe_cmd(Path("input.mkv"))
        str_cmd = [str(c) for c in cmd]
        f_idx = str_cmd.index("-f")
        assert str_cmd[f_idx + 1] == "hevc"
        assert str_cmd[-1] == "-"

    def test_quiet_loglevel(self) -> None:
        """Producer chatter would spam the log; -loglevel error keeps
        only true failures while still surfacing them.
        """
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        cmd = adapter._build_ffmpeg_pipe_cmd(Path("input.mkv"))
        str_cmd = [str(c) for c in cmd]
        ll_idx = str_cmd.index("-loglevel")
        assert str_cmd[ll_idx + 1] == "error"


class TestDoviExtractCmd:
    def test_copy_mode_no_m_flag(self) -> None:
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        cmd = adapter._build_extract_cmd(Path("RPU.bin"), DvMode.COPY)
        str_cmd = [str(c) for c in cmd]
        assert str_cmd[0] == str(DOVI)
        assert "-m" not in str_cmd
        assert "extract-rpu" in str_cmd

    def test_to_8_1_mode_adds_m_2(self) -> None:
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        cmd = adapter._build_extract_cmd(Path("RPU.bin"), DvMode.TO_8_1)
        str_cmd = [str(c) for c in cmd]
        m_idx = str_cmd.index("-m")
        assert str_cmd[m_idx + 1] == "2"

    def test_reads_from_stdin(self) -> None:
        """Bug fix: the consumer must NOT receive a container path —
        dovi_tool reads the HEVC stream produced by the ffmpeg producer
        over stdin (`-`).
        """
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        cmd = adapter._build_extract_cmd(Path("RPU.bin"), DvMode.COPY)
        str_cmd = [str(c) for c in cmd]
        ex_idx = str_cmd.index("extract-rpu")
        assert str_cmd[ex_idx + 1] == "-"

    def test_output_flag_points_to_rpu(self) -> None:
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        rpu_path = Path(tempfile.gettempdir()) / "RPU.bin"
        cmd = adapter._build_extract_cmd(rpu_path, DvMode.COPY)
        str_cmd = [str(c) for c in cmd]
        o_idx = str_cmd.index("-o")
        assert str_cmd[o_idx + 1] == str(rpu_path)


class TestDoviExtractRpuExecution:
    """extract_rpu wires both builders into run_pipeline."""

    def _patch_pipeline(
        self, captured: dict[str, Any], rc: int = 0,
    ) -> Any:
        def fake(
            producer_cmd: Any,
            consumer_cmd: Any,
            on_output: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            captured["producer"] = list(producer_cmd)
            captured["consumer"] = list(consumer_cmd)
            captured["log_path"] = log_path
            captured["on_output"] = on_output
            return rc, ""

        return patch(
            "furnace.adapters.dovi_tool.run_pipeline",
            side_effect=fake,
        )

    def test_returns_pipeline_rc(self) -> None:
        captured: dict[str, Any] = {}
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        with self._patch_pipeline(captured, rc=0):
            rc = adapter.extract_rpu(
                Path("input.mkv"), Path("rpu.bin"), DvMode.COPY,
            )
        assert rc == 0

    def test_propagates_nonzero_rc(self) -> None:
        captured: dict[str, Any] = {}
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        with self._patch_pipeline(captured, rc=42):
            rc = adapter.extract_rpu(
                Path("input.mkv"), Path("rpu.bin"), DvMode.COPY,
            )
        assert rc == 42

    def test_passes_ffmpeg_pipe_cmd_as_producer(self) -> None:
        captured: dict[str, Any] = {}
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        with self._patch_pipeline(captured):
            adapter.extract_rpu(
                Path("movie.mkv"), Path("rpu.bin"), DvMode.COPY,
            )
        producer = [str(c) for c in captured["producer"]]
        assert producer[0] == str(FFMPEG)
        assert "-bsf:v" in producer
        assert producer[-1] == "-"
        assert "movie.mkv" in producer

    def test_passes_dovi_extract_cmd_as_consumer(self) -> None:
        captured: dict[str, Any] = {}
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        with self._patch_pipeline(captured):
            adapter.extract_rpu(
                Path("movie.mkv"), Path("out.bin"), DvMode.TO_8_1,
            )
        consumer = [str(c) for c in captured["consumer"]]
        assert consumer[0] == str(DOVI)
        assert "extract-rpu" in consumer
        m_idx = consumer.index("-m")
        assert consumer[m_idx + 1] == "2"
        o_idx = consumer.index("-o")
        assert consumer[o_idx + 1] == "out.bin"

    def test_log_path_set_when_log_dir_configured(
        self, tmp_path: Path,
    ) -> None:
        captured: dict[str, Any] = {}
        adapter = DoviToolAdapter(DOVI, FFMPEG, log_dir=tmp_path)
        with self._patch_pipeline(captured):
            adapter.extract_rpu(
                Path("a.mkv"), Path("rpu.bin"), DvMode.TO_8_1,
            )
        assert captured["log_path"] == tmp_path / "dovi_tool_extract.log"

    def test_log_path_none_when_no_log_dir(self) -> None:
        captured: dict[str, Any] = {}
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        with self._patch_pipeline(captured):
            adapter.extract_rpu(
                Path("a.mkv"), Path("rpu.bin"), DvMode.COPY,
            )
        assert captured["log_path"] is None

    def test_on_output_propagates(self) -> None:
        captured: dict[str, Any] = {}

        def output_fn(_line: str) -> None:
            return None

        adapter = DoviToolAdapter(DOVI, FFMPEG, on_output=output_fn)
        with self._patch_pipeline(captured):
            adapter.extract_rpu(
                Path("a.mkv"), Path("rpu.bin"), DvMode.COPY,
            )
        assert captured["on_output"] is output_fn


class TestSetLogDir:
    def test_set_log_dir(self, tmp_path: Path) -> None:
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        adapter.set_log_dir(tmp_path)
        assert adapter._log_dir == tmp_path

    def test_set_log_dir_none(self, tmp_path: Path) -> None:
        adapter = DoviToolAdapter(DOVI, FFMPEG, log_dir=tmp_path)
        adapter.set_log_dir(None)
        assert adapter._log_dir is None
```

- [ ] **Step 3: Extend `test_executor_fn_with_dovi_tool` in `tests/test_cli.py`**

Locate `test_executor_fn_with_dovi_tool` (starts at `tests/test_cli.py:938`). The existing assertion block at the end of the test reads:

```python
        mock_dovi.assert_called_once()
        mock_executor_cls.return_value.run.assert_called_once()
```

Replace **only** the `mock_dovi.assert_called_once()` line with:

```python
        mock_dovi.assert_called_once()
        dovi_args = mock_dovi.call_args.args
        assert dovi_args[0] == cfg.dovi_tool
        assert dovi_args[1] == cfg.ffmpeg
```

Leave the `mock_executor_cls.return_value.run.assert_called_once()` line below it intact. No other test in this file changes.

- [ ] **Step 4: Run `make test` — expected RED**

Run: `make test`
Expected: tests fail. Specifically:

- `tests/adapters/test_subprocess_pipeline.py::*` — `ImportError: cannot import name 'run_pipeline' from 'furnace.adapters._subprocess'` (function doesn't exist yet).
- `tests/adapters/test_dovi_tool.py::*` — `TypeError: __init__() takes ... arguments` on every constructor call (new ffmpeg_path positional). Once the constructor signature is fixed in Task 2 the builder/execution tests will fail until `_build_ffmpeg_pipe_cmd` exists and `_build_extract_cmd` drops `input_path`.
- `tests/test_cli.py::TestExecutor::test_executor_fn_with_dovi_tool` — `IndexError: tuple index out of range` on `dovi_args[1]` (the adapter is currently called with one positional arg).

Confirm the failures are in those three files only and trace back to the spec'd API changes. If anything else fails, stop and investigate before continuing.

---

## Task 2: Implement `run_pipeline`, rewrite the adapter, wire CLI

**Files:**
- Modify: `furnace/adapters/_subprocess.py` (append new function)
- Modify: `furnace/adapters/dovi_tool.py` (full rewrite)
- Modify: `furnace/cli.py:561-562`

Rationale: tests from Task 1 lock the contracts. This task makes them pass by adding the runner, rewriting the adapter, and updating the one CLI line. We do all three before the next `make test` because they're tightly coupled (the adapter imports `run_pipeline`; the CLI imports the new `DoviToolAdapter` ctor signature).

- [ ] **Step 1: Append `run_pipeline` to `furnace/adapters/_subprocess.py`**

Add the following function at the end of `furnace/adapters/_subprocess.py` (after `run_tool`, before EOF — keep `run_tool` untouched):

```python
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
        producer.wait()
        for t in threads:
            t.join(timeout=5)
        for t in threads:
            if t.is_alive():
                logger.warning(
                    "run_pipeline: reader thread did not finish in 5s",
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
```

No other change to `_subprocess.py`. `run_tool` and the existing imports stay exactly as they are.

- [ ] **Step 2: Replace `furnace/adapters/dovi_tool.py` in full**

Overwrite the entire file with this content:

```python
from __future__ import annotations

import logging
from pathlib import Path

from furnace.core.models import DvMode

from ._subprocess import OutputCallback, run_pipeline

logger = logging.getLogger(__name__)


class DoviToolAdapter:
    """Implements DoviProcessor port via dovi_tool CLI.

    `dovi_tool extract-rpu` only accepts an HEVC elementary stream (file
    or stdin), not a container. We pipe the source through ffmpeg so a
    `.mkv` (or any container ffmpeg can read) becomes Annex B HEVC on
    dovi_tool's stdin.
    """

    def __init__(
        self,
        dovi_tool_path: Path,
        ffmpeg_path: Path,
        on_output: OutputCallback = None,
        log_dir: Path | None = None,
    ) -> None:
        self._dovi_tool = dovi_tool_path
        self._ffmpeg = ffmpeg_path
        self._on_output = on_output
        self._log_dir = log_dir

    def set_log_dir(self, log_dir: Path | None) -> None:
        self._log_dir = log_dir

    def _build_ffmpeg_pipe_cmd(self, input_path: Path) -> list[str | Path]:
        # `-bsf:v hevc_mp4toannexb` is a no-op on already-Annex-B inputs
        # (raw .hevc) and converts MP4-style length-prefixed NALs to
        # start-codes when the source is in a container. Robust either
        # way, so a single command serves every supported input.
        return [
            self._ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-i", input_path,
            "-map", "0:v:0",
            "-c", "copy",
            "-bsf:v", "hevc_mp4toannexb",
            "-f", "hevc",
            "-",
        ]

    def _build_extract_cmd(
        self,
        output_rpu: Path,
        mode: DvMode,
    ) -> list[str | Path]:
        cmd: list[str | Path] = [self._dovi_tool]
        if mode == DvMode.TO_8_1:
            cmd += ["-m", "2"]
        cmd += ["extract-rpu", "-", "-o", output_rpu]
        return cmd

    def extract_rpu(
        self,
        input_path: Path,
        output_rpu: Path,
        mode: DvMode,
    ) -> int:
        """Extract RPU from a container or raw HEVC stream."""
        producer = self._build_ffmpeg_pipe_cmd(input_path)
        consumer = self._build_extract_cmd(output_rpu, mode)
        logger.debug(
            "dovi_tool pipeline: %s | %s",
            " ".join(str(c) for c in producer),
            " ".join(str(c) for c in consumer),
        )
        log_path = (
            self._log_dir / "dovi_tool_extract.log"
            if self._log_dir
            else None
        )
        rc, _out = run_pipeline(
            producer,
            consumer,
            on_output=self._on_output,
            log_path=log_path,
        )
        return rc
```

- [ ] **Step 3: Update CLI wiring in `furnace/cli.py`**

In `furnace/cli.py`, locate the block at lines 560-562:

```python
        dovi_adapter: DoviToolAdapter | None = None
        if cfg.dovi_tool is not None:
            dovi_adapter = DoviToolAdapter(cfg.dovi_tool, on_output=tool_output)
```

Replace the `DoviToolAdapter(...)` call with:

```python
            dovi_adapter = DoviToolAdapter(
                cfg.dovi_tool, cfg.ffmpeg, on_output=tool_output,
            )
```

The `cfg.dovi_tool is not None` guard is preserved — only the constructor call changes. `cfg.ffmpeg` is required by `ToolPaths` (validated at config load) so no None-check is needed.

- [ ] **Step 4: Run `make test` — expected GREEN**

Run: `make test`
Expected: full suite passes, coverage ≥ 100 %.

If any test fails, the most likely causes:

- Indentation drift in the new `run_pipeline` body — its function header is at module top level, all body four spaces.
- A leftover reference to the old single-arg `_build_extract_cmd(input_path, output_rpu, mode)` somewhere — `grep -rn "_build_extract_cmd" furnace/ tests/` should return only the new two-arg call sites.
- Coverage gap on a new branch — see Task 4 coverage notes.

Fix and re-run until green.

---

## Task 3: Bump version to 1.14.5

**Files:**
- Modify: `furnace/__init__.py`
- Modify: `pyproject.toml`

Rationale: per `CLAUDE.md`, every change to user-facing behaviour bumps SemVer. DV jobs that previously hung indefinitely now complete. That is a bugfix to a fully-broken feature, hence PATCH (1.14.4 → 1.14.5).

- [ ] **Step 1: Bump `furnace/__init__.py`**

Change the `VERSION` line from `VERSION = "1.14.4"` to:

```python
VERSION = "1.14.5"
```

- [ ] **Step 2: Bump `pyproject.toml`**

Change the `version = "1.14.4"` line under `[project]` to:

```toml
version = "1.14.5"
```

---

## Task 4: Final `make check`

**Rationale:** `make check` is the single source of truth (`CLAUDE.md`). Lint, typecheck, and the full test suite with 100 % line + branch coverage on `furnace/` and `tests/` must all pass before the work is considered done.

- [ ] **Step 1: Run `make check`**

Run: `make check`
Expected: ruff clean, mypy strict clean, pytest with 100 % line + branch coverage.

Coverage checklist for the new code paths (every one should be hit by tests added in Task 1):

- `run_pipeline` happy path — `test_happy_path_pipes_data`, `test_both_rc_zero_returns_zero`.
- `run_pipeline` rc-priority branches:
  - consumer rc != 0 (with producer rc 0) — `test_consumer_nonzero_rc_returned`.
  - consumer rc 0, producer rc != 0 — `test_producer_nonzero_rc_returned_when_consumer_ok`.
  - both rc != 0, consumer wins — `test_consumer_rc_takes_priority_over_producer`.
  - both rc 0 — `test_both_rc_zero_returns_zero`.
- `run_pipeline` reader paths (covers `_read_stream`'s split-on-`\r\n` else branch and the EOF flush) — `test_producer_stderr_captured`, `test_consumer_stderr_captured`, `test_consumer_stdout_captured`.
- `run_pipeline` byte passthrough — `test_binary_safe_byte_passthrough`.
- `run_pipeline` log-file truthy branch (write commands + exit codes) — `test_log_file_contains_both_commands_and_exit_codes`.
- `run_pipeline` log-file falsy branch — `test_no_log_path_no_file`.
- `run_pipeline` `on_output is None` branch — `test_no_on_output_smoke`.
- `run_pipeline` `cwd` truthy — `test_cwd_forwarded_to_producer`, `test_cwd_forwarded_to_consumer`.
- `run_pipeline` pipe-validation guards (4 raise paths) — the four `TestRunPipelinePipeValidation` cases.
- `DoviToolAdapter._build_ffmpeg_pipe_cmd` — six `TestFfmpegPipeCmd` cases.
- `DoviToolAdapter._build_extract_cmd` mode branch (TO_8_1 vs COPY) — `test_to_8_1_mode_adds_m_2`, `test_copy_mode_no_m_flag`.
- `DoviToolAdapter._build_extract_cmd` stdin / output flag — `test_reads_from_stdin`, `test_output_flag_points_to_rpu`.
- `DoviToolAdapter.extract_rpu` log-dir truthy / falsy — `test_log_path_set_when_log_dir_configured`, `test_log_path_none_when_no_log_dir`.
- `DoviToolAdapter.extract_rpu` rc propagation — `test_returns_pipeline_rc`, `test_propagates_nonzero_rc`.
- `DoviToolAdapter.extract_rpu` producer / consumer cmd wiring — `test_passes_ffmpeg_pipe_cmd_as_producer`, `test_passes_dovi_extract_cmd_as_consumer`.
- `DoviToolAdapter.extract_rpu` callback propagation — `test_on_output_propagates`.
- `DoviToolAdapter.set_log_dir` set / clear — `test_set_log_dir`, `test_set_log_dir_none`.
- CLI wiring with both paths — extended `test_executor_fn_with_dovi_tool`.

If `make check` reports an uncovered line or branch outside this list, add a focused test that hits exactly that path. Re-run `make check` until clean.

Coverage gotchas to watch for:

- The `run_pipeline` `_read_stream` reader thread has an `OSError` warning branch (`except OSError as exc:`). The same branch in `run_tool` is uncovered in the existing suite (no negative test), so we don't need to add one here either — the project's existing coverage policy treats it as unreachable in normal operation. If `make check` flags it as a new gap (e.g., the coverage config uses `--no-skip-covered` or similar), add `# pragma: no cover` rather than fabricating an OSError.
- The reader-`is_alive` warning branch (`for t in threads: if t.is_alive(): logger.warning(...)`) is similarly defensive — same handling.

- [ ] **Step 2: Confirm clean output**

Re-read the `make check` summary. Every line: PASS. Every percentage: 100. The work is done — stop here. Do not commit; the user will commit explicitly.

---

## Self-review notes

Cross-check against the spec (`docs/superpowers/specs/2026-05-10-dovi-tool-mkv-pipe-design.md`):

- **`run_pipeline` helper exists and matches signature/semantics:** Task 2 step 1 + Task 1 step 1 (`tests/adapters/test_subprocess_pipeline.py`).
- **`DoviToolAdapter` constructor takes `ffmpeg_path`:** Task 2 step 2 + every test in Task 1 step 2.
- **`_build_ffmpeg_pipe_cmd` produces the spec'd argv (input, map, copy, bsf, format, stdout):** Task 2 step 2 + `TestFfmpegPipeCmd` six cases.
- **`_build_extract_cmd` reads stdin (`-`) and supports both DV modes:** Task 2 step 2 + `TestDoviExtractCmd` four cases.
- **`extract_rpu` public signature unchanged:** Task 2 step 2 — same `(input_path, output_rpu, mode) -> int`. Confirmed by the executor remaining untouched.
- **CLI passes both paths to the adapter:** Task 2 step 3 + extended `test_executor_fn_with_dovi_tool` in Task 1 step 3.
- **rc surfacing — consumer wins on conflict, producer surfaces when consumer is fine:** Task 1 step 1 four rc cases.
- **Binary-safe passthrough (HEVC bytes survive intact):** Task 1 step 1 `test_binary_safe_byte_passthrough`.
- **Log file format (header + exit codes for both):** Task 1 step 1 `test_log_file_contains_both_commands_and_exit_codes`.
- **Pipe-validation defense:** four `TestRunPipelinePipeValidation` cases mirroring `TestRunToolPipeValidation`.
- **Version bump:** Task 3.
- **100 % coverage gate:** Task 4.

Untouched on purpose:

- `furnace/core/ports.py` — `DoviProcessor.extract_rpu` signature unchanged.
- `furnace/services/executor.py` — calls the same kwargs on the port.
- `tests/services/test_executor.py` — every existing dovi mock asserts the same `extract_rpu(input_path=..., output_rpu=..., mode=...)` shape.
- `furnace/adapters/_subprocess.py::run_tool` — independent code path; not modified, not re-tested.
