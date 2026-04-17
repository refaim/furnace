from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from furnace.adapters._subprocess import run_tool
from furnace.core.progress import ProgressSample


class TestRunToolPipeValidation:
    def test_raises_when_stdout_missing(self) -> None:
        mock_process = MagicMock()
        mock_process.stdout = None
        mock_process.stderr = MagicMock()
        with patch("furnace.adapters._subprocess.subprocess.Popen", return_value=mock_process), \
             pytest.raises(RuntimeError, match="pipes"):
            run_tool(["echo", "x"])

    def test_raises_when_stderr_missing(self) -> None:
        mock_process = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stderr = None
        with patch("furnace.adapters._subprocess.subprocess.Popen", return_value=mock_process), \
             pytest.raises(RuntimeError, match="pipes"):
            run_tool(["echo", "x"])


class TestRunToolRealProcess:
    """Tests using real subprocesses (no mocking Popen)."""

    def test_happy_path_stdout(self) -> None:
        rc, output = run_tool([sys.executable, "-c", "print('hello world')"])
        assert rc == 0
        assert "hello world" in output

    def test_stderr_captured(self) -> None:
        rc, output = run_tool([
            sys.executable, "-c",
            "import sys; sys.stderr.write('err msg\\n')",
        ])
        assert rc == 0
        assert "err msg" in output

    def test_nonzero_rc(self) -> None:
        rc, _output = run_tool([
            sys.executable, "-c", "import sys; sys.exit(42)",
        ])
        assert rc == 42

    def test_cr_splitting(self) -> None:
        rc, output = run_tool([
            sys.executable, "-c",
            "import sys; sys.stdout.write('aaa\\rbbb\\n'); sys.stdout.flush()",
        ])
        assert rc == 0
        assert "aaa" in output
        assert "bbb" in output

    def test_on_output_callback(self) -> None:
        received: list[str] = []
        rc, _output = run_tool(
            [sys.executable, "-c", "print('line1'); print('line2')"],
            on_output=received.append,
        )
        assert rc == 0
        assert "line1" in received
        assert "line2" in received

    def test_progress_routing(self) -> None:
        samples: list[ProgressSample] = []

        def parser(line: str) -> bool:
            if line.strip() == "50":
                samples.append(ProgressSample(fraction=0.5))
                return True
            return False

        rc, output = run_tool(
            [sys.executable, "-c", "print('50'); print('keep')"],
            on_progress_line=parser,
        )
        assert rc == 0
        assert len(samples) == 1
        assert samples[0].fraction == 0.5
        assert "50" not in output
        assert "keep" in output

    def test_log_file(self, tmp_path: Path) -> None:
        log = tmp_path / "test.log"
        rc, _output = run_tool(
            [sys.executable, "-c", "print('logged')"],
            log_path=log,
        )
        assert rc == 0
        contents = log.read_text(encoding="utf-8")
        assert contents.startswith("$")
        assert "logged" in contents
        assert "exit code: 0" in contents

    def test_cwd_parameter(self, tmp_path: Path) -> None:
        rc, output = run_tool(
            [sys.executable, "-c", "import os; print(os.getcwd())"],
            cwd=tmp_path,
        )
        assert rc == 0
        # Resolve to handle symlinks / case differences across platforms
        assert Path(output.strip()).resolve() == tmp_path.resolve()


class _SyncThread:
    """Fake Thread that runs target synchronously in the main thread.

    Used to bring `_read_stream` into the traced main thread so coverage
    can see its body, and to let us force `is_alive()` True for the
    thread-race branch.
    """

    is_alive_return = False

    def __init__(
        self,
        target: object = None,
        args: tuple[object, ...] = (),
        daemon: bool = False,  # noqa: ARG002 — matches threading.Thread signature
    ) -> None:
        self._target = target
        self._args = args
        self._started = False

    def start(self) -> None:
        self._started = True
        if callable(self._target):
            self._target(*self._args)

    def join(self, timeout: float | None = None) -> None:  # noqa: ARG002
        return

    def is_alive(self) -> bool:
        return type(self).is_alive_return


class TestRunToolThreadRace:
    """Covers the reader-thread-did-not-finish warning branch and
    brings `_read_stream` into the main thread for coverage tracing.
    """

    def test_read_stream_covered_via_sync_thread(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Run readers synchronously so coverage traces `_read_stream`.

        Produces output covering:
          * EOF-with-buffer path (no trailing newline)
          * \\n-split path (newline-terminated)
          * empty-buffer skip on consecutive newlines (`\\n\\n`)
        """

        class SyncOK(_SyncThread):
            is_alive_return = False

        monkeypatch.setattr(
            "furnace.adapters._subprocess.threading.Thread",
            SyncOK,
        )

        rc, output = run_tool([
            sys.executable,
            "-c",
            # stdout: no trailing newline -> EOF-with-buffer path
            # stderr: `\n\n` -> covers empty-buf skip branch at line 101
            "import sys; "
            "sys.stdout.write('partial'); sys.stdout.flush(); "
            "sys.stderr.write('done\\n\\nmore\\n'); sys.stderr.flush()",
        ])

        assert rc == 0
        assert "partial" in output
        assert "done" in output
        assert "more" in output

    def test_read_stream_oserror_logged(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Cover the OSError handler in `_read_stream`.

        Patches `threading.Thread` to run synchronously, then hands a
        stream whose `read()` raises OSError so the except branch fires.
        """
        class BoomStream:
            def read(self, _n: int) -> bytes:
                raise OSError("boom")

        # stdout raises, stderr is benign
        mock_process = MagicMock()
        mock_process.stdout = BoomStream()
        mock_process.stderr = BoomStream()
        mock_process.wait.return_value = 0
        mock_process.returncode = 0

        class SyncOK(_SyncThread):
            is_alive_return = False

        monkeypatch.setattr(
            "furnace.adapters._subprocess.threading.Thread",
            SyncOK,
        )
        monkeypatch.setattr(
            "furnace.adapters._subprocess.subprocess.Popen",
            lambda *a, **kw: mock_process,
        )

        with caplog.at_level(logging.WARNING, logger="furnace.adapters._subprocess"):
            rc, _output = run_tool(["dummy"])

        assert rc == 0
        assert any("reader died with" in rec.message for rec in caplog.records)

    def test_warns_when_reader_thread_stays_alive(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        class SyncAlive(_SyncThread):
            is_alive_return = True

        monkeypatch.setattr(
            "furnace.adapters._subprocess.threading.Thread",
            SyncAlive,
        )

        with caplog.at_level(logging.WARNING, logger="furnace.adapters._subprocess"):
            rc, _output = run_tool([sys.executable, "-c", "print('x')"])

        assert rc == 0
        assert any(
            "reader thread did not finish in 5s" in rec.message for rec in caplog.records
        )
