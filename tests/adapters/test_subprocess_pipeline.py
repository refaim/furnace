from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from furnace.adapters._subprocess import run_pipeline


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


class TestRunPipelineRealProcess:
    def test_happy_path_pipes_data(self) -> None:
        producer = _py("print('hello')")
        consumer = _py(
            "import sys; data = sys.stdin.read(); sys.exit(0 if 'hello' in data else 1)",
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
            "import sys; sys.stderr.write('PRODERR\\n'); sys.stdout.write('x')",
        )
        consumer = _py("import sys; sys.stdin.read()")
        rc, _ = run_pipeline(producer, consumer, on_output=received.append)
        assert rc == 0
        assert any("PRODERR" in line for line in received)

    def test_consumer_stderr_captured(self) -> None:
        received: list[str] = []
        producer = _py("print('x')")
        consumer = _py(
            "import sys; sys.stdin.read(); sys.stderr.write('CONERR\\n')",
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
        marker = tmp_path / "marker.bin"
        producer = _py(
            "import sys; sys.stdout.buffer.write(b'AB\\x00CD\\xff')",
        )
        consumer = _py(
            f"import sys; open({str(marker)!r}, 'wb').write(sys.stdin.buffer.read())",
        )
        rc, _ = run_pipeline(producer, consumer)
        assert rc == 0
        assert marker.read_bytes() == b"AB\x00CD\xff"

    def test_log_file_contains_both_commands_and_exit_codes(
        self,
        tmp_path: Path,
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
            "import os; open('prod.txt', 'w').write(os.getcwd()); print('done')",
        )
        consumer = _py("import sys; sys.stdin.read()")
        rc, _ = run_pipeline(producer, consumer, cwd=tmp_path)
        assert rc == 0
        out = (tmp_path / "prod.txt").read_text(encoding="utf-8")
        assert Path(out).resolve() == tmp_path.resolve()

    def test_cwd_forwarded_to_consumer(self, tmp_path: Path) -> None:
        producer = _py("print('x')")
        consumer = _py(
            "import os, sys; sys.stdin.read(); open('cons.txt', 'w').write(os.getcwd())",
        )
        rc, _ = run_pipeline(producer, consumer, cwd=tmp_path)
        assert rc == 0
        out = (tmp_path / "cons.txt").read_text(encoding="utf-8")
        assert Path(out).resolve() == tmp_path.resolve()

    def test_producer_killed_when_it_outlives_consumer(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        real_popen = subprocess.Popen
        spawned: list[subprocess.Popen[bytes]] = []

        def fake_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
            proc: subprocess.Popen[bytes] = real_popen(*args, **kwargs)
            spawned.append(proc)
            if len(spawned) == 1:
                real_wait = proc.wait
                calls = {"n": 0}

                def fake_wait(timeout: float | None = None) -> int:
                    calls["n"] += 1
                    if calls["n"] == 1 and timeout is not None:
                        raise subprocess.TimeoutExpired(cmd="producer", timeout=timeout)
                    return int(real_wait())

                proc.wait = fake_wait  # type: ignore[method-assign]
            return proc

        producer = _py(
            "import signal, sys, time; "
            "signal.signal(signal.SIGPIPE, signal.SIG_IGN); "
            "sys.stdout.write('x'); sys.stdout.flush(); "
            "time.sleep(60)",
        )
        consumer = _py("import sys; sys.stdin.read(1); sys.exit(0)")

        with (
            patch(
                "furnace.adapters._subprocess.subprocess.Popen",
                side_effect=fake_popen,
            ),
            caplog.at_level(
                logging.WARNING,
                logger="furnace.adapters._subprocess",
            ),
        ):
            rc, _ = run_pipeline(producer, consumer)

        assert rc != 0
        warn_records = [rec for rec in caplog.records if "did not exit 10s after consumer" in rec.message]
        assert len(warn_records) == 1
        assert spawned[0].returncode is not None
        assert spawned[0].returncode != 0


class TestRunPipelinePipeValidation:
    def _mock_proc(self, *, stdout: object, stderr: object) -> MagicMock:
        m = MagicMock()
        m.stdout = stdout
        m.stderr = stderr
        return m

    def test_raises_when_producer_stdout_missing(self) -> None:
        prod = self._mock_proc(stdout=None, stderr=MagicMock())
        cons = self._mock_proc(stdout=MagicMock(), stderr=MagicMock())
        with (
            patch(
                "furnace.adapters._subprocess.subprocess.Popen",
                side_effect=[prod, cons],
            ),
            pytest.raises(RuntimeError, match="pipes"),
        ):
            run_pipeline(["x"], ["y"])

    def test_raises_when_producer_stderr_missing(self) -> None:
        prod = self._mock_proc(stdout=MagicMock(), stderr=None)
        cons = self._mock_proc(stdout=MagicMock(), stderr=MagicMock())
        with (
            patch(
                "furnace.adapters._subprocess.subprocess.Popen",
                side_effect=[prod, cons],
            ),
            pytest.raises(RuntimeError, match="pipes"),
        ):
            run_pipeline(["x"], ["y"])

    def test_raises_when_consumer_stdout_missing(self) -> None:
        prod = self._mock_proc(stdout=MagicMock(), stderr=MagicMock())
        cons = self._mock_proc(stdout=None, stderr=MagicMock())
        with (
            patch(
                "furnace.adapters._subprocess.subprocess.Popen",
                side_effect=[prod, cons],
            ),
            pytest.raises(RuntimeError, match="pipes"),
        ):
            run_pipeline(["x"], ["y"])

    def test_raises_when_consumer_stderr_missing(self) -> None:
        prod = self._mock_proc(stdout=MagicMock(), stderr=MagicMock())
        cons = self._mock_proc(stdout=MagicMock(), stderr=None)
        with (
            patch(
                "furnace.adapters._subprocess.subprocess.Popen",
                side_effect=[prod, cons],
            ),
            pytest.raises(RuntimeError, match="pipes"),
        ):
            run_pipeline(["x"], ["y"])

    def test_producer_pipe_guard_closes_log_file(
        self,
        tmp_path: Path,
    ) -> None:
        log = tmp_path / "pipe.log"
        prod = self._mock_proc(stdout=None, stderr=MagicMock())
        cons = self._mock_proc(stdout=MagicMock(), stderr=MagicMock())
        with (
            patch(
                "furnace.adapters._subprocess.subprocess.Popen",
                side_effect=[prod, cons],
            ),
            pytest.raises(RuntimeError, match="pipes"),
        ):
            run_pipeline(["x"], ["y"], log_path=log)
        assert log.exists()

    def test_consumer_pipe_guard_closes_log_file(
        self,
        tmp_path: Path,
    ) -> None:
        log = tmp_path / "pipe.log"
        prod = self._mock_proc(stdout=MagicMock(), stderr=MagicMock())
        cons = self._mock_proc(stdout=MagicMock(), stderr=None)
        with (
            patch(
                "furnace.adapters._subprocess.subprocess.Popen",
                side_effect=[prod, cons],
            ),
            pytest.raises(RuntimeError, match="pipes"),
        ):
            run_pipeline(["x"], ["y"], log_path=log)
        assert log.exists()


class _SyncThread:
    is_alive_return = False

    def __init__(
        self,
        target: Callable[..., object],
        args: tuple[object, ...] = (),
        daemon: bool = False,  # noqa: ARG002 — matches threading.Thread signature
    ) -> None:
        self._target = target
        self._args = args
        self._started = False

    def start(self) -> None:
        self._started = True
        self._target(*self._args)

    def join(self, timeout: float | None = None) -> None:  # noqa: ARG002
        return

    def is_alive(self) -> bool:
        return type(self).is_alive_return


class TestRunPipelineThreadRace:
    def test_read_stream_covered_via_sync_thread(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:

        class SyncOK(_SyncThread):
            is_alive_return = False

        monkeypatch.setattr(
            "furnace.adapters._subprocess.threading.Thread",
            SyncOK,
        )

        producer = _py(
            "import sys; "
            "sys.stderr.write('done\\n\\nmore\\n'); sys.stderr.flush(); "
            "sys.stderr.write('partial'); sys.stderr.flush(); "
            "sys.stdout.write('x'); sys.stdout.flush()",
        )
        consumer = _py("import sys; sys.stdin.read()")
        rc, output = run_pipeline(producer, consumer)

        assert rc == 0
        assert "partial" in output
        assert "done" in output
        assert "more" in output

    def test_read_stream_oserror_logged(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:

        class BoomStream:
            def read(self, _n: int) -> bytes:
                raise OSError("boom")

        prod = MagicMock()
        prod.stdout = MagicMock()
        prod.stderr = BoomStream()
        prod.wait.return_value = 0
        prod.returncode = 0

        cons = MagicMock()
        cons.stdout = BoomStream()
        cons.stderr = BoomStream()
        cons.wait.return_value = 0
        cons.returncode = 0

        class SyncOK(_SyncThread):
            is_alive_return = False

        monkeypatch.setattr(
            "furnace.adapters._subprocess.threading.Thread",
            SyncOK,
        )
        monkeypatch.setattr(
            "furnace.adapters._subprocess.subprocess.Popen",
            lambda *a, **kw: prod if a[0] == ["p"] else cons,
        )

        with caplog.at_level(
            logging.WARNING,
            logger="furnace.adapters._subprocess",
        ):
            rc, _output = run_pipeline(["p"], ["c"])

        assert rc == 0
        assert any("run_pipeline reader died with" in rec.message for rec in caplog.records)

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

        producer = _py("import sys; sys.stdout.write('x')")
        consumer = _py("import sys; sys.stdin.read()")
        with caplog.at_level(
            logging.WARNING,
            logger="furnace.adapters._subprocess",
        ):
            rc, _output = run_pipeline(producer, consumer)

        assert rc == 0
        warn_records = [rec for rec in caplog.records if "reader thread(s) did not finish in 5s" in rec.message]
        assert len(warn_records) == 1
        msg = warn_records[0].getMessage()
        assert "producer.stderr" in msg
        assert "consumer.stdout" in msg
        assert "consumer.stderr" in msg
