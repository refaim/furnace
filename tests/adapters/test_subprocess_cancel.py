from __future__ import annotations

import sys
import threading
import time

from furnace.adapters._subprocess import run_tool


def test_cancel_event_kills_child_promptly() -> None:
    cancel = threading.Event()

    def _trigger() -> None:
        time.sleep(0.2)
        cancel.set()

    threading.Thread(target=_trigger, daemon=True).start()

    start = time.monotonic()
    rc, _ = run_tool(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        cancel_event=cancel,
    )
    elapsed = time.monotonic() - start
    assert elapsed < 1.0
    assert rc != 0


def test_no_cancel_event_runs_to_completion() -> None:
    rc, output = run_tool(
        [sys.executable, "-c", "print('hello')"],
    )
    assert rc == 0
    assert "hello" in output


def test_cancel_event_unset_lets_process_finish_naturally() -> None:
    # cancel_event provided but never set: the polled-wait loop must exit
    # when the child completes on its own (covers the `while ... is None`
    # False-branch through to the trailing `process.wait()`).
    cancel = threading.Event()
    rc, output = run_tool(
        [sys.executable, "-c", "print('done')"],
        cancel_event=cancel,
    )
    assert rc == 0
    assert "done" in output
    assert not cancel.is_set()
