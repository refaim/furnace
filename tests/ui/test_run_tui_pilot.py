"""Textual Pilot tests for `furnace.ui.run_tui` widgets and `RunApp`.

These exercise the declarative widget classes (HeaderWidget, SourceWidget,
TargetWidget, StepsWidget, OutputLog, ProgressWidget) and the public API
of RunApp end-to-end via Textual's async headless harness
(`App.run_test() -> Pilot`). Because the test suite is driven by plain
pytest (no pytest-asyncio plugin), each test wraps its async body in
`asyncio.run(...)`.
"""
from __future__ import annotations

import asyncio
import threading
from unittest.mock import patch

from textual.app import App, ComposeResult

from furnace.core.progress import TrackerSnapshot
from furnace.ui.run_tui import (
    HeaderWidget,
    OutputLog,
    ProgressWidget,
    RunApp,
    SourceWidget,
    StepsWidget,
    TargetWidget,
)
from tests.conftest import make_job

# ---------------------------------------------------------------------------
# Widget smoke tests — mount each Static-subclass in a minimal host App
# ---------------------------------------------------------------------------


class _HostApp(App[None]):
    """Tiny host App that simply mounts the widgets we want to render."""

    def __init__(self) -> None:
        super().__init__()

    def compose(self) -> ComposeResult:
        yield HeaderWidget("header-text", id="header")
        yield SourceWidget("source-text", id="source")
        yield TargetWidget("target-text", id="target")
        yield StepsWidget("steps-text", id="steps")
        yield OutputLog(id="output")
        yield ProgressWidget("progress-text", id="progress")


def test_widgets_mount_and_render_without_error() -> None:
    """Each widget class must mount inside a live App and render cleanly."""

    async def _run() -> None:
        app = _HostApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one("#header", HeaderWidget) is not None
            assert app.query_one("#source", SourceWidget) is not None
            assert app.query_one("#target", TargetWidget) is not None
            assert app.query_one("#steps", StepsWidget) is not None
            assert app.query_one("#output", OutputLog) is not None
            assert app.query_one("#progress", ProgressWidget) is not None

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# RunApp pilot tests — drive the public API and verify state transitions
# ---------------------------------------------------------------------------


def _make_runapp(executor_fn: object = lambda _progress: None) -> RunApp:
    """Construct a RunApp with safe defaults for tests."""
    return RunApp(
        total_jobs=1,
        shutdown_event=threading.Event(),
        executor_fn=executor_fn,  # type: ignore[arg-type]
        vmaf_enabled=False,
    )


def test_runapp_mounts_and_composes_widgets() -> None:
    """RunApp.compose() yields all six widgets; on_mount spawns a worker."""
    executor_called = threading.Event()

    def _executor(_progress: object) -> None:
        executor_called.set()

    async def _run() -> None:
        app = _make_runapp(executor_fn=_executor)
        async with app.run_test() as pilot:
            await pilot.pause()
            # All six widgets are mounted
            assert app.query_one("#header", HeaderWidget) is not None
            assert app.query_one("#source", SourceWidget) is not None
            assert app.query_one("#target", TargetWidget) is not None
            assert app.query_one("#steps", StepsWidget) is not None
            assert app.query_one("#output", OutputLog) is not None
            assert app.query_one("#progress", ProgressWidget) is not None

    asyncio.run(_run())
    # The executor worker thread must have been invoked by on_mount
    assert executor_called.wait(timeout=2.0)


def test_runapp_start_job_populates_widgets() -> None:
    """start_job updates header, source, target, steps and resets progress."""
    job = make_job()

    async def _run() -> None:
        app = _make_runapp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._do_start_job(job, 0)
            await pilot.pause()
            header = app.query_one("#header", HeaderWidget)
            # Header renders "[1/1] <filename>"
            assert "1/1" in str(header.content)
            assert "movie.mkv" in str(header.content)
            # Steps widget has content
            steps = app.query_one("#steps", StepsWidget)
            assert str(steps.content) != ""

    asyncio.run(_run())


def test_runapp_progress_flow_and_finish() -> None:
    """update_status -> update_progress -> update_output_size -> finish_job."""
    job = make_job()

    async def _run() -> None:
        app = _make_runapp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._do_start_job(job, 0)
            await pilot.pause()

            # Advance step
            app._do_update_status("Encoding...")
            await pilot.pause()

            # Push a progress snapshot with speed + ETA
            snap = TrackerSnapshot(fraction=0.5, speed=1.25, eta_s=30.0)
            app._do_update_progress(snap)
            await pilot.pause()
            progress = app.query_one("#progress", ProgressWidget)
            assert "%" in str(progress.content)

            # Snapshot without speed/eta — still renders
            snap2 = TrackerSnapshot(fraction=0.75, speed=None, eta_s=None)
            app._do_update_progress(snap2)
            await pilot.pause()

            # Output log line
            app._do_add_tool_line("ffmpeg: running")
            await pilot.pause()

            # Output size updates
            app._do_update_output_size(123_456)
            await pilot.pause()
            target = app.query_one("#target", TargetWidget)
            assert "Size:" in str(target.content)

            # Output size == 0 renders "..."
            app._do_update_output_size(0)
            await pilot.pause()

            # Finish
            app._do_finish_job(job)
            await pilot.pause()
            progress2 = app.query_one("#progress", ProgressWidget)
            assert "Done" in str(progress2.content)

    asyncio.run(_run())


def test_runapp_update_status_does_not_overflow_step_list() -> None:
    """_do_update_status must not advance past the last step."""
    job = make_job()

    async def _run() -> None:
        app = _make_runapp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._do_start_job(job, 0)
            await pilot.pause()
            # Jump to the final step, then call _do_update_status: must stay put.
            app._current_step_idx = len(app._steps) - 1
            app._do_update_status("Terminal step")
            await pilot.pause()
            assert app._current_step_idx == len(app._steps) - 1

    asyncio.run(_run())


def test_runapp_refresh_progress_noop_when_no_snapshot() -> None:
    """_refresh_progress returns early if snapshot is None."""

    async def _run() -> None:
        app = _make_runapp()
        async with app.run_test() as pilot:
            await pilot.pause()
            # _snapshot starts as None; direct call must be a no-op
            app._refresh_progress()
            await pilot.pause()

    asyncio.run(_run())


def test_runapp_safe_call_swallows_exceptions_after_exit() -> None:
    """_safe_call silently ignores errors if the app has exited."""

    async def _run() -> None:
        app = _make_runapp()
        async with app.run_test() as pilot:
            await pilot.pause()
        # After exit, call_from_thread raises — _safe_call must swallow it
        app._safe_call(lambda: None)

    asyncio.run(_run())


def test_runapp_public_api_methods_before_mount() -> None:
    """Public thread-safe API methods must not raise even if app is not mounted.

    These exercise the thin wrappers (start_job, update_progress, update_status,
    add_tool_line, finish_job, update_output_size, stop) so their bodies are
    covered without needing a real worker thread.
    """
    app = _make_runapp()
    job = make_job()
    snap = TrackerSnapshot(fraction=0.1, speed=None, eta_s=None)
    # None of these should raise — _safe_call swallows the error cleanly when
    # the app has no event loop yet.
    app.start_job(job, 0)
    app.update_progress(snap)
    app.update_status("step")
    app.add_tool_line("line")
    app.finish_job(job)
    app.update_output_size(42)
    app.stop()


def test_runapp_action_quit_app_sets_shutdown_and_exits() -> None:
    """ESC triggers action_quit_app: shutdown event set, os._exit called."""
    shutdown = threading.Event()

    app = RunApp(
        total_jobs=1,
        shutdown_event=shutdown,
        executor_fn=lambda _p: None,
        vmaf_enabled=False,
    )

    with (
        patch("furnace.ui.run_tui.os._exit") as m_exit,
        patch("furnace.ui.run_tui.psutil.Process") as m_proc,
    ):
        # psutil.Process(...).children(recursive=True) -> one fake child
        fake_child = m_proc.return_value.children.return_value = [
            type("FakeChild", (), {"kill": lambda self: None})(),
        ]
        _ = fake_child  # silence linter: the fake is used by the mock attr chain
        app.action_quit_app()

    assert shutdown.is_set()
    m_exit.assert_called_once_with(0)


def test_runapp_action_quit_app_skips_dead_children() -> None:
    """action_quit_app tolerates psutil.NoSuchProcess from killed children."""
    import psutil

    shutdown = threading.Event()
    app = RunApp(
        total_jobs=1,
        shutdown_event=shutdown,
        executor_fn=lambda _p: None,
        vmaf_enabled=False,
    )

    class _DeadChild:
        def kill(self) -> None:
            raise psutil.NoSuchProcess(pid=-1)

    with (
        patch("furnace.ui.run_tui.os._exit") as m_exit,
        patch("furnace.ui.run_tui.psutil.Process") as m_proc,
    ):
        m_proc.return_value.children.return_value = [_DeadChild()]
        app.action_quit_app()

    assert shutdown.is_set()
    m_exit.assert_called_once_with(0)
