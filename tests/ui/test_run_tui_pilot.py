from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock, patch

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


class _HostApp(App[None]):
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


def _make_runapp(executor_fn: object = lambda _progress: None) -> RunApp:
    return RunApp(
        total_jobs=1,
        shutdown_event=threading.Event(),
        executor_fn=executor_fn,  # type: ignore[arg-type]
    )


def test_runapp_mounts_and_composes_widgets() -> None:
    executor_called = threading.Event()

    def _executor(_progress: object) -> None:
        executor_called.set()

    async def _run() -> None:
        app = _make_runapp(executor_fn=_executor)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one("#header", HeaderWidget) is not None
            assert app.query_one("#source", SourceWidget) is not None
            assert app.query_one("#target", TargetWidget) is not None
            assert app.query_one("#steps", StepsWidget) is not None
            assert app.query_one("#output", OutputLog) is not None
            assert app.query_one("#progress", ProgressWidget) is not None

    asyncio.run(_run())
    assert executor_called.wait(timeout=2.0)


def test_runapp_start_job_populates_widgets() -> None:
    job = make_job()

    async def _run() -> None:
        app = _make_runapp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._do_start_job(job, 0)
            await pilot.pause()
            header = app.query_one("#header", HeaderWidget)
            assert "1/1" in str(header.content)
            assert "movie.mkv" in str(header.content)
            steps = app.query_one("#steps", StepsWidget)
            assert str(steps.content) != ""

    asyncio.run(_run())


def test_runapp_progress_flow_and_finish() -> None:
    job = make_job()

    async def _run() -> None:
        app = _make_runapp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._do_set_chosen_quality(99)
            app._do_start_job(job, 0)
            await pilot.pause()

            app._do_set_chosen_quality(28)
            await pilot.pause()
            target0 = app.query_one("#target", TargetWidget)
            assert "28" in str(target0.content)

            app._do_update_status("Encoding...")
            await pilot.pause()

            snap = TrackerSnapshot(fraction=0.5, speed=1.25, eta_s=30.0)
            app._do_update_progress(snap)
            await pilot.pause()
            progress = app.query_one("#progress", ProgressWidget)
            assert "%" in str(progress.content)

            snap2 = TrackerSnapshot(fraction=0.75, speed=None, eta_s=None)
            app._do_update_progress(snap2)
            await pilot.pause()

            app._do_add_tool_line("ffmpeg: running")
            await pilot.pause()

            app._do_update_output_size(123_456)
            await pilot.pause()
            target = app.query_one("#target", TargetWidget)
            assert "Size:" in str(target.content)

            app._do_update_output_size(0)
            await pilot.pause()

            app._do_finish_job(job)
            await pilot.pause()
            progress2 = app.query_one("#progress", ProgressWidget)
            assert "Done" in str(progress2.content)

    asyncio.run(_run())


def test_runapp_update_status_does_not_overflow_step_list() -> None:
    job = make_job()

    async def _run() -> None:
        app = _make_runapp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._do_start_job(job, 0)
            await pilot.pause()
            app._current_step_idx = len(app._steps) - 1
            app._do_update_status("Terminal step")
            await pilot.pause()
            assert app._current_step_idx == len(app._steps) - 1

    asyncio.run(_run())


def test_runapp_refresh_progress_noop_when_no_snapshot() -> None:

    async def _run() -> None:
        app = _make_runapp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._refresh_progress()
            await pilot.pause()

    asyncio.run(_run())


def test_runapp_safe_call_swallows_exceptions_after_exit() -> None:

    async def _run() -> None:
        app = _make_runapp()
        async with app.run_test() as pilot:
            await pilot.pause()
        app._safe_call(lambda: None)

    asyncio.run(_run())


def test_runapp_public_api_methods_before_mount() -> None:
    app = _make_runapp()
    job = make_job()
    snap = TrackerSnapshot(fraction=0.1, speed=None, eta_s=None)
    app.start_job(job, 0)
    app.update_progress(snap)
    app.update_status("step")
    app.add_tool_line("line")
    app.finish_job(job)
    app.update_output_size(42)
    app.set_chosen_quality(28)
    app.stop()


def test_runapp_tool_output_respects_mute() -> None:
    app = _make_runapp()
    app._safe_call = MagicMock()  # type: ignore[method-assign]

    app.tool_output("before")
    app.mute_tool_output()
    app.tool_output("during")
    app.unmute_tool_output()
    app.tool_output("after")

    scheduled = [c.args[1] for c in app._safe_call.call_args_list]
    assert scheduled == ["before", "after"]


def test_runapp_add_tool_line_is_never_muted() -> None:
    app = _make_runapp()
    app._safe_call = MagicMock()  # type: ignore[method-assign]

    app.mute_tool_output()
    app.add_tool_line("[furnace] narration")

    scheduled = [c.args[1] for c in app._safe_call.call_args_list]
    assert scheduled == ["[furnace] narration"]


def test_runapp_tool_output_methods_before_mount() -> None:
    app = _make_runapp()
    app.tool_output("line")
    app.mute_tool_output()
    app.tool_output("dropped")
    app.unmute_tool_output()
    app.tool_output("line2")


def test_runapp_action_quit_app_sets_shutdown_and_exits() -> None:
    shutdown = threading.Event()

    app = RunApp(
        total_jobs=1,
        shutdown_event=shutdown,
        executor_fn=lambda _p: None,
    )

    with (
        patch("furnace.ui.run_tui.os._exit") as m_exit,
        patch("furnace.ui.run_tui.psutil.Process") as m_proc,
    ):
        fake_child = m_proc.return_value.children.return_value = [
            type("FakeChild", (), {"kill": lambda self: None})(),
        ]
        _ = fake_child
        app.action_quit_app()

    assert shutdown.is_set()
    m_exit.assert_called_once_with(0)


def test_runapp_action_quit_app_skips_dead_children() -> None:
    import psutil

    shutdown = threading.Event()
    app = RunApp(
        total_jobs=1,
        shutdown_event=shutdown,
        executor_fn=lambda _p: None,
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
