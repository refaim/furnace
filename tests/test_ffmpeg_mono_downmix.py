from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from furnace.adapters.ffmpeg import FFmpegAdapter
from furnace.core.progress import ProgressSample

PAN_STEREO = "pan=mono|c0=0.5*FL+0.5*FR"


@pytest.fixture
def adapter() -> FFmpegAdapter:
    return FFmpegAdapter(
        ffmpeg_path=Path("ffmpeg"),
        ffprobe_path=Path("ffprobe"),
    )


def _af_value(call_args: Any) -> str:
    cmd: list[str] = call_args[0][0]
    return cmd[cmd.index("-af") + 1]


def _invoke(
    adapter: FFmpegAdapter,
    tmp_path: Path,
    *,
    delay_ms: int = 0,
) -> str:
    with patch("furnace.adapters.ffmpeg.run_tool") as run_tool:
        run_tool.return_value = (0, "")
        adapter.stereo_to_mono_wav(
            input_path=tmp_path / "a.mkv",
            stream_index=1,
            output_wav=tmp_path / "out.wav",
            delay_ms=delay_ms,
        )
    return _af_value(run_tool.call_args)


def _invoke_cmd(adapter: FFmpegAdapter, tmp_path: Path) -> list[str]:
    with patch("furnace.adapters.ffmpeg.run_tool") as run_tool:
        run_tool.return_value = (0, "")
        adapter.stereo_to_mono_wav(
            input_path=tmp_path / "a.mkv",
            stream_index=1,
            output_wav=tmp_path / "out.wav",
            delay_ms=0,
        )
    return [str(c) for c in run_tool.call_args[0][0]]


def test_stereo_averages_fronts(
    adapter: FFmpegAdapter,
    tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path)
    assert PAN_STEREO in af


def test_mono_wav_asks_for_24_bit(
    adapter: FFmpegAdapter,
    tmp_path: Path,
) -> None:
    cmd = _invoke_cmd(adapter, tmp_path)
    assert ("-c:a", "pcm_s24le") in pairwise(cmd)


def test_no_alimiter(
    adapter: FFmpegAdapter,
    tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path)
    assert "alimiter" not in af


def test_no_layout_normalizer(
    adapter: FFmpegAdapter,
    tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path)
    assert "aformat=" not in af


def test_zero_delay_has_no_delay_filter(
    adapter: FFmpegAdapter,
    tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path, delay_ms=0)
    assert "adelay" not in af
    assert "atrim" not in af


def test_positive_delay_appends_adelay(
    adapter: FFmpegAdapter,
    tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path, delay_ms=50)
    assert "adelay=50" in af
    assert PAN_STEREO in af


def test_negative_delay_appends_atrim(
    adapter: FFmpegAdapter,
    tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path, delay_ms=-50)
    assert "atrim=start=0.050" in af
    assert PAN_STEREO in af
    assert "adelay" not in af


def test_returns_run_tool_exit_code(
    adapter: FFmpegAdapter,
    tmp_path: Path,
) -> None:
    with patch("furnace.adapters.ffmpeg.run_tool") as run_tool:
        run_tool.return_value = (42, "")
        rc = adapter.stereo_to_mono_wav(
            input_path=tmp_path / "a.mkv",
            stream_index=1,
            output_wav=tmp_path / "out.wav",
            delay_ms=0,
        )
    assert rc == 42


def test_log_path_uses_log_dir_when_set(
    adapter: FFmpegAdapter,
    tmp_path: Path,
) -> None:
    adapter.set_log_dir(tmp_path)
    with patch("furnace.adapters.ffmpeg.run_tool") as run_tool:
        run_tool.return_value = (0, "")
        adapter.stereo_to_mono_wav(
            input_path=tmp_path / "a.mkv",
            stream_index=7,
            output_wav=tmp_path / "out.wav",
            delay_ms=0,
        )
    assert run_tool.call_args.kwargs["log_path"] == tmp_path / "ffmpeg_mono_s7.log"


def test_command_has_progress_pipe(
    adapter: FFmpegAdapter,
    tmp_path: Path,
) -> None:
    with patch("furnace.adapters.ffmpeg.run_tool") as run_tool:
        run_tool.return_value = (0, "")
        adapter.stereo_to_mono_wav(
            input_path=tmp_path / "a.mkv",
            stream_index=1,
            output_wav=tmp_path / "out.wav",
            delay_ms=0,
        )
    cmd: list[str] = run_tool.call_args[0][0]
    assert "-progress" in cmd
    assert cmd[cmd.index("-progress") + 1] == "pipe:1"


def test_passes_on_progress_line_hook_to_run_tool(
    adapter: FFmpegAdapter,
    tmp_path: Path,
) -> None:
    with patch("furnace.adapters.ffmpeg.run_tool") as run_tool:
        run_tool.return_value = (0, "")
        adapter.stereo_to_mono_wav(
            input_path=tmp_path / "a.mkv",
            stream_index=1,
            output_wav=tmp_path / "out.wav",
            delay_ms=0,
        )
    hook = run_tool.call_args.kwargs.get("on_progress_line")
    assert hook is not None
    assert callable(hook)


def test_on_progress_callback_fires_on_progress_block(
    adapter: FFmpegAdapter,
    tmp_path: Path,
) -> None:
    samples: list[ProgressSample] = []

    def cb(s: ProgressSample) -> None:
        samples.append(s)

    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> tuple[int, str]:
        captured.update(kwargs)
        return (0, "")

    with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run):
        adapter.stereo_to_mono_wav(
            input_path=tmp_path / "a.mkv",
            stream_index=1,
            output_wav=tmp_path / "out.wav",
            delay_ms=0,
            on_progress=cb,
        )

    hook = captured["on_progress_line"]
    hook("out_time_us=1000000")
    hook("speed=1.5x")
    hook("fps=24")
    hook("progress=continue")
    assert len(samples) == 1


def test_on_progress_line_returns_true_for_kv_false_otherwise(
    adapter: FFmpegAdapter,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> tuple[int, str]:
        captured.update(kwargs)
        return (0, "")

    with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run):
        adapter.stereo_to_mono_wav(
            input_path=tmp_path / "a.mkv",
            stream_index=1,
            output_wav=tmp_path / "out.wav",
            delay_ms=0,
        )

    hook = captured["on_progress_line"]
    assert hook("frame=42") is True
    assert hook("Estimating duration from bitrate") is False


def test_on_progress_not_called_when_callback_none(
    adapter: FFmpegAdapter,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> tuple[int, str]:
        captured.update(kwargs)
        return (0, "")

    with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run):
        adapter.stereo_to_mono_wav(
            input_path=tmp_path / "a.mkv",
            stream_index=1,
            output_wav=tmp_path / "out.wav",
            delay_ms=0,
        )

    hook = captured["on_progress_line"]
    hook("out_time_us=1000000")
    hook("progress=continue")
