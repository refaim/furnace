"""Test the filter chain built by FFmpegAdapter.stereo_to_mono_wav.

``run_tool`` is patched in every test — no real ffmpeg invocation. We only
verify the command line the adapter builds (stereo pan formula, delay
handling, exit-code propagation, log-path behaviour).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from furnace.adapters.ffmpeg import FFmpegAdapter

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
    """Run the adapter with run_tool patched, return the -af value."""
    with patch("furnace.adapters.ffmpeg.run_tool") as run_tool:
        run_tool.return_value = (0, "")
        adapter.stereo_to_mono_wav(
            input_path=tmp_path / "a.mkv",
            stream_index=1,
            output_wav=tmp_path / "out.wav",
            delay_ms=delay_ms,
        )
    return _af_value(run_tool.call_args)


def test_stereo_averages_fronts(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path)
    assert PAN_STEREO in af


def test_no_alimiter(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path)
    assert "alimiter" not in af


def test_no_layout_normalizer(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path)
    assert "aformat=" not in af


def test_zero_delay_has_no_delay_filter(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path, delay_ms=0)
    assert "adelay" not in af
    assert "atrim" not in af


def test_positive_delay_appends_adelay(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path, delay_ms=50)
    assert "adelay=50" in af
    assert PAN_STEREO in af


def test_negative_delay_appends_atrim(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    """delay_ms<0 appends atrim=start=<abs(ms)/1000:.3f> to trim lead-in."""
    af = _invoke(adapter, tmp_path, delay_ms=-50)
    assert "atrim=start=0.050" in af
    assert PAN_STEREO in af
    assert "adelay" not in af


def test_returns_run_tool_exit_code(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    """Propagating ffmpeg's exit code lets the executor branch on failure."""
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
    adapter: FFmpegAdapter, tmp_path: Path,
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
