"""Test the filter chain built by FFmpegAdapter.downmix_to_mono_wav.

``run_tool`` is patched in every test — no real ffmpeg invocation. We only
verify the command line the adapter builds (pan formula, aformat layout
normalizer, alimiter peak guard, delay handling, channel-count gate).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from furnace.adapters.ffmpeg import FFmpegAdapter

PAN_5_1 = "pan=mono|c0=0.707*FC+0.5*FL+0.5*FR+0.354*BL+0.354*BR"
PAN_7_1 = (
    "pan=mono|c0=0.707*FC+0.5*FL+0.5*FR"
    "+0.354*SL+0.354*SR+0.354*BL+0.354*BR"
)
PAN_STEREO = "pan=mono|c0=0.5*FL+0.5*FR"
ALIMITER = "alimiter=limit=0.99"


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
    channels: int,
    delay_ms: int = 0,
) -> str:
    """Run the adapter with run_tool patched, return the -af value."""
    with patch("furnace.adapters.ffmpeg.run_tool") as run_tool:
        run_tool.return_value = (0, "")
        adapter.downmix_to_mono_wav(
            input_path=tmp_path / "a.mkv",
            stream_index=1,
            channels=channels,
            output_wav=tmp_path / "out.wav",
            delay_ms=delay_ms,
        )
    return _af_value(run_tool.call_args)


def test_5_1_uses_itu_formula(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path, channels=6)
    assert PAN_5_1 in af


def test_7_1_uses_itu_formula(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path, channels=8)
    assert PAN_7_1 in af


def test_stereo_averages_fronts(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path, channels=2)
    assert PAN_STEREO in af


def test_5_1_appends_alimiter(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path, channels=6)
    assert ALIMITER in af


def test_7_1_appends_alimiter(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path, channels=8)
    assert ALIMITER in af


def test_stereo_has_no_alimiter(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path, channels=2)
    assert "alimiter" not in af


def test_5_1_prepends_layout_normalizer(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path, channels=6)
    assert "aformat=channel_layouts=5.1" in af
    # aformat must come before the pan filter so canonical channel names
    # are available to the pan expression.
    assert af.index("aformat=channel_layouts=5.1") < af.index(PAN_5_1)


def test_7_1_prepends_layout_normalizer(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path, channels=8)
    assert "aformat=channel_layouts=7.1" in af
    assert af.index("aformat=channel_layouts=7.1") < af.index(PAN_7_1)


def test_stereo_has_no_layout_normalizer(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path, channels=2)
    assert "aformat=" not in af


def test_positive_delay_applied_after_downmix(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path, channels=6, delay_ms=50)
    assert "adelay=50" in af
    assert PAN_5_1 in af
    assert ALIMITER in af


def test_negative_delay_trims_lead_in(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    """delay_ms<0 appends atrim=start=<abs(ms)/1000:.3f> to trim lead-in."""
    af = _invoke(adapter, tmp_path, channels=6, delay_ms=-50)
    assert "atrim=start=0.050" in af
    assert PAN_5_1 in af
    # adelay must NOT appear for a negative delay.
    assert "adelay" not in af


def test_rejects_unsupported_channel_count(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="channels"):
        adapter.downmix_to_mono_wav(
            input_path=tmp_path / "a.mkv",
            stream_index=1,
            channels=3,
            output_wav=tmp_path / "out.wav",
            delay_ms=0,
        )


def test_returns_run_tool_exit_code(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    """Propagating ffmpeg's exit code lets the executor branch on failure."""
    with patch("furnace.adapters.ffmpeg.run_tool") as run_tool:
        run_tool.return_value = (42, "")
        rc = adapter.downmix_to_mono_wav(
            input_path=tmp_path / "a.mkv",
            stream_index=1,
            channels=6,
            output_wav=tmp_path / "out.wav",
            delay_ms=0,
        )
    assert rc == 42
