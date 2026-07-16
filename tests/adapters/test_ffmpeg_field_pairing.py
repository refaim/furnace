"""Tests for ``FFmpegAdapter.sample_field_pairing`` (field-separated probing).

The adapter counts decoded frames against demuxed packets over one window via
``ffprobe -count_frames -count_packets``; the pure ratio math lives in
``core.detect``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from furnace.adapters.ffmpeg import FFmpegAdapter


def _streams_json(frames: Any, packets: Any) -> str:
    return json.dumps({"streams": [{"nb_read_frames": frames, "nb_read_packets": packets}]})


def _fake_result(stdout: str, returncode: int = 0) -> MagicMock:
    result = MagicMock()
    result.stdout = stdout
    result.returncode = returncode
    return result


def test_sample_field_pairing_returns_frame_and_packet_counts() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))

    with patch(
        "furnace.adapters.ffmpeg.subprocess.run",
        return_value=_fake_result(_streams_json("1500", "3000")),
    ):
        assert adapter.sample_field_pairing(Path("v.mkv")) == (1500, 3000)


def test_sample_field_pairing_command_shape() -> None:
    """One window from the start: a seek would strand packets before the first
    keyframe with no decoded frame to pair them against and skew the ratio."""
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))

    with patch(
        "furnace.adapters.ffmpeg.subprocess.run",
        return_value=_fake_result(_streams_json("1500", "3000")),
    ) as mock_run:
        adapter.sample_field_pairing(Path("v.mkv"))

    cmd = mock_run.call_args_list[0].args[0]
    assert cmd[0] == "ffprobe"
    assert "-count_frames" in cmd
    assert "-count_packets" in cmd
    assert cmd[cmd.index("-show_entries") + 1] == "stream=nb_read_frames,nb_read_packets"
    assert cmd[cmd.index("-select_streams") + 1] == "v:0"
    assert cmd[cmd.index("-read_intervals") + 1] == "%+60"
    assert cmd[-1] == "v.mkv"


def test_sample_field_pairing_ffprobe_failure_is_empty() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))

    with patch(
        "furnace.adapters.ffmpeg.subprocess.run",
        return_value=_fake_result("", returncode=1),
    ):
        assert adapter.sample_field_pairing(Path("v.mkv")) == (0, 0)


def test_sample_field_pairing_unparseable_json_is_empty() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))

    with patch(
        "furnace.adapters.ffmpeg.subprocess.run",
        return_value=_fake_result("not json"),
    ):
        assert adapter.sample_field_pairing(Path("v.mkv")) == (0, 0)


def test_sample_field_pairing_no_streams_is_empty() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    payload: dict[str, Any] = {"streams": []}

    with patch(
        "furnace.adapters.ffmpeg.subprocess.run",
        return_value=_fake_result(json.dumps(payload)),
    ):
        assert adapter.sample_field_pairing(Path("v.mkv")) == (0, 0)


def test_sample_field_pairing_missing_keys_is_empty() -> None:
    """ffprobe omits the counters when it cannot decode the stream at all."""
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    payload: dict[str, Any] = {"streams": [{}]}

    with patch(
        "furnace.adapters.ffmpeg.subprocess.run",
        return_value=_fake_result(json.dumps(payload)),
    ):
        assert adapter.sample_field_pairing(Path("v.mkv")) == (0, 0)


def test_sample_field_pairing_non_integer_counter_is_empty() -> None:
    """ffprobe reports an unknown counter as "N/A", not a number."""
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))

    with patch(
        "furnace.adapters.ffmpeg.subprocess.run",
        return_value=_fake_result(_streams_json("N/A", "3000")),
    ):
        assert adapter.sample_field_pairing(Path("v.mkv")) == (0, 0)
