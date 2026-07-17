from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from furnace.adapters.ffmpeg import FFmpegAdapter


def _frames_json(flags: list[int]) -> str:
    return json.dumps({"frames": [{"repeat_pict": f} for f in flags]})


def _fake_result(stdout: str, returncode: int = 0) -> MagicMock:
    result = MagicMock()
    result.stdout = stdout
    result.returncode = returncode
    return result


def test_sample_repeat_pict_probes_five_points_and_concatenates() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))

    with patch(
        "furnace.adapters.ffmpeg.subprocess.run",
        return_value=_fake_result(_frames_json([0, 1, 0, 1])),
    ) as mock_run:
        flags = adapter.sample_repeat_pict(Path("v.mkv"), duration_s=1000.0)

    assert mock_run.call_count == 5
    assert flags == [0, 1, 0, 1] * 5
    seeks = []
    for call in mock_run.call_args_list:
        cmd = call.args[0]
        seeks.append(cmd[cmd.index("-read_intervals") + 1])
    assert seeks == [
        "100.00%+#500",
        "300.00%+#500",
        "500.00%+#500",
        "700.00%+#500",
        "900.00%+#500",
    ]


def test_sample_repeat_pict_command_shape() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))

    with patch(
        "furnace.adapters.ffmpeg.subprocess.run",
        return_value=_fake_result(_frames_json([])),
    ) as mock_run:
        adapter.sample_repeat_pict(Path("v.mkv"), duration_s=1000.0)

    cmd = mock_run.call_args_list[0].args[0]
    assert cmd[0] == "ffprobe"
    assert "-show_frames" in cmd
    assert cmd[cmd.index("-show_entries") + 1] == "frame=repeat_pict"
    assert cmd[cmd.index("-select_streams") + 1] == "v:0"
    assert cmd[-1] == "v.mkv"


def test_sample_repeat_pict_missing_key_counts_as_zero() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    stdout = json.dumps({"frames": [{"repeat_pict": 1}, {}]})

    with patch(
        "furnace.adapters.ffmpeg.subprocess.run",
        return_value=_fake_result(stdout),
    ):
        flags = adapter.sample_repeat_pict(Path("v.mkv"), duration_s=1000.0)

    assert flags == [1, 0] * 5


def test_sample_repeat_pict_skips_failed_window() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    results = [
        _fake_result(_frames_json([0, 1])),
        _fake_result("", returncode=1),
        _fake_result(_frames_json([1, 0])),
        _fake_result(_frames_json([0, 0])),
        _fake_result(_frames_json([1, 1])),
    ]

    with patch("furnace.adapters.ffmpeg.subprocess.run", side_effect=results):
        flags = adapter.sample_repeat_pict(Path("v.mkv"), duration_s=1000.0)

    assert flags == [0, 1, 1, 0, 0, 0, 1, 1]


def test_sample_repeat_pict_skips_unparseable_window() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    results = [
        _fake_result("not json"),
        _fake_result(_frames_json([1, 0])),
        _fake_result(_frames_json([0, 1])),
        _fake_result(_frames_json([0, 0])),
        _fake_result(_frames_json([1, 1])),
    ]

    with patch("furnace.adapters.ffmpeg.subprocess.run", side_effect=results):
        flags = adapter.sample_repeat_pict(Path("v.mkv"), duration_s=1000.0)

    assert flags == [1, 0, 0, 1, 0, 0, 1, 1]


def test_sample_repeat_pict_skips_window_with_non_integer_flag() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    results = [
        _fake_result(_frames_json([0, 1])),
        _fake_result(json.dumps({"frames": [{"repeat_pict": 1}, {"repeat_pict": "x"}]})),
        _fake_result(_frames_json([1, 0])),
        _fake_result(_frames_json([0, 0])),
        _fake_result(_frames_json([1, 1])),
    ]

    with patch("furnace.adapters.ffmpeg.subprocess.run", side_effect=results):
        flags = adapter.sample_repeat_pict(Path("v.mkv"), duration_s=1000.0)

    assert flags == [0, 1, 1, 0, 0, 0, 1, 1]


def test_sample_repeat_pict_all_windows_failed_returns_empty() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))

    with patch(
        "furnace.adapters.ffmpeg.subprocess.run",
        return_value=_fake_result("", returncode=1),
    ):
        flags = adapter.sample_repeat_pict(Path("v.mkv"), duration_s=1000.0)

    assert flags == []


def test_sample_repeat_pict_no_frames_key() -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    payload: dict[str, Any] = {}

    with patch(
        "furnace.adapters.ffmpeg.subprocess.run",
        return_value=_fake_result(json.dumps(payload)),
    ):
        flags = adapter.sample_repeat_pict(Path("v.mkv"), duration_s=1000.0)

    assert flags == []
