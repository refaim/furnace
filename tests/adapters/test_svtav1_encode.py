"""Tests for SvtAv1Adapter.encode orchestration: command run, log wiring, progress.

``encode`` runs a single ffmpeg + libsvtav1 pass and returns an EncodeResult
(return code + ENCODER_SETTINGS tag). These tests verify that contract: the
encode command is built and run, progress/output callbacks are forwarded, the
log path is wired, and the ``rpu_path`` argument is accepted but ignored.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from furnace.adapters.svtav1 import SvtAv1Adapter
from furnace.core.models import CropRect, VideoParams
from furnace.core.progress import ProgressSample


def _make_vp(
    *,
    crop: CropRect | None = None,
    deinterlace: bool = False,
    source_width: int = 1920,
    source_height: int = 1080,
    sar_num: int = 1,
    sar_den: int = 1,
) -> VideoParams:
    return VideoParams(
        cq=23, crop=crop, deinterlace=deinterlace,
        color_matrix="bt709", color_range="tv",
        color_transfer="bt709", color_primaries="bt709",
        hdr=None, gop=120, fps_num=24000, fps_den=1001,
        source_width=source_width, source_height=source_height,
        source_codec="mpeg2video", source_bitrate=8_000_000,
        sar_num=sar_num, sar_den=sar_den, grain=True,
    )


class _FakeRunTool:
    """Records the single ``run_tool`` encode invocation."""

    def __init__(self, *, encode_rc: int = 0) -> None:
        self.calls: list[dict[str, Any]] = []
        self.encode_rc = encode_rc

    def __call__(
        self,
        cmd: Any,
        on_output: Any = None,
        on_progress_line: Any = None,
        log_path: Any = None,
        cwd: Any = None,
    ) -> tuple[int, str]:
        self.calls.append(
            {
                "cmd": [str(c) for c in cmd],
                "on_output": on_output,
                "on_progress_line": on_progress_line,
                "log_path": log_path,
                "cwd": cwd,
            },
        )
        return self.encode_rc, ""

    @property
    def encode_call(self) -> dict[str, Any]:
        return self.calls[0]


def _run(
    adapter: SvtAv1Adapter,
    fake: _FakeRunTool,
    tmp_path: Path,
    *,
    rpu_path: Path | None = None,
    on_progress: Any = None,
) -> Any:
    with patch("furnace.adapters.svtav1.run_tool", side_effect=fake):
        return adapter.encode(
            tmp_path / "input.mkv", tmp_path / "output.obu", _make_vp(),
            rpu_path=rpu_path, on_progress=on_progress,
        )


class TestEncodeBasics:
    def test_returns_rc_and_settings(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool(encode_rc=0)
        result = _run(adapter, fake, tmp_path)
        assert result.return_code == 0
        assert result.encoder_settings.startswith("av1_svt")

    def test_returns_rc_from_run_tool(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool(encode_rc=7)
        result = _run(adapter, fake, tmp_path)
        assert result.return_code == 7

    def test_encode_command_is_svtav1(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool()
        _run(adapter, fake, tmp_path)
        assert "libsvtav1" in fake.encode_call["cmd"]

    def test_forwards_on_output(self, tmp_path: Path) -> None:
        lines: list[str] = []
        cb = lines.append
        adapter = SvtAv1Adapter(Path("ffmpeg"), on_output=cb)
        fake = _FakeRunTool()
        _run(adapter, fake, tmp_path)
        assert fake.encode_call["on_output"] is cb

    def test_forwards_progress_through_ffmpeg_handler(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        samples: list[ProgressSample] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                on_progress_line("out_time_us=5000000")
                on_progress_line("progress=continue")
            return 0, ""

        with patch("furnace.adapters.svtav1.run_tool", side_effect=fake_run_tool):
            adapter.encode(
                tmp_path / "in.mkv", tmp_path / "out.obu", _make_vp(),
                on_progress=samples.append,
            )
        assert len(samples) == 1
        assert samples[0].processed_s == 5.0


class TestEncodeLogWiring:
    def test_encode_log_wired_when_log_dir_set(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"), log_dir=tmp_path)
        fake = _FakeRunTool()
        _run(adapter, fake, tmp_path)
        assert fake.encode_call["log_path"] == tmp_path / "svt_encode.log"

    def test_encode_log_none_when_no_log_dir(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool()
        _run(adapter, fake, tmp_path)
        assert fake.encode_call["log_path"] is None


class TestRpuIgnored:
    def test_rpu_path_ignored_no_crash(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool()
        result = _run(adapter, fake, tmp_path, rpu_path=Path("rpu.bin"))
        assert result.return_code == 0

    def test_rpu_not_in_encode_command(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool()
        _run(adapter, fake, tmp_path, rpu_path=Path("rpu.bin"))
        for call in fake.calls:
            assert "rpu.bin" not in " ".join(call["cmd"])
