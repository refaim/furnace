"""Tests for Eac3toAdapter.decode_lossless downmix flag emission."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from furnace.adapters.eac3to import Eac3toAdapter
from furnace.core.models import DownmixMode


def _run_and_capture(downmix: DownmixMode | None, delay_ms: int = 0) -> list[str]:
    """Invoke decode_lossless with given downmix and return the captured argv."""
    captured: dict[str, object] = {}

    def fake_run_tool(
        cmd: list[str | Path],
        on_output: object = None,
        on_progress_line: object = None,
        log_path: object = None,
        cwd: object = None,
    ) -> tuple[int, str]:
        captured["cmd"] = [str(c) for c in cmd]
        return (0, "")

    adapter = Eac3toAdapter(Path("C:/Tools/eac3to.exe"))
    with patch("furnace.adapters.eac3to.run_tool", side_effect=fake_run_tool):
        adapter.decode_lossless(
            Path("/src/audio.thd"),
            Path(tempfile.gettempdir()) / "out.wav",
            delay_ms,
            on_progress=None,
            downmix=downmix,
        )
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    return cmd


class TestDecodeLosslessDownmixFlags:
    def test_no_downmix_emits_no_downmix_flags(self) -> None:
        """Regression guard: calls without downmix must not emit downmix flags."""
        cmd = _run_and_capture(downmix=None)
        assert "-downStereo" not in cmd
        assert "-mixlfe" not in cmd
        assert "-down6" not in cmd

    def test_stereo_mode_emits_down_stereo(self) -> None:
        cmd = _run_and_capture(downmix=DownmixMode.STEREO)
        assert "-downStereo" in cmd
        assert "-mixlfe" not in cmd
        assert "-down6" not in cmd

    def test_down6_mode_emits_down6(self) -> None:
        cmd = _run_and_capture(downmix=DownmixMode.DOWN6)
        assert "-down6" in cmd
        assert "-downStereo" not in cmd
        assert "-mixlfe" not in cmd

    def test_remove_dialnorm_still_present_with_downmix(self) -> None:
        """Downmix flags augment, not replace, -removeDialnorm."""
        cmd = _run_and_capture(downmix=DownmixMode.STEREO)
        assert "-removeDialnorm" in cmd

    def test_delay_still_applied_with_downmix(self) -> None:
        """Delay arg is independent of downmix."""
        cmd = _run_and_capture(downmix=DownmixMode.STEREO, delay_ms=50)
        assert "-downStereo" in cmd
        assert any(arg == "+50ms" for arg in cmd)

    def test_negative_delay_with_downmix(self) -> None:
        cmd = _run_and_capture(downmix=DownmixMode.DOWN6, delay_ms=-30)
        assert "-down6" in cmd
        assert any(arg == "-30ms" for arg in cmd)
