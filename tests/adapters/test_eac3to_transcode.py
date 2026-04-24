"""Tests for Eac3toAdapter.transcode_to_flac (Wave64 -> FLAC conversion)."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

from furnace.adapters.eac3to import Eac3toAdapter
from furnace.core.ports import PcmTranscoder
from furnace.core.progress import ProgressSample


def _run_and_capture(
    rc: int = 0,
    on_progress: Callable[[ProgressSample], None] | None = None,
    log_dir: Path | None = None,
) -> dict[str, object]:
    """Invoke transcode_to_flac with mocked run_tool; return what was captured."""
    captured: dict[str, object] = {}

    def fake_run_tool(
        cmd: list[str | Path],
        on_output: object = None,
        on_progress_line: object = None,
        log_path: object = None,
        cwd: object = None,
    ) -> tuple[int, str]:
        captured["cmd"] = [str(c) for c in cmd]
        captured["on_progress_line"] = on_progress_line
        captured["log_path"] = log_path
        return (rc, "")

    adapter = Eac3toAdapter(Path("C:/Tools/eac3to.exe"), log_dir=log_dir)
    with patch("furnace.adapters.eac3to.run_tool", side_effect=fake_run_tool):
        captured["rc"] = adapter.transcode_to_flac(
            Path("/src/audio.w64"),
            Path("/out/audio.flac"),
            on_progress=on_progress,
        )
    return captured


class TestTranscodeToFlac:
    def test_builds_expected_cmd(self) -> None:
        """Positional args are [eac3to, input.w64, output.flac, -progressnumbers]."""
        cap = _run_and_capture()
        cmd = cap["cmd"]
        assert isinstance(cmd, list)
        # eac3to path is arg 0
        assert cmd[0].endswith("eac3to.exe")
        # Input and output paths appear as positional args
        stringified = [c.replace("\\", "/") for c in cmd]
        assert "/src/audio.w64" in stringified
        assert "/out/audio.flac" in stringified
        assert "-progressnumbers" in cmd

    def test_no_remove_dialnorm_emitted(self) -> None:
        """PCM has no dialnorm metadata; -removeDialnorm must not be sent."""
        cap = _run_and_capture()
        cmd = cap["cmd"]
        assert isinstance(cmd, list)
        assert "-removeDialnorm" not in cmd

    def test_no_downmix_flags_emitted(self) -> None:
        """Transcode is bit-perfect; no channel manipulation flags."""
        cap = _run_and_capture()
        cmd = cap["cmd"]
        assert isinstance(cmd, list)
        for flag in ("-downStereo", "-down6", "-mixlfe"):
            assert flag not in cmd

    def test_propagates_rc_zero(self) -> None:
        cap = _run_and_capture(rc=0)
        assert cap["rc"] == 0

    def test_propagates_rc_nonzero(self) -> None:
        cap = _run_and_capture(rc=3)
        assert cap["rc"] == 3

    def test_progress_callback_wired(self) -> None:
        """on_progress triggered by a progress line routed through _run."""
        progress_sink = MagicMock()
        cap = _run_and_capture(on_progress=progress_sink)
        on_progress_line = cap["on_progress_line"]
        assert callable(on_progress_line)
        # Simulate eac3to emitting a progress line; the closure should parse it
        # and invoke progress_sink with a ProgressSample (fraction=0.5).
        consumed = on_progress_line("process: 50%")
        assert consumed is True  # progress lines are suppressed from the log
        assert progress_sink.called
        sample = progress_sink.call_args[0][0]
        assert sample.fraction == 0.5

    def test_progress_line_non_progress_not_consumed(self) -> None:
        """Non-progress lines must pass through (return False) so they still log."""
        cap = _run_and_capture(on_progress=MagicMock())
        on_progress_line = cap["on_progress_line"]
        assert callable(on_progress_line)
        assert on_progress_line("Decoding frame 12") is False

    def test_log_path_uses_transcode_label(self, tmp_path: Path) -> None:
        """log_path is <log_dir>/eac3to_w64_to_flac.log when log_dir is set."""
        cap = _run_and_capture(log_dir=tmp_path)
        log_path = cap["log_path"]
        assert isinstance(log_path, Path)
        assert log_path == tmp_path / "eac3to_w64_to_flac.log"

    def test_log_path_none_without_log_dir(self) -> None:
        """Without log_dir configured, log_path is None."""
        cap = _run_and_capture()
        assert cap["log_path"] is None


class TestAdapterImplementsPcmTranscoder:
    def test_eac3to_adapter_satisfies_pcm_transcoder(self) -> None:
        """Eac3toAdapter satisfies the PcmTranscoder structural protocol."""
        adapter = Eac3toAdapter(Path("C:/Tools/eac3to.exe"))
        assert isinstance(adapter, PcmTranscoder)
