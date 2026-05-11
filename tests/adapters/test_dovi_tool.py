"""Tests for DoviToolAdapter — ffmpeg pipe + dovi_tool consumer.

run_pipeline is patched in every command-execution test (no real
subprocess). Builders are tested directly. The adapter signature now
requires both dovi_tool_path and ffmpeg_path, since the bug fix routes
the source MKV through ffmpeg before dovi_tool sees it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from furnace.adapters.dovi_tool import DoviToolAdapter
from furnace.core.models import DvMode

DOVI = Path("dovi_tool.exe")
FFMPEG = Path("ffmpeg.exe")


class TestFfmpegPipeCmd:
    def test_input_flag_points_to_source(self) -> None:
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        cmd = adapter._build_ffmpeg_pipe_cmd(Path("input.mkv"))
        str_cmd = [str(c) for c in cmd]
        assert str_cmd[0] == str(FFMPEG)
        i_idx = str_cmd.index("-i")
        assert str_cmd[i_idx + 1] == "input.mkv"

    def test_maps_first_video_stream(self) -> None:
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        cmd = adapter._build_ffmpeg_pipe_cmd(Path("input.mkv"))
        str_cmd = [str(c) for c in cmd]
        m_idx = str_cmd.index("-map")
        assert str_cmd[m_idx + 1] == "0:v:0"

    def test_copies_codec_no_reencode(self) -> None:
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        cmd = adapter._build_ffmpeg_pipe_cmd(Path("input.mkv"))
        str_cmd = [str(c) for c in cmd]
        c_idx = str_cmd.index("-c")
        assert str_cmd[c_idx + 1] == "copy"

    def test_applies_annexb_bitstream_filter(self) -> None:
        """Without hevc_mp4toannexb, MP4-style length-prefixed NALs
        reach dovi_tool which only understands Annex B start codes."""
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        cmd = adapter._build_ffmpeg_pipe_cmd(Path("input.mkv"))
        str_cmd = [str(c) for c in cmd]
        bsf_idx = str_cmd.index("-bsf:v")
        assert str_cmd[bsf_idx + 1] == "hevc_mp4toannexb"

    def test_emits_raw_hevc_to_stdout(self) -> None:
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        cmd = adapter._build_ffmpeg_pipe_cmd(Path("input.mkv"))
        str_cmd = [str(c) for c in cmd]
        f_idx = str_cmd.index("-f")
        assert str_cmd[f_idx + 1] == "hevc"
        assert str_cmd[-1] == "-"

    def test_quiet_loglevel(self) -> None:
        """Producer chatter would spam the log; -loglevel error keeps
        only true failures while still surfacing them.
        """
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        cmd = adapter._build_ffmpeg_pipe_cmd(Path("input.mkv"))
        str_cmd = [str(c) for c in cmd]
        ll_idx = str_cmd.index("-loglevel")
        assert str_cmd[ll_idx + 1] == "error"


class TestDoviExtractCmd:
    def test_copy_mode_no_m_flag(self) -> None:
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        cmd = adapter._build_extract_cmd(Path("RPU.bin"), DvMode.COPY)
        str_cmd = [str(c) for c in cmd]
        assert str_cmd[0] == str(DOVI)
        assert "-m" not in str_cmd
        assert "extract-rpu" in str_cmd

    def test_to_8_1_mode_adds_m_2(self) -> None:
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        cmd = adapter._build_extract_cmd(Path("RPU.bin"), DvMode.TO_8_1)
        str_cmd = [str(c) for c in cmd]
        m_idx = str_cmd.index("-m")
        assert str_cmd[m_idx + 1] == "2"

    def test_reads_from_stdin(self) -> None:
        """Bug fix: the consumer must NOT receive a container path —
        dovi_tool reads the HEVC stream produced by the ffmpeg producer
        over stdin (`-`).
        """
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        cmd = adapter._build_extract_cmd(Path("RPU.bin"), DvMode.COPY)
        str_cmd = [str(c) for c in cmd]
        ex_idx = str_cmd.index("extract-rpu")
        assert str_cmd[ex_idx + 1] == "-"

    def test_output_flag_points_to_rpu(self) -> None:
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        rpu_path = Path(tempfile.gettempdir()) / "RPU.bin"
        cmd = adapter._build_extract_cmd(rpu_path, DvMode.COPY)
        str_cmd = [str(c) for c in cmd]
        o_idx = str_cmd.index("-o")
        assert str_cmd[o_idx + 1] == str(rpu_path)


class TestDoviExtractRpuExecution:
    """extract_rpu wires both builders into run_pipeline."""

    def _patch_pipeline(
        self, captured: dict[str, Any], rc: int = 0,
    ) -> Any:
        def fake(
            producer_cmd: Any,
            consumer_cmd: Any,
            on_output: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            captured["producer"] = list(producer_cmd)
            captured["consumer"] = list(consumer_cmd)
            captured["log_path"] = log_path
            captured["on_output"] = on_output
            return rc, ""

        return patch(
            "furnace.adapters.dovi_tool.run_pipeline",
            side_effect=fake,
        )

    def test_returns_pipeline_rc(self) -> None:
        captured: dict[str, Any] = {}
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        with self._patch_pipeline(captured, rc=0):
            rc = adapter.extract_rpu(
                Path("input.mkv"), Path("rpu.bin"), DvMode.COPY,
            )
        assert rc == 0

    def test_propagates_nonzero_rc(self) -> None:
        captured: dict[str, Any] = {}
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        with self._patch_pipeline(captured, rc=42):
            rc = adapter.extract_rpu(
                Path("input.mkv"), Path("rpu.bin"), DvMode.COPY,
            )
        assert rc == 42

    def test_passes_ffmpeg_pipe_cmd_as_producer(self) -> None:
        captured: dict[str, Any] = {}
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        with self._patch_pipeline(captured):
            adapter.extract_rpu(
                Path("movie.mkv"), Path("rpu.bin"), DvMode.COPY,
            )
        producer = [str(c) for c in captured["producer"]]
        assert producer[0] == str(FFMPEG)
        assert "-bsf:v" in producer
        assert producer[-1] == "-"
        assert "movie.mkv" in producer

    def test_passes_dovi_extract_cmd_as_consumer(self) -> None:
        captured: dict[str, Any] = {}
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        with self._patch_pipeline(captured):
            adapter.extract_rpu(
                Path("movie.mkv"), Path("out.bin"), DvMode.TO_8_1,
            )
        consumer = [str(c) for c in captured["consumer"]]
        assert consumer[0] == str(DOVI)
        assert "extract-rpu" in consumer
        m_idx = consumer.index("-m")
        assert consumer[m_idx + 1] == "2"
        o_idx = consumer.index("-o")
        assert consumer[o_idx + 1] == "out.bin"

    def test_log_path_set_when_log_dir_configured(
        self, tmp_path: Path,
    ) -> None:
        captured: dict[str, Any] = {}
        adapter = DoviToolAdapter(DOVI, FFMPEG, log_dir=tmp_path)
        with self._patch_pipeline(captured):
            adapter.extract_rpu(
                Path("a.mkv"), Path("rpu.bin"), DvMode.TO_8_1,
            )
        assert captured["log_path"] == tmp_path / "dovi_tool_extract.log"

    def test_log_path_none_when_no_log_dir(self) -> None:
        captured: dict[str, Any] = {}
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        with self._patch_pipeline(captured):
            adapter.extract_rpu(
                Path("a.mkv"), Path("rpu.bin"), DvMode.COPY,
            )
        assert captured["log_path"] is None

    def test_on_output_propagates(self) -> None:
        captured: dict[str, Any] = {}

        def output_fn(_line: str) -> None:
            return None

        adapter = DoviToolAdapter(DOVI, FFMPEG, on_output=output_fn)
        with self._patch_pipeline(captured):
            adapter.extract_rpu(
                Path("a.mkv"), Path("rpu.bin"), DvMode.COPY,
            )
        assert captured["on_output"] is output_fn


class TestSetLogDir:
    def test_set_log_dir(self, tmp_path: Path) -> None:
        adapter = DoviToolAdapter(DOVI, FFMPEG)
        adapter.set_log_dir(tmp_path)
        assert adapter._log_dir == tmp_path

    def test_set_log_dir_none(self, tmp_path: Path) -> None:
        adapter = DoviToolAdapter(DOVI, FFMPEG, log_dir=tmp_path)
        adapter.set_log_dir(None)
        assert adapter._log_dir is None
