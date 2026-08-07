from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from furnace.adapters.hdr10plus_tool import Hdr10PlusToolAdapter

HDR10PLUS = Path("hdr10plus_tool.exe")
FFMPEG = Path("ffmpeg.exe")


class TestFfmpegPipeCmd:
    def test_input_flag_points_to_source(self) -> None:
        adapter = Hdr10PlusToolAdapter(HDR10PLUS, FFMPEG)
        cmd = adapter._build_ffmpeg_pipe_cmd(Path("input.mkv"))
        str_cmd = [str(c) for c in cmd]
        assert str_cmd[0] == str(FFMPEG)
        i_idx = str_cmd.index("-i")
        assert str_cmd[i_idx + 1] == "input.mkv"

    def test_maps_first_video_stream(self) -> None:
        adapter = Hdr10PlusToolAdapter(HDR10PLUS, FFMPEG)
        cmd = adapter._build_ffmpeg_pipe_cmd(Path("input.mkv"))
        str_cmd = [str(c) for c in cmd]
        m_idx = str_cmd.index("-map")
        assert str_cmd[m_idx + 1] == "0:v:0"

    def test_copies_codec_no_reencode(self) -> None:
        adapter = Hdr10PlusToolAdapter(HDR10PLUS, FFMPEG)
        cmd = adapter._build_ffmpeg_pipe_cmd(Path("input.mkv"))
        str_cmd = [str(c) for c in cmd]
        c_idx = str_cmd.index("-c")
        assert str_cmd[c_idx + 1] == "copy"

    def test_applies_annexb_bitstream_filter(self) -> None:
        adapter = Hdr10PlusToolAdapter(HDR10PLUS, FFMPEG)
        cmd = adapter._build_ffmpeg_pipe_cmd(Path("input.mkv"))
        str_cmd = [str(c) for c in cmd]
        bsf_idx = str_cmd.index("-bsf:v")
        assert str_cmd[bsf_idx + 1] == "hevc_mp4toannexb"

    def test_emits_raw_hevc_to_stdout(self) -> None:
        adapter = Hdr10PlusToolAdapter(HDR10PLUS, FFMPEG)
        cmd = adapter._build_ffmpeg_pipe_cmd(Path("input.mkv"))
        str_cmd = [str(c) for c in cmd]
        f_idx = str_cmd.index("-f")
        assert str_cmd[f_idx + 1] == "hevc"
        assert str_cmd[-1] == "-"

    def test_quiet_loglevel(self) -> None:
        adapter = Hdr10PlusToolAdapter(HDR10PLUS, FFMPEG)
        cmd = adapter._build_ffmpeg_pipe_cmd(Path("input.mkv"))
        str_cmd = [str(c) for c in cmd]
        ll_idx = str_cmd.index("-loglevel")
        assert str_cmd[ll_idx + 1] == "error"


class TestHdr10PlusExtractCmd:
    def test_tool_is_the_executable(self) -> None:
        adapter = Hdr10PlusToolAdapter(HDR10PLUS, FFMPEG)
        cmd = adapter._build_extract_cmd(Path("hdr10plus.json"))
        str_cmd = [str(c) for c in cmd]
        assert str_cmd[0] == str(HDR10PLUS)
        assert "extract" in str_cmd

    def test_reads_from_stdin(self) -> None:
        adapter = Hdr10PlusToolAdapter(HDR10PLUS, FFMPEG)
        cmd = adapter._build_extract_cmd(Path("hdr10plus.json"))
        str_cmd = [str(c) for c in cmd]
        ex_idx = str_cmd.index("extract")
        assert str_cmd[ex_idx + 1] == "-"

    def test_output_flag_points_to_json(self) -> None:
        adapter = Hdr10PlusToolAdapter(HDR10PLUS, FFMPEG)
        json_path = Path(tempfile.gettempdir()) / "hdr10plus.json"
        cmd = adapter._build_extract_cmd(json_path)
        str_cmd = [str(c) for c in cmd]
        o_idx = str_cmd.index("-o")
        assert str_cmd[o_idx + 1] == str(json_path)


class TestHdr10PlusExtractExecution:
    def _patch_pipeline(
        self,
        captured: dict[str, Any],
        rc: int = 0,
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
            "furnace.adapters.hdr10plus_tool.run_pipeline",
            side_effect=fake,
        )

    def test_returns_pipeline_rc(self) -> None:
        captured: dict[str, Any] = {}
        adapter = Hdr10PlusToolAdapter(HDR10PLUS, FFMPEG)
        with self._patch_pipeline(captured, rc=0):
            rc = adapter.extract(Path("input.mkv"), Path("hdr10plus.json"))
        assert rc == 0

    def test_propagates_nonzero_rc(self) -> None:
        captured: dict[str, Any] = {}
        adapter = Hdr10PlusToolAdapter(HDR10PLUS, FFMPEG)
        with self._patch_pipeline(captured, rc=42):
            rc = adapter.extract(Path("input.mkv"), Path("hdr10plus.json"))
        assert rc == 42

    def test_passes_ffmpeg_pipe_cmd_as_producer(self) -> None:
        captured: dict[str, Any] = {}
        adapter = Hdr10PlusToolAdapter(HDR10PLUS, FFMPEG)
        with self._patch_pipeline(captured):
            adapter.extract(Path("movie.mp4"), Path("out.json"))
        producer = [str(c) for c in captured["producer"]]
        assert producer[0] == str(FFMPEG)
        assert "-bsf:v" in producer
        assert producer[-1] == "-"
        assert "movie.mp4" in producer

    def test_passes_hdr10plus_extract_cmd_as_consumer(self) -> None:
        captured: dict[str, Any] = {}
        adapter = Hdr10PlusToolAdapter(HDR10PLUS, FFMPEG)
        with self._patch_pipeline(captured):
            adapter.extract(Path("movie.mkv"), Path("out.json"))
        consumer = [str(c) for c in captured["consumer"]]
        assert consumer[0] == str(HDR10PLUS)
        ex_idx = consumer.index("extract")
        assert consumer[ex_idx + 1] == "-"
        o_idx = consumer.index("-o")
        assert consumer[o_idx + 1] == "out.json"

    def test_log_path_set_when_log_dir_configured(self, tmp_path: Path) -> None:
        captured: dict[str, Any] = {}
        adapter = Hdr10PlusToolAdapter(HDR10PLUS, FFMPEG, log_dir=tmp_path)
        with self._patch_pipeline(captured):
            adapter.extract(Path("a.mkv"), Path("out.json"))
        assert captured["log_path"] == tmp_path / "hdr10plus_tool_extract.log"

    def test_log_path_none_when_no_log_dir(self) -> None:
        captured: dict[str, Any] = {}
        adapter = Hdr10PlusToolAdapter(HDR10PLUS, FFMPEG)
        with self._patch_pipeline(captured):
            adapter.extract(Path("a.mkv"), Path("out.json"))
        assert captured["log_path"] is None

    def test_on_output_propagates(self) -> None:
        captured: dict[str, Any] = {}
        lines: list[str] = []
        output_fn = lines.append

        adapter = Hdr10PlusToolAdapter(HDR10PLUS, FFMPEG, on_output=output_fn)
        with self._patch_pipeline(captured):
            adapter.extract(Path("a.mkv"), Path("out.json"))
        assert captured["on_output"] is output_fn


class TestSetLogDir:
    def test_set_log_dir(self, tmp_path: Path) -> None:
        adapter = Hdr10PlusToolAdapter(HDR10PLUS, FFMPEG)
        adapter.set_log_dir(tmp_path)
        assert adapter._log_dir == tmp_path

    def test_set_log_dir_none(self, tmp_path: Path) -> None:
        adapter = Hdr10PlusToolAdapter(HDR10PLUS, FFMPEG, log_dir=tmp_path)
        adapter.set_log_dir(None)
        assert adapter._log_dir is None
