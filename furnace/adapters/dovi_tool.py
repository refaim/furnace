from __future__ import annotations

from pathlib import Path

from furnace.core.models import DvMode

from ._subprocess import OutputCallback, run_pipeline


class DoviToolAdapter:
    def __init__(
        self,
        dovi_tool_path: Path,
        ffmpeg_path: Path,
        on_output: OutputCallback = None,
        log_dir: Path | None = None,
    ) -> None:
        self._dovi_tool = dovi_tool_path
        self._ffmpeg = ffmpeg_path
        self._on_output = on_output
        self._log_dir = log_dir

    def set_log_dir(self, log_dir: Path | None) -> None:
        self._log_dir = log_dir

    def _build_ffmpeg_pipe_cmd(self, input_path: Path) -> list[str | Path]:
        return [
            self._ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            input_path,
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-bsf:v",
            "hevc_mp4toannexb",
            "-f",
            "hevc",
            "-",
        ]

    def _build_extract_cmd(
        self,
        output_rpu: Path,
        mode: DvMode,
    ) -> list[str | Path]:
        cmd: list[str | Path] = [self._dovi_tool]
        if mode == DvMode.TO_8_1:
            cmd += ["-m", "2"]
        cmd += ["extract-rpu", "-", "-o", output_rpu]
        return cmd

    def extract_rpu(
        self,
        input_path: Path,
        output_rpu: Path,
        mode: DvMode,
    ) -> int:
        producer = self._build_ffmpeg_pipe_cmd(input_path)
        consumer = self._build_extract_cmd(output_rpu, mode)
        log_path = self._log_dir / "dovi_tool_extract.log" if self._log_dir else None
        rc, _out = run_pipeline(
            producer,
            consumer,
            on_output=self._on_output,
            log_path=log_path,
        )
        return rc
