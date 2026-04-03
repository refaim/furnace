from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..core.models import CropRect, VideoParams
from ._subprocess import OutputCallback, run_tool

logger = logging.getLogger(__name__)


class FFmpegAdapter:
    """Implements Prober + Encoder + AudioExtractor."""

    def __init__(
        self, ffmpeg_path: Path, ffprobe_path: Path,
        on_output: OutputCallback = None, log_dir: Path | None = None,
    ) -> None:
        self._ffmpeg = ffmpeg_path
        self._ffprobe = ffprobe_path
        self._on_output = on_output
        self._log_dir = log_dir

    def set_log_dir(self, log_dir: Path | None) -> None:
        self._log_dir = log_dir

    # ------------------------------------------------------------------
    # Prober
    # ------------------------------------------------------------------

    def probe(self, path: Path) -> dict[str, Any]:
        """ffprobe -v quiet -print_format json -show_format -show_streams -show_chapters path"""
        cmd = [
            str(self._ffprobe),
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            "-show_chapters",
            str(path),
        ]
        logger.debug("probe cmd: %s", cmd)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("ffprobe failed (rc=%d): %s", result.returncode, result.stderr)
            raise RuntimeError(f"ffprobe failed with return code {result.returncode}: {result.stderr}")
        data: dict[str, Any] = json.loads(result.stdout)
        return data

    def detect_crop(self, path: Path, duration_s: float) -> CropRect | None:
        """Run cropdetect at 5 points (10%, 30%, 50%, 70%, 90%), 2 seconds each.
        Returns the mode crop value. Returns None if crop == full frame."""
        crop_values: list[str] = []

        for pct in (0.10, 0.30, 0.50, 0.70, 0.90):
            seek = duration_s * pct
            cmd = [
                str(self._ffmpeg),
                "-hide_banner",
                "-ss", f"{seek:.2f}",
                "-i", str(path),
                "-t", "2",
                "-vf", "cropdetect=24:16:0",
                "-f", "null",
                "-",
            ]
            logger.debug("detect_crop cmd: %s", cmd)
            result = subprocess.run(cmd, capture_output=True, text=True)
            # cropdetect writes to stderr
            for line in result.stderr.splitlines():
                m = re.search(r"crop=(\d+:\d+:\d+:\d+)", line)
                if m:
                    crop_values.append(m.group(1))

        if not crop_values:
            return None

        # Mode: most frequent crop value
        mode_crop = Counter(crop_values).most_common(1)[0][0]
        parts = mode_crop.split(":")
        if len(parts) != 4:
            return None
        w, h, x, y = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        return CropRect(w=w, h=h, x=x, y=y)

    def get_encoder_tag(self, path: Path) -> str | None:
        """Read format.tags.ENCODER from probe output."""
        try:
            data = self.probe(path)
        except RuntimeError:
            return None
        tags = data.get("format", {}).get("tags", {})
        # Tags can be ENCODER or encoder (case varies)
        for key in ("ENCODER", "encoder"):
            if key in tags:
                return str(tags[key])
        return None

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def encode(
        self,
        input_path: Path,
        output_path: Path,
        video_params: VideoParams,
        source_size: int,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> int:
        """Encode with hevc_nvenc. Parses -progress pipe:1 for fps/speed/progress.
        Performs mid-encoding bloat check: if progress > 5% and output >= source_size,
        terminates the process and returns non-zero."""
        cmd = self._build_encode_cmd(input_path, output_path, video_params)
        logger.debug("encode cmd: %s", " ".join(str(c) for c in cmd))

        # Open per-tool log file
        encode_log = None
        if self._log_dir:
            encode_log = (self._log_dir / "ffmpeg_encode.log").open("w", encoding="utf-8")
            encode_log.write(f"$ {' '.join(str(c) for c in cmd)}\n\n")
            encode_log.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Read stderr in background thread (prevents deadlock + feeds output display)
        import threading
        stderr_lines: list[str] = []

        def _read_stderr() -> None:
            assert process.stderr is not None
            for err_line in process.stderr:
                err_line = err_line.rstrip()
                if err_line:
                    stderr_lines.append(err_line)
                    if self._on_output is not None:
                        self._on_output(err_line)
                    if encode_log is not None:
                        encode_log.write(err_line + "\n")
                        encode_log.flush()

        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()

        progress_data: dict[str, str] = {}
        bloat_abort = False
        encode_duration_us: int = 0
        duration_probed = False

        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if "=" in line:
                key, _, value = line.partition("=")
                progress_data[key.strip()] = value.strip()

            # Calculate progress percentage from out_time_us
            progress_pct = 0.0

            # Parse out_time_us
            out_time_us_str = progress_data.get("out_time_us", "0")
            try:
                out_us = max(0, int(out_time_us_str))
            except ValueError:
                out_us = 0

            # Probe duration once lazily
            if not duration_probed:
                duration_probed = True
                try:
                    probe_data = self.probe(input_path)
                    dur_s = float(probe_data.get("format", {}).get("duration", 0))
                    encode_duration_us = int(dur_s * 1_000_000)
                except Exception:
                    encode_duration_us = 0

            if encode_duration_us > 0 and out_us > 0:
                progress_pct = min(100.0, out_us / encode_duration_us * 100.0)

            fps = progress_data.get("fps", "0")
            speed = progress_data.get("speed", "N/A")
            status_line = (
                f"progress={progress_pct:.1f}% fps={fps} speed={speed}"
            )

            if on_progress is not None:
                on_progress(progress_pct, status_line)

            # Mid-encoding bloat check (section 12.11)
            if self._check_mid_encoding_bloat(output_path, source_size, progress_pct):
                logger.warning(
                    "Mid-encoding bloat detected at %.1f%% progress, terminating ffmpeg", progress_pct
                )
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                # Remove partial output
                if output_path.exists():
                    output_path.unlink()
                bloat_abort = True
                break

        process.wait()
        stderr_thread.join(timeout=5)

        if encode_log is not None:
            encode_log.write(f"\n--- exit code: {process.returncode} ---\n")
            encode_log.close()

        if bloat_abort:
            logger.error(
                "Encoding aborted: mid-encoding bloat detected (output >= source_size=%d)", source_size
            )
            return 1

        if process.returncode != 0:
            logger.error("ffmpeg encode failed (rc=%d): %s", process.returncode, "\n".join(stderr_lines[-10:]))

        return process.returncode

    # Codecs supported by NVIDIA NVDEC hardware decoder
    _NVDEC_CODECS: set[str] = {
        "h264", "hevc", "mpeg2video", "mpeg4", "vp8", "vp9", "vc1", "av1",
    }

    def _build_encode_cmd(self, input_path: Path, output_path: Path, vp: VideoParams) -> list[str]:
        """Build the full ffmpeg NVENC encode command.

        Hybrid CUDA/CPU path:
        - CUDA: when source_codec is nvdec-supported AND no crop (no crop_cuda in ffmpeg)
        - CPU:  otherwise (fallback)
        """
        use_cuda = (
            vp.source_codec in self._NVDEC_CODECS
            and vp.crop is None
        )

        cmd: list[str] = [str(self._ffmpeg), "-hide_banner", "-loglevel", "warning"]

        # Input with optional CUDA hardware acceleration
        if use_cuda:
            cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        cmd += ["-i", str(input_path)]

        # Video codec and NVENC params
        cmd += [
            "-c:v", "hevc_nvenc",
            "-profile:v", "main10",
        ]
        # CPU path needs explicit pixel format; CUDA path GPU handles it
        if not use_cuda:
            cmd += ["-pix_fmt", "p010le"]
        cmd += [
            "-preset", "p5",
            "-tune", "uhq",
            "-rc", "vbr",
            "-cq", str(vp.cq),
            "-spatial-aq", "1",
            "-temporal-aq", "1",
            "-rc-lookahead", "32",
            "-multipass", "qres",
            "-forced-idr", "1",
            "-g", str(vp.gop),
        ]

        # Video filters (CUDA or CPU variants)
        vf_parts: list[str] = []
        if vp.deinterlace:
            if use_cuda:
                vf_parts.append("bwdif_cuda=mode=send_frame:parity=auto")
            else:
                vf_parts.append("bwdif=mode=send_frame:parity=auto")
        if vp.crop is not None:
            # crop is always CPU path (use_cuda is False when crop is set)
            c = vp.crop
            vf_parts.append(f"crop={c.w}:{c.h}:{c.x}:{c.y}")
        if vf_parts:
            cmd += ["-vf", ",".join(vf_parts)]

        # Color parameters
        cmd += ["-color_range", vp.color_range]
        if vp.color_primaries:
            cmd += ["-color_primaries", vp.color_primaries]
        if vp.color_transfer:
            cmd += ["-color_trc", vp.color_transfer]

        # Colorspace / matrix coefficients
        from ..core.models import ColorSpace
        if vp.color_space == ColorSpace.BT2020:
            cmd += ["-colorspace", "bt2020nc"]
        elif vp.color_space == ColorSpace.BT709:
            cmd += ["-colorspace", "bt709"]
        elif vp.color_space == ColorSpace.BT601:
            cmd += ["-colorspace", "smpte170m"]

        # No audio/subtitles, no metadata
        cmd += ["-an", "-sn", "-map_metadata", "-1"]

        # Progress output
        cmd += ["-progress", "pipe:1"]

        # Output
        cmd += ["-y", str(output_path)]
        return cmd

    def _check_mid_encoding_bloat(
        self, output_path: Path, source_size: int, progress_pct: float
    ) -> bool:
        """Return True if output has bloated past source size before 5% progress."""
        if progress_pct <= 5.0:
            return False
        if source_size <= 0:
            return False
        try:
            current_size = output_path.stat().st_size
        except FileNotFoundError:
            return False
        return current_size >= source_size

    # ------------------------------------------------------------------
    # VMAF
    # ------------------------------------------------------------------

    def compute_vmaf(
        self,
        reference: Path,
        distorted: Path,
        duration_s: float,
        on_progress: Callable[[float, str], None] | None = None,
        crop: CropRect | None = None,
    ) -> float | None:
        """Calculate VMAF score using libvmaf filter. Returns mean VMAF or None.

        If crop is provided, reference is cropped to match distorted dimensions.
        """
        import json as _json
        import tempfile
        import threading

        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)  # noqa: SIM115
        tmp.close()
        log_path_lavfi = tmp.name.replace("\\", "/").replace(":", "\\\\:")
        n_threads = max(1, (os.cpu_count() or 4) - 2)

        # If crop was applied to distorted, apply same crop to reference for matching dimensions
        if crop is not None:
            lavfi = (
                f"[0:v]crop={crop.w}:{crop.h}:{crop.x}:{crop.y}[ref];"
                f"[ref][1:v]libvmaf=log_fmt=json:n_threads={n_threads}:log_path={log_path_lavfi}"
            )
        else:
            lavfi = f"libvmaf=log_fmt=json:n_threads={n_threads}:log_path={log_path_lavfi}"

        cmd = [
            str(self._ffmpeg), "-hide_banner", "-v", "error",
            "-hwaccel", "cuda", "-i", str(reference),
            "-hwaccel", "cuda", "-i", str(distorted),
            "-lavfi", lavfi,
            "-f", "null",
            "-progress", "pipe:1",
            "-",
        ]
        logger.debug("compute_vmaf cmd: %s", " ".join(cmd))

        # Write to per-tool log
        vmaf_log = None
        if self._log_dir:
            vmaf_log = (self._log_dir / "ffmpeg_vmaf.log").open("w", encoding="utf-8")
            vmaf_log.write(f"$ {' '.join(cmd)}\n\n")
            vmaf_log.flush()

        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
            )

            # Read stderr in background
            stderr_lines: list[str] = []

            def _read_stderr() -> None:
                assert process.stderr is not None
                for err_line in process.stderr:
                    err_line = err_line.rstrip()
                    if err_line:
                        stderr_lines.append(err_line)
                        if self._on_output is not None:
                            self._on_output(err_line)
                        if vmaf_log is not None:
                            vmaf_log.write(err_line + "\n")
                            vmaf_log.flush()

            stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
            stderr_thread.start()

            # Parse progress from stdout
            assert process.stdout is not None
            for line in process.stdout:
                line = line.strip()
                if line.startswith("out_time_us="):
                    try:
                        us = int(line.split("=", 1)[1])
                        if duration_s > 0:
                            pct = min(100.0, (us / 1_000_000) / duration_s * 100)
                            if on_progress is not None:
                                on_progress(pct, f"VMAF {pct:.1f}%")
                    except ValueError:
                        pass

            process.wait()
            stderr_thread.join(timeout=5)

            if vmaf_log is not None:
                vmaf_log.write(f"\n--- exit code: {process.returncode} ---\n")
                vmaf_log.close()

            if process.returncode != 0:
                logger.error("VMAF failed (rc=%d): %s", process.returncode, "\n".join(stderr_lines[-5:]))
                return None

            with Path(tmp.name).open(encoding="utf-8") as f:
                vmaf_data = _json.load(f)
            return float(vmaf_data["pooled_metrics"]["vmaf"]["mean"])

        except Exception:
            logger.exception("VMAF computation failed")
            return None
        finally:
            try:
                Path(tmp.name).unlink(missing_ok=True)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # AudioExtractor
    # ------------------------------------------------------------------

    def extract_track(
        self,
        input_path: Path,
        stream_index: int,
        output_path: Path,
        codec: str,
    ) -> int:
        """ffmpeg -i input -map 0:{index} -c copy output"""
        cmd = [
            str(self._ffmpeg),
            "-hide_banner", "-loglevel", "warning",
            "-i", str(input_path),
            "-map", f"0:{stream_index}",
            "-c", "copy",
            "-y", str(output_path),
        ]
        log_path = self._log_dir / f"ffmpeg_extract_s{stream_index}.log" if self._log_dir else None
        rc, _out = run_tool(cmd, on_output=self._on_output, log_path=log_path)
        return rc

    def ffmpeg_to_wav(
        self,
        input_path: Path,
        stream_index: int,
        output_wav: Path,
    ) -> int:
        """ffmpeg -i input -map 0:{index} -f wav -rf64 auto output.wav"""
        cmd = [
            str(self._ffmpeg),
            "-hide_banner", "-loglevel", "warning",
            "-i", str(input_path),
            "-map", f"0:{stream_index}",
            "-f", "wav",
            "-rf64", "auto",
            "-y", str(output_wav),
        ]
        log_path = self._log_dir / f"ffmpeg_to_wav_s{stream_index}.log" if self._log_dir else None
        rc, _out = run_tool(cmd, on_output=self._on_output, log_path=log_path)
        return rc
