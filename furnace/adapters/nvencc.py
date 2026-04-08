"""NVEncC (rigaya) video encoder adapter.

Implements the Encoder protocol using NVEncC for HEVC encoding with NVDEC
hardware decode, HDR/DV passthrough, and built-in quality metrics.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

from ..core.models import (
    CropRect,
    EncodeResult,
    VideoParams,
)
from ..core.quality import align_dimensions, correct_sar
from ._subprocess import OutputCallback

logger = logging.getLogger(__name__)

# NVEncC color range: ffmpeg "tv" -> NVEncC "limited", "pc" -> "full"
_COLOR_RANGE_MAP: dict[str, str] = {
    "tv": "limited",
    "pc": "full",
}


def _parse_content_light(content_light: str) -> tuple[str, str] | None:
    """Parse HdrMetadata.content_light string into (MaxCLL, MaxFALL).

    Expected format: "MaxCLL=X,MaxFALL=Y" or "MaxCLL=X, MaxFALL=Y".
    Returns tuple of string values or None if parsing fails.
    """
    m = re.match(r"MaxCLL=(\d+)\s*,\s*MaxFALL=(\d+)", content_light)
    if m:
        return m.group(1), m.group(2)
    return None


def _convert_crop(crop: CropRect, source_width: int, source_height: int) -> tuple[int, int, int, int]:
    """Convert CropRect(w,h,x,y) to NVEncC --crop left,top,right,bottom.

    CropRect: w=active width, h=active height, x=left offset, y=top offset.
    NVEncC --crop: pixels to remove from each side.
    """
    left = crop.x
    top = crop.y
    right = source_width - crop.x - crop.w
    bottom = source_height - crop.y - crop.h
    return left, top, right, bottom


# Codecs supported by NVDEC hardware decoder
_NVDEC_CODECS: set[str] = {
    "h264", "hevc", "mpeg2video", "mpeg4", "vp8", "vp9", "vc1", "av1",
    "mpeg1video",
}


class NVEncCAdapter:
    """Implements Encoder protocol via NVEncC (rigaya) CLI.

    Outputs raw HEVC bitstream (not MKV) -- mkvmerge handles muxing.
    """

    def __init__(
        self,
        nvencc_path: Path,
        on_output: OutputCallback = None,
        log_dir: Path | None = None,
    ) -> None:
        self._nvencc = nvencc_path
        self._on_output = on_output
        self._log_dir = log_dir

    def set_log_dir(self, log_dir: Path | None) -> None:
        self._log_dir = log_dir

    # ------------------------------------------------------------------
    # Version
    # ------------------------------------------------------------------

    def _get_version(self) -> str:
        """Get NVEncC version string. Cached after first call."""
        cached: str | None = getattr(self, "_version_cached", None)
        if cached is not None:
            return cached
        try:
            result = subprocess.run(
                [str(self._nvencc), "--version"],
                capture_output=True, text=True, timeout=10,
            )
            # NVEncC outputs: "NVEncC (x64) X.YY (rZZZZ)" on first line
            m = re.search(r"(\d+\.\d+)", result.stdout)
            self._version_cached: str = m.group(1) if m else ""
        except Exception:
            self._version_cached = ""
        return self._version_cached

    # ------------------------------------------------------------------
    # Encoder settings string
    # ------------------------------------------------------------------

    def _build_encoder_settings(self, vp: VideoParams) -> str:
        """Build ENCODER_SETTINGS string for MKV global tag.

        Format: slash-separated, NVEncC params always present, filters only when applied.
        """
        version = self._get_version()
        parts: list[str] = ["hevc_nvenc"]
        if version:
            parts.append(f"NVEncC={version}")
        parts += [
            "main10",
            f"qvbr={vp.cq}",
            "preset=P5",
            "tune=uhq",
            "aq",
            "aq-temporal",
            "lookahead=32",
            "lookahead-level=3",
            "multipass=2pass-quarter",
        ]

        if vp.deinterlace:
            parts.append("deinterlace=nnedi(nns=64,nsize=32x6,slow)")

        if vp.crop is not None:
            left, top, right, bottom = _convert_crop(vp.crop, vp.source_width, vp.source_height)
            parts.append(f"crop={top}:{bottom}:{left}:{right}")

        if vp.sar_num != vp.sar_den:
            cur_w = vp.crop.w if vp.crop is not None else vp.source_width
            cur_h = vp.crop.h if vp.crop is not None else vp.source_height
            display_w, display_h = correct_sar(cur_w, cur_h, vp.sar_num, vp.sar_den)
            parts.append(f"sar={display_w}x{display_h}")

        if vp.dv_mode is not None:
            parts.append("dolby-vision=8.1")

        return " / ".join(parts)

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------

    def _build_encode_cmd(
        self,
        input_path: Path,
        output_path: Path,
        vp: VideoParams,
        *,
        vmaf_enabled: bool = False,
        rpu_path: Path | None = None,
    ) -> list[str | Path]:
        """Build the full NVEncC encode command."""
        cmd: list[str | Path] = [self._nvencc]

        # Hardware decode: NVDEC for supported codecs, sw decode otherwise.
        # Also fall back to avsw when left crop > 0 (NVEncC limitation).
        use_hwdec = (
            vp.source_codec in _NVDEC_CODECS
            and (vp.crop is None or vp.crop.x == 0)
        )
        cmd.append("--avhw" if use_hwdec else "--avsw")

        # Codec, profile, bit depth, tier
        cmd += ["-c", "hevc", "--profile", "main10", "--output-depth", "10"]
        cmd += ["--tier", "high"]

        # Quality / rate control
        cmd += [
            "--preset", "P5",
            "--tune", "uhq",
            "--qvbr", str(vp.cq),
            "--aq", "--aq-temporal",
            "--lookahead", "32",
            "--lookahead-level", "3",
            "--multipass", "2pass-quarter",
        ]

        # GOP structure
        cmd += ["--gop-len", str(vp.gop)]
        cmd += ["--strict-gop", "--repeat-headers"]

        # --- Crop ---
        if vp.crop is not None:
            left, top, right, bottom = _convert_crop(
                vp.crop, vp.source_width, vp.source_height,
            )
            cmd += ["--crop", f"{left},{top},{right},{bottom}"]
            # Align final dimensions to mod-8 for HEVC CU
            final_w = vp.crop.w
            final_h = vp.crop.h
            aligned = align_dimensions(final_w, final_h)
            if aligned.w != final_w or aligned.h != final_h:
                cmd += ["--output-res", f"{aligned.w}x{aligned.h}"]

        # --- Deinterlace ---
        if vp.deinterlace:
            cmd += ["--vpp-nnedi", "nns=64,nsize=32x6,quality=slow"]

        # --- SAR correction ---
        if vp.sar_num != vp.sar_den:
            cur_w = vp.crop.w if vp.crop is not None else vp.source_width
            cur_h = vp.crop.h if vp.crop is not None else vp.source_height
            display_w, display_h = correct_sar(cur_w, cur_h, vp.sar_num, vp.sar_den)
            aligned = align_dimensions(display_w, display_h)
            cmd += ["--output-res", f"{aligned.w}x{aligned.h}"]
            cmd += ["--vpp-resize", "spline64"]
            cmd += ["--sar", "1:1"]

        # --- Color metadata ---
        nvencc_range = _COLOR_RANGE_MAP.get(vp.color_range)
        if nvencc_range:
            cmd += ["--colorrange", nvencc_range]

        cmd += ["--colorprim", vp.color_primaries]
        cmd += ["--transfer", vp.color_transfer]
        cmd += ["--colormatrix", vp.color_matrix]

        # --- HDR metadata ---
        if vp.hdr is not None:
            if vp.hdr.content_light:
                parsed = _parse_content_light(vp.hdr.content_light)
                if parsed:
                    cll, fall = parsed
                    cmd += ["--max-cll", f"{cll},{fall}"]

            if vp.hdr.mastering_display:
                cmd += ["--master-display", vp.hdr.mastering_display]

        # --- Dolby Vision ---
        if rpu_path is not None:
            cmd += ["--dolby-vision-rpu", str(rpu_path)]
            cmd += ["--dolby-vision-profile", "8.1"]
            # Adjust active area when crop is applied
            if vp.crop is not None:
                cmd += ["--dolby-vision-rpu-prm", "crop=true"]

        # --- Quality metrics ---
        if vmaf_enabled:
            n_threads = max(1, (os.cpu_count() or 4) - 2)
            # Select VMAF model by resolution
            pixel_area = vp.crop.w * vp.crop.h if vp.crop is not None else vp.source_width * vp.source_height
            model = "vmaf_4k_v0.6.1" if pixel_area >= 3_686_400 else "vmaf_v0.6.1"
            cmd.append("--ssim")
            cmd += ["--vmaf", f"model={model},threads={n_threads},subsample=8"]

        # --- Input / Output ---
        cmd += ["-i", str(input_path)]
        cmd += ["-o", str(output_path)]

        return cmd

    # ------------------------------------------------------------------
    # Encode execution
    # ------------------------------------------------------------------

    def encode(
        self,
        input_path: Path,
        output_path: Path,
        video_params: VideoParams,
        source_size: int,
        on_progress: Callable[[float, str], None] | None = None,
        vmaf_enabled: bool = False,
        rpu_path: Path | None = None,
    ) -> EncodeResult:
        """Encode video via NVEncC. Parses stderr for progress."""
        cmd = self._build_encode_cmd(
            input_path, output_path, video_params,
            vmaf_enabled=vmaf_enabled, rpu_path=rpu_path,
        )
        str_cmd = [str(c) for c in cmd]
        logger.debug("nvencc cmd: %s", " ".join(str_cmd))

        encoder_settings = self._build_encoder_settings(video_params)

        # Open per-job log file
        encode_log = None
        if self._log_dir:
            encode_log = (self._log_dir / "nvencc_encode.log").open("w", encoding="utf-8")
            encode_log.write(f"$ {' '.join(str_cmd)}\n\n")
            encode_log.flush()

        process = subprocess.Popen(
            str_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,  # unbuffered binary pipes
        )
        # Close stdin immediately so NVEncC doesn't wait for input.
        # Any prompt (e.g. overwrite confirmation) will go to stderr
        # where we can see it in the TUI/logs.
        assert process.stdin is not None
        process.stdin.close()

        # NVEncC writes progress to stderr with \r (no \n).
        # We read binary, decode manually, split on \r and \n.
        progress_re = re.compile(r"\[(\d+\.?\d*)%\]")
        ssim_re = re.compile(r"All:\s*(\d+\.\d+)")
        vmaf_re = re.compile(r"VMAF\s+Score\s+(\d+\.\d+)", re.IGNORECASE)

        ssim_score: float | None = None
        vmaf_score: float | None = None
        stdout_lines: list[str] = []

        def _process_line(line: str) -> None:
            nonlocal ssim_score, vmaf_score
            line = line.strip()
            if not line:
                return

            if encode_log is not None:
                encode_log.write(line + "\n")
                encode_log.flush()

            if self._on_output is not None:
                self._on_output(line)

            m = progress_re.search(line)
            if m:
                pct = float(m.group(1))
                if on_progress is not None:
                    on_progress(pct, line)

            if "SSIM" in line:
                sm = ssim_re.search(line)
                if sm:
                    ssim_score = float(sm.group(1))

            if "VMAF" in line:
                vm = vmaf_re.search(line)
                if vm:
                    vmaf_score = float(vm.group(1))

        def _read_stdout() -> None:
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if line:
                    stdout_lines.append(line)
                    if encode_log is not None:
                        encode_log.write(f"[stdout] {line}\n")
                        encode_log.flush()

        def _read_stderr() -> None:
            """Read stderr byte-by-byte, split on \\r and \\n.

            NVEncC writes progress with \\r (no \\n), so line-based
            reading would block. Byte-by-byte in a daemon thread is safe:
            when the process is killed the pipe closes and read() returns b''.
            """
            assert process.stderr is not None
            raw_buf = bytearray()
            while True:
                byte = process.stderr.read(1)
                if not byte:
                    break
                if byte in (b"\r", b"\n"):
                    if raw_buf:
                        _process_line(raw_buf.decode("utf-8", errors="replace"))
                        raw_buf.clear()
                else:
                    raw_buf.extend(byte)
            if raw_buf:
                _process_line(raw_buf.decode("utf-8", errors="replace"))

        stdout_thread = threading.Thread(target=_read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        process.wait()
        stderr_thread.join(timeout=5)
        stdout_thread.join(timeout=5)

        # Also check stdout for metrics
        for sline in stdout_lines:
            if ssim_score is None and "SSIM" in sline:
                sm = ssim_re.search(sline)
                if sm:
                    ssim_score = float(sm.group(1))
            if vmaf_score is None and "VMAF" in sline:
                vm = vmaf_re.search(sline)
                if vm:
                    vmaf_score = float(vm.group(1))

        if encode_log is not None:
            encode_log.write(f"\n--- exit code: {process.returncode} ---\n")
            encode_log.close()

        if process.returncode != 0:
            logger.error("NVEncC encode failed (rc=%d)", process.returncode)

        return EncodeResult(
            return_code=process.returncode,
            encoder_settings=encoder_settings,
            vmaf_score=vmaf_score,
            ssim_score=ssim_score,
        )

