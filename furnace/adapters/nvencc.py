"""NVEncC (rigaya) video encoder adapter.

Implements the Encoder protocol using NVEncC for AV1 encoding with NVDEC
hardware decode, HDR/DV passthrough, and built-in quality metrics.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from furnace.core.models import CropRect, EncodeResult, VideoParams
from furnace.core.progress import ProgressSample
from furnace.core.quality import final_output_dimensions

from ._subprocess import OutputCallback, run_tool

logger = logging.getLogger(__name__)

# 1440p pixel area (2560x1440). At or above this we pick the 4K-tuned VMAF model.
_VMAF_4K_MIN_PIXEL_AREA = 2560 * 1440

_NVENCC_PCT_RE = re.compile(r"\[(\d+\.?\d*)%\]")
_NVENCC_FPS_RE = re.compile(r"(\d+\.?\d*)\s*fps,")


def _parse_nvencc_progress_line(
    line: str,
    src_fps: float | None = None,
) -> ProgressSample | None:
    """Convert one NVEncC progress line into a sample.

    `src_fps` is the source video frame rate; when passed, the encoder's
    current fps is divided by it to compute a speed multiplier. Without it,
    `speed` is left as None.
    """
    m_pct = _NVENCC_PCT_RE.search(line)
    if not m_pct:
        return None
    fraction = float(m_pct.group(1)) / 100.0
    speed: float | None = None
    if src_fps and src_fps > 0:
        m_fps = _NVENCC_FPS_RE.search(line)
        if m_fps:
            speed = float(m_fps.group(1)) / src_fps
    return ProgressSample(fraction=fraction, speed=speed)


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
    "h264",
    "hevc",
    "mpeg4",
    "vp8",
    "vp9",
    "vc1",
    "av1",
}
# mpeg1video and mpeg2video are deliberately NOT in this set: NVDEC's
# MPEG1/2 path is unreliable on interlaced DVD sources (encoder stops
# after ~40 frames with a clean exit code on PAL DVD MPEG2 with
# field-picture flags set). Software decoding of MPEG1/2 is trivial on
# any modern CPU, so we always fall back to --avsw for those codecs.

# Minimum NVEncC version for AV1 Dolby Vision (Profile 10.x RPU): 8.00.
# Profiles 10.x added in 8.00b4; AV1 RPU interleaving fixed in 8.00b5.
_MIN_DV_VERSION = (8, 0)


def _version_tuple(version: str) -> tuple[int, int] | None:
    """Parse 'MAJOR.MINOR' into a comparable (major, minor) tuple, or None."""
    m = re.match(r"(\d+)\.(\d+)", version)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


class NVEncCAdapter:
    """Implements Encoder protocol via NVEncC (rigaya) CLI.

    Outputs a raw AV1 OBU elementary stream (not MKV) -- mkvmerge handles
    muxing, which also lets the Dolby Vision RPU (a T.35 metadata OBU) survive.
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
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            # NVEncC outputs: "NVEncC (x64) X.YY (rZZZZ)" on first line
            m = re.search(r"(\d+\.\d+)", result.stdout)
            self._version_cached: str = m.group(1) if m else ""
        except (OSError, subprocess.SubprocessError):
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
        parts: list[str] = ["av1_nvenc"]
        if version:
            parts.append(f"NVEncC={version}")
        parts += ["main", "output-depth=10"]
        # Rate-control fields must mirror those chosen in _build_encode_cmd
        # so the ENCODER_SETTINGS tag is never misleading.
        parts += [
            f"qvbr={vp.cq}",
            "preset=P4",
            "tune=uhq",
            "aq",
            "aq-temporal",
            "lookahead=32",
            "multipass=2pass-quarter",
        ]

        if vp.deinterlace:
            parts.append("deinterlace=nnedi(nns=64,nsize=32x6,slow)")

        if vp.crop is not None:
            left, top, right, bottom = _convert_crop(vp.crop, vp.source_width, vp.source_height)
            parts.append(f"crop={top}:{bottom}:{left}:{right}")

        if vp.dv_mode is not None:
            parts.append("dolby-vision=10.1")

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
        use_hwdec = vp.source_codec in _NVDEC_CODECS and (vp.crop is None or vp.crop.x == 0)
        cmd.append("--avhw" if use_hwdec else "--avsw")

        # Codec, profile, bit depth (AV1: no tier)
        cmd += ["-c", "av1", "--profile", "main", "--output-depth", "10"]

        # Quality / rate control (AV1: no lookahead-level). QVBR perceptual
        # profile for all jobs; grain sources route to SVT-AV1 at the executor.
        cmd += [
            "--preset", "P4", "--tune", "uhq", "--qvbr", str(vp.cq),
            "--aq", "--aq-temporal", "--lookahead", "32",
            "--multipass", "2pass-quarter",
        ]

        # GOP structure
        cmd += ["--gop-len", str(vp.gop)]
        cmd += ["--strict-gop", "--repeat-headers"]

        # --- Crop ---
        if vp.crop is not None:
            left, top, right, bottom = _convert_crop(
                vp.crop,
                vp.source_width,
                vp.source_height,
            )
            cmd += ["--crop", f"{left},{top},{right},{bottom}"]

        # --- Deinterlace ---
        if vp.deinterlace:
            cmd += ["--vpp-nnedi", "nns=64,nsize=32x6,quality=slow"]

        # --- Output resolution (single source of truth: helper) ---
        final_w, final_h = final_output_dimensions(vp)
        pre_resize_w = vp.crop.w if vp.crop is not None else vp.source_width
        pre_resize_h = vp.crop.h if vp.crop is not None else vp.source_height
        if (final_w, final_h) != (pre_resize_w, pre_resize_h):
            cmd += ["--output-res", f"{final_w}x{final_h}"]

        # --- SAR correction (anamorphic only) ---
        # When SAR is non-square, NVEncC must rescale via a high-quality
        # filter and override the sample aspect ratio of the encoded stream
        # back to 1:1. For pure mod-8 trims (square SAR) the default resize
        # is fine -- no spline64 / sar override needed.
        if vp.sar_num != vp.sar_den:
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
            cmd += ["--dolby-vision-profile", "10.1"]
            # Adjust active area when crop is applied
            if vp.crop is not None:
                cmd += ["--dolby-vision-rpu-prm", "crop=true"]

        # --- Quality metrics ---
        if vmaf_enabled:
            n_threads = max(1, (os.cpu_count() or 4) - 2)
            # Select VMAF model by resolution
            pixel_area = vp.crop.w * vp.crop.h if vp.crop is not None else vp.source_width * vp.source_height
            model = "vmaf_4k_v0.6.1" if pixel_area >= _VMAF_4K_MIN_PIXEL_AREA else "vmaf_v0.6.1"
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
        *,
        on_progress: Callable[[ProgressSample], None] | None = None,
        vmaf_enabled: bool = False,
        rpu_path: Path | None = None,
    ) -> EncodeResult:
        """Encode video via NVEncC. Parses stderr for progress via the unified parser."""
        if rpu_path is not None:
            version = self._get_version()
            parsed = _version_tuple(version)
            if parsed is not None and parsed < _MIN_DV_VERSION:
                raise RuntimeError(
                    f"AV1 Dolby Vision requires NVEncC >= 8.00 (detected {version})."
                )

        cmd = self._build_encode_cmd(
            input_path,
            output_path,
            video_params,
            vmaf_enabled=vmaf_enabled,
            rpu_path=rpu_path,
        )
        str_cmd = [str(c) for c in cmd]
        logger.debug("nvencc cmd: %s", " ".join(str_cmd))

        encoder_settings = self._build_encoder_settings(video_params)

        src_fps = video_params.fps_num / video_params.fps_den if video_params.fps_den else 0.0
        ssim_score: float | None = None
        vmaf_score: float | None = None
        ssim_re = re.compile(r"All:\s*(\d+\.\d+)")
        vmaf_re = re.compile(r"VMAF\s+Score\s+(\d+\.\d+)", re.IGNORECASE)

        def _on_output(line: str) -> None:
            nonlocal ssim_score, vmaf_score
            if self._on_output is not None:
                self._on_output(line)
            if "SSIM" in line:
                m = ssim_re.search(line)
                if m:
                    ssim_score = float(m.group(1))
            if "VMAF" in line:
                m = vmaf_re.search(line)
                if m:
                    vmaf_score = float(m.group(1))

        def _on_progress_line(line: str) -> bool:
            sample = _parse_nvencc_progress_line(line, src_fps=src_fps)
            if sample is None:
                return False
            if on_progress is not None:
                on_progress(sample)
            return True

        log_path = self._log_dir / "nvencc_encode.log" if self._log_dir else None
        rc, _out = run_tool(
            str_cmd,
            on_output=_on_output,
            on_progress_line=_on_progress_line,
            log_path=log_path,
        )

        return EncodeResult(
            return_code=rc,
            encoder_settings=encoder_settings,
            ssim_score=ssim_score,
            vmaf_score=vmaf_score,
        )
