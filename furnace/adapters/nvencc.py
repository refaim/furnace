from __future__ import annotations

import logging
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from furnace.core.models import CropRect, EncodeResult, VideoParams
from furnace.core.progress import ProgressSample
from furnace.core.quality import aligned_crop, final_output_dimensions

from ._subprocess import OutputCallback, run_tool

logger = logging.getLogger(__name__)

_VMAF_4K_MIN_PIXEL_AREA = 2560 * 1440

_PROBE_METRIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "vmaf": re.compile(r"VMAF\s+Score\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE),
    "ssimulacra2": re.compile(r"SSIMU2\s+Score\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE),
    "cvvdp": re.compile(r"CVVDP\s+Score\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE),
}

_NVENCC_PCT_RE = re.compile(r"\[(\d+\.?\d*)%\]")
_NVENCC_FPS_RE = re.compile(r"(\d+\.?\d*)\s*fps,")


def _parse_nvencc_progress_line(
    line: str,
    src_fps: float | None = None,
) -> ProgressSample | None:
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


_COLOR_RANGE_MAP: dict[str, str] = {
    "tv": "limited",
    "pc": "full",
}


def _parse_content_light(content_light: str) -> tuple[str, str] | None:
    m = re.match(r"MaxCLL=(\d+)\s*,\s*MaxFALL=(\d+)", content_light)
    if m:
        return m.group(1), m.group(2)
    return None


def _convert_crop(crop: CropRect, source_width: int, source_height: int) -> tuple[int, int, int, int]:
    left = crop.x
    top = crop.y
    right = source_width - crop.x - crop.w
    bottom = source_height - crop.y - crop.h
    return left, top, right, bottom


_NVDEC_CODECS: set[str] = {
    "h264",
    "hevc",
    "mpeg4",
    "vp8",
    "vp9",
    "vc1",
    "av1",
}

_MIN_DV_VERSION = (8, 0)


def _version_tuple(version: str) -> tuple[int, int] | None:
    m = re.match(r"(\d+)\.(\d+)", version)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


class NVEncCAdapter:
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

    def _get_version(self) -> str:
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
            m = re.search(r"(\d+\.\d+)", result.stdout)
            self._version_cached: str = m.group(1) if m else ""
        except (OSError, subprocess.SubprocessError):
            self._version_cached = ""
        return self._version_cached

    def _build_encoder_settings(self, vp: VideoParams, *, cq_override: int | None = None) -> str:
        version = self._get_version()
        effective_cq = cq_override if cq_override is not None else vp.cq
        parts: list[str] = ["av1_nvenc"]
        if version:
            parts.append(f"NVEncC={version}")
        parts += ["main", "output-depth=10"]
        parts += [
            f"qvbr={effective_cq}",
            "preset=P4",
            "tune=uhq",
            "aq",
            "aq-temporal",
            "lookahead=32",
            "multipass=2pass-quarter",
        ]

        if vp.deinterlace:
            parts.append("deinterlace=nnedi(nns=64,nsize=32x6,slow)")

        crop = aligned_crop(vp)
        if crop is not None:
            left, top, right, bottom = _convert_crop(crop, vp.source_width, vp.source_height)
            parts.append(f"crop={top}:{bottom}:{left}:{right}")

        if vp.dv_mode is not None:
            parts.append("dolby-vision=10.1")

        return " / ".join(parts)

    def _probe_metric_flags(self, metric: str, vp: VideoParams) -> list[str]:
        if metric == "ssimulacra2":
            return ["--vship-ssimulacra2"]
        if metric == "cvvdp":
            return ["--vship-cvvdp"]
        if metric == "vmaf":
            n_threads = max(1, (os.cpu_count() or 4) - 2)
            final_w, final_h = final_output_dimensions(vp)
            model = "vmaf_4k_v0.6.1" if final_w * final_h >= _VMAF_4K_MIN_PIXEL_AREA else "vmaf_v0.6.1"
            return ["--vmaf", f"model={model},threads={n_threads},subsample=8"]
        raise ValueError(f"unknown probe metric {metric!r}")

    def _build_encode_cmd(
        self,
        input_path: Path,
        output_path: Path,
        vp: VideoParams,
        *,
        rpu_path: Path | None = None,
        cq_override: int | None = None,
        probe_metric: str | None = None,
    ) -> list[str | Path]:
        cmd: list[str | Path] = [self._nvencc]

        crop = aligned_crop(vp)
        use_hwdec = vp.source_codec in _NVDEC_CODECS and (crop is None or crop.x == 0)
        cmd.append("--avhw" if use_hwdec else "--avsw")

        cmd += ["-c", "av1", "--profile", "main", "--output-depth", "10"]

        effective_cq = cq_override if cq_override is not None else vp.cq
        cmd += [
            "--preset",
            "P4",
            "--tune",
            "uhq",
            "--qvbr",
            str(effective_cq),
            "--aq",
            "--aq-temporal",
            "--lookahead",
            "32",
            "--multipass",
            "2pass-quarter",
        ]

        cmd += ["--gop-len", str(vp.gop)]
        cmd += ["--strict-gop", "--repeat-headers"]

        if crop is not None:
            left, top, right, bottom = _convert_crop(
                crop,
                vp.source_width,
                vp.source_height,
            )
            cmd += ["--crop", f"{left},{top},{right},{bottom}"]

        if vp.deinterlace:
            cmd += ["--vpp-nnedi", "nns=64,nsize=32x6,quality=slow"]

        final_w, final_h = final_output_dimensions(vp)
        pre_resize_w = crop.w if crop is not None else vp.source_width
        pre_resize_h = crop.h if crop is not None else vp.source_height
        if (final_w, final_h) != (pre_resize_w, pre_resize_h):
            cmd += ["--output-res", f"{final_w}x{final_h}"]

        if vp.sar_num != vp.sar_den:
            cmd += ["--vpp-resize", "spline64"]
            cmd += ["--sar", "1:1"]

        nvencc_range = _COLOR_RANGE_MAP.get(vp.color_range)
        if nvencc_range:
            cmd += ["--colorrange", nvencc_range]

        cmd += ["--colorprim", vp.color_primaries]
        cmd += ["--transfer", vp.color_transfer]
        cmd += ["--colormatrix", vp.color_matrix]

        if vp.hdr is not None:
            if vp.hdr.content_light:
                parsed = _parse_content_light(vp.hdr.content_light)
                if parsed:
                    cll, fall = parsed
                    cmd += ["--max-cll", f"{cll},{fall}"]

            if vp.hdr.mastering_display:
                cmd += ["--master-display", vp.hdr.mastering_display]

        if rpu_path is not None:
            cmd += ["--dolby-vision-rpu", str(rpu_path)]
            cmd += ["--dolby-vision-profile", "10.1"]
            if crop is not None:
                cmd += ["--dolby-vision-rpu-prm", "crop=true"]

        if probe_metric is not None:
            cmd += self._probe_metric_flags(probe_metric, vp)

        cmd += ["-i", str(input_path)]
        cmd += ["-o", str(output_path)]

        return cmd

    def encode(
        self,
        input_path: Path,
        output_path: Path,
        video_params: VideoParams,
        *,
        on_progress: Callable[[ProgressSample], None] | None = None,
        rpu_path: Path | None = None,
        cq_override: int | None = None,
    ) -> EncodeResult:
        if rpu_path is not None:
            version = self._get_version()
            parsed = _version_tuple(version)
            if parsed is not None and parsed < _MIN_DV_VERSION:
                raise RuntimeError(f"AV1 Dolby Vision requires NVEncC >= 8.00 (detected {version}).")

        cmd = self._build_encode_cmd(
            input_path,
            output_path,
            video_params,
            rpu_path=rpu_path,
            cq_override=cq_override,
        )
        str_cmd = [str(c) for c in cmd]
        logger.debug("nvencc cmd: %s", " ".join(str_cmd))

        encoder_settings = self._build_encoder_settings(video_params, cq_override=cq_override)

        src_fps = video_params.fps_num / video_params.fps_den if video_params.fps_den else 0.0

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
            on_output=self._on_output,
            on_progress_line=_on_progress_line,
            log_path=log_path,
        )

        return EncodeResult(return_code=rc, encoder_settings=encoder_settings)

    def probe(
        self,
        input_path: Path,
        output_path: Path,
        video_params: VideoParams,
        *,
        qvbr: int,
        metric: str,
    ) -> float:
        cmd = self._build_encode_cmd(
            input_path,
            output_path,
            video_params,
            cq_override=qvbr,
            probe_metric=metric,
        )
        pattern = _PROBE_METRIC_PATTERNS[metric]
        str_cmd = [str(c) for c in cmd]
        logger.debug("nvencc probe cmd: %s", " ".join(str_cmd))

        score: float | None = None

        def _on_output(line: str) -> None:
            nonlocal score
            if self._on_output is not None:
                self._on_output(line)
            m = pattern.search(line)
            if m:
                score = float(m.group(1))

        log_path = self._log_dir / f"nvencc_probe_{metric}_q{qvbr}.log" if self._log_dir else None
        rc, _out = run_tool(str_cmd, on_output=_on_output, log_path=log_path)
        if rc != 0:
            raise RuntimeError(f"NVEncC probe failed (rc={rc}) at qvbr={qvbr} metric={metric}")
        if score is None:
            raise RuntimeError(f"NVEncC probe reported no {metric} score at qvbr={qvbr}")
        return score
