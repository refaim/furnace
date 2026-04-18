from __future__ import annotations

import json
import logging
import math
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from furnace.core.audio_profile import AudioMetrics
from furnace.core.detect import cluster_crop_values
from furnace.core.models import CropRect
from furnace.core.progress import ProgressSample

from ._subprocess import OutputCallback, run_tool

logger = logging.getLogger(__name__)

_PROFILE_WINDOW_SEC = 20.0
_PROFILE_STEREO_POINTS: tuple[float, ...] = (0.30, 0.60)
_PROFILE_MULTI_POINTS: tuple[float, ...] = (0.15, 0.35, 0.55, 0.75)
_PROFILE_SAMPLE_RATE = 48000
_DIGITAL_SILENCE_DB = -120.0
_ZERO_NORM_EPS = 1e-9
_CHANNELS_STEREO = 2
_CHANNELS_5_1 = 6
_CHANNELS_7_1 = 8


def _rms_db(x: np.ndarray) -> float:
    """RMS in dB. Empty input or near-zero signal → -120 dB floor."""
    if x.size == 0:
        return _DIGITAL_SILENCE_DB
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-30))
    if rms < _ZERO_NORM_EPS:
        return _DIGITAL_SILENCE_DB
    return 20.0 * math.log10(rms)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, zero-norm safe (returns 0.0 on empty or constant input)."""
    if a.size == 0 or b.size == 0:
        return 0.0
    a64 = a.astype(np.float64) - float(a.mean())
    b64 = b.astype(np.float64) - float(b.mean())
    na = float(np.sqrt(float((a64 * a64).sum())))
    nb = float(np.sqrt(float((b64 * b64).sum())))
    if na < _ZERO_NORM_EPS or nb < _ZERO_NORM_EPS:
        return 0.0
    return float((a64 * b64).sum() / (na * nb))


def _parse_ffmpeg_progress_block(kv: dict[str, str]) -> ProgressSample | None:
    """Convert one completed ffmpeg `-progress pipe:1` key=value block into a sample.

    `kv` is expected to contain the keys emitted between two `progress=` lines
    (inclusive). Returns `None` if `out_time_us` is missing, `"N/A"`, or
    unparseable.
    """
    out_time_us = kv.get("out_time_us")
    if out_time_us is None or out_time_us == "N/A":
        return None
    try:
        processed_s = int(out_time_us) / 1_000_000
    except ValueError:
        return None
    speed: float | None = None
    speed_str = kv.get("speed", "").strip()
    if speed_str and speed_str.endswith("x"):
        try:
            speed = float(speed_str[:-1])
        except ValueError:
            speed = None
    return ProgressSample(processed_s=processed_s, speed=speed)


class FFmpegAdapter:
    """Implements Prober + AudioExtractor."""

    def __init__(
        self,
        ffmpeg_path: Path,
        ffprobe_path: Path,
        on_output: OutputCallback = None,
        log_dir: Path | None = None,
    ) -> None:
        self._ffmpeg = ffmpeg_path
        self._ffprobe = ffprobe_path
        self._on_output = on_output
        self._log_dir = log_dir

    def set_log_dir(self, log_dir: Path | None) -> None:
        self._log_dir = log_dir

    def _get_ffmpeg_version(self) -> str:
        """Get ffmpeg version string (e.g. '7.1'). Cached after first call."""
        cached: str | None = getattr(self, "_ffmpeg_version_cached", None)
        if cached is not None:
            return cached
        try:
            result = subprocess.run(
                [str(self._ffmpeg), "-version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            m = re.match(r"ffmpeg version (\S+)", result.stdout)
            self._ffmpeg_version_cached: str = m.group(1) if m else ""
        except (OSError, subprocess.SubprocessError):
            self._ffmpeg_version_cached = ""
        return self._ffmpeg_version_cached

    # ------------------------------------------------------------------
    # Prober
    # ------------------------------------------------------------------

    def probe(self, path: Path) -> dict[str, Any]:
        """ffprobe -v quiet -print_format json -show_format -show_streams -show_chapters path"""
        cmd = [
            str(self._ffprobe),
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-show_chapters",
            str(path),
        ]
        logger.debug("probe cmd: %s", cmd)
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        if result.returncode != 0:
            logger.error("ffprobe failed (rc=%d): %s", result.returncode, result.stderr)
            raise RuntimeError(f"ffprobe failed with return code {result.returncode}: {result.stderr}")
        data: dict[str, Any] = json.loads(result.stdout)
        return data

    _CROP_SAMPLE_POINTS: tuple[float, ...] = (
        0.05,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    )
    _CROP_SAMPLE_POINTS_DVD: tuple[float, ...] = (
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.75,
        0.85,
        0.90,
    )

    def detect_crop(
        self,
        path: Path,
        duration_s: float,
        *,
        interlaced: bool = False,
        is_dvd: bool = False,
    ) -> CropRect | None:
        """Run cropdetect at multiple points across the timeline.

        Returns the median crop of the dominant cluster only if the cluster
        contains >50 % of samples.  Returns None otherwise.
        """
        points = self._CROP_SAMPLE_POINTS_DVD if is_dvd else self._CROP_SAMPLE_POINTS
        vf = "yadif,cropdetect=24:16:0" if interlaced else "cropdetect=24:16:0"

        crop_values: list[CropRect] = []

        for pct in points:
            seek = duration_s * pct
            cmd = [
                str(self._ffmpeg),
                "-hide_banner",
                "-ss",
                f"{seek:.2f}",
                "-i",
                str(path),
                "-t",
                "2",
                "-vf",
                vf,
                "-f",
                "null",
                "-",
            ]
            logger.debug("detect_crop cmd: %s", cmd)
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace", check=False,
            )
            last_crop: str | None = None
            for line in result.stderr.splitlines():
                m = re.search(r"crop=(\d+:\d+:\d+:\d+)", line)
                if m:
                    last_crop = m.group(1)
            if last_crop is not None:
                parts = last_crop.split(":")
                # Regex `crop=(\d+:\d+:\d+:\d+)` structurally guarantees 4 parts.
                crop_values.append(
                    CropRect(
                        w=int(parts[0]),
                        h=int(parts[1]),
                        x=int(parts[2]),
                        y=int(parts[3]),
                    )
                )

        if not crop_values:
            return None

        median_crop, cluster_size = cluster_crop_values(crop_values)
        if cluster_size <= len(crop_values) // 2:
            logger.info(
                "Crop not reliable: cluster %d:%d:%d:%d has %d/%d samples",
                median_crop.w,
                median_crop.h,
                median_crop.x,
                median_crop.y,
                cluster_size,
                len(crop_values),
            )
            return None

        return median_crop

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

    def run_idet(self, path: Path, duration_s: float) -> float:
        """Run idet filter at multiple points across the timeline.

        Samples 1000 frames at 10%, 30%, 50%, 70%, 90% of duration.
        Returns the ratio of interlaced frames (0.0 to 1.0).
        """
        total_interlaced = 0
        total_prog = 0

        for pct in (0.10, 0.30, 0.50, 0.70, 0.90):
            seek = duration_s * pct
            cmd = [
                str(self._ffmpeg),
                "-hide_banner",
                "-ss",
                f"{seek:.2f}",
                "-i",
                str(path),
                "-vf",
                "idet",
                "-frames:v",
                "1000",
                "-f",
                "null",
                "-",
            ]
            logger.debug("run_idet cmd: %s", cmd)
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace", check=False,
            )

            for line in result.stderr.splitlines():
                m = re.search(
                    r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*Progressive:\s*(\d+)",
                    line,
                )
                if m:
                    total_interlaced += int(m.group(1)) + int(m.group(2))
                    total_prog += int(m.group(3))

        total = total_interlaced + total_prog
        if total == 0:
            return 0.0

        return total_interlaced / total

    def probe_hdr_side_data(self, path: Path) -> list[dict[str, Any]]:
        """Read side_data_list from the first video frame.

        Uses: ffprobe -v quiet -print_format json -select_streams v:0
              -show_frames -read_intervals "%+#1" path
        """
        cmd = [
            str(self._ffprobe),
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-read_intervals",
            "%+#1",
            str(path),
        ]
        logger.debug("probe_hdr_side_data cmd: %s", cmd)
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        if result.returncode != 0:
            logger.warning("probe_hdr_side_data failed (rc=%d), returning []", result.returncode)
            return []
        data: dict[str, Any] = json.loads(result.stdout)
        frames = data.get("frames", [])
        if not frames:
            return []
        side_data: list[dict[str, Any]] = frames[0].get("side_data_list", [])
        return side_data

    # ------------------------------------------------------------------
    # AudioExtractor
    # ------------------------------------------------------------------

    def extract_track(
        self,
        input_path: Path,
        stream_index: int,
        output_path: Path,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int:
        """ffmpeg -i input -map 0:{index} -c copy -progress pipe:1 output"""
        # loglevel=fatal: -c copy is byte-copy, and ffmpeg's TrueHD "non
        # monotonically increasing dts" spam is logged at ERROR level despite
        # being cosmetic. -progress pipe:1 writes key=value blocks to stdout
        # which is parsed by the on_progress_line hook below; it's independent
        # of the loglevel.
        cmd = [
            str(self._ffmpeg),
            "-hide_banner",
            "-loglevel",
            "fatal",
            "-i",
            str(input_path),
            "-map",
            f"0:{stream_index}",
            "-c",
            "copy",
            "-progress",
            "pipe:1",
            "-y",
            str(output_path),
        ]
        log_path = self._log_dir / f"ffmpeg_extract_s{stream_index}.log" if self._log_dir else None

        kv_buf: dict[str, str] = {}

        def _on_progress_line(line: str) -> bool:
            # Every line of `-progress pipe:1` output is a `key=value` pair.
            # Consume (return True) to keep it out of the log and the TUI.
            if "=" not in line:
                return False
            key, _, val = line.partition("=")
            key = key.strip()
            kv_buf[key] = val.strip()
            if key == "progress":
                sample = _parse_ffmpeg_progress_block(kv_buf)
                kv_buf.clear()
                if sample is not None and on_progress is not None:
                    on_progress(sample)
            return True

        rc, _out = run_tool(
            cmd,
            on_output=self._on_output,
            on_progress_line=_on_progress_line,
            log_path=log_path,
        )
        return rc

    def ffmpeg_to_wav(
        self,
        input_path: Path,
        stream_index: int,
        output_wav: Path,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int:
        """ffmpeg -i input -map 0:{index} -f wav -rf64 auto -progress pipe:1 output.wav"""
        cmd = [
            str(self._ffmpeg),
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(input_path),
            "-map",
            f"0:{stream_index}",
            "-f",
            "wav",
            "-rf64",
            "auto",
            "-progress",
            "pipe:1",
            "-y",
            str(output_wav),
        ]
        log_path = self._log_dir / f"ffmpeg_to_wav_s{stream_index}.log" if self._log_dir else None

        kv_buf: dict[str, str] = {}

        def _on_progress_line(line: str) -> bool:
            if "=" not in line:
                return False
            key, _, val = line.partition("=")
            key = key.strip()
            kv_buf[key] = val.strip()
            if key == "progress":
                sample = _parse_ffmpeg_progress_block(kv_buf)
                kv_buf.clear()
                if sample is not None and on_progress is not None:
                    on_progress(sample)
            return True

        rc, _out = run_tool(
            cmd,
            on_output=self._on_output,
            on_progress_line=_on_progress_line,
            log_path=log_path,
        )
        return rc

    def _decode_pcm_window(
        self,
        path: Path,
        stream_index: int,
        channels: int,
        layout: str,
        start_s: float,
        dur_s: float,
    ) -> np.ndarray:
        """Decode one PCM window to f32le via ffmpeg stdout pipe."""
        cmd = [
            str(self._ffmpeg),
            "-v", "error", "-nostdin",
            "-ss", f"{start_s:.2f}",
            "-i", str(path),
            "-map", f"0:{stream_index}",
            "-t", f"{dur_s:.2f}",
            "-af", f"aformat=channel_layouts={layout}:sample_rates={_PROFILE_SAMPLE_RATE}",
            "-f", "f32le", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, check=False)
        if result.returncode != 0:
            logger.warning(
                "profile_audio_track ffmpeg rc=%d at %.1fs: %s",
                result.returncode, start_s,
                result.stderr.decode("utf-8", errors="replace")[:200],
            )
            return np.empty((0, channels), dtype=np.float32)
        buf = np.frombuffer(result.stdout, dtype=np.float32)
        n = buf.size // channels
        if n == 0:
            return np.empty((0, channels), dtype=np.float32)
        return buf[: n * channels].reshape(n, channels)

    def profile_audio_track(
        self,
        path: Path,
        stream_index: int,
        channels: int,
        duration_s: float,
    ) -> AudioMetrics:
        """Sample PCM windows and compute per-channel RMS + pairwise Pearson."""
        if channels == _CHANNELS_STEREO:
            layout = "stereo"
            points = _PROFILE_STEREO_POINTS
        elif channels == _CHANNELS_5_1:
            layout = "5.1"
            points = _PROFILE_MULTI_POINTS
        elif channels == _CHANNELS_7_1:
            layout = "7.1"
            points = _PROFILE_MULTI_POINTS
        else:
            raise ValueError(f"profile_audio_track: unsupported channels={channels}")

        chunks: list[np.ndarray] = []
        for frac in points:
            start = max(0.0, duration_s * frac - _PROFILE_WINDOW_SEC / 2)
            window = self._decode_pcm_window(
                path, stream_index, channels, layout, start, _PROFILE_WINDOW_SEC,
            )
            if window.size > 0:
                chunks.append(window)

        if not chunks:
            raise RuntimeError(
                f"profile_audio_track: no windows decoded from {path} stream {stream_index}",
            )

        data = np.concatenate(chunks, axis=0)
        cols = [data[:, i] for i in range(channels)]

        if channels == _CHANNELS_STEREO:
            left, right = cols
            return AudioMetrics(
                channels=_CHANNELS_STEREO,
                rms_l=_rms_db(left), rms_r=_rms_db(right),
                rms_c=None, rms_lfe=None, rms_ls=None, rms_rs=None,
                rms_lb=None, rms_rb=None,
                corr_lr=_pearson(left, right),
                corr_ls_l=None, corr_rs_r=None, corr_ls_rs=None,
                corr_lb_ls=None, corr_rb_rs=None,
            )

        if channels == _CHANNELS_5_1:
            left, right, center, lfe, ls, rs = cols
            return AudioMetrics(
                channels=_CHANNELS_5_1,
                rms_l=_rms_db(left), rms_r=_rms_db(right),
                rms_c=_rms_db(center), rms_lfe=_rms_db(lfe),
                rms_ls=_rms_db(ls), rms_rs=_rms_db(rs),
                rms_lb=None, rms_rb=None,
                corr_lr=_pearson(left, right),
                corr_ls_l=_pearson(ls, left),
                corr_rs_r=_pearson(rs, right),
                corr_ls_rs=_pearson(ls, rs),
                corr_lb_ls=None, corr_rb_rs=None,
            )

        # 7.1 — canonical order after aformat: [L, R, C, LFE, Lb, Rb, Ls, Rs]
        left, right, center, lfe, lb, rb, ls, rs = cols
        return AudioMetrics(
            channels=_CHANNELS_7_1,
            rms_l=_rms_db(left), rms_r=_rms_db(right),
            rms_c=_rms_db(center), rms_lfe=_rms_db(lfe),
            rms_ls=_rms_db(ls), rms_rs=_rms_db(rs),
            rms_lb=_rms_db(lb), rms_rb=_rms_db(rb),
            corr_lr=_pearson(left, right),
            corr_ls_l=_pearson(ls, left),
            corr_rs_r=_pearson(rs, right),
            corr_ls_rs=_pearson(ls, rs),
            corr_lb_ls=_pearson(lb, ls),
            corr_rb_rs=_pearson(rb, rs),
        )

    def downmix_to_mono_wav(
        self,
        input_path: Path,
        stream_index: int,
        channels: int,
        output_wav: Path,
        delay_ms: int,
    ) -> int:
        """ffmpeg pan filter -> mono WAV. Bypasses eac3to.

        Stereo sources are averaged (0.5*FL + 0.5*FR). Multichannel
        sources use an ITU-R BS.775 / Dolby Lo downmix, normalized via
        ``aformat=channel_layouts=<layout>`` and peak-protected via
        ``alimiter=limit=0.99``. LFE is excluded per ITU standard.
        """
        if channels == _CHANNELS_STEREO:
            filters = ["pan=mono|c0=0.5*FL+0.5*FR"]
        elif channels == _CHANNELS_5_1:
            filters = [
                "aformat=channel_layouts=5.1",
                "pan=mono|c0=0.707*FC+0.5*FL+0.5*FR+0.354*BL+0.354*BR",
                "alimiter=limit=0.99",
            ]
        elif channels == _CHANNELS_7_1:
            filters = [
                "aformat=channel_layouts=7.1",
                (
                    "pan=mono|c0=0.707*FC+0.5*FL+0.5*FR"
                    "+0.354*SL+0.354*SR+0.354*BL+0.354*BR"
                ),
                "alimiter=limit=0.99",
            ]
        else:
            raise ValueError(f"downmix_to_mono_wav: unsupported channels={channels}")

        if delay_ms > 0:
            filters.append(f"adelay={delay_ms}")
        elif delay_ms < 0:
            seconds = abs(delay_ms) / 1000.0
            filters.append(f"atrim=start={seconds:.3f}")

        af_value = ",".join(filters)

        cmd = [
            str(self._ffmpeg),
            "-hide_banner", "-loglevel", "warning",
            "-i", str(input_path),
            "-map", f"0:{stream_index}",
            "-af", af_value,
            "-ac", "1",
            "-f", "wav",
            "-rf64", "auto",
            "-y", str(output_wav),
        ]
        log_path = self._log_dir / f"ffmpeg_mono_s{stream_index}.log" if self._log_dir else None
        rc, _out = run_tool(cmd, on_output=self._on_output, log_path=log_path)
        return rc
