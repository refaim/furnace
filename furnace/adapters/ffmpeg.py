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
from furnace.core.detect import aggregate_crop
from furnace.core.models import CropRect, VideoParams
from furnace.core.progress import ProgressSample

from ._geometry import build_vf
from ._subprocess import OutputCallback, run_tool

logger = logging.getLogger(__name__)

_PROFILE_WINDOW_SEC = 20.0
_PROFILE_STEREO_POINTS: tuple[float, ...] = tuple(i / 13 for i in range(1, 13))
_PROFILE_MULTI_POINTS: tuple[float, ...] = (0.15, 0.35, 0.55, 0.75)
_PROFILE_SAMPLE_RATE = 48000
_DIGITAL_SILENCE_DB = -120.0
_ZERO_NORM_EPS = 1e-9
_CHANNELS_STEREO = 2
_CHANNELS_5_1 = 6
_CHANNELS_7_1 = 8

_FIELD_PAIRING_SECONDS = 60

_GRAIN_WINDOWS: tuple[float, ...] = (0.10, 0.30, 0.50, 0.70, 0.90)
_GRAIN_FRAMES = 24
_GRAIN_BLOCK = 16
_GRAIN_LUMA_MIN, _GRAIN_LUMA_MAX = 30.0, 220.0
_GRAIN_MIN_VALID_BLOCKS = 20
_GRAIN_STATIC_QUANTILE = 0.15
_GRAIN_MIN_FRAMES = 8
_GRAIN_MIN_BLOCK_FLICKER = 0.01

_Y4M_HEADER_RE = re.compile(rb"YUV4MPEG2 W([1-9][0-9]*) H([1-9][0-9]*)[^\n]*\n")
_Y4M_FRAME = b"FRAME\n"


def _rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return _DIGITAL_SILENCE_DB
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-30))
    if rms < _ZERO_NORM_EPS:
        return _DIGITAL_SILENCE_DB
    return 20.0 * math.log10(rms)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    a64 = a.astype(np.float64) - float(a.mean())
    b64 = b.astype(np.float64) - float(b.mean())
    na = float(np.sqrt(float((a64 * a64).sum())))
    nb = float(np.sqrt(float((b64 * b64).sum())))
    if na < _ZERO_NORM_EPS or nb < _ZERO_NORM_EPS:
        return 0.0
    return float((a64 * b64).sum() / (na * nb))


def _parse_y4m_gray(buf: bytes) -> np.ndarray | None:
    header = _Y4M_HEADER_RE.match(buf)
    if header is None:
        return None
    width, height = int(header.group(1)), int(header.group(2))
    frame_px = width * height

    frames: list[np.ndarray] = []
    pos = header.end()
    while buf.startswith(_Y4M_FRAME, pos):
        start = pos + len(_Y4M_FRAME)
        stop = start + frame_px
        if stop > len(buf):
            break
        frames.append(np.frombuffer(buf, dtype=np.uint8, count=frame_px, offset=start))
        pos = stop
    if not frames:
        return None
    return np.stack(frames).reshape(len(frames), height, width)


def _grain_window_value(frames: np.ndarray) -> float | None:
    n, height, width = frames.shape
    h_blocks = height // _GRAIN_BLOCK
    w_blocks = width // _GRAIN_BLOCK
    used_h = h_blocks * _GRAIN_BLOCK
    used_w = w_blocks * _GRAIN_BLOCK
    cropped = frames[:, :used_h, :used_w]

    luma = cropped.reshape(n, h_blocks, _GRAIN_BLOCK, w_blocks, _GRAIN_BLOCK).mean(axis=(0, 2, 4))

    step_diff = np.abs(np.diff(cropped, axis=0))
    block_flicker = step_diff.reshape(n - 1, h_blocks, _GRAIN_BLOCK, w_blocks, _GRAIN_BLOCK).mean(axis=(2, 4))
    block_median = np.median(block_flicker, axis=0)

    valid = (luma > _GRAIN_LUMA_MIN) & (luma < _GRAIN_LUMA_MAX) & (block_median > _GRAIN_MIN_BLOCK_FLICKER)
    valid_medians = block_median[valid]
    if valid_medians.size < _GRAIN_MIN_VALID_BLOCKS:
        return None
    return float(np.quantile(valid_medians, _GRAIN_STATIC_QUANTILE))


def _parse_ffmpeg_progress_block(kv: dict[str, str]) -> ProgressSample | None:
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


def _make_ffmpeg_progress_handler(
    on_progress: Callable[[ProgressSample], None] | None,
) -> Callable[[str], bool]:
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

    return _on_progress_line


class FFmpegAdapter:
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

    def probe(self, path: Path) -> dict[str, Any]:
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

    _CROP_BATCH_HD = 10
    _CROP_BATCH_DVD = 15
    _CROP_MAX_BATCHES = 4

    _CROP_DETECT_LIMIT = 40

    _CROP_DETECT_ROUND = 2

    @staticmethod
    def _crop_sample_batches(per_batch: int, max_batches: int) -> list[list[float]]:
        total = per_batch * max_batches
        fracs = [(i + 0.5) / total for i in range(total)]
        return [fracs[b::max_batches] for b in range(max_batches)]

    def detect_crop(
        self,
        path: Path,
        duration_s: float,
        *,
        interlaced: bool = False,
        is_dvd: bool = False,
        hdr_transfer: str | None = None,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> CropRect | None:
        per_batch = self._CROP_BATCH_DVD if is_dvd else self._CROP_BATCH_HD
        batches = self._crop_sample_batches(per_batch, self._CROP_MAX_BATCHES)
        total_points = per_batch * self._CROP_MAX_BATCHES

        parts: list[str] = []
        if interlaced:
            parts.append("yadif")
        if hdr_transfer is not None:
            parts.append(
                f"zscale=tin={hdr_transfer}:min=2020_ncl:pin=2020:t=linear:npl=100",
            )
            parts.append(
                "zscale=tin=linear:min=2020_ncl:pin=2020:t=bt709:m=bt709:p=bt709:r=tv",
            )
        parts.append("format=yuv420p")
        parts.append(
            f"cropdetect={self._CROP_DETECT_LIMIT}:{self._CROP_DETECT_ROUND}:0",
        )
        vf = ",".join(parts)

        crop_values: list[CropRect] = []
        done = 0
        prev: CropRect | None = None
        result: CropRect | None = None

        for batch in batches:
            for pct in batch:
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
                run = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
                last_crop: str | None = None
                for line in run.stderr.splitlines():
                    m = re.search(r"crop=(\d+:\d+:\d+:\d+)", line)
                    if m:
                        last_crop = m.group(1)
                if last_crop is not None:
                    parts_crop = last_crop.split(":")
                    crop_values.append(
                        CropRect(
                            w=int(parts_crop[0]),
                            h=int(parts_crop[1]),
                            x=int(parts_crop[2]),
                            y=int(parts_crop[3]),
                        )
                    )
                done += 1
                if on_progress is not None:
                    on_progress(ProgressSample(fraction=done / total_points))

            if not crop_values:
                continue
            current = aggregate_crop(crop_values)
            if current == prev:
                result = current
                break
            prev = current
        else:
            result = prev

        if on_progress is not None:
            on_progress(ProgressSample(fraction=1.0))
        return result

    def get_encoder_tag(self, path: Path) -> str | None:
        try:
            data = self.probe(path)
        except RuntimeError:
            return None
        tags = data.get("format", {}).get("tags", {})
        for key in ("ENCODER", "encoder"):
            if key in tags:
                return str(tags[key])
        return None

    def run_idet(
        self,
        path: Path,
        duration_s: float,
        *,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> float:
        points = (0.10, 0.30, 0.50, 0.70, 0.90)
        total_interlaced = 0
        total_prog = 0

        for i, pct in enumerate(points, start=1):
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
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

            for line in result.stderr.splitlines():
                m = re.search(
                    r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*Progressive:\s*(\d+)",
                    line,
                )
                if m:
                    total_interlaced += int(m.group(1)) + int(m.group(2))
                    total_prog += int(m.group(3))

            if on_progress is not None:
                on_progress(ProgressSample(fraction=i / len(points)))

        total = total_interlaced + total_prog
        if total == 0:
            return 0.0

        return total_interlaced / total

    def probe_hdr_side_data(self, path: Path) -> list[dict[str, Any]]:
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

    def sample_repeat_pict(self, path: Path, duration_s: float) -> list[int]:
        points = (0.10, 0.30, 0.50, 0.70, 0.90)
        flags: list[int] = []

        for pct in points:
            seek = duration_s * pct
            cmd = [
                str(self._ffprobe),
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_entries",
                "frame=repeat_pict",
                "-read_intervals",
                f"{seek:.2f}%+#500",
                str(path),
            ]
            logger.debug("sample_repeat_pict cmd: %s", cmd)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if result.returncode != 0:
                logger.warning("sample_repeat_pict window at %.2fs failed (rc=%d)", seek, result.returncode)
                continue
            try:
                data: dict[str, Any] = json.loads(result.stdout)
                window_flags = [int(frame.get("repeat_pict", 0)) for frame in data.get("frames", [])]
            except ValueError:
                logger.warning("sample_repeat_pict window at %.2fs returned unparseable data", seek)
                continue
            flags.extend(window_flags)

        return flags

    def sample_field_pairing(self, path: Path) -> tuple[int, int]:
        cmd = [
            str(self._ffprobe),
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_frames,nb_read_packets",
            "-read_intervals",
            f"%+{_FIELD_PAIRING_SECONDS}",
            str(path),
        ]
        logger.debug("sample_field_pairing cmd: %s", cmd)
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        if result.returncode != 0:
            logger.warning("sample_field_pairing failed (rc=%d), returning (0, 0)", result.returncode)
            return (0, 0)
        try:
            data: dict[str, Any] = json.loads(result.stdout)
            streams: list[dict[str, Any]] = data.get("streams", [])
            if not streams:
                logger.warning("sample_field_pairing found no video stream, returning (0, 0)")
                return (0, 0)
            frames = int(streams[0]["nb_read_frames"])
            packets = int(streams[0]["nb_read_packets"])
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("sample_field_pairing returned unusable data (%s), returning (0, 0)", exc)
            return (0, 0)
        return (frames, packets)

    def sample_grain(self, path: Path, duration_s: float) -> list[float]:
        values: list[float] = []

        for pct in _GRAIN_WINDOWS:
            seek = duration_s * pct
            cmd = [
                str(self._ffmpeg),
                "-v",
                "error",
                "-ss",
                f"{seek:.2f}",
                "-i",
                str(path),
                "-frames:v",
                str(_GRAIN_FRAMES),
                "-vf",
                "format=gray",
                "-strict",
                "-1",
                "-f",
                "yuv4mpegpipe",
                "-pix_fmt",
                "gray",
                "-",
            ]
            logger.debug("sample_grain cmd: %s", cmd)
            result = subprocess.run(cmd, capture_output=True, check=False)
            if result.returncode != 0:
                logger.warning(
                    "sample_grain window at %.2fs failed (rc=%d)",
                    seek,
                    result.returncode,
                )
                continue
            decoded = _parse_y4m_gray(result.stdout)
            if decoded is None:
                logger.warning(
                    "sample_grain window at %.2fs returned no readable y4m frames",
                    seek,
                )
                continue
            n = decoded.shape[0]
            if n < _GRAIN_MIN_FRAMES:
                logger.warning(
                    "sample_grain window at %.2fs decoded only %d frame(s) (< %d)",
                    seek,
                    n,
                    _GRAIN_MIN_FRAMES,
                )
                continue
            window_value = _grain_window_value(decoded.astype(np.float32))
            if window_value is None:
                logger.debug("sample_grain window at %.2fs had too few static blocks", seek)
                continue
            values.append(window_value)

        return values

    def copy_video(
        self,
        input_path: Path,
        output_path: Path,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int:
        cmd = [
            str(self._ffmpeg),
            "-hide_banner",
            "-loglevel",
            "fatal",
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-progress",
            "pipe:1",
            "-y",
            str(output_path),
        ]
        log_path = self._log_dir / "ffmpeg_copy_video.log" if self._log_dir else None

        rc, _out = run_tool(
            cmd,
            on_output=self._on_output,
            on_progress_line=_make_ffmpeg_progress_handler(on_progress),
            log_path=log_path,
        )
        return rc

    def extract_window(
        self,
        input_path: Path,
        output_path: Path,
        *,
        start_s: float,
        frames: int,
    ) -> int:
        cmd = [
            str(self._ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_s:.3f}",
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-frames:v",
            str(frames),
            "-c:v",
            "copy",
            "-an",
            "-sn",
            "-y",
            str(output_path),
        ]
        log_path = self._log_dir / "ffmpeg_extract_window.log" if self._log_dir else None
        rc, _out = run_tool(cmd, on_output=self._on_output, log_path=log_path)
        return rc

    def build_reference(
        self,
        input_path: Path,
        output_path: Path,
        video_params: VideoParams,
    ) -> int:
        cmd = [
            str(self._ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-vf",
            build_vf(video_params),
            "-c:v",
            "ffv1",
            "-color_range",
            video_params.color_range,
            "-color_primaries",
            video_params.color_primaries,
            "-color_trc",
            video_params.color_transfer,
            "-colorspace",
            video_params.color_matrix,
            "-r",
            f"{video_params.fps_num}/{video_params.fps_den}",
            "-an",
            "-sn",
            "-y",
            str(output_path),
        ]
        log_path = self._log_dir / "ffmpeg_build_reference.log" if self._log_dir else None
        rc, _out = run_tool(cmd, on_output=self._on_output, log_path=log_path)
        return rc

    def window_bitrates(self, source: Path, window_s: float) -> list[tuple[float, float]]:
        cmd = [
            str(self._ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time,size",
            "-of",
            "csv=p=0",
            str(source),
        ]
        logger.debug("window_bitrates cmd: %s", cmd)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            logger.warning("window_bitrates ffprobe failed (rc=%d)", result.returncode)
            return []
        totals: dict[int, int] = {}
        for line in result.stdout.splitlines():
            pts_str, sep, size_str = line.partition(",")
            if not sep:
                continue
            try:
                pts = float(pts_str)
                size = int(size_str)
            except ValueError:
                continue
            if pts < 0.0:
                continue
            bin_idx = int(pts / window_s)
            totals[bin_idx] = totals.get(bin_idx, 0) + size
        return [(b * window_s, totals[b] / 1024.0) for b in sorted(totals)]

    def extract_track(
        self,
        input_path: Path,
        stream_index: int,
        output_path: Path,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int:
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

        rc, _out = run_tool(
            cmd,
            on_output=self._on_output,
            on_progress_line=_make_ffmpeg_progress_handler(on_progress),
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
        cmd = [
            str(self._ffmpeg),
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(input_path),
            "-map",
            f"0:{stream_index}",
            "-c:a",
            "pcm_s24le",
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

        rc, _out = run_tool(
            cmd,
            on_output=self._on_output,
            on_progress_line=_make_ffmpeg_progress_handler(on_progress),
            log_path=log_path,
        )
        return rc

    def decode_full_wav(
        self,
        input_path: Path,
        stream_index: int,
        output_wav: Path,
        *,
        disable_drc: bool = False,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int:
        decode_opts = ["-err_detect", "ignore_err"]
        if disable_drc:
            decode_opts += ["-drc_scale", "0"]
        cmd = [
            str(self._ffmpeg),
            "-hide_banner",
            "-loglevel",
            "warning",
            *decode_opts,
            "-i",
            str(input_path),
            "-map",
            f"0:{stream_index}",
            "-c:a",
            "pcm_s24le",
            "-f",
            "wav",
            "-rf64",
            "auto",
            "-progress",
            "pipe:1",
            "-y",
            str(output_wav),
        ]
        log_path = self._log_dir / f"ffmpeg_full_decode_s{stream_index}.log" if self._log_dir else None

        rc, _out = run_tool(
            cmd,
            on_output=self._on_output,
            on_progress_line=_make_ffmpeg_progress_handler(on_progress),
            log_path=log_path,
        )
        return rc

    def transcode_to_flac(
        self,
        input_path: Path,
        output_path: Path,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int:
        cmd = [
            str(self._ffmpeg),
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(input_path),
            "-map",
            "0:a:0",
            "-c:a",
            "flac",
            "-progress",
            "pipe:1",
            "-y",
            str(output_path),
        ]
        log_path = self._log_dir / f"ffmpeg_to_flac_{output_path.stem}.log" if self._log_dir else None

        rc, _out = run_tool(
            cmd,
            on_output=self._on_output,
            on_progress_line=_make_ffmpeg_progress_handler(on_progress),
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
        cmd = [
            str(self._ffmpeg),
            "-v",
            "error",
            "-nostdin",
            "-ss",
            f"{start_s:.2f}",
            "-i",
            str(path),
            "-map",
            f"0:{stream_index}",
            "-t",
            f"{dur_s:.2f}",
            "-af",
            f"aformat=channel_layouts={layout}:sample_rates={_PROFILE_SAMPLE_RATE}",
            "-f",
            "f32le",
            "-",
        ]
        result = subprocess.run(cmd, capture_output=True, check=False)
        if result.returncode != 0:
            logger.warning(
                "profile_audio_track ffmpeg rc=%d at %.1fs: %s",
                result.returncode,
                start_s,
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
        *,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> AudioMetrics:
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
        for i, frac in enumerate(points, start=1):
            start = max(0.0, duration_s * frac - _PROFILE_WINDOW_SEC / 2)
            window = self._decode_pcm_window(
                path,
                stream_index,
                channels,
                layout,
                start,
                _PROFILE_WINDOW_SEC,
            )
            if window.size > 0:
                chunks.append(window)
            if on_progress is not None:
                on_progress(ProgressSample(fraction=i / len(points)))

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
                rms_l=_rms_db(left),
                rms_r=_rms_db(right),
                rms_c=None,
                rms_lfe=None,
                rms_ls=None,
                rms_rs=None,
                rms_lb=None,
                rms_rb=None,
                corr_lr=_pearson(left, right),
                corr_ls_l=None,
                corr_rs_r=None,
                corr_ls_rs=None,
                corr_lb_ls=None,
                corr_rb_rs=None,
            )

        if channels == _CHANNELS_5_1:
            left, right, center, lfe, ls, rs = cols
            return AudioMetrics(
                channels=_CHANNELS_5_1,
                rms_l=_rms_db(left),
                rms_r=_rms_db(right),
                rms_c=_rms_db(center),
                rms_lfe=_rms_db(lfe),
                rms_ls=_rms_db(ls),
                rms_rs=_rms_db(rs),
                rms_lb=None,
                rms_rb=None,
                corr_lr=_pearson(left, right),
                corr_ls_l=_pearson(ls, left),
                corr_rs_r=_pearson(rs, right),
                corr_ls_rs=_pearson(ls, rs),
                corr_lb_ls=None,
                corr_rb_rs=None,
            )

        left, right, center, lfe, lb, rb, ls, rs = cols
        return AudioMetrics(
            channels=_CHANNELS_7_1,
            rms_l=_rms_db(left),
            rms_r=_rms_db(right),
            rms_c=_rms_db(center),
            rms_lfe=_rms_db(lfe),
            rms_ls=_rms_db(ls),
            rms_rs=_rms_db(rs),
            rms_lb=_rms_db(lb),
            rms_rb=_rms_db(rb),
            corr_lr=_pearson(left, right),
            corr_ls_l=_pearson(ls, left),
            corr_rs_r=_pearson(rs, right),
            corr_ls_rs=_pearson(ls, rs),
            corr_lb_ls=_pearson(lb, ls),
            corr_rb_rs=_pearson(rb, rs),
        )

    def stereo_to_mono_wav(
        self,
        input_path: Path,
        stream_index: int,
        output_wav: Path,
        delay_ms: int,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int:
        filters = ["pan=mono|c0=0.5*FL+0.5*FR"]

        if delay_ms > 0:
            filters.append(f"adelay={delay_ms}")
        elif delay_ms < 0:
            seconds = abs(delay_ms) / 1000.0
            filters.append(f"atrim=start={seconds:.3f}")

        af_value = ",".join(filters)

        cmd = [
            str(self._ffmpeg),
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(input_path),
            "-map",
            f"0:{stream_index}",
            "-af",
            af_value,
            "-ac",
            "1",
            "-c:a",
            "pcm_s24le",
            "-f",
            "wav",
            "-rf64",
            "auto",
            "-progress",
            "pipe:1",
            "-y",
            str(output_wav),
        ]
        log_path = self._log_dir / f"ffmpeg_mono_s{stream_index}.log" if self._log_dir else None

        rc, _out = run_tool(
            cmd,
            on_output=self._on_output,
            on_progress_line=_make_ffmpeg_progress_handler(on_progress),
            log_path=log_path,
        )
        return rc
