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
# 12 evenly-spaced interior points (i/13). Two windows carried enough sampling
# variance to land on quiet, centered-dialogue scenes and false-flag real
# stereo as mono; 12 windows stabilise the aggregate L/R correlation.
_PROFILE_STEREO_POINTS: tuple[float, ...] = tuple(i / 13 for i in range(1, 13))
_PROFILE_MULTI_POINTS: tuple[float, ...] = (0.15, 0.35, 0.55, 0.75)
_PROFILE_SAMPLE_RATE = 48000
_DIGITAL_SILENCE_DB = -120.0
_ZERO_NORM_EPS = 1e-9
_CHANNELS_STEREO = 2
_CHANNELS_5_1 = 6
_CHANNELS_7_1 = 8

# --- Field-pairing probe (interlaced sources reporting a field rate) -------
# One window from the start of the stream, long enough that a boundary packet
# cannot move the packets-per-frame ratio off 2.0 by more than the tolerance
# ``core.detect.detect_field_separated`` allows (~1500 frames at 25 fps).
_FIELD_PAIRING_SECONDS = 60

# --- Film-grain probe (SDR sources, any resolution) ------------------------
# Five short windows across the timeline; each pipes a handful of luma-only
# frames and measures how much the calmest 16x16 blocks flicker frame to frame.
# Real film grain keeps those blocks jittering; a denoised transfer holds them
# still. The pure ``core.detect.classify_grain`` turns the per-window values
# into a GRAINY verdict.
_GRAIN_WINDOWS: tuple[float, ...] = (0.10, 0.30, 0.50, 0.70, 0.90)
_GRAIN_FRAMES = 24
_GRAIN_W, _GRAIN_H = 480, 270
_GRAIN_BLOCK = 16
_GRAIN_LUMA_MIN, _GRAIN_LUMA_MAX = 30.0, 220.0
_GRAIN_MIN_VALID_BLOCKS = 20
_GRAIN_STATIC_QUANTILE = 0.15
# A window that decodes fewer than this many frames is too short for a
# trustworthy temporal median — skip it.
_GRAIN_MIN_FRAMES = 8
# At or below this per-block temporal median the block is digital black (or a
# frozen duplicate) rather than grain; excluded so dead areas never read static.
_GRAIN_MIN_BLOCK_FLICKER = 0.01


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


def _grain_window_value(frames: np.ndarray) -> float | None:
    """Static-block temporal flicker for one window of gray frames ``(n, H, W)``.

    Splits each frame into 16x16 blocks and takes, per block, its mean luma
    (over every frame) and its temporal median absolute frame-to-frame
    difference. Blocks that are letterbox/black (luma <= 30), blown highlights
    (luma >= 220) or frozen/digital-black (median flicker <= 0.01) are dropped.
    With fewer than 20 valid blocks the window is untrustworthy and ``None`` is
    returned; otherwise the value is the 0.15 quantile of the valid block
    medians — the flicker of the calmest blocks, i.e. the grain floor.

    The per-block *median* over time (not the mean) gives scene-cut immunity: a
    single hard cut in the window spikes one temporal step, which the median of
    the remaining steps ignores.
    """
    n = frames.shape[0]
    h_blocks = _GRAIN_H // _GRAIN_BLOCK
    w_blocks = _GRAIN_W // _GRAIN_BLOCK
    used_h = h_blocks * _GRAIN_BLOCK
    used_w = w_blocks * _GRAIN_BLOCK
    cropped = frames[:, :used_h, :used_w]

    # Mean luma per block (over every frame and every pixel in the block).
    luma = cropped.reshape(n, h_blocks, _GRAIN_BLOCK, w_blocks, _GRAIN_BLOCK).mean(axis=(0, 2, 4))

    # Temporal abs-diff per step, averaged within each block -> (n-1, hb, wb),
    # then per-block median over the temporal axis.
    step_diff = np.abs(np.diff(cropped, axis=0))
    block_flicker = step_diff.reshape(n - 1, h_blocks, _GRAIN_BLOCK, w_blocks, _GRAIN_BLOCK).mean(axis=(2, 4))
    block_median = np.median(block_flicker, axis=0)

    valid = (
        (luma > _GRAIN_LUMA_MIN)
        & (luma < _GRAIN_LUMA_MAX)
        & (block_median > _GRAIN_MIN_BLOCK_FLICKER)
    )
    valid_medians = block_median[valid]
    if valid_medians.size < _GRAIN_MIN_VALID_BLOCKS:
        return None
    return float(np.quantile(valid_medians, _GRAIN_STATIC_QUANTILE))


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


def _make_ffmpeg_progress_handler(
    on_progress: Callable[[ProgressSample], None] | None,
) -> Callable[[str], bool]:
    """Build an ``on_progress_line`` hook for ffmpeg ``-progress pipe:1`` output.

    Every line is a ``key=value`` pair; the hook accumulates them until a
    ``progress=`` line closes the block, parses it via
    :func:`_parse_ffmpeg_progress_block`, and forwards the sample to
    ``on_progress``. Returns ``True`` for consumed ``key=value`` lines (kept out
    of the log and the TUI) and ``False`` for anything else.
    """
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
    """Implements Prober + AudioExtractor + VideoCopier."""

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

    # Adaptive cropdetect sampling: take a batch of points across the
    # timeline, aggregate, then keep adding batches until the crop stops
    # changing (converged -> confident) or the cap is hit. Dark episodes need
    # more samples to catch enough well-lit frames; clean ones converge after
    # the minimum two batches.
    _CROP_BATCH_HD = 10
    _CROP_BATCH_DVD = 15  # DVD is cheap to decode and noisier -> denser batches
    _CROP_MAX_BATCHES = 4

    # cropdetect luma threshold (8-bit). Higher than ffmpeg's default 24 because
    # old DVD/analog transfers carry a dim *grey* letterbox (~40, not pure black)
    # that 24 mistakes for picture and leaves uncropped. 40 catches it while the
    # consensus aggregation (aggregate_crop) absorbs the extra per-window
    # over-crop a higher threshold causes on genuinely dark scenes. Validated to
    # leave pure-black-bar content (e.g. pillarboxed cartoons) unchanged.
    _CROP_DETECT_LIMIT = 40

    # cropdetect rounds the detected w/h DOWN to a multiple of this, shaving
    # real picture to fit the grid. Must be 2 (the minimum), NOT the
    # macroblock-era 16: 4:2:0 only requires *even* dimensions, and 1080 is not
    # a multiple of 16, so round=16 crops every bar-free 1080p source to 1072
    # (1080 % 16 = 8 -> 4px off each edge) even with zero black bars. AV1
    # superblocks are 64/128 and padded internally regardless, so 16-alignment
    # buys no compression -- it only discards picture. aggregate_crop snaps the
    # final offsets to even so the whole crop stays yuv420-valid.
    _CROP_DETECT_ROUND = 2

    @staticmethod
    def _crop_sample_batches(per_batch: int, max_batches: int) -> list[list[float]]:
        """Timeline fractions split into ``max_batches`` interleaved batches.

        Every batch spans the whole timeline (stride ``max_batches``), so even
        the first batch is well distributed and later batches fill the gaps --
        a prefix of batches is never clustered in one part of the runtime.
        """
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
        """Run cropdetect across the timeline, sampling adaptively.

        Samples a batch of points, aggregates them (see ``aggregate_crop``),
        and keeps adding batches until the aggregate stops changing between
        batches (converged) or the batch cap is reached. Returns the converged
        crop, or None if no sample produced a crop at all. Whether the result
        counts as "no black bars" (crop equals the full frame) is decided
        downstream by the planner.

        Propagates ``ValueError`` from ``aggregate_crop`` if the samples are too
        inconsistent to form a crop (the planner catches it and treats it as no
        crop). This needs non-physical cropdetect output and never happens in
        practice.

        ``hdr_transfer`` is the source's color transfer ('smpte2084' or
        'arib-std-b67') when the input needs HDR tonemapping before
        cropdetect (PQ/HLG -> linear -> bt709, then ``format=yuv420p`` so
        the 8-bit ``limit`` keeps its intended meaning -- cropdetect does
        NOT auto-scale ``limit`` to bit depth). DV Profile 5 (single-layer
        dvhe.05) is also tagged as smpte2084 in container metadata; zscale
        mis-handles its IPT-PQ-C2 colors but luma magnitude near zero is
        identical to YCbCr black, so cropdetect still returns the right
        geometry.

        ``on_progress`` is called after each sample point with a fraction
        (``points_done / cap``), and once more with ``1.0`` when detection
        finishes (so an early-converged run still completes the bar).
        """
        per_batch = self._CROP_BATCH_DVD if is_dvd else self._CROP_BATCH_HD
        batches = self._crop_sample_batches(per_batch, self._CROP_MAX_BATCHES)
        total_points = per_batch * self._CROP_MAX_BATCHES

        parts: list[str] = []
        if interlaced:
            parts.append("yadif")
        if hdr_transfer is not None:
            # PQ/HLG -> linear (npl=100 normalises to SDR peak; clips highlights
            # but leaves shadows untouched, which is all cropdetect cares about).
            # tin/min/pin set explicitly on BOTH stages: zscale auto-detects from
            # frame metadata otherwise, and on -ss seeks the parser can land
            # before VUI propagates from the keyframe -- without explicit tin
            # zscale falls back to bt709 and npl is silently ignored.
            # DV Profile 5 (single-layer dvhe.05) is also tagged smpte2084 in
            # container metadata; zscale mis-handles its IPT-PQ-C2 colors but
            # luma magnitude near zero is identical to YCbCr black, so cropdetect
            # still returns the right geometry by accident.
            parts.append(
                f"zscale=tin={hdr_transfer}:min=2020_ncl:pin=2020:t=linear:npl=100",
            )
            parts.append(
                "zscale=tin=linear:min=2020_ncl:pin=2020:"
                "t=bt709:m=bt709:p=bt709:r=tv",
            )
            # format=yuv420p is load-bearing: cropdetect's `limit` is
            # bit-depth-naive -- see libavfilter/vf_cropdetect.c -- so 10-bit
            # input would compare against code 40/1023 (~10 in 8-bit), well
            # below limited-range black. Force 8-bit so the threshold keeps
            # its intended meaning.
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
                    cmd, capture_output=True, text=True,
                    encoding="utf-8", errors="replace", check=False,
                )
                last_crop: str | None = None
                for line in run.stderr.splitlines():
                    m = re.search(r"crop=(\d+:\d+:\d+:\d+)", line)
                    if m:
                        last_crop = m.group(1)
                if last_crop is not None:
                    parts_crop = last_crop.split(":")
                    # Regex `crop=(\d+:\d+:\d+:\d+)` guarantees 4 parts.
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
            # Cap reached without convergence: keep the last (best) estimate.
            result = prev

        if on_progress is not None:
            on_progress(ProgressSample(fraction=1.0))
        return result

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

    def run_idet(
        self,
        path: Path,
        duration_s: float,
        *,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> float:
        """Run idet filter at multiple points across the timeline.

        Samples 1000 frames at 10%, 30%, 50%, 70%, 90% of duration.
        Returns the ratio of interlaced frames (0.0 to 1.0). After each
        sample point, calls ``on_progress`` with a fraction.
        """
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

            if on_progress is not None:
                on_progress(ProgressSample(fraction=i / len(points)))

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

    def sample_repeat_pict(self, path: Path, duration_s: float) -> list[int]:
        """Sample repeat_pict flags: 500 frames at 10/30/50/70/90% of duration.

        Uses: ffprobe -v quiet -print_format json -select_streams v:0
              -show_frames -show_entries frame=repeat_pict
              -read_intervals "SEEK%+#500" path

        Fail-soft per window (ffprobe error or unparseable JSON skips that
        window): a partial sample still lets the caller detect pulldown, and
        an empty one degrades to "no telecine detected".
        """
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
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace", check=False,
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
        """Count decoded frames vs demuxed packets over one window from the start.

        Uses: ffprobe -v quiet -print_format json -select_streams v:0
              -count_frames -count_packets
              -show_entries stream=nb_read_frames,nb_read_packets
              -read_intervals "%+<seconds>" path

        The window starts at zero deliberately: on a seek, ffprobe reads the
        packets between the seek point and the first keyframe but decodes no
        frame from them, inflating the ratio the caller measures. From the
        start every packet pairs with the frame it belongs to, so the ratio is
        exact. One window suffices -- how a muxer stores fields is a property
        of the whole track, not of a moment in the timeline.

        Fail-soft: an ffprobe error, unparseable JSON, a missing stream, or a
        counter ffprobe could not fill (it reports "N/A") all return ``(0, 0)``,
        which ``core.detect.detect_field_separated`` reads as an untrustworthy
        sample and answers by keeping the container's reported rate.
        """
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
        """Measure film-grain amplitude via static-block temporal flicker.

        Pipes five short luma-only windows (``scale=480:270,format=gray``
        rawvideo, 24 frames at 10/30/50/70/90% of ``duration_s``) out of ffmpeg
        and, per window, measures how much the calmest 16x16 blocks flicker
        between frames (see :func:`_grain_window_value`) — real grain keeps them
        jittering, a denoised transfer holds them still. Returns one flicker
        value per window; ``core.detect.classify_grain`` reduces the list to a
        boolean verdict.

        Fail-soft per window: an ffmpeg error, a truncated read (< 8 frames), or
        a window with too few valid static blocks (all letterbox/black/blown)
        contributes nothing, so the list may be shorter than five — empty when
        every window failed.
        """
        frame_px = _GRAIN_H * _GRAIN_W
        values: list[float] = []

        for pct in _GRAIN_WINDOWS:
            seek = duration_s * pct
            cmd = [
                str(self._ffmpeg),
                "-v", "error",
                "-ss", f"{seek:.2f}",
                "-i", str(path),
                "-frames:v", str(_GRAIN_FRAMES),
                "-vf", f"scale={_GRAIN_W}:{_GRAIN_H},format=gray",
                "-f", "rawvideo",
                "-pix_fmt", "gray",
                "-",
            ]
            logger.debug("sample_grain cmd: %s", cmd)
            result = subprocess.run(cmd, capture_output=True, check=False)
            if result.returncode != 0:
                logger.warning(
                    "sample_grain window at %.2fs failed (rc=%d)",
                    seek, result.returncode,
                )
                continue
            buf = np.frombuffer(result.stdout, dtype=np.uint8)
            n = buf.size // frame_px
            if n < _GRAIN_MIN_FRAMES:
                logger.warning(
                    "sample_grain window at %.2fs decoded only %d frame(s) (< %d)",
                    seek, n, _GRAIN_MIN_FRAMES,
                )
                continue
            frames = buf[: n * frame_px].astype(np.float32).reshape(n, _GRAIN_H, _GRAIN_W)
            window_value = _grain_window_value(frames)
            if window_value is None:
                logger.debug("sample_grain window at %.2fs had too few static blocks", seek)
                continue
            values.append(window_value)

        return values

    # ------------------------------------------------------------------
    # VideoCopier
    # ------------------------------------------------------------------

    def copy_video(
        self,
        input_path: Path,
        output_path: Path,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int:
        """ffmpeg -map 0:v:0 -c:v copy -progress pipe:1 output (passthrough).

        loglevel=fatal: byte-for-byte copy emits the same cosmetic dts spam
        as ``extract_track``. ``-progress pipe:1`` is parsed independently of
        the loglevel by :func:`_make_ffmpeg_progress_handler`.
        """
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

    # ------------------------------------------------------------------
    # WindowExtractor (target-quality probing)
    # ------------------------------------------------------------------

    def extract_window(
        self,
        input_path: Path,
        output_path: Path,
        *,
        start_s: float,
        frames: int,
    ) -> int:
        """Stream-copy ``frames`` video frames from ``start_s`` into an MKV.

        A fast, lossless window extractor for target-quality probing: ``-ss``
        before ``-i`` seeks to the nearest keyframe at/before ``start_s`` and
        ``-frames:v`` copies that many packets verbatim (``-c:v copy``, no
        re-encode). MKV output is the most robust container for arbitrary source
        codecs (per ab-av1's sampling notes). Audio and subtitles are dropped;
        the probe encoder only needs the video. Returns the ffmpeg exit code.
        """
        cmd = [
            str(self._ffmpeg),
            "-hide_banner",
            "-loglevel", "error",
            "-ss", f"{start_s:.3f}",
            "-i", str(input_path),
            "-map", "0:v:0",
            "-frames:v", str(frames),
            "-c:v", "copy",
            "-an", "-sn",
            "-y", str(output_path),
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
        """Materialise a LOSSLESS reference at the encoded geometry for the grain
        perceptual metric.

        Runs the SAME geometry filtergraph the SVT-AV1 encode uses
        (:func:`furnace.adapters._geometry.build_vf`: deinterlace -> crop ->
        scale -> 10-bit -> square SAR) but writes FFV1 (mathematically lossless)
        instead of libsvtav1, and pins the coded frame rate exactly as the encode
        does. The result therefore differs from the encoded AV1 OBU ONLY by AV1's
        lossy compression -- same resampler, same crop offsets, same deinterlace
        parity -- so the metric compares like-for-like and a crop can't
        phase-shift the reference against the encode (which would otherwise
        collapse SSIMULACRA2 and rail the CRF search). The colour description is
        stamped (matching the encode) so the reference and the OBU carry the same
        range/primaries/transfer: the metric passes the matrix to the RGBS
        conversion explicitly, but the range is read from the clip's frame props,
        so the two must agree here. Returns the ffmpeg exit code.
        """
        cmd = [
            str(self._ffmpeg),
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(input_path),
            "-map", "0:v:0",
            "-vf", build_vf(video_params),
            "-c:v", "ffv1",
            "-color_range", video_params.color_range,
            "-color_primaries", video_params.color_primaries,
            "-color_trc", video_params.color_transfer,
            "-colorspace", video_params.color_matrix,
            "-r", f"{video_params.fps_num}/{video_params.fps_den}",
            "-an", "-sn",
            "-y", str(output_path),
        ]
        log_path = self._log_dir / "ffmpeg_build_reference.log" if self._log_dir else None
        rc, _out = run_tool(cmd, on_output=self._on_output, log_path=log_path)
        return rc

    def window_bitrates(self, source: Path, window_s: float) -> list[tuple[float, float]]:
        """Per-window source bitrate proxy for grain hard-scene selection.

        Reads the video packet sizes once (``ffprobe ... -show_entries
        packet=pts_time,size``, no decode) and bins them into consecutive
        non-overlapping ``window_s`` windows. Returns ``(window_start_s, kbytes)``
        per populated window in time order. Empty if the packets can't be read (a
        broken source -> the caller falls back to even sampling).
        """
        cmd = [
            str(self._ffprobe),
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "packet=pts_time,size",
            "-of", "csv=p=0",
            str(source),
        ]
        logger.debug("window_bitrates cmd: %s", cmd)
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
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
        # which are parsed by _make_ffmpeg_progress_handler; it's independent
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
        *,
        on_progress: Callable[[ProgressSample], None] | None = None,
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
        for i, frac in enumerate(points, start=1):
            start = max(0.0, duration_s * frac - _PROFILE_WINDOW_SEC / 2)
            window = self._decode_pcm_window(
                path, stream_index, channels, layout, start, _PROFILE_WINDOW_SEC,
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

    def stereo_to_mono_wav(
        self,
        input_path: Path,
        stream_index: int,
        output_wav: Path,
        delay_ms: int,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int:
        """Average a stereo PCM stream to mono WAV via the ffmpeg ``pan`` filter.

        Averaging two channels at 0.5 each cannot exceed unity for normalised PCM,
        so no limiter is needed. Multichannel collapse is the caller's
        responsibility (typically eac3to ``-downStereo``). Delay handling: if
        ``delay_ms > 0`` an ``adelay`` filter is appended; if ``delay_ms < 0``
        an ``atrim=start=<seconds>`` trims the leading audio; zero adds nothing.

        ``-progress pipe:1`` is enabled and consumed by
        :func:`_make_ffmpeg_progress_handler`;
        ffmpeg's stderr (warnings/errors) still flows to ``self._on_output`` and
        ends up in the TUI log. ``on_progress`` receives a ``ProgressSample``
        per progress block so the per-step bar advances.
        """
        filters = ["pan=mono|c0=0.5*FL+0.5*FR"]

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
            "-progress", "pipe:1",
            "-y", str(output_wav),
        ]
        log_path = self._log_dir / f"ffmpeg_mono_s{stream_index}.log" if self._log_dir else None

        rc, _out = run_tool(
            cmd,
            on_output=self._on_output,
            on_progress_line=_make_ffmpeg_progress_handler(on_progress),
            log_path=log_path,
        )
        return rc
