"""SVT-AV1 (libsvtav1 via bundled ffmpeg) video encoder adapter.

Implements the Encoder protocol using ffmpeg + libsvtav1 for grainy SD sources
that NVENC's psychovisual profile smooths over. Outputs a raw AV1 OBU elementary
stream (``-f obu``) -- mkvmerge handles muxing downstream, matching the NVEncC
adapter's contract so the executor can swap encoders transparently.

The tuned recipe (preset/CRF/svtav1-params) is load-bearing: it targets
grain-preserving, near-transparent quality with mainline SVT-AV1 knobs only
(no fork-specific psy-rd/spy-rd/noise-norm params). Do not edit the constants
without re-measuring.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

from furnace.core.color import CICP_MATRIX, CICP_PRIMARIES, CICP_TRANSFER
from furnace.core.models import EncodeResult, VideoParams
from furnace.core.progress import ProgressSample
from furnace.core.quality import final_output_dimensions

from ._subprocess import OutputCallback, run_tool
from .ffmpeg import _make_ffmpeg_progress_handler

logger = logging.getLogger(__name__)

# --- Load-bearing SVT-AV1 recipe (mainline libsvtav1 knobs only) ------------
# preset 4: slow enough for high fidelity, still tractable for SD grain jobs.
# crf 23: constant-quality anchor for grain preservation.
# svtav1-params: variance boost + quant-matrices + luma/AC biases tuned to keep
# fine film grain alive. Deliberately excludes fork-only params (psy-rd, spy-rd,
# noise-norm-strength) so the bundled mainline ffmpeg accepts the string.
_SVT_PRESET = "4"
_SVT_CRF = "23"
_SVT_PARAMS = (
    "tune=0:enable-variance-boost=1:variance-boost-strength=3:"
    "enable-qm=1:qm-min=0:luminance-qp-bias=50:ac-bias=6.0"
)

# AV1 video_full_range_flag: 0 = studio/limited swing, 1 = full swing.
_SVT_COLOR_RANGE: dict[str, int] = {"tv": 0, "pc": 1}


def _color_svtav1_params(vp: VideoParams) -> str:
    """CICP color-description appended to ``-svtav1-params``.

    libsvtav1 does not propagate ffmpeg's ``-color_primaries`` / ``-color_trc``
    into the AV1 sequence header (only matrix + range survive the ffmpeg flags),
    so the full description is pinned here. Values are CICP code points; the
    ``color-range`` key is the AV1 ``video_full_range_flag`` (0 studio / 1 full),
    NOT the Matroska range enum.

    ``resolve_color_metadata`` only emits mapped values for the SD grain sources
    this encoder handles (real DVD MPEG-2 signals bt709/bt470bg/smpte170m), but
    ``transfer``/``primaries`` pass through source tags unvalidated, so a
    mistagged input carrying a CICP value furnace has no code point for raises a
    clear ValueError instead of a cryptic KeyError or a malformed OBU header.
    """
    try:
        return (
            f"color-primaries={CICP_PRIMARIES[vp.color_primaries]}:"
            f"transfer-characteristics={CICP_TRANSFER[vp.color_transfer]}:"
            f"matrix-coefficients={CICP_MATRIX[vp.color_matrix]}:"
            f"color-range={_SVT_COLOR_RANGE[vp.color_range]}"
        )
    except KeyError as exc:
        raise ValueError(
            f"no CICP code point for color value {exc.args[0]!r} "
            f"(primaries={vp.color_primaries!r} transfer={vp.color_transfer!r} "
            f"matrix={vp.color_matrix!r} range={vp.color_range!r})"
        ) from exc


def _geometry_filters(vp: VideoParams) -> list[str]:
    """Shared crop/scale/deinterlace prefix for the encode and the VMAF pass.

    Order matters: deinterlace on fields first, then crop, then a single
    high-quality rescale (only when the final encoded size differs from the
    pre-resize size). This is the *geometry* only -- it deliberately omits the
    fixed 10-bit / square-SAR tail so the VMAF reference chain can reuse it to
    match the encoded frames' geometry without forcing a pixel format. The
    encode path (:func:`_build_vf`) appends the tail; the two never drift.
    """
    parts: list[str] = []

    # Deinterlace first -- must run on interlaced fields before any spatial op.
    # send_frame = SINGLE-RATE (one output frame per input frame), matching
    # NVEncC's nnedi. bwdif's default (send_field) is double-rate: 2 frames per
    # frame, which would desync against the single-rate --default-duration the
    # executor pins at mux time (video would play at half speed).
    if vp.deinterlace:
        parts.append("bwdif=send_frame")

    if vp.crop is not None:
        parts.append(f"crop={vp.crop.w}:{vp.crop.h}:{vp.crop.x}:{vp.crop.y}")

    # Single source of truth for the encoded size (crop -> SAR -> mod-8).
    final_w, final_h = final_output_dimensions(vp)
    pre_w = vp.crop.w if vp.crop is not None else vp.source_width
    pre_h = vp.crop.h if vp.crop is not None else vp.source_height
    if (final_w, final_h) != (pre_w, pre_h):
        parts.append(f"scale={final_w}:{final_h}:flags=spline")

    return parts


def _build_vf(vp: VideoParams) -> str:
    """Build the ffmpeg ``-vf`` filtergraph string (comma-joined).

    The geometry prefix (:func:`_geometry_filters`) followed by the fixed
    10-bit / square-SAR tail that every SVT-AV1 encode needs.
    """
    parts = [*_geometry_filters(vp), "format=yuv420p10le", "setsar=1"]
    return ",".join(parts)


class SvtAv1Adapter:
    """Implements the Encoder protocol via ffmpeg + libsvtav1.

    Outputs a raw AV1 OBU elementary stream (not MKV) -- mkvmerge handles
    muxing, mirroring NVEncCAdapter so the executor treats both encoders alike.
    """

    def __init__(
        self,
        ffmpeg: Path,
        *,
        on_output: OutputCallback = None,
        log_dir: Path | None = None,
    ) -> None:
        self._ffmpeg = ffmpeg
        self._on_output = on_output
        self._log_dir = log_dir

    def set_log_dir(self, log_dir: Path | None) -> None:
        self._log_dir = log_dir

    # ------------------------------------------------------------------
    # Encoder settings string
    # ------------------------------------------------------------------

    def _build_encoder_settings(self, vp: VideoParams) -> str:
        """Build the ENCODER_SETTINGS string for the MKV global tag.

        Slash-separated: the SVT-AV1 recipe is always present, filters only
        when applied (mirrors NVEncCAdapter's convention).
        """
        parts: list[str] = [
            "av1_svt",
            "SVT-AV1",
            f"preset={_SVT_PRESET}",
            f"crf={_SVT_CRF}",
            _SVT_PARAMS,
        ]
        if vp.deinterlace:
            parts.append("bwdif=send_frame")
        if vp.crop is not None:
            parts.append(f"crop={vp.crop.w}:{vp.crop.h}:{vp.crop.x}:{vp.crop.y}")
        return " / ".join(parts)

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------

    def _build_encode_cmd(
        self,
        input_path: Path,
        output_path: Path,
        vp: VideoParams,
    ) -> list[str | Path]:
        """Build the full ffmpeg + libsvtav1 encode command (raw AV1 OBU out).

        The output ``-r vp.fps_num/vp.fps_den`` pins the OBU to the coded film
        rate that mkvmerge later pins the container to. For soft-telecine
        NTSC-DVD sources plain ffmpeg applies the 2:3 pulldown on decode
        (inflating 23.976 -> 29.97); this drops the duplicated frames. For
        native content the input already decodes at that rate, so it is a
        harmless no-op (no dup/drop).
        """
        return [
            self._ffmpeg, "-hide_banner", "-i", input_path,
            "-vf", _build_vf(vp),
            "-c:v", "libsvtav1", "-preset", _SVT_PRESET, "-crf", _SVT_CRF,
            "-g", str(vp.gop),
            "-svtav1-params", f"{_SVT_PARAMS}:{_color_svtav1_params(vp)}",
            "-color_range", vp.color_range, "-color_primaries", vp.color_primaries,
            "-color_trc", vp.color_transfer, "-colorspace", vp.color_matrix,
            "-r", f"{vp.fps_num}/{vp.fps_den}",
            "-progress", "pipe:1", "-f", "obu", "-y", output_path,
        ]

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
        """Encode via ffmpeg + libsvtav1, parsing ffmpeg ``-progress`` for progress.

        When ``vmaf_enabled`` and the encode succeeds, a second ffmpeg
        ``libvmaf`` pass compares the encoded OBU against the source and the
        pooled mean VMAF is returned; any failure of that pass is fail-soft
        (``vmaf_score`` stays None -- a metrics failure never fails the encode).
        ``ssim_score`` is always None: the SVT path surfaces only VMAF.
        ``rpu_path`` is accepted for Encoder-protocol parity but ignored --
        SVT-AV1 grain jobs are SDR (no Dolby Vision path).
        """
        _ = rpu_path
        cmd = self._build_encode_cmd(input_path, output_path, video_params)
        str_cmd = [str(c) for c in cmd]
        logger.debug("svtav1 cmd: %s", " ".join(str_cmd))

        log_path = self._log_dir / "svt_encode.log" if self._log_dir else None
        rc, _out = run_tool(
            cmd,
            on_output=self._on_output,
            on_progress_line=_make_ffmpeg_progress_handler(on_progress),
            log_path=log_path,
        )

        encoder_settings = self._build_encoder_settings(video_params)
        vmaf_score: float | None = None
        if vmaf_enabled and rc == 0:
            vmaf_score = self._run_vmaf(input_path, output_path, video_params)

        return EncodeResult(
            return_code=rc,
            encoder_settings=encoder_settings,
            vmaf_score=vmaf_score,
            ssim_score=None,
        )

    # ------------------------------------------------------------------
    # VMAF metrics pass
    # ------------------------------------------------------------------

    def _run_vmaf(
        self,
        source: Path,
        encoded_obu: Path,
        vp: VideoParams,
    ) -> float | None:
        """Measure pooled mean VMAF of ``encoded_obu`` against ``source``.

        Frame-exact alignment in three steps, so distorted frame N is always
        compared against the source frame N that produced it:

        1. The reference is brought to the *encoded* geometry -- the
           crop/scale/deinterlace from :func:`_geometry_filters` when the source
           needs any (a plain source needs none), but never the 10-bit / setsar
           tail, so no pixel format is forced.
        2. ``fps=<coded rate>`` decimates the reference to the OBU's frame
           *count*. This matters for soft-telecine sources: the OBU is at the
           coded rate (encode ``-r``) but the source decodes with 2:3 pulldown
           applied, so without decimation the two carry different frame counts.
        3. ``setpts=N`` resets BOTH streams' timestamps to their frame index, so
           libvmaf pairs index N against index N. Pairing by *timestamp* instead
           silently drifts apart on PAL DVD sources whose demuxed reference PTS
           carry a start offset / jitter versus the OBU's clean 0-based grid --
           that drift collapsed the score to ~35 over a feature-length run while
           the encode itself was pristine. Index pairing is identical to
           timestamp pairing on clean soft-telecine (the decimated dupes are
           bit-identical), so it fixes the PAL case without regressing telecine.

        libvmaf writes a JSON log which is parsed for
        ``pooled_metrics.vmaf.mean``.

        Fail-soft: a non-zero ffmpeg rc, or a missing / unparseable / key-less
        JSON, returns None. The log filename (not the full path) is passed to
        libvmaf and ffmpeg runs with ``cwd`` set to the log's directory -- a
        full Windows path would carry a drive-letter colon that libvmaf's
        ``:``-delimited option parser would mangle.
        """
        json_path = encoded_obu.with_name(f"{encoded_obu.stem}.vmaf.json")
        n_threads = max(1, (os.cpu_count() or 4) - 2)
        # The raw OBU is rateless (ffmpeg would assume 25fps), so it is read with
        # ``-r fps`` to stamp it at the coded rate. Reference chain: geometry ->
        # ``fps=`` decimation to that same rate -> ``setpts=N`` re-index. The
        # distorted OBU is likewise re-indexed with ``setpts=N`` so libvmaf pairs
        # frame-by-frame (index N vs index N), never by timestamp.
        fps = f"{vp.fps_num}/{vp.fps_den}"
        ref_chain = ",".join([*_geometry_filters(vp), f"fps={fps}", "setpts=N"])
        lavfi = (
            f"[0:v]setpts=N[d];[1:v]{ref_chain}[r];"
            f"[d][r]libvmaf=log_path={json_path.name}:log_fmt=json:"
            f"n_threads={n_threads}"
        )
        # Absolute inputs: run_tool runs this pass with cwd=json_path.parent
        # (the log_path colon workaround), so a relative source/OBU path -- as
        # the executor passes -- would not resolve from that cwd. resolve()
        # against the current furnace cwd before the subprocess switches.
        cmd: list[str | Path] = [
            self._ffmpeg, "-hide_banner",
            "-r", fps, "-i", encoded_obu.resolve(), "-i", source.resolve(),
            "-lavfi", lavfi,
            "-f", "null", "-",
        ]
        log_path = self._log_dir / "svt_vmaf.log" if self._log_dir else None
        rc, _out = run_tool(
            cmd,
            on_output=self._on_output,
            log_path=log_path,
            cwd=json_path.parent,
        )
        if rc != 0:
            logger.warning("svtav1 VMAF pass failed (rc=%d); no score recorded", rc)
            return None
        try:
            with json_path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            return float(data["pooled_metrics"]["vmaf"]["mean"])
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            logger.warning("svtav1 VMAF metrics unavailable; continuing without a score")
            return None
