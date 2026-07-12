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

import logging
from collections.abc import Callable
from pathlib import Path

from furnace.core.color import CICP_MATRIX, CICP_PRIMARIES, CICP_TRANSFER
from furnace.core.models import EncodeResult, MetricScores, VideoParams
from furnace.core.ports import PerceptualMetrics
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
    """Shared crop/scale/deinterlace prefix for the SVT-AV1 encode filtergraph.

    Order matters: deinterlace on fields first, then crop, then a single
    high-quality rescale (only when the final encoded size differs from the
    pre-resize size). This is the *geometry* only -- it deliberately omits the
    fixed 10-bit / square-SAR tail, which :func:`_build_vf` appends. The
    perceptual-metrics reference reproduces this same crop+scale geometry in
    VapourSynth (see :mod:`furnace.adapters.vship_metrics`); the two are kept
    consistent by the planner refusing interlaced grain jobs, so metrics never
    face a deinterlaced reference.
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
        metrics: PerceptualMetrics | None = None,
    ) -> None:
        self._ffmpeg = ffmpeg
        self._on_output = on_output
        self._log_dir = log_dir
        self._metrics = metrics

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

        When ``vmaf_enabled`` and the encode succeeds, GPU perceptual metrics
        (SSIMULACRA2 / Butteraugli / CVVDP) are measured against the source via
        the injected VapourSynth + Vship adapter; any failure is fail-soft (the
        scores stay None -- a metrics failure never fails the encode).
        ``vmaf_score`` is always None here: the grain path deliberately drops
        VMAF (grain-blind) for the perceptual metrics. ``rpu_path`` is accepted
        for Encoder-protocol parity but ignored -- SVT-AV1 grain jobs are SDR.
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
        scores = MetricScores()
        if vmaf_enabled and rc == 0 and self._metrics is not None:
            final_w, final_h = final_output_dimensions(video_params)
            scores = self._metrics.measure(
                input_path,
                output_path,
                crop=video_params.crop,
                final_width=final_w,
                final_height=final_h,
                matrix=video_params.color_matrix,
                fps_num=video_params.fps_num,
                fps_den=video_params.fps_den,
            )

        return EncodeResult(
            return_code=rc,
            encoder_settings=encoder_settings,
            ssimulacra2_score=scores.ssimulacra2,
            butteraugli_score=scores.butteraugli,
            cvvdp_score=scores.cvvdp,
        )
