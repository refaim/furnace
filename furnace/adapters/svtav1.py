from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from furnace.core.color import CICP_MATRIX, CICP_PRIMARIES, CICP_TRANSFER
from furnace.core.models import EncodeResult, VideoParams
from furnace.core.progress import ProgressSample
from furnace.core.quality import aligned_crop

from ._geometry import build_vf
from ._subprocess import OutputCallback, run_tool
from .ffmpeg import _make_ffmpeg_progress_handler

logger = logging.getLogger(__name__)

_SVT_PRESET = "4"
_SVT_CRF = "23"
_SVT_PARAMS = (
    "tune=0:enable-variance-boost=1:variance-boost-strength=3:enable-qm=1:qm-min=0:luminance-qp-bias=50:ac-bias=6.0"
)

_SVT_COLOR_RANGE: dict[str, int] = {"tv": 0, "pc": 1}


def _color_svtav1_params(vp: VideoParams) -> str:
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


class SvtAv1Adapter:
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

    def _build_encoder_settings(self, vp: VideoParams, *, cq_override: int | None = None) -> str:
        effective_crf = str(cq_override) if cq_override is not None else _SVT_CRF
        parts: list[str] = [
            "av1_svt",
            "SVT-AV1",
            f"preset={_SVT_PRESET}",
            f"crf={effective_crf}",
            _SVT_PARAMS,
        ]
        if vp.deinterlace:
            parts.append("bwdif=send_frame")
        crop = aligned_crop(vp)
        if crop is not None:
            parts.append(f"crop={crop.w}:{crop.h}:{crop.x}:{crop.y}")
        return " / ".join(parts)

    def _build_encode_cmd(
        self,
        input_path: Path,
        output_path: Path,
        vp: VideoParams,
        *,
        cq_override: int | None = None,
    ) -> list[str | Path]:
        effective_crf = str(cq_override) if cq_override is not None else _SVT_CRF
        return [
            self._ffmpeg,
            "-hide_banner",
            "-i",
            input_path,
            "-map",
            "0:v:0",
            "-vf",
            build_vf(vp),
            "-c:v",
            "libsvtav1",
            "-preset",
            _SVT_PRESET,
            "-crf",
            effective_crf,
            "-g",
            str(vp.gop),
            "-svtav1-params",
            f"{_SVT_PARAMS}:{_color_svtav1_params(vp)}",
            "-color_range",
            vp.color_range,
            "-color_primaries",
            vp.color_primaries,
            "-color_trc",
            vp.color_transfer,
            "-colorspace",
            vp.color_matrix,
            "-r",
            f"{vp.fps_num}/{vp.fps_den}",
            "-progress",
            "pipe:1",
            "-f",
            "obu",
            "-y",
            output_path,
        ]

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
        _ = rpu_path
        cmd = self._build_encode_cmd(input_path, output_path, video_params, cq_override=cq_override)
        str_cmd = [str(c) for c in cmd]
        logger.debug("svtav1 cmd: %s", " ".join(str_cmd))

        log_path = self._log_dir / "svt_encode.log" if self._log_dir else None
        rc, _out = run_tool(
            cmd,
            on_output=self._on_output,
            on_progress_line=_make_ffmpeg_progress_handler(on_progress),
            log_path=log_path,
        )

        encoder_settings = self._build_encoder_settings(video_params, cq_override=cq_override)
        return EncodeResult(return_code=rc, encoder_settings=encoder_settings)
