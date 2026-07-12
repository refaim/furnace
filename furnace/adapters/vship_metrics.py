"""GPU perceptual metrics (SSIMULACRA2 / Butteraugli / CVVDP) via VapourSynth + Vship.

The SVT-AV1 grain path cannot compute these with ffmpeg (no such avfilter exists)
and the standalone FFVship binary cannot apply geometry transforms. So Vship is
driven as a VapourSynth plugin, in process: BestSource opens the reference (the
original source) and the encoded AV1 OBU, the reference is brought to the encoded
geometry with bundled crop/scale nodes, both are converted to RGBS, and
``clip.vship.<METRIC>`` scores them frame-on-demand on the GPU -- no lossless
intermediate is ever written to disk.

Deinterlacing is intentionally absent: the planner refuses interlaced grain jobs
(no bwdif VS plugin is provisioned yet), so this graph only needs crop + scale --
which is exactly the geometry ``svtav1._geometry_filters`` applies for a
progressive/soft-telecine grain source. That invariant (grain job => no
deinterlace) is what keeps this reference geometry equal to the encoder's input.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from furnace.core.models import CropRect, MetricScores

logger = logging.getLogger(__name__)

# furnace (ffmpeg-style) colour matrix -> VapourSynth resize matrix token. This
# mirrors core.color.CICP_MATRIX -- the exact set _color_svtav1_params accepts,
# so any matrix that can reach an encoded OBU maps here (no silent default).
_VS_MATRIX: dict[str, str] = {
    "bt709": "709",
    "bt470bg": "470bg",
    "smpte170m": "170m",
    "smpte240m": "240m",
    "bt2020nc": "2020ncl",
    "bt2020c": "2020cl",
}
_DEFAULT_MATRIX = "170m"  # SD fallback; unreachable for a validly-encoded source

# Per-frame vship frame-property names (verified against libvship 5.0.2).
_PROP_SSIMULACRA2 = "_SSIMULACRA2"
_PROP_BUTTERAUGLI = "_BUTTERAUGLI_3Norm"  # community-standard 3-norm aggregate
_PROP_CVVDP = "_CVVDP"


class VshipMetricsAdapter:
    """Compute GPU perceptual metrics for the SVT-AV1 grain path.

    ``bestsource_dll`` and ``vship_dll`` are the VapourSynth plugin binaries;
    they are loaded per call so a missing/incompatible plugin degrades to
    all-None scores instead of crashing the encode.
    """

    def __init__(self, bestsource_dll: Path, vship_dll: Path) -> None:
        self._bestsource = bestsource_dll
        self._vship = vship_dll

    def measure(
        self,
        reference: Path,
        distorted: Path,
        *,
        crop: CropRect | None,
        final_width: int,
        final_height: int,
        matrix: str,
        fps_num: int,
        fps_den: int,
    ) -> MetricScores:
        """Score ``distorted`` against ``reference`` brought to the encoded geometry.

        Fail-soft: any VapourSynth / GPU / plugin error returns an all-None
        ``MetricScores`` so a metrics failure never fails the encode.
        """
        try:
            return self._measure(
                reference, distorted, crop, final_width, final_height, matrix, fps_num, fps_den,
            )
        except Exception:  # noqa: BLE001 -- metrics are best-effort; never fail the encode
            logger.warning(
                "Vship perceptual metrics unavailable; continuing without scores",
                exc_info=True,
            )
            return MetricScores()

    def _measure(
        self,
        reference: Path,
        distorted: Path,
        crop: CropRect | None,
        final_width: int,
        final_height: int,
        matrix: str,
        fps_num: int,
        fps_den: int,
    ) -> MetricScores:
        import vapoursynth as vs  # noqa: PLC0415 -- optional heavy dependency, imported lazily

        core: Any = vs.core
        core.std.LoadPlugin(str(self._bestsource))
        core.std.LoadPlugin(str(self._vship))

        # rff=0: yield coded frames (never apply 2:3 pulldown), so a soft-telecine
        # source lines up 1:1 with the coded-rate OBU without decimation.
        ref = core.bs.VideoSource(str(reference), rff=0)
        dist = core.bs.VideoSource(str(distorted), rff=0)

        # Bring the reference to the encoded geometry: crop, then a single rescale
        # to the final encoded size (mirrors svtav1._geometry_filters ordering).
        # Note: the encode scales with ffmpeg spline vs VS Spline36 here -- both
        # are high-quality splines but not bit-identical, a small accepted bias.
        if crop is not None:
            ref = core.std.Crop(
                ref,
                left=crop.x,
                top=crop.y,
                right=ref.width - crop.x - crop.w,
                bottom=ref.height - crop.y - crop.h,
            )
        if (ref.width, ref.height) != (final_width, final_height):
            ref = core.resize.Spline36(ref, width=final_width, height=final_height)

        # Frame-exact index pairing: trim both to the shorter length so vship
        # compares frame i against frame i.
        n = min(ref.num_frames, dist.num_frames)
        ref = core.std.Trim(ref, first=0, last=n - 1)
        dist = core.std.Trim(dist, first=0, last=n - 1)

        # Stamp both at the coded rate: CVVDP is temporal (its masking model
        # depends on frame rate) and the raw OBU is rateless.
        ref = core.std.AssumeFPS(ref, fpsnum=fps_num, fpsden=fps_den)
        dist = core.std.AssumeFPS(dist, fpsnum=fps_num, fpsden=fps_den)

        token = _VS_MATRIX.get(matrix, _DEFAULT_MATRIX)
        ref_rgb = core.resize.Bicubic(ref, format=vs.RGBS, matrix_in_s=token)
        dist_rgb = core.resize.Bicubic(dist, format=vs.RGBS, matrix_in_s=token)

        s2 = core.vship.SSIMULACRA2(ref_rgb, dist_rgb)
        ba = core.vship.BUTTERAUGLI(ref_rgb, dist_rgb)
        cv = core.vship.CVVDP(ref_rgb, dist_rgb)

        # One interleaved pass: requesting all three metrics per frame index lets
        # VapourSynth's frame cache decode each source/OBU frame once (not 3x). A
        # synchronous get_frame loop (vs clip.frames() prefetch) is chosen to keep
        # that single-decode dedup; on SD grain sources the preset-4 encode, not
        # this GPU metric pass, is the throughput bottleneck.
        total_s2 = total_ba = total_cv = 0.0
        for i in range(n):
            total_s2 += float(s2.get_frame(i).props[_PROP_SSIMULACRA2])
            total_ba += float(ba.get_frame(i).props[_PROP_BUTTERAUGLI])
            total_cv += float(cv.get_frame(i).props[_PROP_CVVDP])

        return MetricScores(
            ssimulacra2=total_s2 / n,
            butteraugli=total_ba / n,
            cvvdp=total_cv / n,
        )
