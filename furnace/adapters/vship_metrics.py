"""GPU perceptual metrics (SSIMULACRA2 / Butteraugli / CVVDP) via VapourSynth + Vship.

The SVT-AV1 grain path cannot compute these with ffmpeg (no such avfilter exists)
and the standalone FFVship binary cannot apply geometry transforms. So Vship is
driven as a VapourSynth plugin, in process: BestSource opens the reference and the
encoded AV1 OBU, both are converted to RGBS, and ``clip.vship.<METRIC>`` scores
them frame-on-demand on the GPU -- no lossless intermediate is written here.

This adapter is a PURE COMPARATOR: it assumes ``reference`` and ``distorted`` are
ALREADY at the same geometry. The grain path guarantees that upstream by building
the reference through the encode's own ffmpeg geometry filtergraph
(:meth:`furnace.adapters.ffmpeg.FFmpegAdapter.build_reference` -> the shared
:func:`furnace.adapters._geometry.build_vf`), so the reference differs from the
OBU only by AV1's lossy compression -- same deinterlace, same crop, same
resampler. Re-doing crop/scale here in VapourSynth (as an earlier version did)
phase-shifted a Spline36 reference against the ffmpeg-spline encode whenever a
crop forced a 2-D anamorphic scale, collapsing SSIMULACRA2 and railing the CRF
search; comparing two already-matched clips removes that failure mode entirely.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from furnace.core.models import METRIC_NAMES, MetricPool, MetricScores

logger = logging.getLogger(__name__)

# Worst-case percentile for MetricPool.LOW (the CRF search). 5th percentile =
# "95% of frames score at least this well"; less noise-prone than the 1st. The
# exact value is a calibration knob (target-quality Phase 4).
_LOW_PERCENTILE = 5.0


def _pool_scores(scores: list[float], pool: MetricPool) -> float:
    """Reduce per-frame scores to one value by the chosen pooling."""
    arr = np.asarray(scores, dtype=np.float64)
    if pool is MetricPool.LOW:
        return float(np.percentile(arr, _LOW_PERCENTILE))
    return float(arr.mean())

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

# Metric name (see core ``METRIC_NAMES``) -> (vship node constructor attr,
# per-frame prop). Canonical order; the CRF search asks for only its driver
# metric so the unneeded GPU kernels are skipped (a search reads only one metric).
_METRIC_NODES: dict[str, tuple[str, str]] = {
    "ssimulacra2": ("SSIMULACRA2", _PROP_SSIMULACRA2),
    "butteraugli": ("BUTTERAUGLI", _PROP_BUTTERAUGLI),
    "cvvdp": ("CVVDP", _PROP_CVVDP),
}


class VshipMetricsAdapter:
    """Compute GPU perceptual metrics for the SVT-AV1 grain path.

    ``bestsource_dll`` and ``vship_dll`` are the VapourSynth plugin binaries;
    they are loaded per call so a missing/incompatible plugin degrades to
    all-None scores instead of crashing the encode. No deinterlace/crop/scale is
    done here -- the reference is handed in already at the encoded geometry (see
    the module docstring), so the adapter only opens both clips and scores them.
    """

    def __init__(self, bestsource_dll: Path, vship_dll: Path) -> None:
        self._bestsource = bestsource_dll
        self._vship = vship_dll

    def measure(
        self,
        reference: Path,
        distorted: Path,
        *,
        matrix: str,
        fps_num: int,
        fps_den: int,
        pool: MetricPool = MetricPool.MEAN,
        metrics: frozenset[str] = METRIC_NAMES,
    ) -> MetricScores:
        """Score ``distorted`` against ``reference`` (both already at the same geometry).

        ``pool`` selects mean (readout) or low-percentile (worst-case, CRF search)
        frame pooling. ``metrics`` selects which perceptual metrics to compute
        (the others stay None) -- the CRF search asks for only its driver metric so
        the unneeded GPU kernels are skipped. Fail-soft: any VapourSynth / GPU /
        plugin error returns an all-None ``MetricScores`` so a metrics failure
        never fails the encode. One check is deliberately *loud* and sits outside
        the fail-soft guard: an unknown metric name is a caller bug and raises
        rather than degrading silently.
        """
        unknown = metrics - METRIC_NAMES
        if unknown:
            raise ValueError(f"unknown perceptual metric(s): {sorted(unknown)}")
        try:
            return self._measure(
                reference, distorted,
                matrix=matrix, fps_num=fps_num, fps_den=fps_den,
                pool=pool, metrics=metrics,
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
        *,
        matrix: str,
        fps_num: int,
        fps_den: int,
        pool: MetricPool,
        metrics: frozenset[str],
    ) -> MetricScores:
        import vapoursynth as vs  # noqa: PLC0415 -- optional heavy dependency, imported lazily

        core: Any = vs.core
        # VapourSynth's core is a process-global singleton and a plugin can be
        # loaded only once per process -- a second LoadPlugin of the same plugin
        # raises "already loaded". The grain CRF search calls measure() many times
        # in one run (per window x per probed knob), so load each plugin only when
        # its namespace isn't already present (the idiomatic hasattr guard).
        if not hasattr(core, "bs"):
            core.std.LoadPlugin(str(self._bestsource))
        if not hasattr(core, "vship"):
            core.std.LoadPlugin(str(self._vship))

        # rff=0: yield coded frames (never apply 2:3 pulldown), so a soft-telecine
        # source lines up 1:1 with the coded-rate OBU without decimation. Both the
        # reference (built with ffmpeg -r at the coded rate) and the OBU are read
        # at that rate; no crop/scale/deinterlace is applied here -- the reference
        # already carries the encoded geometry.
        ref = core.bs.VideoSource(str(reference), rff=0)
        dist = core.bs.VideoSource(str(distorted), rff=0)

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

        # Build a vship node only for each requested metric (canonical order), so a
        # single-metric search skips the other GPU kernels entirely.
        nodes: dict[str, tuple[Any, str]] = {}
        for name, (ctor, prop) in _METRIC_NODES.items():
            if name in metrics:
                nodes[name] = (getattr(core.vship, ctor)(ref_rgb, dist_rgb), prop)

        # One interleaved pass: requesting the wanted metrics per frame index lets
        # VapourSynth's frame cache decode each source/OBU frame once (not once per
        # metric). A synchronous get_frame loop (vs clip.frames() prefetch) is
        # chosen to keep that single-decode dedup; on SD grain sources the preset-4
        # encode, not this GPU metric pass, is the throughput bottleneck.
        frames: dict[str, list[float]] = {name: [] for name in nodes}
        for i in range(n):
            for name, (node, prop) in nodes.items():
                frames[name].append(float(node.get_frame(i).props[prop]))

        return MetricScores(
            ssimulacra2=_pool_scores(frames["ssimulacra2"], pool) if "ssimulacra2" in frames else None,
            butteraugli=_pool_scores(frames["butteraugli"], pool) if "butteraugli" in frames else None,
            cvvdp=_pool_scores(frames["cvvdp"], pool) if "cvvdp" in frames else None,
        )
