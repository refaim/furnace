from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from furnace.core.models import METRIC_NAMES, MetricPool, MetricScores

logger = logging.getLogger(__name__)

_LOW_PERCENTILE = 5.0


def _pool_scores(scores: list[float], pool: MetricPool) -> float:
    arr = np.asarray(scores, dtype=np.float64)
    if pool is MetricPool.LOW:
        return float(np.percentile(arr, _LOW_PERCENTILE))
    return float(arr.mean())


_VS_MATRIX: dict[str, str] = {
    "bt709": "709",
    "bt470bg": "470bg",
    "smpte170m": "170m",
    "smpte240m": "240m",
    "bt2020nc": "2020ncl",
    "bt2020c": "2020cl",
}
_DEFAULT_MATRIX = "170m"

_PROP_SSIMULACRA2 = "_SSIMULACRA2"
_PROP_BUTTERAUGLI = "_BUTTERAUGLI_3Norm"
_PROP_CVVDP = "_CVVDP"

_METRIC_NODES: dict[str, tuple[str, str]] = {
    "ssimulacra2": ("SSIMULACRA2", _PROP_SSIMULACRA2),
    "butteraugli": ("BUTTERAUGLI", _PROP_BUTTERAUGLI),
    "cvvdp": ("CVVDP", _PROP_CVVDP),
}


class VshipMetricsAdapter:
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
        unknown = metrics - METRIC_NAMES
        if unknown:
            raise ValueError(f"unknown perceptual metric(s): {sorted(unknown)}")
        try:
            return self._measure(
                reference,
                distorted,
                matrix=matrix,
                fps_num=fps_num,
                fps_den=fps_den,
                pool=pool,
                metrics=metrics,
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
        if not hasattr(core, "bs"):
            core.std.LoadPlugin(str(self._bestsource))
        if not hasattr(core, "vship"):
            core.std.LoadPlugin(str(self._vship))

        ref = core.bs.VideoSource(str(reference), rff=0)
        dist = core.bs.VideoSource(str(distorted), rff=0)

        n = min(ref.num_frames, dist.num_frames)
        ref = core.std.Trim(ref, first=0, last=n - 1)
        dist = core.std.Trim(dist, first=0, last=n - 1)

        ref = core.std.AssumeFPS(ref, fpsnum=fps_num, fpsden=fps_den)
        dist = core.std.AssumeFPS(dist, fpsnum=fps_num, fpsden=fps_den)

        token = _VS_MATRIX.get(matrix, _DEFAULT_MATRIX)
        ref_rgb = core.resize.Bicubic(ref, format=vs.RGBS, matrix_in_s=token)
        dist_rgb = core.resize.Bicubic(dist, format=vs.RGBS, matrix_in_s=token)

        nodes: dict[str, tuple[Any, str]] = {}
        for name, (ctor, prop) in _METRIC_NODES.items():
            if name in metrics:
                nodes[name] = (getattr(core.vship, ctor)(ref_rgb, dist_rgb), prop)

        frames: dict[str, list[float]] = {name: [] for name in nodes}
        for i in range(n):
            for name, (node, prop) in nodes.items():
                frames[name].append(float(node.get_frame(i).props[prop]))

        return MetricScores(
            ssimulacra2=_pool_scores(frames["ssimulacra2"], pool) if "ssimulacra2" in frames else None,
            butteraugli=_pool_scores(frames["butteraugli"], pool) if "butteraugli" in frames else None,
            cvvdp=_pool_scores(frames["cvvdp"], pool) if "cvvdp" in frames else None,
        )
