"""Tests for VshipMetricsAdapter: the in-process VapourSynth + Vship graph.

The adapter is exercised against a hand-written fake VapourSynth core injected
via ``sys.modules`` (no GPU / real plugins needed). The tests assert the graph
it builds -- plugins loaded, crop/scale geometry, matrix token, frame trimming,
per-frame pooling -- and the fail-soft contract (any error -> all-None scores).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

from furnace.adapters.vship_metrics import VshipMetricsAdapter
from furnace.core.models import CropRect, MetricScores


class _Frame:
    def __init__(self, props: dict[str, float]) -> None:
        self.props = props


class _Clip:
    """A fake VapourSynth clip node with per-frame metric props (cyclic values)."""

    def __init__(
        self,
        *,
        width: int = 1920,
        height: int = 1080,
        num_frames: int = 4,
        prop: str | None = None,
        values: tuple[float, ...] = (),
    ) -> None:
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self._prop = prop
        self._values = values

    def get_frame(self, i: int) -> _Frame:
        if self._prop is None:
            return _Frame({})
        return _Frame({self._prop: self._values[i % len(self._values)]})


def _metric(prop: str, values: tuple[float, ...]) -> Any:
    def _fn(ref: Any, dist: Any) -> _Clip:
        return _Clip(prop=prop, values=values)

    return _fn


class _FakeCore:
    """Records the graph the adapter builds and returns configured clips."""

    def __init__(
        self,
        *,
        ref: _Clip,
        dist: _Clip,
        cropped: _Clip | None = None,
        scaled: _Clip | None = None,
        raise_in: str | None = None,
    ) -> None:
        self._sources = [ref, dist]
        self._cropped = cropped
        self._scaled = scaled
        self._raise_in = raise_in
        self.loaded: list[str] = []
        self.crop_kwargs: dict[str, int] | None = None
        self.spline_kwargs: dict[str, int] | None = None
        self.trim_lasts: list[int] = []
        self.assumefps_kwargs: list[dict[str, int]] = []
        self.bicubic_matrix: list[str] = []
        self.std = types.SimpleNamespace(
            LoadPlugin=self._load, Crop=self._crop, Trim=self._trim, AssumeFPS=self._assumefps,
        )
        self.bs = types.SimpleNamespace(VideoSource=self._source)
        self.resize = types.SimpleNamespace(Spline36=self._spline, Bicubic=self._bicubic)
        self.vship = types.SimpleNamespace(
            SSIMULACRA2=_metric("_SSIMULACRA2", (80.0, 90.0)),
            BUTTERAUGLI=_metric("_BUTTERAUGLI_3Norm", (1.5, 2.5)),
            CVVDP=_metric("_CVVDP", (9.0, 8.0)),
        )

    def _load(self, path: str) -> None:
        self.loaded.append(path)

    def _source(self, path: str, **kw: int) -> _Clip:  # noqa: ARG002 -- fake mirrors real signature
        if self._raise_in == "source":
            raise RuntimeError("boom")
        return self._sources.pop(0)

    def _crop(self, clip: _Clip, **kw: int) -> _Clip:  # noqa: ARG002
        self.crop_kwargs = kw
        assert self._cropped is not None
        return self._cropped

    def _spline(self, clip: _Clip, **kw: int) -> _Clip:  # noqa: ARG002
        self.spline_kwargs = kw
        assert self._scaled is not None
        return self._scaled

    def _trim(self, clip: _Clip, *, first: int, last: int) -> _Clip:  # noqa: ARG002
        self.trim_lasts.append(last)
        return clip

    def _assumefps(self, clip: _Clip, **kw: int) -> _Clip:
        self.assumefps_kwargs.append(kw)
        return clip

    def _bicubic(self, clip: _Clip, **kw: Any) -> _Clip:
        self.bicubic_matrix.append(kw["matrix_in_s"])
        return clip


def _measure(
    core: _FakeCore,
    *,
    crop: CropRect | None = None,
    final_width: int = 1904,
    final_height: int = 792,
    matrix: str = "bt709",
) -> MetricScores:
    adapter = VshipMetricsAdapter(Path("BestSource.dll"), Path("libvship.dll"))
    fake_vs = types.SimpleNamespace(core=core, RGBS="RGBS")
    with patch.dict(sys.modules, {"vapoursynth": fake_vs}):
        return adapter.measure(
            Path("ref.mkv"), Path("dist.obu"),
            crop=crop, final_width=final_width, final_height=final_height,
            matrix=matrix, fps_num=24000, fps_den=1001,
        )


class TestMeasure:
    def test_full_pipeline_crop_and_scale(self) -> None:
        core = _FakeCore(
            ref=_Clip(width=1920, height=1080, num_frames=4),
            dist=_Clip(width=1904, height=792, num_frames=4),
            cropped=_Clip(width=1910, height=798, num_frames=4),
            scaled=_Clip(width=1904, height=792, num_frames=4),
        )
        scores = _measure(core, crop=CropRect(w=1910, h=798, x=5, y=141))
        assert scores.ssimulacra2 == 85.0  # mean(80, 90)
        assert scores.butteraugli == 2.0  # mean(1.5, 2.5)
        assert scores.cvvdp == 8.5  # mean(9.0, 8.0)

    def test_loads_both_plugins(self) -> None:
        core = _FakeCore(ref=_Clip(width=1904, height=792), dist=_Clip(width=1904, height=792))
        _measure(core)
        assert core.loaded == ["BestSource.dll", "libvship.dll"]

    def test_crop_geometry_converted(self) -> None:
        core = _FakeCore(
            ref=_Clip(width=1920, height=1080, num_frames=4),
            dist=_Clip(width=1904, height=792, num_frames=4),
            cropped=_Clip(width=1910, height=798, num_frames=4),
            scaled=_Clip(width=1904, height=792, num_frames=4),
        )
        _measure(core, crop=CropRect(w=1910, h=798, x=5, y=141))
        # CropRect(w,h,x,y) -> VS Crop(left,top,right,bottom) pixels removed.
        assert core.crop_kwargs == {"left": 5, "top": 141, "right": 5, "bottom": 141}

    def test_no_crop_and_no_scale(self) -> None:
        """crop None + reference already at final size -> neither node is built."""
        core = _FakeCore(
            ref=_Clip(width=1904, height=792, num_frames=4),
            dist=_Clip(width=1904, height=792, num_frames=4),
        )
        scores = _measure(core, crop=None)
        assert core.crop_kwargs is None
        assert core.spline_kwargs is None
        assert scores.ssimulacra2 == 85.0

    def test_assumefps_stamps_coded_rate(self) -> None:
        core = _FakeCore(ref=_Clip(width=1904, height=792), dist=_Clip(width=1904, height=792))
        _measure(core)
        # Both reference and distorted are stamped at the coded rate for CVVDP.
        assert core.assumefps_kwargs == [
            {"fpsnum": 24000, "fpsden": 1001},
            {"fpsnum": 24000, "fpsden": 1001},
        ]

    def test_matrix_token_known(self) -> None:
        core = _FakeCore(ref=_Clip(width=1904, height=792), dist=_Clip(width=1904, height=792))
        _measure(core, matrix="bt709")
        assert core.bicubic_matrix == ["709", "709"]

    def test_matrix_token_unknown_defaults_to_sd(self) -> None:
        core = _FakeCore(ref=_Clip(width=1904, height=792), dist=_Clip(width=1904, height=792))
        _measure(core, matrix="something-exotic")
        assert core.bicubic_matrix == ["170m", "170m"]

    def test_trims_to_shorter_length(self) -> None:
        core = _FakeCore(
            ref=_Clip(width=1904, height=792, num_frames=5),
            dist=_Clip(width=1904, height=792, num_frames=3),
        )
        _measure(core)
        # min(5, 3) = 3 -> Trim last = 2 on both clips.
        assert core.trim_lasts == [2, 2]

    def test_fail_soft_returns_all_none(self) -> None:
        core = _FakeCore(
            ref=_Clip(width=1904, height=792), dist=_Clip(width=1904, height=792),
            raise_in="source",
        )
        scores = _measure(core)
        assert scores == MetricScores()
        assert scores.ssimulacra2 is None
        assert scores.butteraugli is None
        assert scores.cvvdp is None
