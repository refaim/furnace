"""Tests for VshipMetricsAdapter: the in-process VapourSynth + Vship comparator.

The adapter is exercised against a hand-written fake VapourSynth core injected
via ``sys.modules`` (no GPU / real plugins needed). It is a PURE COMPARATOR --
``reference`` and ``distorted`` arrive already at the same geometry (the grain
path builds the reference through the encode's own ffmpeg filtergraph), so the
adapter does NO crop/scale/deinterlace: these tests assert the graph it builds
(plugins loaded, matrix token, frame trimming, coded-rate stamping, per-frame
pooling, metric selection) and the fail-soft contract (any error -> all-None).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from furnace.adapters.vship_metrics import VshipMetricsAdapter
from furnace.core.models import MetricPool, MetricScores


class _Frame:
    def __init__(self, props: dict[str, float]) -> None:
        self.props = props


class _Clip:
    """A fake VapourSynth clip node with per-frame metric props (cyclic values)."""

    def __init__(
        self,
        *,
        num_frames: int = 4,
        prop: str | None = None,
        values: tuple[float, ...] = (),
    ) -> None:
        self.num_frames = num_frames
        self._prop = prop
        self._values = values

    def get_frame(self, i: int) -> _Frame:
        props: dict[str, float] = {}
        if self._prop is not None:
            props[self._prop] = self._values[i % len(self._values)]
        return _Frame(props)


class _FakeCore:
    """Records the graph the adapter builds and returns configured clips."""

    def __init__(
        self,
        *,
        ref: _Clip,
        dist: _Clip,
        raise_in: str | None = None,
    ) -> None:
        self._sources = [ref, dist]
        self._source_calls = 0
        self._raise_in = raise_in
        self.loaded: list[str] = []
        self.trim_lasts: list[int] = []
        self.assumefps_kwargs: list[dict[str, int]] = []
        self.bicubic_matrix: list[str] = []
        self.vship_built: list[str] = []
        # Built-in namespaces are always present; plugin namespaces (bs / vship)
        # appear only after LoadPlugin, mirroring real VapourSynth so the adapter's
        # load-once hasattr guard is exercised faithfully.
        self.std = types.SimpleNamespace(
            LoadPlugin=self._load, Trim=self._trim, AssumeFPS=self._assumefps,
        )
        self.resize = types.SimpleNamespace(Bicubic=self._bicubic)

    def _vship_metric(self, name: str, prop: str, values: tuple[float, ...]) -> Any:
        """A vship metric constructor that records it was built (so tests can
        assert which metrics the adapter chose to compute)."""

        def _fn(ref: Any, dist: Any) -> _Clip:
            self.vship_built.append(name)
            return _Clip(prop=prop, values=values)

        return _fn

    def _load(self, path: str) -> None:
        # Real VapourSynth loads a plugin at most once per process; a second load
        # of the same plugin raises. Model that, and register the plugin's
        # namespace so the adapter's hasattr(core, ...) guard sees it.
        if path in self.loaded:
            raise RuntimeError(f"Plugin {path} already loaded")
        self.loaded.append(path)
        low = path.lower()
        if "bestsource" in low:
            self.bs = types.SimpleNamespace(VideoSource=self._source)
        elif "vship" in low:
            self.vship = types.SimpleNamespace(
                SSIMULACRA2=self._vship_metric("ssimulacra2", "_SSIMULACRA2", (80.0, 90.0)),
                BUTTERAUGLI=self._vship_metric("butteraugli", "_BUTTERAUGLI_3Norm", (1.5, 2.5)),
                CVVDP=self._vship_metric("cvvdp", "_CVVDP", (9.0, 8.0)),
            )

    def _source(self, path: str, **kw: int) -> _Clip:  # noqa: ARG002 -- fake mirrors real signature
        if self._raise_in == "source":
            raise RuntimeError("boom")
        clip = self._sources[self._source_calls % len(self._sources)]
        self._source_calls += 1
        return clip

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
    matrix: str = "bt709",
    pool: MetricPool = MetricPool.MEAN,
    metrics: frozenset[str] | None = None,
) -> MetricScores:
    adapter = VshipMetricsAdapter(Path("BestSource.dll"), Path("libvship.dll"))
    fake_vs = types.SimpleNamespace(core=core, RGBS="RGBS")
    # metrics=None exercises the adapter's own default (compute all three).
    extra = {} if metrics is None else {"metrics": metrics}
    with patch.dict(sys.modules, {"vapoursynth": fake_vs}):
        return adapter.measure(
            Path("ref.mkv"), Path("dist.obu"),
            matrix=matrix, fps_num=24000, fps_den=1001,
            pool=pool,
            **extra,
        )


class TestMeasure:
    def test_scores_both_clips(self) -> None:
        core = _FakeCore(ref=_Clip(num_frames=4), dist=_Clip(num_frames=4))
        scores = _measure(core)
        assert scores.ssimulacra2 == 85.0  # mean(80, 90)
        assert scores.butteraugli == 2.0  # mean(1.5, 2.5)
        assert scores.cvvdp == 8.5  # mean(9.0, 8.0)

    def test_loads_both_plugins(self) -> None:
        core = _FakeCore(ref=_Clip(), dist=_Clip())
        _measure(core)
        assert core.loaded == ["BestSource.dll", "libvship.dll"]

    def test_assumefps_stamps_coded_rate(self) -> None:
        core = _FakeCore(ref=_Clip(), dist=_Clip())
        _measure(core)
        # Both reference and distorted are stamped at the coded rate for CVVDP.
        assert core.assumefps_kwargs == [
            {"fpsnum": 24000, "fpsden": 1001},
            {"fpsnum": 24000, "fpsden": 1001},
        ]

    def test_matrix_token_known(self) -> None:
        core = _FakeCore(ref=_Clip(), dist=_Clip())
        _measure(core, matrix="bt709")
        assert core.bicubic_matrix == ["709", "709"]

    def test_matrix_token_unknown_defaults_to_sd(self) -> None:
        core = _FakeCore(ref=_Clip(), dist=_Clip())
        _measure(core, matrix="something-exotic")
        assert core.bicubic_matrix == ["170m", "170m"]

    def test_trims_to_shorter_length(self) -> None:
        core = _FakeCore(ref=_Clip(num_frames=5), dist=_Clip(num_frames=3))
        _measure(core)
        # min(5, 3) = 3 -> Trim last = 2 on both clips.
        assert core.trim_lasts == [2, 2]

    def test_fail_soft_returns_all_none(self) -> None:
        core = _FakeCore(ref=_Clip(), dist=_Clip(), raise_in="source")
        scores = _measure(core)
        assert scores == MetricScores()
        assert scores.ssimulacra2 is None
        assert scores.butteraugli is None
        assert scores.cvvdp is None

    def test_default_pool_is_mean(self) -> None:
        """MetricPool.MEAN (the default) averages per-frame scores."""
        core = _FakeCore(ref=_Clip(), dist=_Clip())
        scores = _measure(core, pool=MetricPool.MEAN)
        assert scores.ssimulacra2 == 85.0  # mean(80, 90)
        assert scores.cvvdp == 8.5  # mean(9.0, 8.0)

    def test_low_pool_takes_worst_case_percentile(self) -> None:
        """MetricPool.LOW takes the 5th-percentile (worst-case) per metric.
        Frames cycle (80,90) -> [80,90,80,90]; p5 -> 80.0."""
        core = _FakeCore(ref=_Clip(), dist=_Clip())
        scores = _measure(core, pool=MetricPool.LOW)
        assert scores.ssimulacra2 == 80.0  # p5 of [80,90,80,90]
        assert scores.butteraugli == 1.5  # p5 of [1.5,2.5,1.5,2.5]
        assert scores.cvvdp == 8.0  # p5 of [9,8,9,8]

    def test_loads_each_plugin_only_once_across_calls(self) -> None:
        """VapourSynth's core is a process-global singleton -- a plugin loads only
        once. A second measure() on the same core must reuse the loaded plugins,
        not reload them (a reload raises "already loaded"). The grain CRF search
        calls measure() many times per run, so a reload would abort every search
        after the first probe."""
        core = _FakeCore(ref=_Clip(), dist=_Clip())
        adapter = VshipMetricsAdapter(Path("BestSource.dll"), Path("libvship.dll"))
        fake_vs = types.SimpleNamespace(core=core, RGBS="RGBS")
        kw: dict[str, Any] = {"matrix": "bt709", "fps_num": 24000, "fps_den": 1001}
        with patch.dict(sys.modules, {"vapoursynth": fake_vs}):
            first = adapter.measure(Path("ref.mkv"), Path("d1.obu"), **kw)
            second = adapter.measure(Path("ref.mkv"), Path("d2.obu"), **kw)
        assert first.ssimulacra2 is not None
        assert second.ssimulacra2 is not None  # None if it reloaded -> fail-soft
        assert core.loaded == ["BestSource.dll", "libvship.dll"]  # loaded once total


class TestMetricSelection:
    """The CRF search reads only one metric, so ``measure`` computes only the
    requested ones -- skipping the other GPU metric kernels (perf)."""

    def test_default_computes_all_three(self) -> None:
        """With no selection, every metric node is built (backward compatible)."""
        core = _FakeCore(ref=_Clip(), dist=_Clip())
        scores = _measure(core)
        assert core.vship_built == ["ssimulacra2", "butteraugli", "cvvdp"]
        assert scores.ssimulacra2 is not None
        assert scores.butteraugli is not None
        assert scores.cvvdp is not None

    def test_single_metric_skips_the_others(self) -> None:
        """Requesting just SSIMULACRA2 builds only that node; the rest stay None."""
        core = _FakeCore(ref=_Clip(), dist=_Clip())
        scores = _measure(core, metrics=frozenset({"ssimulacra2"}))
        assert core.vship_built == ["ssimulacra2"]
        assert scores.ssimulacra2 == 85.0  # mean(80, 90)
        assert scores.butteraugli is None
        assert scores.cvvdp is None

    def test_unknown_metric_raises_loudly(self) -> None:
        """An unknown metric name is a caller bug -> loud, outside the fail-soft
        guard (never silently degraded to all-None)."""
        core = _FakeCore(ref=_Clip(), dist=_Clip())
        with pytest.raises(ValueError, match="unknown perceptual metric"):
            _measure(core, metrics=frozenset({"bogus"}))
