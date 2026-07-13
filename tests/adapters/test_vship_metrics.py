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

import pytest

from furnace.adapters.vship_metrics import VshipMetricsAdapter
from furnace.core.models import CropRect, MetricPool, MetricScores


class _Frame:
    def __init__(self, props: dict[str, float]) -> None:
        self.props = props


class _Clip:
    """A fake VapourSynth clip node with per-frame metric props (cyclic values).

    ``field_based`` (when set) is surfaced on every frame as the ``_FieldBased``
    property, mirroring what BestSource stamps on an interlaced source so the
    adapter can pick the bwdif single-rate parity.
    """

    def __init__(
        self,
        *,
        width: int = 1920,
        height: int = 1080,
        num_frames: int = 4,
        prop: str | None = None,
        values: tuple[float, ...] = (),
        field_based: int | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self._prop = prop
        self._values = values
        self._field_based = field_based

    def get_frame(self, i: int) -> _Frame:
        props: dict[str, float] = {}
        if self._prop is not None:
            props[self._prop] = self._values[i % len(self._values)]
        if self._field_based is not None:
            props["_FieldBased"] = self._field_based
        return _Frame(props)


class _FakeCore:
    """Records the graph the adapter builds and returns configured clips."""

    def __init__(
        self,
        *,
        ref: _Clip,
        dist: _Clip,
        cropped: _Clip | None = None,
        scaled: _Clip | None = None,
        deinterlaced: _Clip | None = None,
        raise_in: str | None = None,
    ) -> None:
        self._sources = [ref, dist]
        self._source_calls = 0
        self._cropped = cropped
        self._scaled = scaled
        self._deinterlaced = deinterlaced
        self._raise_in = raise_in
        self.loaded: list[str] = []
        self.crop_kwargs: dict[str, int] | None = None
        self.spline_kwargs: dict[str, int] | None = None
        self.trim_lasts: list[int] = []
        self.assumefps_kwargs: list[dict[str, int]] = []
        self.bicubic_matrix: list[str] = []
        self.bwdif_kwargs: dict[str, int] | None = None
        self.bwdif_applied_to: _Clip | None = None
        self.crop_applied_to: _Clip | None = None
        self.vship_built: list[str] = []
        # Built-in namespaces are always present; plugin namespaces (bs / vship /
        # bwdif) appear only after LoadPlugin, mirroring real VapourSynth so the
        # adapter's load-once hasattr guard is exercised faithfully.
        self.std = types.SimpleNamespace(
            LoadPlugin=self._load, Crop=self._crop, Trim=self._trim, AssumeFPS=self._assumefps,
        )
        self.resize = types.SimpleNamespace(Spline36=self._spline, Bicubic=self._bicubic)

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
        elif "bwdif" in low:
            self.bwdif = types.SimpleNamespace(Bwdif=self._bwdif)

    def _source(self, path: str, **kw: int) -> _Clip:  # noqa: ARG002 -- fake mirrors real signature
        if self._raise_in == "source":
            raise RuntimeError("boom")
        clip = self._sources[self._source_calls % len(self._sources)]
        self._source_calls += 1
        return clip

    def _bwdif(self, clip: _Clip, **kw: int) -> _Clip:
        self.bwdif_kwargs = kw
        self.bwdif_applied_to = clip
        assert self._deinterlaced is not None
        return self._deinterlaced

    def _crop(self, clip: _Clip, **kw: int) -> _Clip:
        self.crop_kwargs = kw
        self.crop_applied_to = clip
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
    deinterlace: bool = False,
    bwdif: Path | None = None,
    pool: MetricPool = MetricPool.MEAN,
    metrics: frozenset[str] | None = None,
) -> MetricScores:
    adapter = VshipMetricsAdapter(Path("BestSource.dll"), Path("libvship.dll"), bwdif)
    fake_vs = types.SimpleNamespace(core=core, RGBS="RGBS")
    # metrics=None exercises the adapter's own default (compute all three).
    extra = {} if metrics is None else {"metrics": metrics}
    with patch.dict(sys.modules, {"vapoursynth": fake_vs}):
        return adapter.measure(
            Path("ref.mkv"), Path("dist.obu"),
            crop=crop, deinterlace=deinterlace,
            final_width=final_width, final_height=final_height,
            matrix=matrix, fps_num=24000, fps_den=1001,
            pool=pool,
            **extra,
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

    def test_default_pool_is_mean(self) -> None:
        """MetricPool.MEAN (the default) averages per-frame scores."""
        core = _FakeCore(ref=_Clip(width=1904, height=792), dist=_Clip(width=1904, height=792))
        scores = _measure(core, pool=MetricPool.MEAN)
        assert scores.ssimulacra2 == 85.0  # mean(80, 90)
        assert scores.cvvdp == 8.5  # mean(9.0, 8.0)

    def test_low_pool_takes_worst_case_percentile(self) -> None:
        """MetricPool.LOW takes the 5th-percentile (worst-case) per metric.
        Frames cycle (80,90) -> [80,90,80,90]; p5 -> 80.0."""
        core = _FakeCore(ref=_Clip(width=1904, height=792), dist=_Clip(width=1904, height=792))
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
        core = _FakeCore(ref=_Clip(width=1904, height=792), dist=_Clip(width=1904, height=792))
        adapter = VshipMetricsAdapter(Path("BestSource.dll"), Path("libvship.dll"))
        fake_vs = types.SimpleNamespace(core=core, RGBS="RGBS")
        kw: dict[str, Any] = {
            "crop": None, "deinterlace": False, "final_width": 1904, "final_height": 792,
            "matrix": "bt709", "fps_num": 24000, "fps_den": 1001,
        }
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
        core = _FakeCore(ref=_Clip(width=1904, height=792), dist=_Clip(width=1904, height=792))
        scores = _measure(core)
        assert core.vship_built == ["ssimulacra2", "butteraugli", "cvvdp"]
        assert scores.ssimulacra2 is not None
        assert scores.butteraugli is not None
        assert scores.cvvdp is not None

    def test_single_metric_skips_the_others(self) -> None:
        """Requesting just SSIMULACRA2 builds only that node; the rest stay None."""
        core = _FakeCore(ref=_Clip(width=1904, height=792), dist=_Clip(width=1904, height=792))
        scores = _measure(core, metrics=frozenset({"ssimulacra2"}))
        assert core.vship_built == ["ssimulacra2"]
        assert scores.ssimulacra2 == 85.0  # mean(80, 90)
        assert scores.butteraugli is None
        assert scores.cvvdp is None

    def test_unknown_metric_raises_loudly(self) -> None:
        """An unknown metric name is a caller bug -> loud, outside the fail-soft
        guard (never silently degraded to all-None)."""
        core = _FakeCore(ref=_Clip(width=1904, height=792), dist=_Clip(width=1904, height=792))
        with pytest.raises(ValueError, match="unknown perceptual metric"):
            _measure(core, metrics=frozenset({"bogus"}))


class TestDeinterlace:
    """When ``deinterlace`` is set, the reference is bwdif-deinterlaced (single
    rate) *before* crop, with the parity taken from BestSource's ``_FieldBased``
    -- matching the encoder's ffmpeg ``bwdif=send_frame`` (parity=auto)."""

    @staticmethod
    def _core(field_based: int | None) -> _FakeCore:
        # Reference already at the final size -> only bwdif runs (no crop/scale).
        return _FakeCore(
            ref=_Clip(width=1904, height=792, num_frames=4, field_based=field_based),
            dist=_Clip(width=1904, height=792, num_frames=4),
            deinterlaced=_Clip(width=1904, height=792, num_frames=4),
        )

    def test_loads_bwdif_and_scores(self) -> None:
        core = self._core(field_based=2)
        scores = _measure(core, deinterlace=True, bwdif=Path("Bwdif.dll"))
        assert core.loaded == ["BestSource.dll", "libvship.dll", "Bwdif.dll"]
        assert scores.ssimulacra2 == 85.0  # mean(80, 90)

    def test_tff_keeps_top_field(self) -> None:
        core = self._core(field_based=2)  # _FieldBased 2 = top field first
        _measure(core, deinterlace=True, bwdif=Path("Bwdif.dll"))
        assert core.bwdif_kwargs == {"field": 1}

    def test_bff_keeps_bottom_field(self) -> None:
        core = self._core(field_based=1)  # _FieldBased 1 = bottom field first
        _measure(core, deinterlace=True, bwdif=Path("Bwdif.dll"))
        assert core.bwdif_kwargs == {"field": 0}

    def test_unknown_field_order_defaults_to_top(self) -> None:
        core = self._core(field_based=None)  # no _FieldBased property at all
        _measure(core, deinterlace=True, bwdif=Path("Bwdif.dll"))
        assert core.bwdif_kwargs == {"field": 1}

    def test_progressive_flag_defaults_to_top(self) -> None:
        core = self._core(field_based=0)  # _FieldBased 0 = progressive
        _measure(core, deinterlace=True, bwdif=Path("Bwdif.dll"))
        assert core.bwdif_kwargs == {"field": 1}

    def test_bwdif_runs_before_crop(self) -> None:
        ref = _Clip(width=720, height=480, num_frames=4, field_based=2)
        deint = _Clip(width=720, height=480, num_frames=4)
        core = _FakeCore(
            ref=ref, dist=_Clip(width=640, height=480, num_frames=4),
            deinterlaced=deint,
            cropped=_Clip(width=704, height=480, num_frames=4),
            scaled=_Clip(width=640, height=480, num_frames=4),
        )
        _measure(
            core, crop=CropRect(w=704, h=480, x=8, y=0),
            final_width=640, final_height=480,
            deinterlace=True, bwdif=Path("Bwdif.dll"),
        )
        # bwdif consumes the raw source; crop consumes bwdif's output.
        assert core.bwdif_applied_to is ref
        assert core.crop_applied_to is deint

    def test_no_deinterlace_skips_bwdif(self) -> None:
        core = _FakeCore(ref=_Clip(width=1904, height=792), dist=_Clip(width=1904, height=792))
        _measure(core, deinterlace=False, bwdif=Path("Bwdif.dll"))
        assert "Bwdif.dll" not in core.loaded
        assert core.bwdif_kwargs is None

    def test_deinterlace_without_bwdif_raises_loudly(self) -> None:
        # A metrics failure is normally fail-soft (all-None scores); a missing
        # bwdif for an interlaced source is a real config gap -> loud, not silent.
        core = self._core(field_based=2)
        with pytest.raises(RuntimeError, match="bwdif"):
            _measure(core, deinterlace=True, bwdif=None)
