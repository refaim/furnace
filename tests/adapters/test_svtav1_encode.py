"""Tests for SvtAv1Adapter.encode orchestration: log wiring + metric delegation.

``encode`` runs a single ffmpeg + libsvtav1 pass, then -- when ``vmaf_enabled``
and an injected metrics adapter is present -- delegates GPU perceptual scoring
(SSIMULACRA2 / Butteraugli / CVVDP) to that adapter. The metric adapter owns its
own fail-soft behaviour, so these tests only verify the delegation contract:
when it runs, with what geometry, and that its scores land on the result.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from furnace.adapters.svtav1 import SvtAv1Adapter, _build_vf, _geometry_filters
from furnace.core.models import CropRect, MetricScores, VideoParams
from furnace.core.progress import ProgressSample


def _make_vp(
    *,
    crop: CropRect | None = None,
    deinterlace: bool = False,
    source_width: int = 1920,
    source_height: int = 1080,
    sar_num: int = 1,
    sar_den: int = 1,
) -> VideoParams:
    return VideoParams(
        cq=23, crop=crop, deinterlace=deinterlace,
        color_matrix="bt709", color_range="tv",
        color_transfer="bt709", color_primaries="bt709",
        hdr=None, gop=120, fps_num=24000, fps_den=1001,
        source_width=source_width, source_height=source_height,
        source_codec="mpeg2video", source_bitrate=8_000_000,
        sar_num=sar_num, sar_den=sar_den, grain=True,
    )


class _FakeRunTool:
    """Records the single ``run_tool`` encode invocation."""

    def __init__(self, *, encode_rc: int = 0) -> None:
        self.calls: list[dict[str, Any]] = []
        self.encode_rc = encode_rc

    def __call__(
        self,
        cmd: Any,
        on_output: Any = None,
        on_progress_line: Any = None,
        log_path: Any = None,
        cwd: Any = None,
    ) -> tuple[int, str]:
        self.calls.append(
            {
                "cmd": [str(c) for c in cmd],
                "on_output": on_output,
                "on_progress_line": on_progress_line,
                "log_path": log_path,
                "cwd": cwd,
            },
        )
        return self.encode_rc, ""

    @property
    def encode_call(self) -> dict[str, Any]:
        return self.calls[0]


class _FakeMetrics:
    """Stand-in VshipMetricsAdapter: records the measure() call, returns scores."""

    def __init__(self, scores: MetricScores | None = None) -> None:
        self.scores = scores if scores is not None else MetricScores()
        self.calls: list[dict[str, Any]] = []

    def measure(
        self,
        reference: Path,
        distorted: Path,
        *,
        crop: CropRect | None,
        deinterlace: bool,
        final_width: int,
        final_height: int,
        matrix: str,
        fps_num: int,
        fps_den: int,
    ) -> MetricScores:
        self.calls.append(
            {
                "reference": reference,
                "distorted": distorted,
                "crop": crop,
                "deinterlace": deinterlace,
                "final_width": final_width,
                "final_height": final_height,
                "matrix": matrix,
                "fps_num": fps_num,
                "fps_den": fps_den,
            },
        )
        return self.scores


def _run(
    adapter: SvtAv1Adapter,
    fake: _FakeRunTool,
    tmp_path: Path,
    *,
    vmaf_enabled: bool = False,
    rpu_path: Path | None = None,
    on_progress: Any = None,
) -> Any:
    with patch("furnace.adapters.svtav1.run_tool", side_effect=fake):
        return adapter.encode(
            tmp_path / "input.mkv", tmp_path / "output.obu", _make_vp(),
            vmaf_enabled=vmaf_enabled, rpu_path=rpu_path, on_progress=on_progress,
        )


class TestGeometryHelper:
    """`_geometry_filters` is the shared crop+scale+bwdif source; `_build_vf`
    appends the fixed 10-bit / square-SAR tail without changing that prefix."""

    def test_build_vf_unchanged_plain(self) -> None:
        assert _build_vf(_make_vp()) == "format=yuv420p10le,setsar=1"

    def test_build_vf_appends_tail_to_geometry(self) -> None:
        vp = _make_vp(deinterlace=True, crop=CropRect(w=1920, h=800, x=0, y=140))
        assert _build_vf(vp) == ",".join(
            [*_geometry_filters(vp), "format=yuv420p10le", "setsar=1"],
        )

    def test_geometry_excludes_format_and_setsar(self) -> None:
        vp = _make_vp(crop=CropRect(w=1910, h=798, x=5, y=141))
        geom = _geometry_filters(vp)
        assert geom == ["crop=1910:798:5:141", "scale=1904:792:flags=spline"]
        assert "format=yuv420p10le" not in geom
        assert "setsar=1" not in geom

    def test_geometry_empty_for_plain(self) -> None:
        assert _geometry_filters(_make_vp()) == []


class TestEncodeBasics:
    def test_returns_rc_and_settings_no_metrics(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool(encode_rc=0)
        result = _run(adapter, fake, tmp_path)
        assert result.return_code == 0
        assert result.encoder_settings.startswith("av1_svt")
        assert result.vmaf_score is None
        assert result.ssimulacra2_score is None
        assert result.butteraugli_score is None
        assert result.cvvdp_score is None

    def test_returns_rc_from_run_tool(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool(encode_rc=7)
        result = _run(adapter, fake, tmp_path)
        assert result.return_code == 7

    def test_encode_command_is_svtav1(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool()
        _run(adapter, fake, tmp_path)
        assert "libsvtav1" in fake.encode_call["cmd"]

    def test_forwards_on_output(self, tmp_path: Path) -> None:
        lines: list[str] = []
        cb = lines.append
        adapter = SvtAv1Adapter(Path("ffmpeg"), on_output=cb)
        fake = _FakeRunTool()
        _run(adapter, fake, tmp_path)
        assert fake.encode_call["on_output"] is cb

    def test_forwards_progress_through_ffmpeg_handler(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        samples: list[ProgressSample] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                on_progress_line("out_time_us=5000000")
                on_progress_line("progress=continue")
            return 0, ""

        with patch("furnace.adapters.svtav1.run_tool", side_effect=fake_run_tool):
            adapter.encode(
                tmp_path / "in.mkv", tmp_path / "out.obu", _make_vp(),
                on_progress=samples.append,
            )
        assert len(samples) == 1
        assert samples[0].processed_s == 5.0


class TestEncodeLogWiring:
    def test_encode_log_wired_when_log_dir_set(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"), log_dir=tmp_path)
        fake = _FakeRunTool()
        _run(adapter, fake, tmp_path)
        assert fake.encode_call["log_path"] == tmp_path / "svt_encode.log"

    def test_encode_log_none_when_no_log_dir(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool()
        _run(adapter, fake, tmp_path)
        assert fake.encode_call["log_path"] is None


class TestMetrics:
    def test_delegates_and_forwards_scores(self, tmp_path: Path) -> None:
        metrics = _FakeMetrics(MetricScores(ssimulacra2=88.2, butteraugli=1.73, cvvdp=9.1))
        adapter = SvtAv1Adapter(Path("ffmpeg"), metrics=metrics)
        fake = _FakeRunTool()
        result = _run(adapter, fake, tmp_path, vmaf_enabled=True)
        assert len(metrics.calls) == 1
        assert result.ssimulacra2_score == 88.2
        assert result.butteraugli_score == 1.73
        assert result.cvvdp_score == 9.1
        # The grain path deliberately records no VMAF.
        assert result.vmaf_score is None

    def test_measure_receives_encoded_geometry(self, tmp_path: Path) -> None:
        metrics = _FakeMetrics()
        adapter = SvtAv1Adapter(Path("ffmpeg"), metrics=metrics)
        fake = _FakeRunTool()
        vp = _make_vp(crop=CropRect(w=1910, h=798, x=5, y=141))
        with patch("furnace.adapters.svtav1.run_tool", side_effect=fake):
            adapter.encode(tmp_path / "in.mkv", tmp_path / "out.obu", vp, vmaf_enabled=True)
        call = metrics.calls[0]
        assert call["reference"] == tmp_path / "in.mkv"
        assert call["distorted"] == tmp_path / "out.obu"
        assert call["crop"] == CropRect(w=1910, h=798, x=5, y=141)
        assert call["deinterlace"] is False
        assert (call["final_width"], call["final_height"]) == (1904, 792)
        assert call["matrix"] == "bt709"
        assert (call["fps_num"], call["fps_den"]) == (24000, 1001)

    def test_measure_receives_deinterlace_flag(self, tmp_path: Path) -> None:
        metrics = _FakeMetrics()
        adapter = SvtAv1Adapter(Path("ffmpeg"), metrics=metrics)
        fake = _FakeRunTool()
        vp = _make_vp(deinterlace=True)
        with patch("furnace.adapters.svtav1.run_tool", side_effect=fake):
            adapter.encode(tmp_path / "in.mkv", tmp_path / "out.obu", vp, vmaf_enabled=True)
        assert metrics.calls[0]["deinterlace"] is True

    def test_disabled_skips_measure(self, tmp_path: Path) -> None:
        metrics = _FakeMetrics(MetricScores(ssimulacra2=88.2))
        adapter = SvtAv1Adapter(Path("ffmpeg"), metrics=metrics)
        fake = _FakeRunTool()
        result = _run(adapter, fake, tmp_path, vmaf_enabled=False)
        assert metrics.calls == []
        assert result.ssimulacra2_score is None

    def test_no_metrics_adapter_no_scores(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))  # no metrics adapter injected
        fake = _FakeRunTool()
        result = _run(adapter, fake, tmp_path, vmaf_enabled=True)
        assert result.ssimulacra2_score is None
        assert result.butteraugli_score is None
        assert result.cvvdp_score is None

    def test_skipped_when_encode_fails(self, tmp_path: Path) -> None:
        metrics = _FakeMetrics(MetricScores(ssimulacra2=88.2))
        adapter = SvtAv1Adapter(Path("ffmpeg"), metrics=metrics)
        fake = _FakeRunTool(encode_rc=3)
        result = _run(adapter, fake, tmp_path, vmaf_enabled=True)
        assert result.return_code == 3
        assert metrics.calls == []
        assert result.ssimulacra2_score is None


class TestRpuIgnored:
    def test_rpu_path_ignored_no_crash(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool()
        result = _run(adapter, fake, tmp_path, rpu_path=Path("rpu.bin"))
        assert result.return_code == 0

    def test_rpu_not_in_encode_command(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool()
        _run(adapter, fake, tmp_path, rpu_path=Path("rpu.bin"))
        for call in fake.calls:
            assert "rpu.bin" not in " ".join(call["cmd"])
