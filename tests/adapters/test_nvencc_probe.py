"""Tests for NVEncCAdapter.probe — target-quality inline metric probing."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from furnace.core.models import CropRect, DvMode, HdrMetadata, VideoParams


def _make_vp(
    *,
    crop: CropRect | None = None,
    cq: int = 31,
    source_width: int = 3840,
    source_height: int = 2160,
    dv_mode: DvMode | None = None,
    hdr: HdrMetadata | None = None,
) -> VideoParams:
    return VideoParams(
        cq=cq, crop=crop, deinterlace=False,
        color_matrix="bt2020nc", color_range="tv",
        color_transfer="smpte2084", color_primaries="bt2020",
        hdr=hdr, gop=120, fps_num=24000, fps_den=1001,
        source_width=source_width, source_height=source_height, source_codec="hevc",
        source_bitrate=80_000_000, dv_mode=dv_mode,
    )


def _adapter() -> Any:
    from furnace.adapters.nvencc import NVEncCAdapter

    return NVEncCAdapter(Path("NVEncC64.exe"))


def _capture_probe_cmd(vp: VideoParams, *, qvbr: int, metric: str) -> list[str]:
    """Run probe() with a run_tool stub that captures (and does not emit) the cmd,
    then feeds a matching metric line so probe() returns without raising."""
    captured: list[str] = []
    _lines = {
        "cvvdp": "ssim/psnr/vmaf/vship: CVVDP Score 9.30",
        "ssimulacra2": "ssim/psnr/vmaf/vship: SSIMU2 Score 85.00 (Frames: 48)",
        "vmaf": "ssim/psnr/vmaf/vship: VMAF Score 95.00",
    }

    def fake_run_tool(
        cmd: Any,
        on_output: Any = None,
        on_progress_line: Any = None,
        log_path: Any = None,
        cwd: Any = None,
    ) -> tuple[int, str]:
        captured.extend(str(c) for c in cmd)
        if on_output is not None:
            on_output(_lines[metric])
        return 0, ""

    with patch("furnace.adapters.nvencc.run_tool", side_effect=fake_run_tool):
        _adapter().probe(
            Path("window.mkv"), Path("probe.obu"), vp, qvbr=qvbr, metric=metric,
        )
    return captured


class TestNVEncCProbeCommand:
    def test_probe_uses_qvbr_argument(self) -> None:
        cmd = _capture_probe_cmd(_make_vp(cq=31), qvbr=24, metric="cvvdp")
        assert cmd[cmd.index("--qvbr") + 1] == "24"

    def test_probe_cvvdp_single_metric(self) -> None:
        cmd = _capture_probe_cmd(_make_vp(), qvbr=30, metric="cvvdp")
        assert "--vship-cvvdp" in cmd
        assert "--vship-ssimulacra2" not in cmd
        assert "--vship-butteraugli" not in cmd
        assert "--vmaf" not in cmd

    def test_probe_ssimulacra2_single_metric(self) -> None:
        cmd = _capture_probe_cmd(_make_vp(), qvbr=30, metric="ssimulacra2")
        assert "--vship-ssimulacra2" in cmd
        assert "--vship-cvvdp" not in cmd
        assert "--vship-butteraugli" not in cmd
        assert "--vmaf" not in cmd

    def test_probe_vmaf_single_metric_4k_model(self) -> None:
        cmd = _capture_probe_cmd(_make_vp(), qvbr=30, metric="vmaf")
        idx = cmd.index("--vmaf")
        assert "vmaf_4k_v0.6.1" in cmd[idx + 1]
        assert "--vship-cvvdp" not in cmd

    def test_probe_vmaf_single_metric_1080p_model(self) -> None:
        cmd = _capture_probe_cmd(
            _make_vp(source_width=1920, source_height=1080), qvbr=30, metric="vmaf",
        )
        idx = cmd.index("--vmaf")
        assert "vmaf_v0.6.1" in cmd[idx + 1]
        assert "vmaf_4k" not in cmd[idx + 1]

    def test_probe_applies_geometry(self) -> None:
        """Crop/color from the job are applied to the probe encode so the measured
        quality reflects the real geometry pipeline."""
        cmd = _capture_probe_cmd(
            _make_vp(crop=CropRect(w=3560, h=2160, x=140, y=0)), qvbr=30, metric="cvvdp",
        )
        assert cmd[cmd.index("--crop") + 1] == "140,0,140,0"
        assert cmd[cmd.index("--colormatrix") + 1] == "bt2020nc"

    def test_probe_never_emits_dv_flags(self) -> None:
        """Probes skip Dolby Vision (metadata, irrelevant to the quality measure);
        no RPU is threaded, so no DV flags appear even for a DV job."""
        cmd = _capture_probe_cmd(_make_vp(dv_mode=DvMode.COPY), qvbr=30, metric="cvvdp")
        assert "--dolby-vision-rpu" not in cmd
        assert "--dolby-vision-profile" not in cmd

    def test_probe_input_output_paths(self) -> None:
        cmd = _capture_probe_cmd(_make_vp(), qvbr=30, metric="cvvdp")
        assert cmd[cmd.index("-i") + 1] == "window.mkv"
        assert cmd[cmd.index("-o") + 1] == "probe.obu"


class TestNVEncCProbeParsing:
    def _run(self, metric: str, line: str, *, rc: int = 0) -> float:
        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_output is not None and line:
                on_output(line)
            return rc, ""

        with patch("furnace.adapters.nvencc.run_tool", side_effect=fake_run_tool):
            return _adapter().probe(  # type: ignore[no-any-return]
                Path("w.mkv"), Path("p.obu"), _make_vp(), qvbr=30, metric=metric,
            )

    def test_parses_cvvdp(self) -> None:
        score = self._run("cvvdp", "ssim/psnr/vmaf/vship: CVVDP Score 9.783177")
        assert abs(score - 9.783177) < 1e-4

    def test_parses_ssimulacra2(self) -> None:
        score = self._run("ssimulacra2", "ssim/psnr/vmaf/vship: SSIMU2 Score 76.335521 (Frames: 48)")
        assert abs(score - 76.335521) < 1e-4

    def test_parses_vmaf(self) -> None:
        score = self._run("vmaf", "ssim/psnr/vmaf/vship: VMAF Score 99.236780")
        assert abs(score - 99.236780) < 1e-4

    def test_parses_integer_valued_score(self) -> None:
        """A score with no fractional part still parses (fractional part optional)."""
        score = self._run("cvvdp", "ssim/psnr/vmaf/vship: CVVDP Score 10")
        assert score == 10.0

    def test_raises_on_nonzero_return_code(self) -> None:
        with pytest.raises(RuntimeError, match="probe failed"):
            self._run("cvvdp", "ssim/psnr/vmaf/vship: CVVDP Score 9.30", rc=2)

    def test_raises_on_missing_score_line(self) -> None:
        with pytest.raises(RuntimeError, match="no cvvdp score"):
            self._run("cvvdp", "")

    def test_raises_on_unparseable_score(self) -> None:
        with pytest.raises(RuntimeError, match="no cvvdp score"):
            self._run("cvvdp", "ssim/psnr/vmaf/vship: CVVDP Score N/A")

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown probe metric"):
            self._run("butteraugli", "irrelevant")


class TestNVEncCProbeMisc:
    def test_probe_log_path(self, tmp_path: Path) -> None:
        from furnace.adapters.nvencc import NVEncCAdapter

        adapter = NVEncCAdapter(Path("NVEncC64.exe"), log_dir=tmp_path)
        captured_kwargs: dict[str, Any] = {}

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            captured_kwargs["log_path"] = log_path
            if on_output is not None:
                on_output("ssim/psnr/vmaf/vship: CVVDP Score 9.30")
            return 0, ""

        with patch("furnace.adapters.nvencc.run_tool", side_effect=fake_run_tool):
            adapter.probe(Path("w.mkv"), Path("p.obu"), _make_vp(), qvbr=24, metric="cvvdp")
        assert captured_kwargs["log_path"] == tmp_path / "nvencc_probe_cvvdp_q24.log"

    def test_probe_forwards_output_to_on_output(self) -> None:
        from furnace.adapters.nvencc import NVEncCAdapter

        seen: list[str] = []
        adapter = NVEncCAdapter(Path("NVEncC64.exe"), on_output=seen.append)

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_output is not None:
                on_output("ssim/psnr/vmaf/vship: CVVDP Score 9.30")
            return 0, ""

        with patch("furnace.adapters.nvencc.run_tool", side_effect=fake_run_tool):
            adapter.probe(Path("w.mkv"), Path("p.obu"), _make_vp(), qvbr=30, metric="cvvdp")
        assert any("CVVDP" in line for line in seen)


class TestInlineQualityProbeProtocol:
    def test_nvencc_adapter_is_inline_quality_probe(self) -> None:
        from furnace.core.ports import InlineQualityProbe

        assert isinstance(_adapter(), InlineQualityProbe)
