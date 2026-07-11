"""Tests for SvtAv1Adapter.encode orchestration: log wiring, VMAF pass, fail-soft.

Task 3 fleshes out the minimal Task-2 ``encode`` with (a) the encode log file
wired to ``_log_dir``, (b) an optional ffmpeg ``libvmaf`` metrics pass when
``vmaf_enabled``, and (c) fail-soft metrics — a VMAF failure must never fail the
encode. These tests patch ``furnace.adapters.svtav1.run_tool`` and, for the VMAF
pass, write a real JSON log so the JSON parse path is exercised end-to-end.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from furnace.adapters.svtav1 import SvtAv1Adapter, _build_vf, _geometry_filters
from furnace.core.models import CropRect, VideoParams
from furnace.core.progress import ProgressSample

_GOOD_JSON = '{"pooled_metrics": {"vmaf": {"mean": 95.2}}}'


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
    """Records every ``run_tool`` call; dispatches encode vs VMAF by ``-lavfi``.

    For the VMAF pass it reconstructs the libvmaf log target from ``cwd`` plus
    the ``log_path=<basename>`` embedded in the ``-lavfi`` string and writes
    ``vmaf_json`` there (unless ``write_json`` is False), mirroring what a real
    ffmpeg libvmaf run would produce.
    """

    def __init__(
        self,
        *,
        encode_rc: int = 0,
        vmaf_rc: int = 0,
        vmaf_json: str | None = _GOOD_JSON,
        write_json: bool = True,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.encode_rc = encode_rc
        self.vmaf_rc = vmaf_rc
        self.vmaf_json = vmaf_json
        self.write_json = write_json

    def __call__(
        self,
        cmd: Any,
        on_output: Any = None,
        on_progress_line: Any = None,
        log_path: Any = None,
        cwd: Any = None,
    ) -> tuple[int, str]:
        str_cmd = [str(c) for c in cmd]
        self.calls.append(
            {
                "cmd": str_cmd,
                "on_output": on_output,
                "on_progress_line": on_progress_line,
                "log_path": log_path,
                "cwd": cwd,
            },
        )
        if "-lavfi" in str_cmd:
            if self.write_json and self.vmaf_json is not None:
                lavfi = str_cmd[str_cmd.index("-lavfi") + 1]
                m = re.search(r"log_path=([^:]+):log_fmt", lavfi)
                assert m is not None
                target = (Path(cwd) if cwd else Path()) / m.group(1)
                target.write_text(self.vmaf_json, encoding="utf-8")
            return self.vmaf_rc, ""
        return self.encode_rc, ""

    @property
    def encode_call(self) -> dict[str, Any]:
        return next(c for c in self.calls if "-lavfi" not in c["cmd"])

    @property
    def vmaf_call(self) -> dict[str, Any]:
        return next(c for c in self.calls if "-lavfi" in c["cmd"])


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
    def test_returns_rc_and_settings_no_vmaf(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool(encode_rc=0)
        result = _run(adapter, fake, tmp_path)
        assert result.return_code == 0
        assert result.encoder_settings.startswith("av1_svt")
        assert result.vmaf_score is None
        assert result.ssim_score is None

    def test_returns_rc_from_run_tool(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool(encode_rc=7)
        result = _run(adapter, fake, tmp_path)
        assert result.return_code == 7

    def test_vmaf_disabled_skips_pass(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool()
        result = _run(adapter, fake, tmp_path, vmaf_enabled=False)
        assert len(fake.calls) == 1
        assert result.vmaf_score is None

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
        # The encode call receives the adapter's on_output callback.
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


class TestVmafPass:
    def test_vmaf_success_returns_score(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool(vmaf_json=_GOOD_JSON)
        result = _run(adapter, fake, tmp_path, vmaf_enabled=True)
        assert result.vmaf_score == pytest.approx(95.2)
        assert result.ssim_score is None
        assert result.return_code == 0
        assert len(fake.calls) == 2

    def test_vmaf_command_shape(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool()
        _run(adapter, fake, tmp_path, vmaf_enabled=True)
        cmd = fake.vmaf_call["cmd"]
        assert cmd[0] == "ffmpeg"
        assert "-hide_banner" in cmd
        assert cmd[-3:] == ["-f", "null", "-"]
        lavfi = cmd[cmd.index("-lavfi") + 1]
        assert "libvmaf" in lavfi
        assert "log_fmt=json" in lavfi
        # Distorted stream (input 0) is the encoded OBU; reference (input 1) is
        # the source. So -i <output> must precede -i <input>.
        out_i = cmd.index(str((tmp_path / "output.obu").resolve()))
        src_i = cmd.index(str((tmp_path / "input.mkv").resolve()))
        assert out_i < src_i

    def test_vmaf_inputs_are_absolute(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The VMAF pass runs with cwd=json dir, so relative inputs (as the
        executor passes) must be resolved to absolute or the source won't open.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / "sub").mkdir()
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool()
        with patch("furnace.adapters.svtav1.run_tool", side_effect=fake):
            adapter.encode(
                Path("src.mkv"), Path("sub/out.obu"), _make_vp(), vmaf_enabled=True,
            )
        cmd = fake.vmaf_call["cmd"]
        i_positions = [i for i, a in enumerate(cmd) if a == "-i"]
        for i in i_positions:
            assert Path(cmd[i + 1]).is_absolute(), f"input {cmd[i + 1]!r} is not absolute"

    def test_vmaf_reference_vf_matches_geometry_without_tail(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool()
        vp = _make_vp(crop=CropRect(w=1910, h=798, x=5, y=141))
        with patch("furnace.adapters.svtav1.run_tool", side_effect=fake):
            adapter.encode(
                tmp_path / "input.mkv", tmp_path / "output.obu", vp,
                vmaf_enabled=True,
            )
        lavfi = fake.vmaf_call["cmd"][fake.vmaf_call["cmd"].index("-lavfi") + 1]
        assert "crop=1910:798:5:141" in lavfi
        assert "scale=1904:792:flags=spline" in lavfi
        # Geometry -> fps decimation -> re-index, in that order, on the reference.
        assert (
            "[1:v]crop=1910:798:5:141,scale=1904:792:flags=spline,"
            "fps=24000/1001,setpts=N[r]" in lavfi
        )
        # The reference chain must NOT carry the 10-bit / setsar tail.
        assert "format=yuv420p10le" not in lavfi
        assert "setsar=1" not in lavfi

    def test_vmaf_reference_vf_plain_source(self, tmp_path: Path) -> None:
        """No crop/scale/bwdif -> the reference chain decimates to the coded
        rate (``fps=<rate>``) then re-indexes (``setpts=N``) so libvmaf pairs
        frame-by-frame, not by timestamp.
        """
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool()
        _run(adapter, fake, tmp_path, vmaf_enabled=True)
        lavfi = fake.vmaf_call["cmd"][fake.vmaf_call["cmd"].index("-lavfi") + 1]
        assert "[1:v]fps=24000/1001,setpts=N[r]" in lavfi

    def test_vmaf_obu_read_at_coded_rate_and_frame_index_sync(
        self, tmp_path: Path,
    ) -> None:
        """The OBU is rateless, so it's read with ``-r <fps>`` to stamp it at the
        coded rate; the source is decimated to the same rate (``fps=``). BOTH
        streams are then re-indexed with ``setpts=N`` so libvmaf pairs frame N
        against frame N. Timestamp pairing drifted apart on PAL DVD sources whose
        demuxed PTS carry a start offset / jitter, collapsing VMAF to ~35.
        """
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool()
        _run(adapter, fake, tmp_path, vmaf_enabled=True)
        cmd = fake.vmaf_call["cmd"]
        # -r <fps> must precede the OBU input (input 0).
        r_idx = cmd.index("-r")
        obu_idx = cmd.index(str((tmp_path / "output.obu").resolve()))
        assert cmd[r_idx + 1] == "24000/1001"
        assert r_idx < obu_idx
        lavfi = cmd[cmd.index("-lavfi") + 1]
        # Distorted chain (input 0) is re-indexed, then paired against the
        # re-indexed reference: index N <-> index N.
        assert lavfi.startswith("[0:v]setpts=N[d];")
        assert "[d][r]libvmaf" in lavfi
        # fps decimation must come BEFORE setpts on the reference (decimate to
        # the coded rate first, then re-index the surviving frames from 0).
        assert "fps=24000/1001,setpts=N[r]" in lavfi

    def test_vmaf_log_wired_when_log_dir_set(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"), log_dir=tmp_path)
        fake = _FakeRunTool()
        _run(adapter, fake, tmp_path, vmaf_enabled=True)
        assert fake.vmaf_call["log_path"] == tmp_path / "svt_vmaf.log"

    def test_vmaf_log_none_when_no_log_dir(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool()
        _run(adapter, fake, tmp_path, vmaf_enabled=True)
        assert fake.vmaf_call["log_path"] is None


class TestVmafFailSoft:
    def test_vmaf_nonzero_rc_returns_none(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool(vmaf_rc=1, write_json=False)
        result = _run(adapter, fake, tmp_path, vmaf_enabled=True)
        assert result.vmaf_score is None
        assert result.return_code == 0

    def test_vmaf_missing_json_returns_none(self, tmp_path: Path) -> None:
        # rc==0 but no JSON written -> open() raises OSError.
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool(write_json=False)
        result = _run(adapter, fake, tmp_path, vmaf_enabled=True)
        assert result.vmaf_score is None

    def test_vmaf_garbage_json_returns_none(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool(vmaf_json="{not valid json")
        result = _run(adapter, fake, tmp_path, vmaf_enabled=True)
        assert result.vmaf_score is None

    def test_vmaf_missing_key_returns_none(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool(vmaf_json='{"pooled_metrics": {}}')
        result = _run(adapter, fake, tmp_path, vmaf_enabled=True)
        assert result.vmaf_score is None

    def test_vmaf_non_numeric_mean_returns_none(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool(
            vmaf_json='{"pooled_metrics": {"vmaf": {"mean": "oops"}}}',
        )
        result = _run(adapter, fake, tmp_path, vmaf_enabled=True)
        assert result.vmaf_score is None

    def test_vmaf_null_mean_returns_none(self, tmp_path: Path) -> None:
        # mean == null -> float(None) raises TypeError; must degrade fail-soft
        # (no score) rather than propagating and failing the encode.
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool(
            vmaf_json='{"pooled_metrics": {"vmaf": {"mean": null}}}',
        )
        result = _run(adapter, fake, tmp_path, vmaf_enabled=True)
        assert result.vmaf_score is None
        assert result.return_code == 0

    def test_vmaf_skipped_when_encode_fails(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool(encode_rc=3)
        result = _run(adapter, fake, tmp_path, vmaf_enabled=True)
        assert result.return_code == 3
        assert result.vmaf_score is None
        assert len(fake.calls) == 1


class TestRpuIgnored:
    def test_rpu_path_ignored_no_crash(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool()
        result = _run(adapter, fake, tmp_path, rpu_path=Path("rpu.bin"))
        assert result.return_code == 0

    def test_rpu_not_in_encode_command(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"))
        fake = _FakeRunTool()
        _run(adapter, fake, tmp_path, vmaf_enabled=True, rpu_path=Path("rpu.bin"))
        for call in fake.calls:
            assert "rpu.bin" not in " ".join(call["cmd"])
