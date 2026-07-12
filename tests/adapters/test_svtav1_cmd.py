"""Tests for SvtAv1Adapter command construction + Encoder-protocol conformance.

Mirrors ``tests/adapters/test_nvencc_cmd.py`` in style: build the command, str-ify
every element, and pin the recipe flags / -vf filter chain / colors. ``encode`` is
minimal in Task 2 (build + run the command); Task 3 fleshes out VMAF + progress.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from furnace.adapters.svtav1 import (
    _SVT_CRF,
    _SVT_PARAMS,
    _SVT_PRESET,
    SvtAv1Adapter,
)
from furnace.core.models import CropRect, VideoParams
from furnace.core.ports import Encoder
from furnace.core.progress import ProgressSample


def _make_vp(
    *,
    crop: CropRect | None = None,
    deinterlace: bool = False,
    color_matrix: str = "bt709",
    color_transfer: str = "bt709",
    color_primaries: str = "bt709",
    color_range: str = "tv",
    gop: int = 120,
    source_width: int = 1920,
    source_height: int = 1080,
    sar_num: int = 1,
    sar_den: int = 1,
    grain: bool = True,
    fps_num: int = 24000,
    fps_den: int = 1001,
) -> VideoParams:
    return VideoParams(
        cq=23, crop=crop, deinterlace=deinterlace,
        color_matrix=color_matrix, color_range=color_range,
        color_transfer=color_transfer, color_primaries=color_primaries,
        hdr=None, gop=gop, fps_num=fps_num, fps_den=fps_den,
        source_width=source_width, source_height=source_height,
        source_codec="mpeg2video", source_bitrate=8_000_000,
        sar_num=sar_num, sar_den=sar_den, grain=grain,
    )


def _contains_subseq(cmd: list[str], sub: list[str]) -> bool:
    """True if `sub` appears as a contiguous slice of `cmd` (order-preserving)."""
    n = len(sub)
    return any(cmd[i:i + n] == sub for i in range(len(cmd) - n + 1))


def _adapter() -> SvtAv1Adapter:
    return SvtAv1Adapter(Path("ffmpeg"))


def _cmd(vp: VideoParams) -> list[str]:
    """Build the encode command and str-ify every element for easy assertion."""
    raw = _adapter()._build_encode_cmd(Path("input.mkv"), Path("output.obu"), vp)
    return [str(x) for x in raw]


def _vf(vp: VideoParams) -> str:
    """Return the -vf filtergraph value from the built command."""
    cmd = _cmd(vp)
    return cmd[cmd.index("-vf") + 1]


class TestSvtAv1RecipeFlags:
    """The load-bearing SVT-AV1 recipe: codec, preset, crf, params, obu, progress."""

    def test_codec_preset_crf_block(self) -> None:
        cmd = _cmd(_make_vp())
        assert _contains_subseq(
            cmd,
            ["-c:v", "libsvtav1", "-preset", _SVT_PRESET, "-crf", _SVT_CRF],
        )

    def test_preset_is_4(self) -> None:
        assert _SVT_PRESET == "4"

    def test_crf_is_23(self) -> None:
        assert _SVT_CRF == "23"

    def test_svtav1_params_starts_with_recipe(self) -> None:
        # The tuned recipe is preserved verbatim as the prefix; the CICP
        # color-description is appended after it.
        cmd = _cmd(_make_vp())
        idx = cmd.index("-svtav1-params")
        assert cmd[idx + 1].startswith(_SVT_PARAMS + ":")

    def test_svtav1_params_flag_present(self) -> None:
        cmd = _cmd(_make_vp())
        assert "-svtav1-params" in cmd
        assert cmd[cmd.index("-svtav1-params") + 1].startswith(_SVT_PARAMS)

    def test_output_format_obu(self) -> None:
        cmd = _cmd(_make_vp())
        assert _contains_subseq(cmd, ["-f", "obu"])

    def test_progress_pipe(self) -> None:
        cmd = _cmd(_make_vp())
        assert _contains_subseq(cmd, ["-progress", "pipe:1"])

    def test_hide_banner_and_overwrite(self) -> None:
        cmd = _cmd(_make_vp())
        assert "-hide_banner" in cmd
        assert "-y" in cmd

    def test_libsvtav1_params_value_is_exact_recipe(self) -> None:
        """The tuned recipe string is pinned verbatim — no drift allowed."""
        assert _SVT_PARAMS == (
            "tune=0:enable-variance-boost=1:variance-boost-strength=3:"
            "enable-qm=1:qm-min=0:luminance-qp-bias=50:ac-bias=6.0"
        )


class TestSvtAv1ColorDescription:
    """Full CICP color-description is appended to -svtav1-params so the AV1
    bitstream is self-describing. libsvtav1 drops ffmpeg's -color_primaries /
    -color_trc, so without this the OBU carries no primaries/transfer.
    """

    def test_pal_color_description(self) -> None:
        cmd = _cmd(_make_vp(
            color_primaries="bt470bg", color_transfer="smpte170m",
            color_matrix="bt470bg", color_range="tv",
        ))
        params = cmd[cmd.index("-svtav1-params") + 1]
        assert "color-primaries=5" in params
        assert "transfer-characteristics=6" in params
        assert "matrix-coefficients=5" in params
        assert "color-range=0" in params  # tv / studio-swing

    def test_hd_color_description(self) -> None:
        cmd = _cmd(_make_vp(
            color_primaries="bt709", color_transfer="bt709",
            color_matrix="bt709", color_range="tv",
        ))
        params = cmd[cmd.index("-svtav1-params") + 1]
        assert "color-primaries=1" in params
        assert "transfer-characteristics=1" in params
        assert "matrix-coefficients=1" in params

    def test_full_range_maps_to_one(self) -> None:
        cmd = _cmd(_make_vp(color_range="pc"))
        params = cmd[cmd.index("-svtav1-params") + 1]
        assert "color-range=1" in params  # pc / full-swing

    def test_unmapped_color_value_raises_valueerror(self) -> None:
        # transfer/primaries pass through source tags unvalidated; a legit H.273
        # value furnace has no CICP code point for must fail loudly (clear
        # ValueError), not crash with a cryptic KeyError or emit a bad OBU.
        with pytest.raises(ValueError, match="no CICP code point"):
            _cmd(_make_vp(color_primaries="film"))


class TestSvtAv1ForbiddenForkParams:
    """Fork-only knobs must never leak into the -svtav1-params string."""

    def test_psy_rd_absent(self) -> None:
        cmd = _cmd(_make_vp())
        params = cmd[cmd.index("-svtav1-params") + 1]
        assert "psy-rd" not in params

    def test_spy_rd_absent(self) -> None:
        cmd = _cmd(_make_vp())
        params = cmd[cmd.index("-svtav1-params") + 1]
        assert "spy-rd" not in params

    def test_noise_norm_strength_absent(self) -> None:
        cmd = _cmd(_make_vp())
        params = cmd[cmd.index("-svtav1-params") + 1]
        assert "noise-norm-strength" not in params


class TestSvtAv1Gop:
    """-g mirrors vp.gop."""

    def test_gop_value(self) -> None:
        cmd = _cmd(_make_vp(gop=125))
        idx = cmd.index("-g")
        assert cmd[idx + 1] == "125"

    def test_gop_default(self) -> None:
        cmd = _cmd(_make_vp())
        idx = cmd.index("-g")
        assert cmd[idx + 1] == "120"


class TestSvtAv1OutputRate:
    """Output ``-r`` pins the encode to the coded film rate (vp.fps_num/fps_den).

    For soft-telecine NTSC-DVD sources plain ffmpeg applies the 2:3 pulldown on
    decode (inflating to 29.97); ``-r 24000/1001`` drops the duplicated frames so
    the OBU matches the rate mkvmerge pins the container to. For native content
    the input already decodes at that rate, so ``-r`` is a harmless no-op.
    """

    def test_output_rate_telecine(self) -> None:
        cmd = _cmd(_make_vp(fps_num=24000, fps_den=1001))
        assert _contains_subseq(cmd, ["-r", "24000/1001"])

    def test_output_rate_native(self) -> None:
        cmd = _cmd(_make_vp(fps_num=25, fps_den=1))
        assert _contains_subseq(cmd, ["-r", "25/1"])

    def test_output_rate_is_output_option_before_obu(self) -> None:
        """`-r` is an OUTPUT option (drops pulldown dups): after -i, before -f obu."""
        cmd = _cmd(_make_vp())
        r_idx = cmd.index("-r")
        assert r_idx > cmd.index("-i")
        assert r_idx < cmd.index("-f")


class TestSvtAv1Color:
    """Color metadata maps from vp fields; range flows from vp.color_range."""

    def test_color_range_tv(self) -> None:
        cmd = _cmd(_make_vp())
        idx = cmd.index("-color_range")
        assert cmd[idx + 1] == "tv"

    def test_color_range_passthrough_pc(self) -> None:
        """-color_range mirrors vp.color_range (ffmpeg accepts tv/pc), not a
        hardcoded 'tv' -- matches NVEncC deriving range from the same field."""
        cmd = _cmd(_make_vp(color_range="pc"))
        idx = cmd.index("-color_range")
        assert cmd[idx + 1] == "pc"

    def test_bt709_colors(self) -> None:
        cmd = _cmd(_make_vp(
            color_matrix="bt709", color_primaries="bt709", color_transfer="bt709",
        ))
        assert cmd[cmd.index("-color_primaries") + 1] == "bt709"
        assert cmd[cmd.index("-color_trc") + 1] == "bt709"
        assert cmd[cmd.index("-colorspace") + 1] == "bt709"

    def test_bt601_colors(self) -> None:
        cmd = _cmd(_make_vp(
            color_matrix="smpte170m",
            color_primaries="smpte170m",
            color_transfer="smpte170m",
        ))
        assert cmd[cmd.index("-color_primaries") + 1] == "smpte170m"
        assert cmd[cmd.index("-color_trc") + 1] == "smpte170m"
        assert cmd[cmd.index("-colorspace") + 1] == "smpte170m"


class TestSvtAv1InputOutput:
    """Input via -i, output as the final element."""

    def test_input_path(self) -> None:
        cmd = _cmd(_make_vp())
        assert cmd[cmd.index("-i") + 1] == "input.mkv"

    def test_output_path_last(self) -> None:
        cmd = _cmd(_make_vp())
        assert cmd[-1] == "output.obu"


class TestSvtAv1VideoFilter:
    """The -vf filtergraph: format/setsar always, crop/scale/bwdif conditional."""

    def test_always_ends_with_format_and_setsar(self) -> None:
        parts = _vf(_make_vp()).split(",")
        assert parts[-2:] == ["format=yuv420p10le", "setsar=1"]

    def test_no_crop_no_scale_plain_1080p(self) -> None:
        """1920x1080, square SAR, no crop -> only format + setsar."""
        vf = _vf(_make_vp())
        assert "crop=" not in vf
        assert "scale=" not in vf
        assert vf == "format=yuv420p10le,setsar=1"

    def test_crop_present(self) -> None:
        """Crop rect renders as crop=w:h:x:y, and mod-8 dims need no scale."""
        vp = _make_vp(crop=CropRect(w=1920, h=800, x=0, y=140))
        vf = _vf(vp)
        assert "crop=1920:800:0:140" in vf
        assert "scale=" not in vf

    def test_crop_absent(self) -> None:
        vf = _vf(_make_vp(crop=None))
        assert "crop=" not in vf

    def test_crop_with_alignment_scale(self) -> None:
        """Crop to non-mod-8 dims -> scale to the mod-8-aligned size."""
        # CropRect 1910x798 -> mod-8 -> 1904x792.
        vp = _make_vp(crop=CropRect(w=1910, h=798, x=5, y=141))
        vf = _vf(vp)
        assert "crop=1910:798:5:141" in vf
        assert "scale=1904:792:flags=spline" in vf

    def test_scale_from_sar_no_crop(self) -> None:
        """Non-square SAR without crop -> scale to the SAR-corrected size."""
        # 720x480 SAR 4:3 -> 960x480 (both mod-8).
        vp = _make_vp(source_width=720, source_height=480, sar_num=4, sar_den=3)
        vf = _vf(vp)
        assert "crop=" not in vf
        assert "scale=960:480:flags=spline" in vf

    def test_no_scale_when_output_matches_pre_resize(self) -> None:
        vf = _vf(_make_vp())
        assert "scale=" not in vf

    def test_bwdif_first_when_deinterlace(self) -> None:
        parts = _vf(_make_vp(deinterlace=True)).split(",")
        assert parts[0] == "bwdif=send_frame"

    def test_bwdif_single_rate_send_frame(self) -> None:
        """Deinterlace must be SINGLE-RATE (send_frame): one output frame per
        input frame, matching NVEncC's nnedi and the fps-pin contract. The
        double-rate default (send_field) would emit 2N frames pinned at N fps."""
        vf = _vf(_make_vp(deinterlace=True))
        parts = vf.split(",")
        assert parts[0] == "bwdif=send_frame"
        # No bare `bwdif` (default = double-rate) and no explicit send_field.
        assert "bwdif" not in parts
        assert "send_field" not in vf

    def test_bwdif_before_crop(self) -> None:
        vp = _make_vp(deinterlace=True, crop=CropRect(w=1920, h=800, x=0, y=140))
        parts = _vf(vp).split(",")
        assert parts[0] == "bwdif=send_frame"
        assert parts[1] == "crop=1920:800:0:140"

    def test_bwdif_absent_when_not_deinterlaced(self) -> None:
        assert "bwdif" not in _vf(_make_vp(deinterlace=False))


class TestSvtAv1EncoderSettings:
    """The slash-joined ENCODER_SETTINGS tag string."""

    def test_basic_settings(self) -> None:
        settings = _adapter()._build_encoder_settings(_make_vp())
        assert settings.startswith("av1_svt")
        assert "SVT-AV1" in settings
        assert f"preset={_SVT_PRESET}" in settings
        assert f"crf={_SVT_CRF}" in settings
        assert _SVT_PARAMS in settings

    def test_settings_slash_separated(self) -> None:
        settings = _adapter()._build_encoder_settings(_make_vp())
        parts = settings.split(" / ")
        assert parts[0] == "av1_svt"
        assert parts[1] == "SVT-AV1"

    def test_settings_with_deinterlace(self) -> None:
        settings = _adapter()._build_encoder_settings(_make_vp(deinterlace=True))
        assert "bwdif" in settings

    def test_settings_without_deinterlace(self) -> None:
        settings = _adapter()._build_encoder_settings(_make_vp(deinterlace=False))
        assert "bwdif" not in settings

    def test_settings_with_crop(self) -> None:
        vp = _make_vp(crop=CropRect(w=1920, h=800, x=0, y=140))
        settings = _adapter()._build_encoder_settings(vp)
        assert "crop=1920:800:0:140" in settings

    def test_settings_without_crop(self) -> None:
        settings = _adapter()._build_encoder_settings(_make_vp(crop=None))
        assert "crop=" not in settings


class TestSvtAv1SetLogDir:
    def test_set_log_dir(self, tmp_path: Path) -> None:
        adapter = _adapter()
        adapter.set_log_dir(tmp_path)
        assert adapter._log_dir == tmp_path

    def test_init_log_dir(self, tmp_path: Path) -> None:
        adapter = SvtAv1Adapter(Path("ffmpeg"), log_dir=tmp_path)
        assert adapter._log_dir == tmp_path


class TestSvtAv1EncoderProtocol:
    """Runtime-checkable Encoder conformance."""

    def test_isinstance_encoder(self) -> None:
        assert isinstance(SvtAv1Adapter(Path("ffmpeg")), Encoder)


def _fake_run_tool(
    cmd: Any,
    on_output: Any = None,
    on_progress_line: Any = None,
    log_path: Any = None,
    cwd: Any = None,
) -> tuple[int, str]:
    return 0, ""


class TestSvtAv1Encode:
    """Minimal Task-2 encode(): build the command, run it, return EncodeResult."""

    def test_encode_returns_result(self) -> None:
        from unittest.mock import patch

        adapter = _adapter()
        vp = _make_vp()
        with patch("furnace.adapters.svtav1.run_tool", side_effect=_fake_run_tool):
            result = adapter.encode(Path("input.mkv"), Path("output.obu"), vp)
        assert result.return_code == 0
        assert result.encoder_settings.startswith("av1_svt")
        assert result.vmaf_score is None
        assert result.ssim_score is None

    def test_encode_accepts_vmaf_and_rpu_kwargs(self) -> None:
        """Task 2 accepts but ignores vmaf_enabled / rpu_path (Task 3 wires them)."""
        from unittest.mock import patch

        adapter = _adapter()
        vp = _make_vp()
        with patch("furnace.adapters.svtav1.run_tool", side_effect=_fake_run_tool):
            result = adapter.encode(
                Path("input.mkv"), Path("output.obu"), vp,
                vmaf_enabled=True, rpu_path=Path("rpu.bin"),
            )
        assert result.return_code == 0

    def test_encode_forwards_progress(self) -> None:
        from unittest.mock import patch

        adapter = _adapter()
        vp = _make_vp()
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
                Path("input.mkv"), Path("output.obu"), vp,
                on_progress=samples.append,
            )
        assert len(samples) == 1
        assert samples[0].processed_s == 5.0

    def test_encode_forwards_on_output(self) -> None:
        from unittest.mock import patch

        lines: list[str] = []
        adapter = SvtAv1Adapter(Path("ffmpeg"), on_output=lines.append)
        vp = _make_vp()

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_output is not None:
                on_output("frame= 100")
            return 0, ""

        with patch("furnace.adapters.svtav1.run_tool", side_effect=fake_run_tool):
            adapter.encode(Path("input.mkv"), Path("output.obu"), vp)
        assert "frame= 100" in lines
