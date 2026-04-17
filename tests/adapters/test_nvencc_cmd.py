"""Tests for NVEncCAdapter._build_encode_cmd and _build_encoder_settings."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from furnace.adapters.nvencc import NVEncCAdapter, _parse_content_light
from furnace.core.models import CropRect, DvMode, HdrMetadata, VideoParams
from furnace.core.progress import ProgressSample


def _make_vp(
    source_codec: str = "hevc",
    crop: CropRect | None = None,
    deinterlace: bool = False,
    cq: int = 31,
    color_matrix: str = "bt2020nc",
    color_transfer: str = "smpte2084",
    color_primaries: str = "bt2020",
    hdr: HdrMetadata | None = None,
    dv_mode: DvMode | None = None,
    sar_num: int = 1,
    sar_den: int = 1,
) -> VideoParams:
    return VideoParams(
        cq=cq, crop=crop, deinterlace=deinterlace,
        color_matrix=color_matrix, color_range="tv",
        color_transfer=color_transfer, color_primaries=color_primaries,
        hdr=hdr, gop=120, fps_num=24000, fps_den=1001,
        source_width=3840, source_height=2160, source_codec=source_codec,
        source_bitrate=80_000_000, dv_mode=dv_mode,
        sar_num=sar_num, sar_den=sar_den,
    )


def _adapter() -> NVEncCAdapter:
    return NVEncCAdapter(Path("NVEncC64.exe"))


def _cmd(vp: VideoParams, *, vmaf_enabled: bool = False, rpu_path: Path | None = None) -> list[str]:
    """Build command and convert all elements to str for easier assertion."""
    raw = _adapter()._build_encode_cmd(
        Path("input.mkv"), Path("output.hevc"), vp,
        vmaf_enabled=vmaf_enabled, rpu_path=rpu_path,
    )
    return [str(x) for x in raw]


class TestNVEncCBasicCommand:
    """Core encoder flags: codec, profile, output depth, preset, tune, qvbr."""

    def test_hevc_codec_present(self) -> None:
        cmd = _cmd(_make_vp())
        idx = cmd.index("-c")
        assert cmd[idx + 1] == "hevc"

    def test_profile_main10(self) -> None:
        cmd = _cmd(_make_vp())
        idx = cmd.index("--profile")
        assert cmd[idx + 1] == "main10"

    def test_output_depth_10(self) -> None:
        cmd = _cmd(_make_vp())
        idx = cmd.index("--output-depth")
        assert cmd[idx + 1] == "10"

    def test_preset_p5(self) -> None:
        cmd = _cmd(_make_vp())
        idx = cmd.index("--preset")
        assert cmd[idx + 1] == "P5"

    def test_tune_uhq(self) -> None:
        cmd = _cmd(_make_vp())
        idx = cmd.index("--tune")
        assert cmd[idx + 1] == "uhq"

    def test_qvbr_value(self) -> None:
        cmd = _cmd(_make_vp(cq=28))
        idx = cmd.index("--qvbr")
        assert cmd[idx + 1] == "28"

    def test_aq_flags_present(self) -> None:
        cmd = _cmd(_make_vp())
        assert "--aq" in cmd
        assert "--aq-temporal" in cmd

    def test_lookahead_32(self) -> None:
        cmd = _cmd(_make_vp())
        idx = cmd.index("--lookahead")
        assert cmd[idx + 1] == "32"

    def test_multipass_2pass_quarter(self) -> None:
        cmd = _cmd(_make_vp())
        idx = cmd.index("--multipass")
        assert cmd[idx + 1] == "2pass-quarter"

    def test_gop_len(self) -> None:
        cmd = _cmd(_make_vp())
        idx = cmd.index("--gop-len")
        assert cmd[idx + 1] == "120"

    def test_strict_gop(self) -> None:
        cmd = _cmd(_make_vp())
        assert "--strict-gop" in cmd

    def test_repeat_headers(self) -> None:
        cmd = _cmd(_make_vp())
        assert "--repeat-headers" in cmd

    def test_avhw_present(self) -> None:
        cmd = _cmd(_make_vp())
        assert "--avhw" in cmd

    def test_mpeg2_uses_avsw(self) -> None:
        """MPEG2 sources fall back to software decode because NVDEC's MPEG2
        path is unreliable on interlaced DVD streams."""
        cmd = _cmd(_make_vp(source_codec="mpeg2video"))
        assert "--avsw" in cmd
        assert "--avhw" not in cmd

    def test_mpeg1_uses_avsw(self) -> None:
        cmd = _cmd(_make_vp(source_codec="mpeg1video"))
        assert "--avsw" in cmd
        assert "--avhw" not in cmd

    def test_h264_uses_avhw(self) -> None:
        cmd = _cmd(_make_vp(source_codec="h264"))
        assert "--avhw" in cmd

    def test_input_output_paths(self) -> None:
        cmd = _cmd(_make_vp())
        idx_i = cmd.index("-i")
        assert cmd[idx_i + 1] == "input.mkv"
        idx_o = cmd.index("-o")
        assert cmd[idx_o + 1] == "output.hevc"


class TestNVEncCCrop:
    """Crop conversion from CropRect(w,h,x,y) to NVEncC left,top,right,bottom."""

    def test_crop_format(self) -> None:
        """CropRect(3560, 2160, 140, 0) -> left=140, top=0, right=140, bottom=0."""
        vp = _make_vp(crop=CropRect(w=3560, h=2160, x=140, y=0))
        cmd = _cmd(vp)
        idx = cmd.index("--crop")
        assert cmd[idx + 1] == "140,0,140,0"

    def test_crop_with_top_bottom(self) -> None:
        """CropRect(3840, 1600, 0, 280) -> left=0, top=280, right=0, bottom=280."""
        vp = _make_vp(crop=CropRect(w=3840, h=1600, x=0, y=280))
        cmd = _cmd(vp)
        idx = cmd.index("--crop")
        assert cmd[idx + 1] == "0,280,0,280"

    def test_crop_all_sides(self) -> None:
        """CropRect(3680, 1920, 80, 120) -> left=80, top=120, right=80, bottom=120."""
        vp = _make_vp(crop=CropRect(w=3680, h=1920, x=80, y=120))
        cmd = _cmd(vp)
        idx = cmd.index("--crop")
        assert cmd[idx + 1] == "80,120,80,120"

    def test_no_crop_when_none(self) -> None:
        cmd = _cmd(_make_vp(crop=None))
        assert "--crop" not in cmd

    def test_crop_with_alignment(self) -> None:
        """Crop that needs mod-8 alignment should add --output-res."""
        # CropRect that produces non-mod-8 dimensions: 3830x2150
        vp = _make_vp(crop=CropRect(w=3830, h=2150, x=3, y=5))
        cmd = _cmd(vp)
        assert "--crop" in cmd
        assert "--output-res" in cmd


class TestNVEncCDeinterlace:
    """vpp-nnedi deinterlace filter."""

    def test_deinterlace_present(self) -> None:
        cmd = _cmd(_make_vp(deinterlace=True))
        assert "--vpp-nnedi" in cmd

    def test_deinterlace_params(self) -> None:
        cmd = _cmd(_make_vp(deinterlace=True))
        idx = cmd.index("--vpp-nnedi")
        params = cmd[idx + 1]
        assert "nns=64" in params
        assert "nsize=32x6" in params
        assert "quality=slow" in params

    def test_deinterlace_absent(self) -> None:
        cmd = _cmd(_make_vp(deinterlace=False))
        assert "--vpp-nnedi" not in cmd


class TestNVEncCDolbyVision:
    """Dolby Vision RPU injection and profile flags."""

    def test_dv_rpu_present(self) -> None:
        vp = _make_vp(dv_mode=DvMode.TO_8_1)
        cmd = _cmd(vp, rpu_path=Path("rpu.bin"))
        idx = cmd.index("--dolby-vision-rpu")
        assert cmd[idx + 1] == "rpu.bin"

    def test_dv_profile_81(self) -> None:
        vp = _make_vp(dv_mode=DvMode.TO_8_1)
        cmd = _cmd(vp, rpu_path=Path("rpu.bin"))
        idx = cmd.index("--dolby-vision-profile")
        assert cmd[idx + 1] == "8.1"

    def test_no_dv_without_rpu(self) -> None:
        vp = _make_vp(dv_mode=DvMode.TO_8_1)
        cmd = _cmd(vp, rpu_path=None)
        assert "--dolby-vision-rpu" not in cmd
        assert "--dolby-vision-profile" not in cmd

    def test_no_dv_flags_when_no_dv_mode(self) -> None:
        vp = _make_vp(dv_mode=None)
        cmd = _cmd(vp, rpu_path=None)
        assert "--dolby-vision-rpu" not in cmd
        assert "--dolby-vision-profile" not in cmd


class TestNVEncCSar:
    """SAR correction via --output-res and --sar."""

    def test_sar_correction_applied(self) -> None:
        """Non-square SAR -> resize + sar 1:1."""
        vp = _make_vp(sar_num=4, sar_den=3)
        cmd = _cmd(vp)
        assert "--output-res" in cmd
        idx = cmd.index("--sar")
        assert cmd[idx + 1] == "1:1"

    def test_sar_not_applied_when_square(self) -> None:
        vp = _make_vp(sar_num=1, sar_den=1)
        cmd = _cmd(vp)
        assert "--output-res" not in cmd
        # --sar should not be present for square pixels
        # (it may appear as part of another flag, so check carefully)
        sar_indices = [i for i, x in enumerate(cmd) if x == "--sar"]
        assert len(sar_indices) == 0

    def test_sar_resolution_calculation(self) -> None:
        """SAR 4:3 on 3840x2160 -> display_w = 5120, aligned to mod-8."""
        vp = _make_vp(sar_num=4, sar_den=3)
        cmd = _cmd(vp)
        idx = cmd.index("--output-res")
        res = cmd[idx + 1]
        w, h = res.split("x")
        assert int(w) % 8 == 0
        assert int(h) % 8 == 0
        # 3840 * 4/3 = 5120, already mod-8
        assert w == "5120"
        assert h == "2160"


class TestNVEncCColor:
    """Color metadata flags for different color spaces."""

    def test_bt2020_color_flags(self) -> None:
        vp = _make_vp(
            color_matrix="bt2020nc",
            color_primaries="bt2020",
            color_transfer="smpte2084",
        )
        cmd = _cmd(vp)
        idx = cmd.index("--colorrange")
        assert cmd[idx + 1] == "limited"
        idx = cmd.index("--colorprim")
        assert cmd[idx + 1] == "bt2020"
        idx = cmd.index("--transfer")
        assert cmd[idx + 1] == "smpte2084"
        idx = cmd.index("--colormatrix")
        assert cmd[idx + 1] == "bt2020nc"

    def test_bt709_color_flags(self) -> None:
        vp = _make_vp(
            color_matrix="bt709",
            color_primaries="bt709",
            color_transfer="bt709",
        )
        cmd = _cmd(vp)
        idx = cmd.index("--colormatrix")
        assert cmd[idx + 1] == "bt709"
        idx = cmd.index("--colorprim")
        assert cmd[idx + 1] == "bt709"
        idx = cmd.index("--transfer")
        assert cmd[idx + 1] == "bt709"

    def test_bt601_color_flags(self) -> None:
        vp = _make_vp(
            color_matrix="smpte170m",
            color_primaries="smpte170m",
            color_transfer="smpte170m",
        )
        cmd = _cmd(vp)
        idx = cmd.index("--colormatrix")
        assert cmd[idx + 1] == "smpte170m"

    def test_color_range_tv_maps_to_limited(self) -> None:
        cmd = _cmd(_make_vp())
        idx = cmd.index("--colorrange")
        assert cmd[idx + 1] == "limited"



class TestNVEncCHdr:
    """HDR metadata flags: --max-cll and --master-display."""

    def test_max_cll_present(self) -> None:
        hdr = HdrMetadata(content_light="MaxCLL=1000,MaxFALL=400")
        vp = _make_vp(hdr=hdr)
        cmd = _cmd(vp)
        idx = cmd.index("--max-cll")
        assert cmd[idx + 1] == "1000,400"

    def test_master_display_present(self) -> None:
        md = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,0)"
        hdr = HdrMetadata(mastering_display=md)
        vp = _make_vp(hdr=hdr)
        cmd = _cmd(vp)
        idx = cmd.index("--master-display")
        assert cmd[idx + 1] == md

    def test_both_hdr_values(self) -> None:
        hdr = HdrMetadata(
            content_light="MaxCLL=1000,MaxFALL=400",
            mastering_display="G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,0)",
        )
        vp = _make_vp(hdr=hdr)
        cmd = _cmd(vp)
        assert "--max-cll" in cmd
        assert "--master-display" in cmd

    def test_no_hdr_no_flags(self) -> None:
        vp = _make_vp(hdr=None)
        cmd = _cmd(vp)
        assert "--max-cll" not in cmd
        assert "--master-display" not in cmd

    def test_hdr_without_content_light(self) -> None:
        hdr = HdrMetadata(
            mastering_display="G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,0)",
        )
        vp = _make_vp(hdr=hdr)
        cmd = _cmd(vp)
        assert "--max-cll" not in cmd
        assert "--master-display" in cmd

    def test_content_light_with_spaces(self) -> None:
        hdr = HdrMetadata(content_light="MaxCLL=1000, MaxFALL=400")
        vp = _make_vp(hdr=hdr)
        cmd = _cmd(vp)
        idx = cmd.index("--max-cll")
        assert cmd[idx + 1] == "1000,400"


class TestNVEncCVmaf:
    """Quality metrics: --ssim and --vmaf when vmaf_enabled."""

    def test_vmaf_enabled(self) -> None:
        cmd = _cmd(_make_vp(), vmaf_enabled=True)
        assert "--ssim" in cmd
        assert "--vmaf" in cmd

    def test_vmaf_params(self) -> None:
        cmd = _cmd(_make_vp(), vmaf_enabled=True)
        idx = cmd.index("--vmaf")
        params = cmd[idx + 1]
        assert "subsample=8" in params
        assert "vmaf_4k_v0.6.1" in params  # 4K source

    def test_vmaf_model_1080p(self) -> None:
        vp = _make_vp()
        vp.source_width = 1920
        vp.source_height = 1080
        cmd = _cmd(vp, vmaf_enabled=True)
        idx = cmd.index("--vmaf")
        params = cmd[idx + 1]
        assert "vmaf_v0.6.1" in params
        assert "vmaf_4k" not in params

    def test_vmaf_disabled(self) -> None:
        cmd = _cmd(_make_vp(), vmaf_enabled=False)
        assert "--ssim" not in cmd
        assert "--vmaf" not in cmd


class TestNVEncCEncoderSettings:
    """The encoder_settings string format for MKV tags."""

    def test_basic_settings_format(self) -> None:
        adapter = _adapter()
        vp = _make_vp()
        settings = adapter._build_encoder_settings(vp)
        assert settings.startswith("hevc_nvenc")
        assert "main10" in settings
        assert "qvbr=31" in settings
        assert "preset=P5" in settings
        assert "tune=uhq" in settings
        assert "aq" in settings
        assert "aq-temporal" in settings
        assert "lookahead=32" in settings
        assert "multipass=2pass-quarter" in settings

    def test_settings_with_deinterlace(self) -> None:
        adapter = _adapter()
        vp = _make_vp(deinterlace=True)
        settings = adapter._build_encoder_settings(vp)
        assert "deinterlace=nnedi" in settings

    def test_settings_without_deinterlace(self) -> None:
        adapter = _adapter()
        vp = _make_vp(deinterlace=False)
        settings = adapter._build_encoder_settings(vp)
        assert "deinterlace" not in settings

    def test_settings_with_crop(self) -> None:
        adapter = _adapter()
        vp = _make_vp(crop=CropRect(w=3560, h=2160, x=140, y=0))
        settings = adapter._build_encoder_settings(vp)
        # crop in T:B:L:R format in settings string
        assert "crop=0:0:140:140" in settings

    def test_settings_with_dv(self) -> None:
        adapter = _adapter()
        vp = _make_vp(dv_mode=DvMode.TO_8_1)
        settings = adapter._build_encoder_settings(vp)
        assert "dolby-vision=8.1" in settings

    def test_settings_slash_separated(self) -> None:
        adapter = _adapter()
        vp = _make_vp()
        settings = adapter._build_encoder_settings(vp)
        parts = settings.split(" / ")
        assert len(parts) >= 8
        assert parts[0] == "hevc_nvenc"


class TestNVEncCOutputFormat:
    """Output must be raw HEVC, not MKV."""

    def test_output_uses_hevc_extension(self) -> None:
        cmd = _cmd(_make_vp())
        idx = cmd.index("-o")
        output = cmd[idx + 1]
        assert output.endswith(".hevc")

    def test_output_flag_is_dash_o(self) -> None:
        cmd = _cmd(_make_vp())
        assert "-o" in cmd


class TestNVEncCSarInSettings:
    """SAR appears in the encoder_settings string."""

    def test_sar_in_settings(self) -> None:
        adapter = _adapter()
        vp = _make_vp(sar_num=64, sar_den=45)
        settings = adapter._build_encoder_settings(vp)
        assert "sar=" in settings

    def test_no_sar_when_square_in_settings(self) -> None:
        adapter = _adapter()
        vp = _make_vp(sar_num=1, sar_den=1)
        settings = adapter._build_encoder_settings(vp)
        assert "sar=" not in settings


class TestNVEncCDvCropParam:
    """DV crop=true in --dolby-vision-rpu-prm when crop is applied."""

    def test_dv_crop_param(self) -> None:
        vp = _make_vp(
            dv_mode=DvMode.COPY,
            crop=CropRect(w=3560, h=2160, x=140, y=0),
        )
        cmd = _cmd(vp, rpu_path=Path("rpu.bin"))
        idx = cmd.index("--dolby-vision-rpu-prm")
        assert cmd[idx + 1] == "crop=true"

    def test_dv_no_crop_param_when_no_crop(self) -> None:
        vp = _make_vp(dv_mode=DvMode.COPY, crop=None)
        cmd = _cmd(vp, rpu_path=Path("rpu.bin"))
        assert "--dolby-vision-rpu-prm" not in cmd


class TestNVEncCColorRangeFalsy:
    """color_range='unknown' should produce no --colorrange flag."""

    def test_unknown_color_range_no_flag(self) -> None:
        vp = dataclasses.replace(_make_vp(), color_range="unknown")
        cmd = _cmd(vp)
        assert "--colorrange" not in cmd


class TestNVEncCContentLightParseFailure:
    """Invalid content_light string should not produce --max-cll."""

    def test_garbage_content_light(self) -> None:
        hdr = HdrMetadata(content_light="garbage")
        vp = _make_vp(hdr=hdr)
        cmd = _cmd(vp)
        assert "--max-cll" not in cmd


class TestParseContentLight:
    """_parse_content_light pure function."""

    def test_valid_input(self) -> None:
        result = _parse_content_light("MaxCLL=1000,MaxFALL=400")
        assert result == ("1000", "400")

    def test_valid_with_spaces(self) -> None:
        result = _parse_content_light("MaxCLL=1000, MaxFALL=400")
        assert result == ("1000", "400")

    def test_invalid_input(self) -> None:
        result = _parse_content_light("not valid")
        assert result is None


class TestNVEncCGetVersion:
    """_get_version caching and error handling."""

    def test_version_parsed(self) -> None:
        adapter = _adapter()
        mock_result = MagicMock()
        mock_result.stdout = "NVEncC (x64) 7.72 (r2856)"
        with patch("furnace.adapters.nvencc.subprocess.run", return_value=mock_result):
            version = adapter._get_version()
        assert version == "7.72"

    def test_version_cached(self) -> None:
        adapter = _adapter()
        mock_result = MagicMock()
        mock_result.stdout = "NVEncC (x64) 7.72 (r2856)"
        with patch("furnace.adapters.nvencc.subprocess.run", return_value=mock_result) as mock_run:
            v1 = adapter._get_version()
            v2 = adapter._get_version()
        assert v1 == v2
        mock_run.assert_called_once()

    def test_version_oserror_returns_empty(self) -> None:
        adapter = _adapter()
        with patch("furnace.adapters.nvencc.subprocess.run", side_effect=OSError("not found")):
            version = adapter._get_version()
        assert version == ""


class TestNVEncCSetLogDir:
    def test_set_log_dir(self, tmp_path: Path) -> None:
        adapter = _adapter()
        adapter.set_log_dir(tmp_path)
        assert adapter._log_dir == tmp_path


class TestNVEncCEncode:
    """encode() execution with mocked run_tool."""

    def test_encode_returns_result(self) -> None:
        adapter = _adapter()
        vp = _make_vp()

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            return 0, ""

        with patch("furnace.adapters.nvencc.subprocess.run") as mock_sub:
            mock_sub.return_value.stdout = "NVEncC (x64) 7.72 (r2856)"
            with patch("furnace.adapters.nvencc.run_tool", side_effect=fake_run_tool):
                result = adapter.encode(Path("input.mkv"), Path("output.hevc"), vp)
        assert result.return_code == 0
        assert "hevc_nvenc" in result.encoder_settings

    def test_encode_ssim_parsing(self) -> None:
        adapter = _adapter()
        vp = _make_vp()

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_output is not None:
                on_output("SSIM YUV   All: 0.9842")
            return 0, ""

        with patch("furnace.adapters.nvencc.subprocess.run") as mock_sub:
            mock_sub.return_value.stdout = ""
            with patch("furnace.adapters.nvencc.run_tool", side_effect=fake_run_tool):
                result = adapter.encode(Path("input.mkv"), Path("output.hevc"), vp)
        assert result.ssim_score is not None
        assert abs(result.ssim_score - 0.9842) < 0.001

    def test_encode_vmaf_parsing(self) -> None:
        adapter = _adapter()
        vp = _make_vp()

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_output is not None:
                on_output("VMAF Score 95.31")
            return 0, ""

        with patch("furnace.adapters.nvencc.subprocess.run") as mock_sub:
            mock_sub.return_value.stdout = ""
            with patch("furnace.adapters.nvencc.run_tool", side_effect=fake_run_tool):
                result = adapter.encode(Path("input.mkv"), Path("output.hevc"), vp)
        assert result.vmaf_score is not None
        assert abs(result.vmaf_score - 95.31) < 0.01

    def test_encode_progress_callback(self) -> None:
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
                on_progress_line("[50.0%] 1000 frames: 48.0 fps")
            return 0, ""

        with patch("furnace.adapters.nvencc.subprocess.run") as mock_sub:
            mock_sub.return_value.stdout = ""
            with patch("furnace.adapters.nvencc.run_tool", side_effect=fake_run_tool):
                adapter.encode(Path("input.mkv"), Path("output.hevc"), vp, on_progress=samples.append)
        assert len(samples) == 1
        assert abs(samples[0].fraction - 0.5) < 0.01  # type: ignore[operator]

    def test_encode_log_path(self, tmp_path: Path) -> None:
        adapter = NVEncCAdapter(Path("NVEncC64.exe"), log_dir=tmp_path)
        vp = _make_vp()
        captured_kwargs: dict[str, Any] = {}

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            captured_kwargs["log_path"] = log_path
            return 0, ""

        with patch("furnace.adapters.nvencc.subprocess.run") as mock_sub:
            mock_sub.return_value.stdout = ""
            with patch("furnace.adapters.nvencc.run_tool", side_effect=fake_run_tool):
                adapter.encode(Path("input.mkv"), Path("output.hevc"), vp)
        assert captured_kwargs["log_path"] == tmp_path / "nvencc_encode.log"

    def test_encode_no_on_output(self) -> None:
        """Adapter without on_output: SSIM/VMAF lines still parsed via internal callback."""
        adapter = NVEncCAdapter(Path("NVEncC64.exe"), on_output=None)
        vp = _make_vp()

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            # Trigger the on_output callback with SSIM and VMAF lines
            if on_output is not None:
                on_output("SSIM YUV   All: 0.9900")
                on_output("VMAF Score 96.50")
            return 0, ""

        with patch("furnace.adapters.nvencc.subprocess.run") as mock_sub:
            mock_sub.return_value.stdout = ""
            with patch("furnace.adapters.nvencc.run_tool", side_effect=fake_run_tool):
                result = adapter.encode(Path("in.mkv"), Path("out.hevc"), vp)
        assert result.ssim_score is not None
        assert result.vmaf_score is not None

    def test_encode_no_on_progress(self) -> None:
        """encode without on_progress: progress line still consumed, no callback."""
        adapter = _adapter()
        vp = _make_vp()

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                consumed = on_progress_line("[50.0%] frames")
                assert consumed is True
            return 0, ""

        with patch("furnace.adapters.nvencc.subprocess.run") as mock_sub:
            mock_sub.return_value.stdout = ""
            with patch("furnace.adapters.nvencc.run_tool", side_effect=fake_run_tool):
                result = adapter.encode(Path("in.mkv"), Path("out.hevc"), vp, on_progress=None)
        assert result.return_code == 0

    def test_encode_on_output_non_metric_lines(self) -> None:
        """Lines without SSIM/VMAF keywords don't set scores."""
        output_lines: list[str] = []
        adapter = NVEncCAdapter(Path("NVEncC64.exe"), on_output=output_lines.append)
        vp = _make_vp()

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_output is not None:
                on_output("encode started")
                on_output("encode finished")
            return 0, ""

        with patch("furnace.adapters.nvencc.subprocess.run") as mock_sub:
            mock_sub.return_value.stdout = ""
            with patch("furnace.adapters.nvencc.run_tool", side_effect=fake_run_tool):
                result = adapter.encode(Path("in.mkv"), Path("out.hevc"), vp)
        assert result.ssim_score is None
        assert result.vmaf_score is None
        assert "encode started" in output_lines

    def test_encode_non_progress_line_not_consumed(self) -> None:
        """Non-progress lines return False from the progress closure."""
        adapter = _adapter()
        vp = _make_vp()
        results: list[bool] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                results.append(on_progress_line("not a progress line"))
            return 0, ""

        with patch("furnace.adapters.nvencc.subprocess.run") as mock_sub:
            mock_sub.return_value.stdout = ""
            with patch("furnace.adapters.nvencc.run_tool", side_effect=fake_run_tool):
                adapter.encode(Path("in.mkv"), Path("out.hevc"), vp)
        assert results == [False]

    def test_encode_ssim_line_without_numeric_skipped(self) -> None:
        """Line contains 'SSIM' keyword but numeric regex fails → score stays None."""
        adapter = _adapter()
        vp = _make_vp()

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_output is not None:
                # Keyword present, but no `All: <number>` match.
                on_output("SSIM YUV  All: N/A")
            return 0, ""

        with patch("furnace.adapters.nvencc.subprocess.run") as mock_sub:
            mock_sub.return_value.stdout = ""
            with patch("furnace.adapters.nvencc.run_tool", side_effect=fake_run_tool):
                result = adapter.encode(Path("in.mkv"), Path("out.hevc"), vp)
        assert result.ssim_score is None

    def test_encode_vmaf_line_without_numeric_skipped(self) -> None:
        """Line contains 'VMAF' keyword but numeric regex fails → score stays None."""
        adapter = _adapter()
        vp = _make_vp()

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_output is not None:
                # Keyword present, but no `VMAF Score <number>` match.
                on_output("VMAF Score calculation skipped")
            return 0, ""

        with patch("furnace.adapters.nvencc.subprocess.run") as mock_sub:
            mock_sub.return_value.stdout = ""
            with patch("furnace.adapters.nvencc.run_tool", side_effect=fake_run_tool):
                result = adapter.encode(Path("in.mkv"), Path("out.hevc"), vp)
        assert result.vmaf_score is None
