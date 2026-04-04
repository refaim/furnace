"""Tests for FFmpegAdapter._build_encode_cmd — CUDA vs CPU path selection."""
from __future__ import annotations

from pathlib import Path

from furnace.adapters.ffmpeg import FFmpegAdapter, _build_encoder_settings
from furnace.adapters.mkvpropedit import _build_tags_xml
from furnace.core.models import ColorSpace, CropRect, HdrMetadata, VideoParams


def _make_vp(
    source_codec: str = "h264",
    crop: CropRect | None = None,
    deinterlace: bool = False,
    cq: int = 25,
) -> VideoParams:
    return VideoParams(
        cq=cq,
        crop=crop,
        deinterlace=deinterlace,
        color_space=ColorSpace.BT709,
        color_range="tv",
        color_transfer="bt709",
        color_primaries="bt709",
        hdr=None,
        gop=120,
        fps_num=24000,
        fps_den=1001,
        source_width=1920,
        source_height=1080,
        source_codec=source_codec,
    )


def _build(vp: VideoParams) -> list[str]:
    adapter = FFmpegAdapter(Path("ffmpeg.exe"), Path("ffprobe.exe"))
    use_cuda = adapter._should_use_cuda(vp)
    return adapter._build_encode_cmd(Path("input.mkv"), Path("output.mkv"), vp, use_cuda=use_cuda)


class TestCudaPathSelection:
    """CUDA decode used when source_codec is supported AND no crop."""

    def test_h264_no_crop_uses_cuda(self) -> None:
        cmd = _build(_make_vp(source_codec="h264", crop=None))
        assert "-hwaccel" in cmd
        assert "cuda" in cmd

    def test_hevc_no_crop_uses_cuda(self) -> None:
        cmd = _build(_make_vp(source_codec="hevc", crop=None))
        assert "-hwaccel" in cmd

    def test_mpeg2video_no_crop_uses_cuda(self) -> None:
        cmd = _build(_make_vp(source_codec="mpeg2video", crop=None))
        assert "-hwaccel" in cmd

    def test_vp9_no_crop_uses_cuda(self) -> None:
        cmd = _build(_make_vp(source_codec="vp9", crop=None))
        assert "-hwaccel" in cmd

    def test_unknown_codec_uses_cpu(self) -> None:
        cmd = _build(_make_vp(source_codec="theora", crop=None))
        assert "-hwaccel" not in cmd

    def test_h264_with_crop_uses_cpu(self) -> None:
        """Crop requires CPU path because crop_cuda is not available."""
        crop = CropRect(w=1920, h=800, x=0, y=140)
        cmd = _build(_make_vp(source_codec="h264", crop=crop))
        assert "-hwaccel" not in cmd


class TestCudaPathCommand:
    """Verify CUDA path command structure."""

    def test_cuda_has_hwaccel_output_format(self) -> None:
        cmd = _build(_make_vp(source_codec="h264"))
        idx = cmd.index("-hwaccel_output_format")
        assert cmd[idx + 1] == "cuda"

    def test_cuda_no_pix_fmt(self) -> None:
        """CUDA path should NOT have -pix_fmt (GPU handles format)."""
        cmd = _build(_make_vp(source_codec="h264"))
        assert "-pix_fmt" not in cmd

    def test_cuda_deinterlace_uses_bwdif_cuda(self) -> None:
        cmd = _build(_make_vp(source_codec="h264", deinterlace=True))
        vf_idx = cmd.index("-vf")
        vf_value = cmd[vf_idx + 1]
        assert "bwdif_cuda" in vf_value

    def test_cuda_no_filters_when_not_needed(self) -> None:
        cmd = _build(_make_vp(source_codec="h264", deinterlace=False, crop=None))
        assert "-vf" not in cmd


class TestCpuPathCommand:
    """Verify CPU path command structure."""

    def test_cpu_has_pix_fmt_p010le(self) -> None:
        cmd = _build(_make_vp(source_codec="theora"))
        idx = cmd.index("-pix_fmt")
        assert cmd[idx + 1] == "p010le"

    def test_cpu_no_hwaccel(self) -> None:
        cmd = _build(_make_vp(source_codec="theora"))
        assert "-hwaccel" not in cmd
        assert "-hwaccel_output_format" not in cmd

    def test_cpu_deinterlace_uses_bwdif(self) -> None:
        """CPU path uses bwdif (not bwdif_cuda)."""
        cmd = _build(_make_vp(source_codec="theora", deinterlace=True))
        vf_idx = cmd.index("-vf")
        vf_value = cmd[vf_idx + 1]
        assert "bwdif=" in vf_value
        assert "bwdif_cuda" not in vf_value

    def test_cpu_crop(self) -> None:
        crop = CropRect(w=1920, h=800, x=0, y=140)
        cmd = _build(_make_vp(source_codec="h264", crop=crop))
        vf_idx = cmd.index("-vf")
        vf_value = cmd[vf_idx + 1]
        assert "crop=1920:800:0:140" in vf_value

    def test_cpu_deinterlace_and_crop_order(self) -> None:
        """Deinterlace first, then crop."""
        crop = CropRect(w=1920, h=800, x=0, y=140)
        cmd = _build(_make_vp(source_codec="h264", crop=crop, deinterlace=True))
        vf_idx = cmd.index("-vf")
        vf_value = cmd[vf_idx + 1]
        # bwdif should come before crop
        bwdif_pos = vf_value.index("bwdif")
        crop_pos = vf_value.index("crop=")
        assert bwdif_pos < crop_pos


class TestCommonFlags:
    """Flags present in both CUDA and CPU paths."""

    def test_hevc_nvenc(self) -> None:
        cmd = _build(_make_vp())
        assert "hevc_nvenc" in cmd

    def test_main10_profile(self) -> None:
        cmd = _build(_make_vp())
        idx = cmd.index("-profile:v")
        assert cmd[idx + 1] == "main10"

    def test_preset_p5(self) -> None:
        cmd = _build(_make_vp())
        idx = cmd.index("-preset")
        assert cmd[idx + 1] == "p5"

    def test_cq_value(self) -> None:
        cmd = _build(_make_vp(cq=28))
        idx = cmd.index("-cq")
        assert cmd[idx + 1] == "28"

    def test_progress_pipe(self) -> None:
        cmd = _build(_make_vp())
        assert "-progress" in cmd
        idx = cmd.index("-progress")
        assert cmd[idx + 1] == "pipe:1"


class TestBuildEncoderSettings:
    """ENCODER_SETTINGS string generation from VideoParams."""

    def test_no_filters(self) -> None:
        vp = _make_vp(cq=25)
        result = _build_encoder_settings(vp, use_cuda=True)
        assert result == (
            "hevc_nvenc / main10 / cq=25 / preset=p5 / tune=uhq / rc=vbr"
            " / spatial-aq=1 / temporal-aq=1 / rc-lookahead=32 / multipass=qres"
        )

    def test_ffmpeg_version(self) -> None:
        vp = _make_vp(cq=25)
        result = _build_encoder_settings(vp, use_cuda=True, ffmpeg_version="7.1")
        assert result.startswith("hevc_nvenc / ffmpeg=7.1 / main10")

    def test_ffmpeg_version_empty(self) -> None:
        vp = _make_vp(cq=25)
        result = _build_encoder_settings(vp, use_cuda=True, ffmpeg_version="")
        assert "ffmpeg=" not in result

    def test_deinterlace_cuda(self) -> None:
        vp = _make_vp(deinterlace=True)
        result = _build_encoder_settings(vp, use_cuda=True)
        assert result.endswith("multipass=qres / deinterlace=bwdif_cuda")

    def test_deinterlace_cpu(self) -> None:
        vp = _make_vp(deinterlace=True)
        result = _build_encoder_settings(vp, use_cuda=False)
        assert "deinterlace=bwdif" in result
        assert "bwdif_cuda" not in result

    def test_crop(self) -> None:
        """crop=T:B:L:R — pixels removed from each edge."""
        crop = CropRect(w=1920, h=816, x=0, y=132)
        vp = _make_vp(crop=crop)
        result = _build_encoder_settings(vp, use_cuda=False)
        assert "crop=132:132:0:0" in result

    def test_alignment(self) -> None:
        """align=WxH when dimensions aren't multiples of 8."""
        crop = CropRect(w=1916, h=1076, x=2, y=2)
        vp = _make_vp(crop=crop)
        result = _build_encoder_settings(vp, use_cuda=False)
        assert "crop=2:2:2:2" in result
        assert "align=1912x1072" in result

    def test_no_alignment_when_already_aligned(self) -> None:
        crop = CropRect(w=1920, h=816, x=0, y=132)
        vp = _make_vp(crop=crop)
        result = _build_encoder_settings(vp, use_cuda=False)
        assert "align=" not in result

    def test_all_filters(self) -> None:
        crop = CropRect(w=1916, h=1076, x=2, y=2)
        vp = _make_vp(deinterlace=True, crop=crop)
        result = _build_encoder_settings(vp, use_cuda=False)
        parts = result.split(" / ")
        assert "deinterlace=bwdif" in parts
        assert "crop=2:2:2:2" in parts
        assert "align=1912x1072" in parts

    def test_cq_value_varies(self) -> None:
        vp = _make_vp(cq=31)
        result = _build_encoder_settings(vp, use_cuda=True)
        assert "cq=31" in result


class TestBuildTagsXml:
    def test_encoder_only(self) -> None:
        xml = _build_tags_xml("Furnace v1.4.0")
        assert "<Name>ENCODER</Name>" in xml
        assert "<String>Furnace v1.4.0</String>" in xml
        assert "ENCODER_SETTINGS" not in xml

    def test_with_encoder_settings(self) -> None:
        xml = _build_tags_xml("Furnace v1.4.0", "hevc_nvenc / main10 / cq=25")
        assert "<Name>ENCODER</Name>" in xml
        assert "<Name>ENCODER_SETTINGS</Name>" in xml
        assert "<String>hevc_nvenc / main10 / cq=25</String>" in xml
