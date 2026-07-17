from __future__ import annotations

from furnace.adapters._geometry import build_vf, geometry_filters
from furnace.core.models import CropRect, VideoParams


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
        cq=23,
        crop=crop,
        deinterlace=deinterlace,
        color_matrix="bt709",
        color_range="tv",
        color_transfer="bt709",
        color_primaries="bt709",
        hdr=None,
        gop=120,
        fps_num=24000,
        fps_den=1001,
        source_width=source_width,
        source_height=source_height,
        source_codec="mpeg2video",
        source_bitrate=8_000_000,
        sar_num=sar_num,
        sar_den=sar_den,
        grain=True,
    )


class TestGeometryFilters:
    def test_empty_for_plain_square_pixel(self) -> None:
        assert geometry_filters(_make_vp()) == []

    def test_crop_and_anamorphic_scale(self) -> None:
        vp = _make_vp(crop=CropRect(w=1910, h=798, x=5, y=141))
        assert geometry_filters(vp) == ["crop=1910:798:5:141", "scale=1904:792:flags=spline"]

    def test_deinterlace_runs_first(self) -> None:
        vp = _make_vp(deinterlace=True, crop=CropRect(w=1920, h=800, x=0, y=140))
        geom = geometry_filters(vp)
        assert geom[0] == "bwdif=send_frame"
        assert geom[1].startswith("crop=")

    def test_scale_only_when_size_changes(self) -> None:
        vp = _make_vp(crop=CropRect(w=1920, h=1080, x=0, y=0))
        assert geometry_filters(vp) == ["crop=1920:1080:0:0"]

    def test_anamorphic_scale_without_crop(self) -> None:
        vp = _make_vp(source_width=720, source_height=576, sar_num=16, sar_den=15)
        assert geometry_filters(vp) == ["scale=768:576:flags=spline"]


class TestBuildVf:
    def test_tail_only_when_no_geometry(self) -> None:
        assert build_vf(_make_vp()) == "format=yuv420p10le,setsar=1"

    def test_geometry_then_fixed_tail(self) -> None:
        vp = _make_vp(deinterlace=True, crop=CropRect(w=1920, h=800, x=0, y=140))
        assert build_vf(vp) == ",".join(
            [*geometry_filters(vp), "format=yuv420p10le", "setsar=1"],
        )
