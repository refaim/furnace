from __future__ import annotations

from furnace.adapters._geometry import DEINTERLACE_FILTERS, build_vf, geometry_filters
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

    def test_off_grid_crop_is_absorbed_not_scaled(self) -> None:
        vp = _make_vp(crop=CropRect(w=1910, h=798, x=5, y=141))
        assert geometry_filters(vp) == ["crop=1904:792:8:144"]

    def test_off_grid_anamorphic_crop_is_left_for_the_scale(self) -> None:
        vp = _make_vp(
            source_width=720,
            source_height=480,
            crop=CropRect(w=720, h=434, x=0, y=22),
            sar_num=8,
            sar_den=9,
        )
        assert geometry_filters(vp) == ["crop=720:434:0:22", "scale=720:488:flags=spline"]

    def test_off_grid_source_without_crop_is_cropped(self) -> None:
        vp = _make_vp(source_width=1920, source_height=1078)
        assert geometry_filters(vp) == ["crop=1920:1072:0:0"]

    def test_deinterlace_runs_first(self) -> None:
        vp = _make_vp(deinterlace=True, crop=CropRect(w=1920, h=800, x=0, y=140))
        geom = geometry_filters(vp)
        assert geom[0] == "idet"
        assert geom[1].startswith("bwdif=")
        assert geom[2].startswith("crop=")

    def test_deinterlace_only_touches_frames_idet_flags_as_interlaced(self) -> None:
        # A DVD can be 25p film for 85 minutes and 50i video for the last ten.
        # Blanket bwdif softens the film part; idet tags each frame and bwdif
        # then leaves the progressive ones alone.
        geom = geometry_filters(_make_vp(deinterlace=True))
        assert geom == list(DEINTERLACE_FILTERS)
        assert geom.index("idet") < next(i for i, f in enumerate(geom) if f.startswith("bwdif="))
        bwdif = geom[-1]
        assert "deint=interlaced" in bwdif
        assert "mode=send_frame" in bwdif
        assert "parity=auto" in bwdif, "field order comes from idet per frame, not from the container"

    def test_deinterlace_absent_when_not_requested(self) -> None:
        assert "idet" not in geometry_filters(_make_vp(deinterlace=False))

    def test_full_frame_crop_is_dropped_as_a_no_op(self) -> None:
        vp = _make_vp(crop=CropRect(w=1920, h=1080, x=0, y=0))
        assert geometry_filters(vp) == []

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
