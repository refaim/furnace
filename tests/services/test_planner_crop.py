from __future__ import annotations

from pathlib import Path

from furnace.core.detect import is_dvd_resolution
from furnace.core.models import (
    HdrMetadata,
    VideoInfo,
)


def _make_dvd_video(interlaced: bool = True) -> VideoInfo:
    return VideoInfo(
        index=0, codec_name="mpeg2video", width=720, height=576,
        pixel_area=720 * 576, fps_num=25, fps_den=1,
        duration_s=5400.0, interlaced=interlaced, color_matrix_raw="bt470bg",
        color_range="tv", color_transfer="bt709", color_primaries="bt470bg",
        pix_fmt="yuv420p", hdr=HdrMetadata(), source_file=Path("/src/dvd.mkv"),
        bitrate=6_000_000,
    )


def _make_hd_video() -> VideoInfo:
    return VideoInfo(
        index=0, codec_name="h264", width=1920, height=1080,
        pixel_area=1920 * 1080, fps_num=24000, fps_den=1001,
        duration_s=7200.0, interlaced=False, color_matrix_raw="bt709",
        color_range="tv", color_transfer="bt709", color_primaries="bt709",
        pix_fmt="yuv420p", hdr=HdrMetadata(), source_file=Path("/src/hd.mkv"),
        bitrate=20_000_000,
    )


class TestCropDetectDvdFlags:
    def test_dvd_interlaced_is_dvd(self) -> None:
        """DVD interlaced source -> is_dvd_resolution returns True."""
        video = _make_dvd_video(interlaced=True)
        assert is_dvd_resolution(video.width, video.height) is True

    def test_hd_source_not_dvd(self) -> None:
        """HD source -> is_dvd_resolution returns False."""
        video = _make_hd_video()
        assert is_dvd_resolution(video.width, video.height) is False
