from __future__ import annotations

from pathlib import Path

from furnace.core.detect import is_dvd_resolution
from furnace.core.models import VideoInfo
from tests.conftest import make_video_info


def _make_dvd_video(interlaced: bool = True) -> VideoInfo:
    return make_video_info(
        codec_name="mpeg2video", width=720, height=576,
        fps_num=25, fps_den=1,
        interlaced=interlaced, color_matrix_raw="bt470bg",
        color_primaries="bt470bg",
        source_file=Path("/src/dvd.mkv"),
        bitrate=6_000_000,
    )


def _make_hd_video() -> VideoInfo:
    return make_video_info(
        width=1920, height=1080,
        fps_num=24000, fps_den=1001,
        duration_s=7200.0,
        source_file=Path("/src/hd.mkv"),
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
