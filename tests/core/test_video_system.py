from __future__ import annotations

import pytest

from furnace.core.detect import VideoSystem, detect_video_system


class TestDetectVideoSystem:
    def test_pal_576(self) -> None:
        assert detect_video_system(576, 25, 1) == VideoSystem.PAL

    def test_pal_288(self) -> None:
        assert detect_video_system(288, 25, 1) == VideoSystem.PAL

    def test_ntsc_480(self) -> None:
        assert detect_video_system(480, 30000, 1001) == VideoSystem.NTSC

    def test_ntsc_486(self) -> None:
        assert detect_video_system(486, 30000, 1001) == VideoSystem.NTSC

    def test_ntsc_240(self) -> None:
        assert detect_video_system(240, 30000, 1001) == VideoSystem.NTSC

    def test_hd_720(self) -> None:
        assert detect_video_system(720, 24, 1) == VideoSystem.HD

    def test_hd_1080(self) -> None:
        assert detect_video_system(1080, 24000, 1001) == VideoSystem.HD

    def test_hd_2160(self) -> None:
        assert detect_video_system(2160, 24000, 1001) == VideoSystem.HD

    def test_exact_pal_height_authoritative_over_ntsc_fps(self) -> None:
        assert detect_video_system(576, 30000, 1001) == VideoSystem.PAL

    def test_exact_ntsc_height_authoritative_over_pal_fps(self) -> None:
        assert detect_video_system(480, 25, 1) == VideoSystem.NTSC

    def test_nonstandard_height_pal_by_fps_25(self) -> None:
        assert detect_video_system(512, 25, 1) == VideoSystem.PAL

    def test_nonstandard_height_pal_by_fps_50(self) -> None:
        assert detect_video_system(512, 50, 1) == VideoSystem.PAL

    def test_nonstandard_height_ntsc_by_fps_23_976(self) -> None:
        assert detect_video_system(512, 24000, 1001) == VideoSystem.NTSC

    def test_nonstandard_height_ntsc_by_fps_24(self) -> None:
        assert detect_video_system(544, 24, 1) == VideoSystem.NTSC

    def test_nonstandard_height_ntsc_by_fps_29_97(self) -> None:
        assert detect_video_system(544, 30000, 1001) == VideoSystem.NTSC

    def test_nonstandard_height_ntsc_by_fps_30(self) -> None:
        assert detect_video_system(352, 30, 1) == VideoSystem.NTSC

    def test_nonstandard_height_ntsc_by_fps_60(self) -> None:
        assert detect_video_system(400, 60, 1) == VideoSystem.NTSC

    def test_unknown_height_and_fps_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be classified PAL/NTSC"):
            detect_video_system(544, 15, 1)

    def test_unknown_height_odd_fps_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be classified PAL/NTSC"):
            detect_video_system(360, 12, 1)
