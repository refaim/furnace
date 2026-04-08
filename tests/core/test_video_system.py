from __future__ import annotations

import pytest

from furnace.core.detect import VideoSystem, detect_video_system


class TestDetectVideoSystem:
    # PAL standard heights
    def test_pal_576(self) -> None:
        assert detect_video_system(576) == VideoSystem.PAL

    def test_pal_288(self) -> None:
        assert detect_video_system(288) == VideoSystem.PAL

    # NTSC standard heights
    def test_ntsc_480(self) -> None:
        assert detect_video_system(480) == VideoSystem.NTSC

    def test_ntsc_486(self) -> None:
        assert detect_video_system(486) == VideoSystem.NTSC

    def test_ntsc_240(self) -> None:
        assert detect_video_system(240) == VideoSystem.NTSC

    # HD heights
    def test_hd_720(self) -> None:
        assert detect_video_system(720) == VideoSystem.HD

    def test_hd_1080(self) -> None:
        assert detect_video_system(1080) == VideoSystem.HD

    def test_hd_2160(self) -> None:
        assert detect_video_system(2160) == VideoSystem.HD

    # Non-standard SD -> ValueError
    def test_unknown_sd_544(self) -> None:
        with pytest.raises(ValueError):
            detect_video_system(544)

    def test_unknown_sd_352(self) -> None:
        with pytest.raises(ValueError):
            detect_video_system(352)

    def test_unknown_sd_360(self) -> None:
        with pytest.raises(ValueError):
            detect_video_system(360)
