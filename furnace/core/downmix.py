from __future__ import annotations

from enum import StrEnum

STEREO_CHANNELS = 2
SURROUND_5_1_CHANNELS = 6


class DownmixMode(StrEnum):
    STEREO = "stereo"
    MONO = "mono"
    DOWN6 = "down6"
