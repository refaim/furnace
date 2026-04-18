"""Downmix target enum and channel-count landmarks.

Split out of :mod:`furnace.core.models` to break the circular dependency
between :mod:`furnace.core.audio_profile` (needs ``DownmixMode``) and
:mod:`furnace.core.models` (needs ``AudioProfile`` on ``Track``).

``models`` re-exports these names so existing call sites keep working.
"""

from __future__ import annotations

from enum import StrEnum

STEREO_CHANNELS = 2
SURROUND_5_1_CHANNELS = 6


class DownmixMode(StrEnum):
    """Audio downmix target.

    STEREO -> eac3to: -downStereo  (multichannel -> 2.0 AAC)
    MONO   -> ffmpeg pan filter    (stereo/5.1/7.1 -> 1.0 AAC; bypasses eac3to)
    DOWN6  -> eac3to: -down6       (7.1/6.1 -> 5.1 AAC)
    """

    STEREO = "stereo"
    MONO = "mono"
    DOWN6 = "down6"
