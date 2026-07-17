from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass

from .detect import _HD_MIN_HEIGHT, _NTSC_HEIGHTS

_SOFT_QVBR_MIN_HEIGHT = 1440


class Severity(enum.Enum):
    SYNC = ("SYNC", 0)
    FOREIGN = ("FOREIGN", 1)
    QUALITY = ("QUALITY", 2)
    COSMETIC = ("COSMETIC", 3)
    UNREADABLE = ("UNREADABLE", 4)

    def __init__(self, label: str, order: int) -> None:
        self.label = label
        self.order = order


class Fix(enum.Enum):
    NONE = ("—", 0)
    REMUX = ("REMUX", 1)
    RE_ENCODE = ("RE-ENCODE", 2)
    RE_RUN = ("RE-RUN", 3)

    def __init__(self, label: str, strength: int) -> None:
        self.label = label
        self.strength = strength


class EncoderFamily(enum.Enum):
    HEVC_NVENC = "hevc_nvenc"
    AV1_NVENC = "av1_nvenc"
    AV1_SVT = "av1_svt"
    PASSTHROUGH = "passthrough"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Defect:
    reason: str
    severity: Severity
    fix: Fix


_CROP_4PX_FIXED = (2, 1, 2)
_FPS_DRIFT_FIXED = (2, 1, 4)
_SOFT_TELECINE_FIXED = (2, 6, 0)
_SOFT_QVBR_FIXED = (2, 2, 0)
_GRAIN_LOSS_FIXED = (2, 7, 0)
_COLOR_TAGS_FIXED = (2, 7, 2)
_MONO_DOWNMIX_VERSION = (2, 0, 0)


def _matrix_absent(color_matrix: str | None) -> bool:
    return not color_matrix or color_matrix == "unknown"


def classify_outdated(
    *,
    unreadable: bool,
    version: tuple[int, int, int] | None,
    encoder_family: EncoderFamily,
    codec: str | None,
    height: int | None,
    color_matrix: str | None,
    audio_channels: Sequence[int | None],
) -> tuple[Defect, ...]:
    if unreadable:
        return (Defect("unreadable", Severity.UNREADABLE, Fix.NONE),)

    if version is None:
        return (Defect(codec or "unknown", Severity.FOREIGN, Fix.RE_ENCODE),)

    if encoder_family is EncoderFamily.HEVC_NVENC:
        return (Defect("superseded codec", Severity.QUALITY, Fix.RE_ENCODE),)

    defects: list[Defect] = []
    is_av1 = encoder_family in (EncoderFamily.AV1_NVENC, EncoderFamily.AV1_SVT)

    if is_av1 and version < _CROP_4PX_FIXED:
        defects.append(Defect("crop 4px", Severity.QUALITY, Fix.RE_ENCODE))

    if is_av1 and version < _FPS_DRIFT_FIXED:
        defects.append(Defect("fps drift", Severity.SYNC, Fix.REMUX))

    if (
        encoder_family is EncoderFamily.AV1_NVENC
        and _FPS_DRIFT_FIXED <= version < _SOFT_TELECINE_FIXED
        and height in _NTSC_HEIGHTS
    ):
        defects.append(Defect("soft telecine", Severity.SYNC, Fix.REMUX))

    if (
        encoder_family is EncoderFamily.AV1_NVENC
        and version < _SOFT_QVBR_FIXED
        and height is not None
        and height >= _SOFT_QVBR_MIN_HEIGHT
    ):
        defects.append(Defect("soft QVBR", Severity.QUALITY, Fix.RE_ENCODE))

    if (
        encoder_family is EncoderFamily.AV1_NVENC
        and version < _GRAIN_LOSS_FIXED
        and height is not None
        and height < _HD_MIN_HEIGHT
    ):
        defects.append(Defect("grain loss", Severity.QUALITY, Fix.RE_ENCODE))

    if version < _COLOR_TAGS_FIXED and _matrix_absent(color_matrix):
        defects.append(Defect("color tags", Severity.COSMETIC, Fix.REMUX))

    if version == _MONO_DOWNMIX_VERSION and any(ch == 1 for ch in audio_channels):
        defects.append(Defect("mono downmix", Severity.QUALITY, Fix.RE_RUN))

    return tuple(sorted(defects, key=lambda d: d.severity.order))


def row_severity(defects: Sequence[Defect]) -> Severity:
    return min((d.severity for d in defects), key=lambda s: s.order, default=Severity.UNREADABLE)


def row_fix(defects: Sequence[Defect]) -> Fix:
    return max((d.fix for d in defects), key=lambda f: f.strength, default=Fix.NONE)
