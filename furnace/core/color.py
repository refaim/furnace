from __future__ import annotations

import re
from dataclasses import dataclass

CICP_PRIMARIES: dict[str, int] = {
    "bt709": 1,
    "bt470m": 4,
    "bt470bg": 5,
    "smpte170m": 6,
    "smpte240m": 7,
    "bt2020": 9,
}

CICP_TRANSFER: dict[str, int] = {
    "bt709": 1,
    "bt470m": 4,
    "bt470bg": 5,
    "smpte170m": 6,
    "smpte240m": 7,
    "linear": 8,
    "bt2020-10": 14,
    "bt2020-12": 15,
    "smpte2084": 16,
    "arib-std-b67": 18,
}

CICP_MATRIX: dict[str, int] = {
    "bt709": 1,
    "bt470bg": 5,
    "smpte170m": 6,
    "smpte240m": 7,
    "bt2020nc": 9,
    "bt2020c": 10,
}

_CHROMATICITY_UNITS = 50000
_LUMINANCE_UNITS = 10000

_MASTERING_DISPLAY_RE = re.compile(
    r"^G\((\d+),(\d+)\)"
    r"B\((\d+),(\d+)\)"
    r"R\((\d+),(\d+)\)"
    r"WP\((\d+),(\d+)\)"
    r"L\((\d+),(\d+)\)$"
)


@dataclass(frozen=True, slots=True)
class MasteringDisplay:
    red: tuple[float, float]
    green: tuple[float, float]
    blue: tuple[float, float]
    white: tuple[float, float]
    max_luminance: float
    min_luminance: float


def parse_mastering_display(value: str) -> MasteringDisplay:
    match = _MASTERING_DISPLAY_RE.match(value)
    if match is None:
        raise ValueError(f"unrecognized mastering display metadata {value!r}")
    gx, gy, bx, by, rx, ry, wx, wy, lmax, lmin = (int(g) for g in match.groups())
    return MasteringDisplay(
        red=(rx / _CHROMATICITY_UNITS, ry / _CHROMATICITY_UNITS),
        green=(gx / _CHROMATICITY_UNITS, gy / _CHROMATICITY_UNITS),
        blue=(bx / _CHROMATICITY_UNITS, by / _CHROMATICITY_UNITS),
        white=(wx / _CHROMATICITY_UNITS, wy / _CHROMATICITY_UNITS),
        max_luminance=lmax / _LUMINANCE_UNITS,
        min_luminance=lmin / _LUMINANCE_UNITS,
    )
