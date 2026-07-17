from __future__ import annotations

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
