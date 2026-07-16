"""CICP (ITU-T H.273 / ISO/IEC 23091-2) code points for color metadata.

Single source of truth for the integer code points shared by the two places
that stamp color onto the output: the Matroska container tagger
(:mod:`furnace.adapters.mkvmerge`) and the AV1 bitstream encoder
(:mod:`furnace.adapters.svtav1`, via ``svtav1-params``). Keys are furnace's
resolved color-value names (see :func:`furnace.core.detect.resolve_color_metadata`);
values are the CICP enumerations used verbatim by both Matroska colour elements
and the AV1 ``color_config``.

Range is deliberately absent: the CICP ``video_full_range_flag`` (0 studio /
1 full) and Matroska's ``Range`` enum (1 broadcast / 2 full) disagree, so each
adapter maps range with its own container-specific table.
"""

from __future__ import annotations

# ColourPrimaries (H.273 Table 2)
CICP_PRIMARIES: dict[str, int] = {
    "bt709": 1,
    "bt470m": 4,
    "bt470bg": 5,
    "smpte170m": 6,
    "smpte240m": 7,
    "bt2020": 9,
}

# TransferCharacteristics (H.273 Table 3)
# bt2020-10/bt2020-12 are the SDR BT.2020 curves (functionally BT.709's, at 10/12
# bit). They reach the AV1 encoder now that grainy HD/UHD takes the SVT-AV1 grain
# path -- SDR BT.2020 is real there, and ``_color_svtav1_params`` raises on any
# transfer missing from this table.
CICP_TRANSFER: dict[str, int] = {
    "bt709": 1,
    "bt470m": 4,
    "bt470bg": 5,
    "smpte170m": 6,
    "smpte240m": 7,
    "linear": 8,
    "bt2020-10": 14,
    "bt2020-12": 15,
    "smpte2084": 16,  # HDR10 / PQ
    "arib-std-b67": 18,  # HLG
}

# MatrixCoefficients (H.273 Table 4)
CICP_MATRIX: dict[str, int] = {
    "bt709": 1,
    "bt470bg": 5,
    "smpte170m": 6,
    "smpte240m": 7,
    "bt2020nc": 9,
    "bt2020c": 10,
}
