"""The ``furnace scan --outdated`` defect ledger (pure, no I/O).

Classifies one scanned file into the defects that make it a re-encode/remux
candidate: it was produced by a Furnace version with a known baked-in output
defect, or it is a foreign (non-Furnace) file. The ledger keys on two signals
already present in the single ffprobe call per file — the parsed Furnace
``ENCODER`` version and the ``ENCODER_SETTINGS`` encoder family — plus a few
stream fields (height, container matrix tag, audio channel counts).

This module is deliberately decoupled from :mod:`furnace.core.scan`: it takes
explicit primitive arguments rather than a ``ScanRow`` so the dependency only
ever flows scan -> outdated (no import cycle, no ``TYPE_CHECKING`` block, which
the project bans).
"""
from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass

from .detect import _HD_MIN_HEIGHT, _NTSC_HEIGHTS

# 1440p and up: the resolution band where the pre-2.2.0 QVBR anchor was too soft.
_SOFT_QVBR_MIN_HEIGHT = 1440


class Severity(enum.Enum):
    """Defect severity, ordered worst-first for the outdated work-list sort.

    ``order`` is the sort key (lower = worse, sorts first): a SYNC defect (audio
    drifts out of sync — the file is effectively unwatchable) outranks a FOREIGN
    file (not ours, but it plays), which outranks a QUALITY loss, then a COSMETIC
    tag fix, and finally UNREADABLE (kept only for visibility). ``label`` is the
    text shown in the table's Severity cell.
    """

    SYNC = ("SYNC", 0)
    FOREIGN = ("FOREIGN", 1)
    QUALITY = ("QUALITY", 2)
    COSMETIC = ("COSMETIC", 3)
    UNREADABLE = ("UNREADABLE", 4)

    def __init__(self, label: str, order: int) -> None:
        self.label = label
        self.order = order


class Fix(enum.Enum):
    """Remedy for a defect. ``strength`` orders the per-row rollup.

    A row's Fix cell shows the strongest remedy among its defects, where
    REMUX < RE-ENCODE < RE-RUN (a full re-run from source is the heaviest, a
    container remux the lightest). ``NONE`` (``—``) is reserved for unreadable
    rows, which carry no actionable remedy.
    """

    NONE = ("—", 0)
    REMUX = ("REMUX", 1)
    RE_ENCODE = ("RE-ENCODE", 2)
    RE_RUN = ("RE-RUN", 3)

    def __init__(self, label: str, strength: int) -> None:
        self.label = label
        self.strength = strength


class EncoderFamily(enum.Enum):
    """The encoder that produced a Furnace file, parsed from ENCODER_SETTINGS.

    The AV1-era ledger rules gate on ``AV1_NVENC``/``AV1_SVT`` (a real AV1
    encode), so an old ``HEVC_NVENC`` file never stacks them. ``PASSTHROUGH`` is
    a verbatim video copy; ``UNKNOWN`` is a missing or unrecognized tag.
    """

    HEVC_NVENC = "hevc_nvenc"
    AV1_NVENC = "av1_nvenc"
    AV1_SVT = "av1_svt"
    PASSTHROUGH = "passthrough"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Defect:
    """One reason a file is outdated: a display label, its severity and its fix."""

    reason: str
    severity: Severity
    fix: Fix


# AV1 encodes that predate these fixes need the noted remedy. Kept as constants
# so the boundaries read exactly like the ledger.
_CROP_4PX_FIXED = (2, 1, 2)  # crop rounded to 4px before this — visible edge loss
_FPS_DRIFT_FIXED = (2, 1, 4)  # container frame rate wrong before this — audio drift
_SOFT_TELECINE_FIXED = (2, 6, 0)  # NTSC soft-telecine handled from here
_SOFT_QVBR_FIXED = (2, 2, 0)  # QVBR anchor firmed up for >=1440p here
_GRAIN_LOSS_FIXED = (2, 7, 0)  # SD grain routed to SVT-AV1 from here
_COLOR_TAGS_FIXED = (2, 7, 2)  # container color metadata duplicated from here
_MONO_DOWNMIX_VERSION = (2, 0, 0)  # the single release that could mono-downmix


def _matrix_absent(color_matrix: str | None) -> bool:
    """True when the container carries no usable MatrixCoefficients tag.

    ffprobe surfaces the container matrix as the stream ``color_space`` field; a
    missing tag reads as ``None`` or an empty string, and some muxers write the
    literal ``"unknown"`` — all three mean the tag is effectively absent.
    """
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
    """Return every defect that makes one file an outdated re-encode candidate.

    A file may match several rules; all matches are collected and the result is
    returned sorted worst-first by severity (a stable sort, so equal-severity
    defects keep the ledger's own order). An empty tuple means the file is
    current and needs no attention.

    The parameters are deliberately flat keyword primitives rather than a
    ``ScanRow``: this keeps the ``scan -> outdated`` import one-directional (no
    ``ScanRow`` import here, so no import cycle and no banned ``TYPE_CHECKING``).
    """
    if unreadable:
        return (Defect("unreadable", Severity.UNREADABLE, Fix.NONE),)

    if version is None:
        # Foreign file: no valid Furnace tag. Reason is the video codec (or
        # "unknown" with no video stream); a re-encode brings it into the fold.
        return (Defect(codec or "unknown", Severity.FOREIGN, Fix.RE_ENCODE),)

    # superseded codec: a real HEVC encode subsumes every other defect. A
    # from-scratch RE-ENCODE fixes all latent problems (crop, fps, grain, color
    # tags, mono downmix) in one shot, so we report exactly this one reason and
    # skip the rest of the ledger. This early return is what enforces the
    # invariant "a HEVC encode shows ONLY superseded codec"; it applies to real
    # HEVC *encodes* only — a passthrough copy still runs the full ledger below
    # (e.g. a pre-2.7.2 passthrough with an absent matrix still gets color tags).
    if encoder_family is EncoderFamily.HEVC_NVENC:
        return (Defect("superseded codec", Severity.QUALITY, Fix.RE_ENCODE),)

    defects: list[Defect] = []
    is_av1 = encoder_family in (EncoderFamily.AV1_NVENC, EncoderFamily.AV1_SVT)

    # crop 4px: early AV1 rounded crop to a 4px grid, shaving picture edges.
    if is_av1 and version < _CROP_4PX_FIXED:
        defects.append(Defect("crop 4px", Severity.QUALITY, Fix.RE_ENCODE))

    # fps drift: early AV1 muxed the wrong container rate; audio drifts. A remux
    # with the correct rate is enough. Pre-2.1.4 an NTSC source is caught here,
    # which is why soft telecine below only starts at 2.1.4 (no double SYNC).
    if is_av1 and version < _FPS_DRIFT_FIXED:
        defects.append(Defect("fps drift", Severity.SYNC, Fix.REMUX))

    # soft telecine: NTSC-SD NVENC AV1 between the fps-drift fix and the proper
    # soft-telecine handling played at the wrong cadence; a remux corrects it.
    if (
        encoder_family is EncoderFamily.AV1_NVENC
        and _FPS_DRIFT_FIXED <= version < _SOFT_TELECINE_FIXED
        and height in _NTSC_HEIGHTS
    ):
        defects.append(Defect("soft telecine", Severity.SYNC, Fix.REMUX))

    # soft QVBR: the pre-2.2.0 NVENC AV1 QVBR anchor was too soft at >=1440p.
    if (
        encoder_family is EncoderFamily.AV1_NVENC
        and version < _SOFT_QVBR_FIXED
        and height is not None
        and height >= _SOFT_QVBR_MIN_HEIGHT
    ):
        defects.append(Defect("soft QVBR", Severity.QUALITY, Fix.RE_ENCODE))

    # grain loss: SD NVENC AV1 before grain routed to SVT-AV1 smeared film grain.
    if (
        encoder_family is EncoderFamily.AV1_NVENC
        and version < _GRAIN_LOSS_FIXED
        and height is not None
        and height < _HD_MIN_HEIGHT
    ):
        defects.append(Defect("grain loss", Severity.QUALITY, Fix.RE_ENCODE))

    # color tags: Furnace pre-2.7.2 did not duplicate the matrix tag at the
    # container level (a Kodi/TV compatibility fix). A remux re-tags it. Not
    # AV1-gated — a passthrough copy of that era needs it too.
    if version < _COLOR_TAGS_FIXED and _matrix_absent(color_matrix):
        defects.append(Defect("color tags", Severity.COSMETIC, Fix.REMUX))

    # mono downmix: v2.0.0 could collapse a stereo track to mono. Re-running from
    # source is the only fix. Pinned to exactly 2.0.0 — it WILL over-flag content
    # that was genuinely mono (indistinguishable in the output), but the tight
    # version pin keeps the blast radius to a single release, so that is accepted.
    if version == _MONO_DOWNMIX_VERSION and any(ch == 1 for ch in audio_channels):
        defects.append(Defect("mono downmix", Severity.QUALITY, Fix.RE_RUN))

    return tuple(sorted(defects, key=lambda d: d.severity.order))


def row_severity(defects: Sequence[Defect]) -> Severity:
    """The worst (highest-priority) severity among a row's defects.

    An outdated row always carries >=1 defect, but the public renderer accepts
    arbitrary rows, so an empty sequence falls back to ``UNREADABLE`` (lowest
    priority) rather than raising ``ValueError``.
    """
    return min((d.severity for d in defects), key=lambda s: s.order, default=Severity.UNREADABLE)


def row_fix(defects: Sequence[Defect]) -> Fix:
    """The strongest remedy among a row's defects (RE-RUN > RE-ENCODE > REMUX).

    An outdated row always carries >=1 defect, but the public renderer accepts
    arbitrary rows, so an empty sequence falls back to ``NONE`` (``—``) rather
    than raising ``ValueError``.
    """
    return max((d.fix for d in defects), key=lambda f: f.strength, default=Fix.NONE)
