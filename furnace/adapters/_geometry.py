"""Shared ffmpeg geometry filtergraph for the SVT-AV1 grain path.

The SVT-AV1 encode and the perceptual-metrics *reference* must apply
byte-identical geometry (deinterlace -> crop -> scale) so the reference lines up
frame-for-frame, pixel-for-pixel, with the encoded output. A crop that forces a
2-D anamorphic scale otherwise phase-shifts a separately-built reference against
the ffmpeg encode and collapses SSIMULACRA2 (the reference resampler and the
encode resampler disagree by a fraction of a pixel). Both the encoder
(:mod:`furnace.adapters.svtav1`) and the reference builder
(:meth:`furnace.adapters.ffmpeg.FFmpegAdapter.build_reference`) derive their
``-vf`` from THIS single source of truth, so they can never drift.
"""
from __future__ import annotations

from furnace.core.models import VideoParams
from furnace.core.quality import final_output_dimensions


def geometry_filters(vp: VideoParams) -> list[str]:
    """The deinterlace -> crop -> scale prefix (geometry only, no format/SAR tail).

    Order matters: deinterlace on fields first, then crop, then a single
    high-quality rescale -- emitted ONLY when the final encoded size differs from
    the post-crop (pre-resize) size, so a source that needs no scaling gets no
    scale filter at all (and neither the encode nor the reference resamples it).
    """
    parts: list[str] = []

    # Deinterlace first -- must run on interlaced fields before any spatial op.
    # send_frame = SINGLE-RATE (one output frame per input frame), matching
    # NVEncC's nnedi. bwdif's default (send_field) is double-rate: 2 frames per
    # frame, which would desync against the single-rate --default-duration the
    # executor pins at mux time (video would play at half speed).
    if vp.deinterlace:
        parts.append("bwdif=send_frame")

    if vp.crop is not None:
        parts.append(f"crop={vp.crop.w}:{vp.crop.h}:{vp.crop.x}:{vp.crop.y}")

    # Single source of truth for the encoded size (crop -> SAR -> mod-8).
    final_w, final_h = final_output_dimensions(vp)
    pre_w = vp.crop.w if vp.crop is not None else vp.source_width
    pre_h = vp.crop.h if vp.crop is not None else vp.source_height
    if (final_w, final_h) != (pre_w, pre_h):
        parts.append(f"scale={final_w}:{final_h}:flags=spline")

    return parts


def build_vf(vp: VideoParams) -> str:
    """The full ffmpeg ``-vf`` string (comma-joined).

    The geometry prefix (:func:`geometry_filters`) followed by the fixed 10-bit /
    square-SAR tail that every SVT-AV1 encode -- and every matching lossless
    reference -- needs.
    """
    parts = [*geometry_filters(vp), "format=yuv420p10le", "setsar=1"]
    return ",".join(parts)
