from __future__ import annotations

from furnace.core.models import VideoParams
from furnace.core.quality import aligned_crop, final_output_dimensions

# A disc can mix 25p film with 50i video inserts. idet tags every frame with
# what it actually is (and which field leads), so bwdif reconstructs only the
# combed frames and passes the progressive majority through untouched.
DEINTERLACE_FILTERS: tuple[str, ...] = (
    "idet",
    "bwdif=mode=send_frame:parity=auto:deint=interlaced",
)


def geometry_filters(vp: VideoParams) -> list[str]:
    parts: list[str] = []

    if vp.deinterlace:
        parts.extend(DEINTERLACE_FILTERS)

    crop = aligned_crop(vp)
    if crop is not None:
        parts.append(f"crop={crop.w}:{crop.h}:{crop.x}:{crop.y}")

    final_w, final_h = final_output_dimensions(vp)
    pre_w = crop.w if crop is not None else vp.source_width
    pre_h = crop.h if crop is not None else vp.source_height
    if (final_w, final_h) != (pre_w, pre_h):
        parts.append(f"scale={final_w}:{final_h}:flags=spline")

    return parts


def build_vf(vp: VideoParams) -> str:
    parts = [*geometry_filters(vp), "format=yuv420p10le", "setsar=1"]
    return ",".join(parts)
