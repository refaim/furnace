from __future__ import annotations

import bisect
import math

from .models import CropRect, VideoParams

# QVBR anchors for NVENC AV1 (hardware), preset P4 + UHQ tune. NVENC QVBR is a
# constant-quality control (no CRF on NVENC); the value is passed verbatim as
# --qvbr. Preset and anchors are coupled -- re-tuning one means re-tuning the
# other.
#
# The 1440p/4K anchors were eased in 2.2.0 (1440p 38->35, 4K 41->34). The old
# curve grew *more* aggressive with resolution (1080p 36 -> 4K 41); on hard
# content (dark, grainy) that was below transparent. Measured VMAF-vs-qvbr on a
# dark 3840x1920 clip (RTX 5060 Ti, NVEncC 9.22, vmaf_v0.6.1): qvbr 40 -> 93.5,
# 36 -> 94.7, 33 -> 95.3, 31 -> 95.6, 30 -> 95.8 (so the old 4K anchor 41 sat
# under ~93.5). The knee is ~qvbr 34 (interpolated between the measured 36 and
# 33 points), so 4K-class content now targets ~VMAF 95 on its worst scenes
# (near-transparent) at ~2x the old bitrate. Result: 4K is no longer harsher
# than 1080p (the curve is a hump, not monotone). 1080p and below are unchanged.
CQ_ANCHORS: list[tuple[int, int]] = [
    (409_920, 35),  # SD    854x480
    (921_600, 35),  # 720p  1280x720
    (2_073_600, 36),  # 1080p 1920x1080
    (3_686_400, 35),  # 1440p 2560x1440
    (8_294_400, 34),  # 4K    3840x2160
]


def interpolate_cq(pixel_area: int) -> int:
    """Linear interpolation of CQ by pixel area."""
    if pixel_area <= CQ_ANCHORS[0][0]:
        return CQ_ANCHORS[0][1]
    if pixel_area >= CQ_ANCHORS[-1][0]:
        return CQ_ANCHORS[-1][1]
    xs = [a[0] for a in CQ_ANCHORS]
    i = bisect.bisect_left(xs, pixel_area)
    x0, y0 = CQ_ANCHORS[i - 1]
    x1, y1 = CQ_ANCHORS[i]
    t = (pixel_area - x0) / (x1 - x0)
    return round(y0 + t * (y1 - y0))


def calculate_gop(fps_num: int, fps_den: int) -> int:
    """GOP = ceil(fps) * 5 (5-second keyframe interval)."""
    return math.ceil(fps_num / fps_den) * 5


def align_dimensions(w: int, h: int, x: int = 0, y: int = 0) -> CropRect:
    """Align dimensions to multiples of 8.

    AV1 only requires mod-2, but mod-8 is a safe superset kept from the prior
    HEVC pipeline so existing output dimensions are unchanged.

    Trims symmetrically: excess pixels split evenly to offset.
    """
    trim_w = w % 8
    trim_h = h % 8
    return CropRect(
        w=w - trim_w,
        h=h - trim_h,
        x=x + trim_w // 2,
        y=y + trim_h // 2,
    )


def correct_sar(width: int, height: int, sar_num: int, sar_den: int) -> tuple[int, int]:
    """Correct non-square pixel aspect ratio by scaling up the smaller dimension.

    Returns (display_width, display_height) with square pixels.
    """
    if sar_num == sar_den:
        return width, height
    if sar_num > sar_den:
        return round(width * sar_num / sar_den), height
    return width, round(height * sar_den / sar_num)


def final_output_dimensions(vp: VideoParams) -> tuple[int, int]:
    """Return the actual (width, height) that will be encoded in the AV1 track.

    Pipeline: crop (if set) -> SAR correction (if non-square) -> mod-8
    alignment. This is the single source of truth -- UI labels, plan-log
    summaries and the NVEncC ``--output-res`` flag all derive from here.
    """
    cur_w = vp.crop.w if vp.crop is not None else vp.source_width
    cur_h = vp.crop.h if vp.crop is not None else vp.source_height
    if vp.sar_num != vp.sar_den:
        cur_w, cur_h = correct_sar(cur_w, cur_h, vp.sar_num, vp.sar_den)
    aligned = align_dimensions(cur_w, cur_h)
    return aligned.w, aligned.h
