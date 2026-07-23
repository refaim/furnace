from __future__ import annotations

import bisect
import math

from .models import CropRect, VideoParams

CQ_ANCHORS: list[tuple[int, int]] = [
    (409_920, 35),
    (921_600, 35),
    (2_073_600, 36),
    (3_686_400, 35),
    (8_294_400, 34),
]


def interpolate_cq(pixel_area: int) -> int:
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
    return math.ceil(fps_num / fps_den) * 5


def align_dimensions(w: int, h: int, x: int = 0, y: int = 0) -> CropRect:
    trim_w = w % 8
    trim_h = h % 8
    return CropRect(
        w=w - trim_w,
        h=h - trim_h,
        x=x + trim_w // 2,
        y=y + trim_h // 2,
    )


def correct_sar(width: int, height: int, sar_num: int, sar_den: int) -> tuple[int, int]:
    if sar_num == sar_den:
        return width, height
    if sar_num > sar_den:
        return round(width * sar_num / sar_den), height
    return width, round(height * sar_den / sar_num)


def force_16_9_sar(width: int, height: int) -> tuple[int, int]:
    num = 16 * height
    den = 9 * width
    divisor = math.gcd(num, den)
    return num // divisor, den // divisor


def final_output_dimensions(vp: VideoParams) -> tuple[int, int]:
    cur_w = vp.crop.w if vp.crop is not None else vp.source_width
    cur_h = vp.crop.h if vp.crop is not None else vp.source_height
    if vp.sar_num != vp.sar_den:
        cur_w, cur_h = correct_sar(cur_w, cur_h, vp.sar_num, vp.sar_den)
    aligned = align_dimensions(cur_w, cur_h)
    return aligned.w, aligned.h
