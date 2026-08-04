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


ALIGNMENT = 8


def align_down(value: int) -> int:
    return value - value % ALIGNMENT


def _aligned_edge(offset: int, trim: int) -> int:
    # An edge sitting at 0 has no bar to stay centred against, and moving it
    # would cost NVDEC hardware decode, so the whole trim goes to the far edge.
    if trim == 0 or offset == 0:
        return offset
    # Otherwise split the trim across both edges, rounding up to an even offset:
    # 4:2:0 chroma is half resolution and has no sample at an odd one, and
    # rounding down would pull the edge back over a row the crop had excluded.
    return (offset + trim // 2 + 1) & ~1


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


def aligned_crop(vp: VideoParams) -> CropRect | None:
    """The rectangle the encoder should cut out, mod-8 trim included.

    Encoders want dimensions on an 8px grid. Taking the odd few pixels off the
    crop costs nothing; rescaling the frame to reach the same size resamples
    every pixel and skews the aspect, so the trim belongs here and not in a
    resize. Returns None when nothing needs cutting.

    On anamorphic sources correct_sar stretches exactly one axis to reach
    square pixels. That axis is left alone: the resize has to land it on the
    grid anyway, and pre-trimming only moves which multiple of 8 it lands on,
    which the resize then bakes into the aspect -- measured at up to 2.5% of
    the display aspect on NTSC 4:3. The other axis is not resampled at all, so
    it gets the same treatment as a square-pixel source.
    """
    source = CropRect(w=vp.source_width, h=vp.source_height, x=0, y=0)
    crop = vp.crop if vp.crop is not None else source
    trim_w = crop.w % ALIGNMENT if vp.sar_num <= vp.sar_den else 0
    trim_h = crop.h % ALIGNMENT if vp.sar_num >= vp.sar_den else 0
    aligned = CropRect(
        w=crop.w - trim_w,
        h=crop.h - trim_h,
        x=_aligned_edge(crop.x, trim_w),
        y=_aligned_edge(crop.y, trim_h),
    )
    return None if aligned == source else aligned


def final_output_dimensions(vp: VideoParams) -> tuple[int, int]:
    crop = aligned_crop(vp)
    cur_w = crop.w if crop is not None else vp.source_width
    cur_h = crop.h if crop is not None else vp.source_height
    if vp.sar_num == vp.sar_den:
        return cur_w, cur_h
    scaled_w, scaled_h = correct_sar(cur_w, cur_h, vp.sar_num, vp.sar_den)
    return align_down(scaled_w), align_down(scaled_h)
