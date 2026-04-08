from __future__ import annotations

import math

from .models import CropRect

# CQ anchors по спецификации Furnace (отличаются от Crucible)
# Примечание: CQ anchors намеренно отличаются от Crucible.
# Crucible использовал CRF для x264/x265 (software), Furnace использует CQ для NVENC (hardware).
# NVENC CQ шкала не эквивалентна CRF -- значения подобраны для hevc_nvenc с preset p5 + UHQ tune.
CQ_ANCHORS: list[tuple[int, int]] = [
    (409_920, 22),    # SD    854x480
    (921_600, 24),    # 720p  1280x720
    (2_073_600, 25),  # 1080p 1920x1080
    (3_686_400, 28),  # 1440p 2560x1440
    (8_294_400, 31),  # 4K    3840x2160
]


def interpolate_cq(pixel_area: int) -> int:
    """Линейная интерполяция CQ по площади пикселей."""
    if pixel_area <= CQ_ANCHORS[0][0]:
        return CQ_ANCHORS[0][1]
    if pixel_area >= CQ_ANCHORS[-1][0]:
        return CQ_ANCHORS[-1][1]
    for i in range(len(CQ_ANCHORS) - 1):
        x0, y0 = CQ_ANCHORS[i]
        x1, y1 = CQ_ANCHORS[i + 1]
        if x0 <= pixel_area <= x1:
            t = (pixel_area - x0) / (x1 - x0)
            return round(y0 + t * (y1 - y0))
    return CQ_ANCHORS[-1][1]


def calculate_gop(fps_num: int, fps_den: int) -> int:
    """GOP = ceil(fps) * 5 (5-секундный интервал ключевых кадров)."""
    return math.ceil(fps_num / fps_den) * 5


def align_dimensions(w: int, h: int, x: int = 0, y: int = 0) -> CropRect:
    """Align dimensions to multiples of 8 (HEVC CU alignment).

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


