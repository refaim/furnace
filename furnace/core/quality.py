from __future__ import annotations

import math

from .models import ColorSpace, CropRect

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


def align_crop(w: int, h: int, x: int, y: int) -> CropRect:
    """Выравнивание crop по сетке 16x8 (требование NVENC).

    w округляется вниз до кратного 16, h -- до кратного 8.
    x и y корректируются чтобы центрировать обрезку: dw // 2, dh // 2.
    """
    dw = w % 16
    dh = h % 8
    new_w = w - dw
    new_h = h - dh
    new_x = x + dw // 2
    new_y = y + dh // 2
    return CropRect(w=new_w, h=new_h, x=new_x, y=new_y)


def determine_color_space(
    width: int, height: int, source_color_space: str | None
) -> ColorSpace:
    """Определяет цветовое пространство по параметрам видео.

    BT.2020 source -> BT.2020 (passthrough)
    HD (height >= 720) -> BT.709
    SD (height < 720) -> BT.601
    """
    if source_color_space in (ColorSpace.BT2020, ColorSpace.BT2020.value):
        return ColorSpace.BT2020
    if height >= 720:
        return ColorSpace.BT709
    return ColorSpace.BT601
