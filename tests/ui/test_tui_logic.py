"""Tests for extracted business-logic helpers in furnace.ui.tui."""
from __future__ import annotations

from pathlib import Path

import pytest

from furnace.core.models import CropRect, DownmixMode
from furnace.ui.tui import build_downmix_map, parse_crop_value
from tests.conftest import make_track

# ---------------------------------------------------------------------------
# parse_crop_value
# ---------------------------------------------------------------------------


class TestParseCropValue:
    def test_valid_crop(self) -> None:
        result = parse_crop_value("1920:800:0:140", 1920, 1080)
        assert result == CropRect(w=1920, h=800, x=0, y=140)

    def test_wrong_field_count(self) -> None:
        with pytest.raises(ValueError, match="w:h:x:y"):
            parse_crop_value("1920:800:0", 1920, 1080)

    def test_non_integer(self) -> None:
        with pytest.raises(ValueError, match="integers"):
            parse_crop_value("1920:abc:0:0", 1920, 1080)

    def test_zero_width(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            parse_crop_value("0:800:0:0", 1920, 1080)

    def test_negative_offset(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            parse_crop_value("1920:800:-1:0", 1920, 1080)

    def test_exceeds_source(self) -> None:
        with pytest.raises(ValueError, match="exceeds"):
            parse_crop_value("1920:800:0:500", 1920, 1080)

    def test_exact_fit(self) -> None:
        result = parse_crop_value("1920:1080:0:0", 1920, 1080)
        assert result == CropRect(w=1920, h=1080, x=0, y=0)


# ---------------------------------------------------------------------------
# build_downmix_map
# ---------------------------------------------------------------------------


class TestBuildDownmixMap:
    def test_selected_with_downmix_in_map(self) -> None:
        track = make_track(index=1, source_file=Path("/src/movie.mkv"))
        result = build_downmix_map(
            [track],
            [True],
            [DownmixMode.STEREO],
        )
        assert result == {(Path("/src/movie.mkv"), 1): DownmixMode.STEREO}

    def test_unselected_with_downmix_not_in_map(self) -> None:
        track = make_track(index=1, source_file=Path("/src/movie.mkv"))
        result = build_downmix_map(
            [track],
            [False],
            [DownmixMode.STEREO],
        )
        assert result == {}

    def test_selected_without_downmix_not_in_map(self) -> None:
        track = make_track(index=1, source_file=Path("/src/movie.mkv"))
        result = build_downmix_map(
            [track],
            [True],
            [None],
        )
        assert result == {}

    def test_multiple_tracks_mixed(self) -> None:
        tracks = [
            make_track(index=1, source_file=Path("/src/a.mkv")),
            make_track(index=2, source_file=Path("/src/b.mkv")),
            make_track(index=3, source_file=Path("/src/c.mkv")),
        ]
        selected = [True, False, True]
        downmix_list: list[DownmixMode | None] = [
            DownmixMode.STEREO,
            DownmixMode.DOWN6,
            None,
        ]
        result = build_downmix_map(tracks, selected, downmix_list)
        # track 0: selected + stereo -> in map
        # track 1: unselected -> not in map
        # track 2: selected but None downmix -> not in map
        assert result == {(Path("/src/a.mkv"), 1): DownmixMode.STEREO}
