from __future__ import annotations

from pathlib import Path

from furnace.core.models import (
    AudioCodecId,
    CropRect,
    Movie,
    TrackType,
)
from furnace.services.planner import PlannerService
from tests.conftest import make_movie, make_track, make_video_info


def _make_movie(tmp_path: Path, *, width: int = 1920, height: int = 1080) -> Movie:
    main = tmp_path / "movie.mkv"
    main.write_bytes(b"")
    return make_movie(
        main_file=main,
        video=make_video_info(
            width=width,
            height=height,
            source_file=main,
            bitrate=20_000_000,
        ),
        audio_tracks=[
            make_track(
                index=1,
                track_type=TrackType.AUDIO,
                codec_name="aac",
                codec_id=AudioCodecId.AAC_LC,
                language="eng",
                is_default=True,
                source_file=main,
                channels=2,
                bitrate=192_000,
            ),
        ],
    )


class TestPrecomputedCrop:
    def test_precomputed_crop_applied(self, tmp_path: Path) -> None:
        movie = _make_movie(tmp_path)
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            precomputed_crops={movie.main_file: CropRect(w=1920, h=800, x=0, y=140)},
        )

        vp = plan.jobs[0].video_params
        assert vp.crop is not None
        assert (vp.crop.w, vp.crop.h, vp.crop.x, vp.crop.y) == (1920, 800, 0, 140)

    def test_no_entry_for_file_means_no_crop(self, tmp_path: Path) -> None:
        movie = _make_movie(tmp_path)
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            precomputed_crops={Path("/other/film.mkv"): CropRect(w=10, h=10, x=0, y=0)},
        )

        assert plan.jobs[0].video_params.crop is None

    def test_precomputed_crops_none_means_no_crop(self, tmp_path: Path) -> None:
        movie = _make_movie(tmp_path)
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
        )

        assert plan.jobs[0].video_params.crop is None
