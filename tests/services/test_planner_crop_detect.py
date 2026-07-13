"""Planner threads a precomputed crop map into each Job's VideoParams.

The planner no longer runs cropdetect itself (the prober dependency is gone);
it just looks up ``movie.main_file`` in the supplied ``precomputed_crops`` map.
"""
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


def _make_movie(tmp_path: Path, name: str = "movie.mkv", *, width: int = 1920, height: int = 1080) -> Movie:
    main = tmp_path / name
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


class TestPrecomputedCropMap:
    def test_real_crop_from_map_applied(self, tmp_path: Path) -> None:
        """A crop entry keyed by main_file is written into VideoParams verbatim."""
        movie = _make_movie(tmp_path)
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            precomputed_crops={movie.main_file: CropRect(w=1920, h=804, x=0, y=138)},
        )

        assert len(plan.jobs) == 1
        vp = plan.jobs[0].video_params
        assert vp.crop is not None
        assert (vp.crop.w, vp.crop.h, vp.crop.x, vp.crop.y) == (1920, 804, 0, 138)

    def test_crop_applied_only_to_matching_movie(self, tmp_path: Path) -> None:
        """In a multi-movie plan, each job gets only its own file's crop."""
        movie_a = _make_movie(tmp_path, "a.mkv")
        movie_b = _make_movie(tmp_path, "b.mkv")
        planner = PlannerService(previewer=None)

        crops = {movie_a.main_file: CropRect(w=1920, h=816, x=0, y=132)}
        plan = planner.create_plan(
            [(movie_a, tmp_path / "a_out.mkv"), (movie_b, tmp_path / "b_out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            precomputed_crops=crops,
        )

        crop_a = plan.jobs[0].video_params.crop
        assert crop_a is not None
        assert (crop_a.w, crop_a.h) == (1920, 816)
        # movie_b has no entry in the map -> no crop
        assert plan.jobs[1].video_params.crop is None

    def test_empty_map_means_no_crop(self, tmp_path: Path) -> None:
        """An empty crop map -> crop is None (no entry for the file)."""
        movie = _make_movie(tmp_path)
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            precomputed_crops={},
        )

        assert plan.jobs[0].video_params.crop is None

    def test_none_map_means_no_crop(self, tmp_path: Path) -> None:
        """precomputed_crops=None (default) -> crop is None."""
        movie = _make_movie(tmp_path)
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
        )

        assert plan.jobs[0].video_params.crop is None
