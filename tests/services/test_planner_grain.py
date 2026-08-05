from __future__ import annotations

from pathlib import Path

from furnace.core.models import (
    AudioCodecId,
    HdrMetadata,
    Movie,
    TrackType,
    VideoParams,
)
from furnace.services.planner import PlannerService
from tests.conftest import make_movie, make_track, make_video_info


def _make_movie_grain(
    tmp_path: Path,
    *,
    grainy: bool,
    interlaced: bool = False,
    width: int = 720,
    height: int = 480,
) -> Movie:
    main = tmp_path / "movie.mkv"
    main.write_bytes(b"")
    return make_movie(
        main_file=main,
        video=make_video_info(
            codec_name="mpeg2video",
            width=width,
            height=height,
            fps_num=30000,
            fps_den=1001,
            color_matrix_raw="smpte170m",
            color_transfer="smpte170m",
            color_primaries="smpte170m",
            pix_fmt="yuv420p",
            source_file=main,
            bitrate=8_000_000,
            grainy=grainy,
            interlaced=interlaced,
        ),
        audio_tracks=[
            make_track(
                index=1,
                track_type=TrackType.AUDIO,
                codec_name="ac3",
                codec_id=AudioCodecId.AC3,
                language="eng",
                is_default=True,
                source_file=main,
                channels=2,
                bitrate=192_000,
            )
        ],
    )


def _make_movie_hdr(
    tmp_path: Path,
    *,
    grainy: bool,
    color_transfer: str | None,
    width: int = 3840,
    height: int = 2160,
) -> Movie:
    main = tmp_path / "movie.mkv"
    main.write_bytes(b"")
    return make_movie(
        main_file=main,
        video=make_video_info(
            codec_name="hevc",
            width=width,
            height=height,
            fps_num=24000,
            fps_den=1001,
            color_matrix_raw="bt2020nc",
            color_transfer=color_transfer,
            color_primaries="bt2020",
            pix_fmt="yuv420p10le",
            source_file=main,
            bitrate=60_000_000,
            grainy=grainy,
            hdr=HdrMetadata(
                mastering_display="G(...)B(...)R(...)WP(...)L(...)",
                content_light="MaxCLL=1000,MaxFALL=400",
            ),
        ),
        audio_tracks=[
            make_track(
                index=1,
                track_type=TrackType.AUDIO,
                codec_name="ac3",
                codec_id=AudioCodecId.AC3,
                language="eng",
                is_default=True,
                source_file=main,
                channels=2,
                bitrate=192_000,
            )
        ],
    )


class TestGrainOnHdr:
    def _plan_one(
        self,
        tmp_path: Path,
        movie: Movie,
        **kwargs: object,
    ) -> VideoParams:
        planner = PlannerService(previewer=None)
        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            **kwargs,  # type: ignore[arg-type]
        )
        return plan.jobs[0].video_params

    def test_tagged_hdr_hd_keeps_grain(self, tmp_path: Path) -> None:
        movie = _make_movie_hdr(tmp_path, grainy=True, color_transfer="smpte2084")

        assert self._plan_one(tmp_path, movie).grain is True

    def test_tagged_hlg_hd_keeps_grain(self, tmp_path: Path) -> None:
        movie = _make_movie_hdr(tmp_path, grainy=True, color_transfer="arib-std-b67")

        assert self._plan_one(tmp_path, movie).grain is True

    def test_grainy_hdr_hd_encodes_at_the_fixed_knob(self, tmp_path: Path) -> None:
        from furnace.core.target_quality import fixed_grain_knob

        movie = _make_movie_hdr(tmp_path, grainy=True, color_transfer="smpte2084")

        assert fixed_grain_knob(self._plan_one(tmp_path, movie)) == 32

    def test_untagged_hdr_remux_is_not_grain(self, tmp_path: Path) -> None:
        movie = _make_movie_hdr(tmp_path, grainy=True, color_transfer=None)

        vp = self._plan_one(tmp_path, movie)
        assert vp.color_transfer == "smpte2084"
        assert vp.grain is False

    def test_hdr_sd_is_not_grain(self, tmp_path: Path) -> None:
        movie = _make_movie_hdr(
            tmp_path,
            grainy=True,
            color_transfer="smpte2084",
            width=720,
            height=576,
        )

        assert self._plan_one(tmp_path, movie).grain is False

    def test_manual_grain_override_forces_grain_on_tagged_hdr(self, tmp_path: Path) -> None:
        movie = _make_movie_hdr(tmp_path, grainy=False, color_transfer="smpte2084")

        vp = self._plan_one(tmp_path, movie, grain_overrides={movie.main_file: True})
        assert vp.grain is True

    def test_manual_grain_override_beats_the_untagged_hdr_distrust(self, tmp_path: Path) -> None:
        movie = _make_movie_hdr(tmp_path, grainy=False, color_transfer=None)

        vp = self._plan_one(tmp_path, movie, grain_overrides={movie.main_file: True})
        assert vp.grain is True

    def test_manual_grain_override_cannot_force_grain_on_hdr_sd(self, tmp_path: Path) -> None:
        movie = _make_movie_hdr(
            tmp_path,
            grainy=False,
            color_transfer=None,
            width=720,
            height=576,
        )

        vp = self._plan_one(tmp_path, movie, grain_overrides={movie.main_file: True})
        assert vp.grain is False

    def test_manual_grain_override_off_beats_a_grainy_hdr_verdict(self, tmp_path: Path) -> None:
        movie = _make_movie_hdr(tmp_path, grainy=True, color_transfer="smpte2084")

        vp = self._plan_one(tmp_path, movie, grain_overrides={movie.main_file: False})
        assert vp.grain is False

    def test_clean_hdr_job_resolves_a_cvvdp_target(self, tmp_path: Path) -> None:
        from furnace.core.target_quality import resolve_target

        movie = _make_movie_hdr(tmp_path, grainy=False, color_transfer="smpte2084")

        spec = resolve_target(self._plan_one(tmp_path, movie))
        assert spec.metric == "cvvdp"

    def test_passthrough_hdr_grain_stays_off(self, tmp_path: Path) -> None:
        movie = _make_movie_hdr(tmp_path, grainy=True, color_transfer="smpte2084")

        vp = self._plan_one(tmp_path, movie, copy_video=True)
        assert vp.passthrough is True
        assert vp.grain is False

    def test_sdr_sd_grain_survives(self, tmp_path: Path) -> None:
        movie = _make_movie_grain(tmp_path, grainy=True)

        assert self._plan_one(tmp_path, movie).grain is True

    def test_sdr_hd_grain_survives_on_the_fixed_knob(self, tmp_path: Path) -> None:
        from furnace.core.target_quality import fixed_grain_knob

        movie = _make_movie_grain(tmp_path, grainy=True, width=1920, height=1080)

        vp = self._plan_one(tmp_path, movie)
        assert vp.grain is True
        assert fixed_grain_knob(vp) == 32


class TestGrainOverrides:
    def test_grainy_verdict_no_override_true(self, tmp_path: Path) -> None:
        movie = _make_movie_grain(tmp_path, grainy=True)
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
        )

        assert plan.jobs[0].video_params.grain is True

    def test_override_false_beats_grainy_verdict(self, tmp_path: Path) -> None:
        movie = _make_movie_grain(tmp_path, grainy=True)
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            grain_overrides={movie.main_file: False},
        )

        assert plan.jobs[0].video_params.grain is False

    def test_override_true_beats_clean_verdict(self, tmp_path: Path) -> None:
        movie = _make_movie_grain(tmp_path, grainy=False)
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            grain_overrides={movie.main_file: True},
        )

        assert plan.jobs[0].video_params.grain is True

    def test_clean_verdict_no_override_false(self, tmp_path: Path) -> None:
        movie = _make_movie_grain(tmp_path, grainy=False)
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
        )

        assert plan.jobs[0].video_params.grain is False

    def test_grain_overrides_none_behaves_as_empty(self, tmp_path: Path) -> None:
        movie = _make_movie_grain(tmp_path, grainy=True)
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            grain_overrides=None,
        )

        assert plan.jobs[0].video_params.grain is True

    def test_passthrough_forces_grain_false_even_when_grainy(self, tmp_path: Path) -> None:
        movie = _make_movie_grain(tmp_path, grainy=True)
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            copy_video=True,
        )

        vp = plan.jobs[0].video_params
        assert vp.passthrough is True
        assert vp.grain is False

    def test_passthrough_forces_grain_false_even_with_override_true(self, tmp_path: Path) -> None:
        movie = _make_movie_grain(tmp_path, grainy=False)
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            grain_overrides={movie.main_file: True},
            copy_video=True,
        )

        vp = plan.jobs[0].video_params
        assert vp.passthrough is True
        assert vp.grain is False

    def test_movie_video_grainy_not_mutated_by_planner(self, tmp_path: Path) -> None:
        movie = _make_movie_grain(tmp_path, grainy=True)
        planner = PlannerService(previewer=None)

        planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            grain_overrides={movie.main_file: False},
        )

        assert movie.video.grainy is True

    def test_interlaced_grain_plans_normally(self, tmp_path: Path) -> None:
        movie = _make_movie_grain(tmp_path, grainy=True, interlaced=True)
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
        )

        vp = plan.jobs[0].video_params
        assert vp.grain is True
        assert vp.deinterlace is True

    def test_interlaced_without_grain_plans_normally(self, tmp_path: Path) -> None:
        movie = _make_movie_grain(tmp_path, grainy=False, interlaced=True)
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
        )

        vp = plan.jobs[0].video_params
        assert vp.deinterlace is True
        assert vp.grain is False
