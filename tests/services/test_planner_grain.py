"""Tests for the grain-override plumbing: grain_overrides as explicit parameter.

Mirrors ``test_planner_sar.py``. The planner threads a per-file grain decision
from analysis into ``VideoParams.grain``:

- an explicit override (``grain_overrides[source_file]``) wins when present;
- otherwise the analyzer verdict ``video.grainy`` is used;
- a passthrough job (stream copy) forces ``grain=False`` — nothing to encode.

The movie object is never mutated (``video.grainy`` stays put).
"""
from __future__ import annotations

from pathlib import Path

from furnace.core.models import (
    AudioCodecId,
    HdrMetadata,
    Movie,
    TrackType,
)
from furnace.services.planner import PlannerService
from tests.conftest import make_movie, make_track, make_video_info


def _make_movie_grain(tmp_path: Path, *, grainy: bool, interlaced: bool = False) -> Movie:
    main = tmp_path / "movie.mkv"
    main.write_bytes(b"")
    return make_movie(
        main_file=main,
        video=make_video_info(
            codec_name="mpeg2video",
            width=720, height=480,
            fps_num=30000, fps_den=1001,
            color_matrix_raw="smpte170m",
            color_transfer="smpte170m",
            color_primaries="smpte170m",
            pix_fmt="yuv420p",
            source_file=main,
            bitrate=8_000_000,
            grainy=grainy,
            interlaced=interlaced,
        ),
        audio_tracks=[make_track(
            index=1, track_type=TrackType.AUDIO, codec_name="ac3",
            codec_id=AudioCodecId.AC3, language="eng",
            is_default=True, source_file=main,
            channels=2, bitrate=192_000,
        )],
    )


def _make_movie_hdr(
    tmp_path: Path, *, grainy: bool, color_transfer: str | None,
) -> Movie:
    """A 4K HDR remux: mastering-display metadata present. ``color_transfer=None``
    reproduces the real shape mkvmerge writes — ffprobe omits the transfer field
    entirely, and ``resolve_color_metadata`` only then promotes it to PQ."""
    main = tmp_path / "movie.mkv"
    main.write_bytes(b"")
    return make_movie(
        main_file=main,
        video=make_video_info(
            codec_name="hevc",
            width=3840, height=2160,
            fps_num=24000, fps_den=1001,
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
        audio_tracks=[make_track(
            index=1, track_type=TrackType.AUDIO, codec_name="ac3",
            codec_id=AudioCodecId.AC3, language="eng",
            is_default=True, source_file=main,
            channels=2, bitrate=192_000,
        )],
    )


class TestGrainIsSdrOnly:
    """The grain path scores with SSIMULACRA2, which cannot score PQ/HLG, so
    ``resolve_target`` refuses grain+HDR. The planner is the ONLY point where the
    grain decision and the RESOLVED transfer are both known, so it enforces the
    invariant there — the analyzer's probe gate sees the source's RAW transfer and
    a manual override bypasses that gate entirely."""

    def test_untagged_hdr_remux_is_not_grain(self, tmp_path: Path) -> None:
        """THE regression: an HDR remux with NO transfer tag passes the analyzer's
        raw-transfer gate (None reads SDR) and reads GRAINY, but becomes PQ here at
        resolve_color_metadata. Without this guard the job dies in resolve_target."""
        movie = _make_movie_hdr(tmp_path, grainy=True, color_transfer=None)
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
        )

        vp = plan.jobs[0].video_params
        assert vp.color_transfer == "smpte2084"  # promoted from an absent tag
        assert vp.grain is False

    def test_tagged_hdr_is_not_grain(self, tmp_path: Path) -> None:
        movie = _make_movie_hdr(tmp_path, grainy=True, color_transfer="smpte2084")
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
        )

        assert plan.jobs[0].video_params.grain is False

    def test_manual_grain_override_cannot_force_grain_on_hdr(self, tmp_path: Path) -> None:
        """An explicit override normally wins — but not into an unencodable state."""
        movie = _make_movie_hdr(tmp_path, grainy=False, color_transfer=None)
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            grain_overrides={movie.main_file: True},
        )

        assert plan.jobs[0].video_params.grain is False

    def test_hdr_grain_job_resolves_a_target(self, tmp_path: Path) -> None:
        """End-to-end proof the guard closes the hole: the built VideoParams must
        be acceptable to resolve_target, which raises on grain+HDR."""
        from furnace.core.target_quality import resolve_target

        movie = _make_movie_hdr(tmp_path, grainy=True, color_transfer=None)
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
        )

        spec = resolve_target(plan.jobs[0].video_params)  # would raise if grain stuck
        assert spec.metric == "cvvdp"

    def test_sdr_hd_grain_survives(self, tmp_path: Path) -> None:
        """The guard is HDR-specific: grainy HD SDR still takes the grain path."""
        movie = _make_movie_grain(tmp_path, grainy=True)
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
        )

        assert plan.jobs[0].video_params.grain is True


class TestGrainOverrides:
    def test_grainy_verdict_no_override_true(self, tmp_path: Path) -> None:
        """Grainy verdict with no override -> grain preserved."""
        movie = _make_movie_grain(tmp_path, grainy=True)
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
        )

        assert plan.jobs[0].video_params.grain is True

    def test_override_false_beats_grainy_verdict(self, tmp_path: Path) -> None:
        """An explicit False override wins over a grainy verdict."""
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
        """An explicit True override wins over a clean verdict (inverse case)."""
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
        """Clean verdict with no override -> no grain tuning."""
        movie = _make_movie_grain(tmp_path, grainy=False)
        planner = PlannerService(previewer=None)

        plan = planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
        )

        assert plan.jobs[0].video_params.grain is False

    def test_grain_overrides_none_behaves_as_empty(self, tmp_path: Path) -> None:
        """Omitting grain_overrides (None default) falls back to the verdict."""
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
        """A passthrough job copies the stream verbatim -> grain is always False."""
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
        """Passthrough beats even an explicit True override (nothing to encode)."""
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
        """Regression guard: the planner must NOT mutate movie.video.grainy."""
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
        """Interlaced + grain is supported: the metric reference is deinterlaced by
        the encode's own ffmpeg bwdif, so the job plans with both grain and
        deinterlace set."""
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
        """The guard is grain-specific: interlaced clean content still plans (nvenc deint)."""
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
