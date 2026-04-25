from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from furnace.core.models import DiscSource, DiscTitle, DiscType
from furnace.core.progress import ProgressSample
from furnace.services.disc_demuxer import DiscDemuxer
from tests.fakes.recording_reporter import RecordingPlanReporter

OnProgress = Callable[[ProgressSample], None] | None


def _make_disc(tmp_path: Path, name: str, dtype: DiscType) -> DiscSource:
    leaf = "BDMV" if dtype == DiscType.BLURAY else "VIDEO_TS"
    p = tmp_path / name / leaf
    p.mkdir(parents=True)
    return DiscSource(path=p, disc_type=dtype)


def test_cached_disc_emits_one_cached_event(tmp_path: Path) -> None:
    disc = _make_disc(tmp_path, "OldMatrix_BD", DiscType.BLURAY)
    title = DiscTitle(number=2, duration_s=3600.0, raw_label="2) 1:00:00")

    demux_dir = tmp_path / "demux"
    demux_dir.mkdir()
    # Pre-create the .done marker and the resulting MKV
    (demux_dir / "OldMatrix_BD_title_2.done").touch()
    (demux_dir / "OldMatrix_BD_title_2.mkv").write_bytes(b"\x00")

    bd_port = MagicMock()
    dvd_port = MagicMock()
    reporter = RecordingPlanReporter()
    demuxer = DiscDemuxer(
        bd_port=bd_port,
        dvd_port=dvd_port,
        mkvmerge_path=Path("mkvmerge"),
    )
    demuxer.demux(
        discs=[disc],
        selected_titles={disc: [title]},
        demux_dir=demux_dir,
        reporter=reporter,
    )
    methods = [e.method for e in reporter.events]
    assert methods == ["demux_disc_cached"]
    assert reporter.events[0].args == ("OldMatrix_BD",)


def test_fresh_dvd_title_emits_rip_only(tmp_path: Path) -> None:
    disc = _make_disc(tmp_path, "DirtyHarry_DVD", DiscType.DVD)
    title = DiscTitle(number=1, duration_s=3600.0, raw_label="1) 1:00:00")

    demux_dir = tmp_path / "demux"
    demux_dir.mkdir()

    dvd_port = MagicMock()

    def _fake_demux(
        disc_path: Path,
        title_num: int,
        output_dir: Path,
        on_progress: OnProgress = None,
    ) -> list[Path]:
        out = output_dir / "title.mkv"
        out.write_bytes(b"\x00")
        if on_progress is not None:
            on_progress(ProgressSample(fraction=0.5))
            on_progress(ProgressSample(fraction=1.0))
        return [out]

    dvd_port.demux_title.side_effect = _fake_demux

    reporter = RecordingPlanReporter()
    demuxer = DiscDemuxer(
        bd_port=MagicMock(),
        dvd_port=dvd_port,
        mkvmerge_path=Path("mkvmerge"),
    )
    demuxer.demux(
        discs=[disc],
        selected_titles={disc: [title]},
        demux_dir=demux_dir,
        reporter=reporter,
    )
    methods = [e.method for e in reporter.events]
    assert methods[0] == "demux_disc_start"
    assert methods[1] == "demux_title_start"
    triples = [(e.method, e.args, e.kwargs) for e in reporter.events]
    assert ("demux_title_substep", ("rip",), (("has_progress", True),)) in triples
    assert methods[-1] == "demux_title_done"
    # No remux for DVD (single MKV output)
    assert ("demux_title_substep", ("remux",), (("has_progress", True),)) not in triples


def test_demux_failure_emits_failed_then_propagates(tmp_path: Path) -> None:
    disc = _make_disc(tmp_path, "Broken_BD", DiscType.BLURAY)
    title = DiscTitle(number=5, duration_s=3600.0, raw_label="5) 1:00:00")
    demux_dir = tmp_path / "demux"
    demux_dir.mkdir()

    bd_port = MagicMock()
    bd_port.demux_title.side_effect = RuntimeError("eac3to crashed")

    reporter = RecordingPlanReporter()
    demuxer = DiscDemuxer(
        bd_port=bd_port,
        dvd_port=MagicMock(),
        mkvmerge_path=Path("mkvmerge"),
    )
    with pytest.raises(RuntimeError, match="eac3to crashed"):
        demuxer.demux(
            discs=[disc],
            selected_titles={disc: [title]},
            demux_dir=demux_dir,
            reporter=reporter,
        )
    methods = [e.method for e in reporter.events]
    assert "demux_title_failed" in methods


def test_demux_without_reporter_is_silent(tmp_path: Path) -> None:
    """Reporter is optional - pipeline still works without it."""
    disc = _make_disc(tmp_path, "X", DiscType.DVD)
    title = DiscTitle(number=1, duration_s=3600.0, raw_label="1) 1:00:00")
    demux_dir = tmp_path / "demux"
    demux_dir.mkdir()

    dvd_port = MagicMock()

    def _fake_demux(
        disc_path: Path,
        title_num: int,
        output_dir: Path,
        on_progress: OnProgress = None,
    ) -> list[Path]:
        out = output_dir / "x.mkv"
        out.write_bytes(b"\x00")
        return [out]

    dvd_port.demux_title.side_effect = _fake_demux

    demuxer = DiscDemuxer(
        bd_port=MagicMock(),
        dvd_port=dvd_port,
        mkvmerge_path=Path("mkvmerge"),
    )
    # Just don't crash
    demuxer.demux(
        discs=[disc],
        selected_titles={disc: [title]},
        demux_dir=demux_dir,
    )


def test_disc_with_one_cached_one_fresh_runs_disc_start(tmp_path: Path) -> None:
    """If one selected title is cached but another is not, the disc is NOT
    treated as fully cached: demux_disc_start is emitted and the fresh
    title runs through the rip flow.
    """
    disc = _make_disc(tmp_path, "MixedBD", DiscType.BLURAY)
    cached = DiscTitle(number=1, duration_s=3600.0, raw_label="1) 1:00:00")
    fresh = DiscTitle(number=2, duration_s=3600.0, raw_label="2) 1:00:00")

    demux_dir = tmp_path / "demux"
    demux_dir.mkdir()
    # Pre-create cached title
    (demux_dir / "MixedBD_title_1.done").touch()
    (demux_dir / "MixedBD_title_1.mkv").write_bytes(b"\x00")

    bd_port = MagicMock()

    def _fake_demux(
        disc_path: Path,
        title_num: int,
        output_dir: Path,
        on_progress: OnProgress = None,
    ) -> list[Path]:
        out = output_dir / "title.mkv"
        out.write_bytes(b"\x00")
        return [out]

    bd_port.demux_title.side_effect = _fake_demux

    reporter = RecordingPlanReporter()
    demuxer = DiscDemuxer(
        bd_port=bd_port,
        dvd_port=MagicMock(),
        mkvmerge_path=Path("mkvmerge"),
    )
    result = demuxer.demux(
        discs=[disc],
        selected_titles={disc: [cached, fresh]},
        demux_dir=demux_dir,
        reporter=reporter,
    )
    assert len(result) == 2
    methods = [e.method for e in reporter.events]
    # The cached title is silently included before the fresh one is demuxed.
    assert methods[0] == "demux_disc_start"
    # Only the fresh title (#2) gets demux_title_start
    title_starts = [e for e in reporter.events if e.method == "demux_title_start"]
    assert len(title_starts) == 1
    assert title_starts[0].args == (2,)
    assert "demux_title_done" in methods


def test_remux_substep_emitted_when_muxing_required(tmp_path: Path) -> None:
    """BD demux that returns multiple files triggers the 'remux' substep."""
    disc = _make_disc(tmp_path, "MultiBD", DiscType.BLURAY)
    title = DiscTitle(number=3, duration_s=3600.0, raw_label="3) 1:00:00")
    demux_dir = tmp_path / "demux"
    demux_dir.mkdir()

    bd_port = MagicMock()

    def _fake_demux(
        disc_path: Path,
        title_num: int,
        output_dir: Path,
        on_progress: OnProgress = None,
    ) -> list[Path]:
        video = output_dir / "video.h264"
        video.write_bytes(b"v")
        audio = output_dir / "audio [eng].ac3"
        audio.write_bytes(b"a")
        return [video, audio]

    bd_port.demux_title.side_effect = _fake_demux

    reporter = RecordingPlanReporter()
    demuxer = DiscDemuxer(
        bd_port=bd_port,
        dvd_port=MagicMock(),
        mkvmerge_path=Path("mkvmerge"),
    )
    from unittest.mock import patch

    with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")):
        demuxer.demux(
            discs=[disc],
            selected_titles={disc: [title]},
            demux_dir=demux_dir,
            reporter=reporter,
        )

    triples = [(e.method, e.args, e.kwargs) for e in reporter.events]
    assert ("demux_title_substep", ("remux",), (("has_progress", True),)) in triples
    assert ("demux_title_substep", ("rip",), (("has_progress", True),)) in triples
    methods = [e.method for e in reporter.events]
    assert methods[-1] == "demux_title_done"


def test_w64_transcode_emits_substeps_with_indices(tmp_path: Path) -> None:
    """Each .w64 file in demux output emits its own 'transcode N/M' substep
    with progress fractions forwarded to demux_title_progress.
    """
    disc = _make_disc(tmp_path, "WaveBD", DiscType.BLURAY)
    title = DiscTitle(number=4, duration_s=3600.0, raw_label="4) 1:00:00")
    demux_dir = tmp_path / "demux"
    demux_dir.mkdir()

    bd_port = MagicMock()

    def _fake_demux(
        disc_path: Path,
        title_num: int,
        output_dir: Path,
        on_progress: OnProgress = None,
    ) -> list[Path]:
        video = output_dir / "video.h264"
        video.write_bytes(b"v")
        a = output_dir / "audio1 [eng].w64"
        b = output_dir / "audio2 [rus].w64"
        a.write_bytes(b"a")
        b.write_bytes(b"b")
        return [video, a, b]

    bd_port.demux_title.side_effect = _fake_demux

    transcoder = MagicMock()

    def _fake_transcode(
        input_path: Path,
        output_path: Path,
        on_progress: OnProgress = None,
    ) -> int:
        output_path.write_bytes(b"flac")
        if on_progress is not None:
            # None fraction must be ignored, not crash
            on_progress(ProgressSample(fraction=None))
            on_progress(ProgressSample(fraction=0.7))
        return 0

    transcoder.transcode_to_flac.side_effect = _fake_transcode

    reporter = RecordingPlanReporter()
    demuxer = DiscDemuxer(
        bd_port=bd_port,
        dvd_port=MagicMock(),
        mkvmerge_path=Path("mkvmerge"),
        pcm_transcoder=transcoder,
    )
    from unittest.mock import patch

    with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")):
        demuxer.demux(
            discs=[disc],
            selected_titles={disc: [title]},
            demux_dir=demux_dir,
            reporter=reporter,
        )

    triples = [(e.method, e.args, e.kwargs) for e in reporter.events]
    assert (
        "demux_title_substep",
        ("transcode 1/2",),
        (("has_progress", True),),
    ) in triples
    assert (
        "demux_title_substep",
        ("transcode 2/2",),
        (("has_progress", True),),
    ) in triples
    # Per-step progress was forwarded
    progress_events = [e for e in reporter.events if e.method == "demux_title_progress"]
    assert any(e.args == (0.7,) for e in progress_events)
    methods = [e.method for e in reporter.events]
    assert methods[-1] == "demux_title_done"


def test_remux_forwards_mkvmerge_progress(tmp_path: Path) -> None:
    """Progress lines emitted by mkvmerge during the remux substep are
    parsed and forwarded as demux_title_progress events.
    """
    disc = _make_disc(tmp_path, "ProgBD", DiscType.BLURAY)
    title = DiscTitle(number=8, duration_s=3600.0, raw_label="8) 1:00:00")
    demux_dir = tmp_path / "demux"
    demux_dir.mkdir()

    bd_port = MagicMock()

    def _fake_demux(
        disc_path: Path,
        title_num: int,
        output_dir: Path,
        on_progress: OnProgress = None,
    ) -> list[Path]:
        video = output_dir / "video.h264"
        video.write_bytes(b"v")
        audio = output_dir / "audio [eng].ac3"
        audio.write_bytes(b"a")
        return [video, audio]

    bd_port.demux_title.side_effect = _fake_demux

    reporter = RecordingPlanReporter()
    demuxer = DiscDemuxer(
        bd_port=bd_port,
        dvd_port=MagicMock(),
        mkvmerge_path=Path("mkvmerge"),
    )

    def _fake_run_tool(
        cmd: list[str],
        on_output: Callable[[str], None] | None = None,
        on_progress_line: Callable[[str], bool] | None = None,
    ) -> tuple[int, str]:
        # mkvmerge would print progress and unrelated lines
        if on_progress_line is not None:
            assert on_progress_line("Progress: 25%") is True
            assert on_progress_line("garbage line") is False
            assert on_progress_line("Progress: 100%") is True
        return 0, ""

    from unittest.mock import patch

    with patch(
        "furnace.services.disc_demuxer.run_tool", side_effect=_fake_run_tool,
    ):
        demuxer.demux(
            discs=[disc],
            selected_titles={disc: [title]},
            demux_dir=demux_dir,
            reporter=reporter,
        )

    progress_events = [e for e in reporter.events if e.method == "demux_title_progress"]
    fractions = [e.args[0] for e in progress_events]
    assert 0.25 in fractions
    assert 1.0 in fractions


def test_remux_progress_without_reporter_does_not_crash(tmp_path: Path) -> None:
    """If no reporter is wired, mkvmerge progress lines still parse but the
    no-op closure simply returns without forwarding.
    """
    disc = _make_disc(tmp_path, "SilentBD", DiscType.BLURAY)
    title = DiscTitle(number=9, duration_s=3600.0, raw_label="9) 1:00:00")
    demux_dir = tmp_path / "demux"
    demux_dir.mkdir()

    bd_port = MagicMock()

    def _fake_demux(
        disc_path: Path,
        title_num: int,
        output_dir: Path,
        on_progress: OnProgress = None,
    ) -> list[Path]:
        video = output_dir / "video.h264"
        video.write_bytes(b"v")
        audio = output_dir / "audio [eng].ac3"
        audio.write_bytes(b"a")
        return [video, audio]

    bd_port.demux_title.side_effect = _fake_demux

    demuxer = DiscDemuxer(
        bd_port=bd_port,
        dvd_port=MagicMock(),
        mkvmerge_path=Path("mkvmerge"),
    )

    def _fake_run_tool(
        cmd: list[str],
        on_output: Callable[[str], None] | None = None,
        on_progress_line: Callable[[str], bool] | None = None,
    ) -> tuple[int, str]:
        if on_progress_line is not None:
            # Without reporter, demuxer still passes a closure (forwards into a
            # `_rip_progress` that early-returns); just verify no crash.
            on_progress_line("Progress: 50%")
        return 0, ""

    from unittest.mock import patch

    # No reporter argument here.
    with patch(
        "furnace.services.disc_demuxer.run_tool", side_effect=_fake_run_tool,
    ):
        demuxer.demux(
            discs=[disc],
            selected_titles={disc: [title]},
            demux_dir=demux_dir,
        )


def test_mux_to_mkv_without_progress_callback(tmp_path: Path) -> None:
    """Calling _mux_to_mkv directly without on_progress still runs and the
    embedded progress closure no-ops on parsed samples."""
    demuxer = DiscDemuxer(
        bd_port=MagicMock(),
        dvd_port=MagicMock(),
        mkvmerge_path=Path("mkvmerge"),
    )

    def _fake_run_tool(
        cmd: list[str],
        on_output: Callable[[str], None] | None = None,
        on_progress_line: Callable[[str], bool] | None = None,
    ) -> tuple[int, str]:
        if on_progress_line is not None:
            # Drives the on_progress is None branch inside _on_progress_line.
            assert on_progress_line("Progress: 30%") is True
        return 0, ""

    video = tmp_path / "video.h264"
    video.write_bytes(b"v")
    audio = tmp_path / "audio [eng].ac3"
    audio.write_bytes(b"a")
    chapters = tmp_path / "chapters.txt"
    chapters.write_text("CHAPTER01=00:00:00.000\nCHAPTER01NAME=One\n", encoding="utf-8")

    from unittest.mock import patch

    with patch(
        "furnace.services.disc_demuxer.run_tool", side_effect=_fake_run_tool,
    ):
        demuxer._mux_to_mkv([video, audio, chapters], tmp_path / "out.mkv")


def test_w64_single_file_uses_unindexed_label(tmp_path: Path) -> None:
    """A single .w64 file in demux output emits 'transcode' without 1/1 suffix."""
    disc = _make_disc(tmp_path, "OneWaveBD", DiscType.BLURAY)
    title = DiscTitle(number=7, duration_s=3600.0, raw_label="7) 1:00:00")
    demux_dir = tmp_path / "demux"
    demux_dir.mkdir()

    bd_port = MagicMock()

    def _fake_demux(
        disc_path: Path,
        title_num: int,
        output_dir: Path,
        on_progress: OnProgress = None,
    ) -> list[Path]:
        video = output_dir / "video.h264"
        video.write_bytes(b"v")
        a = output_dir / "audio1 [eng].w64"
        a.write_bytes(b"a")
        return [video, a]

    bd_port.demux_title.side_effect = _fake_demux

    transcoder = MagicMock()

    def _fake_transcode(
        input_path: Path,
        output_path: Path,
        on_progress: OnProgress = None,
    ) -> int:
        output_path.write_bytes(b"flac")
        return 0

    transcoder.transcode_to_flac.side_effect = _fake_transcode

    reporter = RecordingPlanReporter()
    demuxer = DiscDemuxer(
        bd_port=bd_port,
        dvd_port=MagicMock(),
        mkvmerge_path=Path("mkvmerge"),
        pcm_transcoder=transcoder,
    )
    from unittest.mock import patch

    with patch("furnace.services.disc_demuxer.run_tool", return_value=(0, "")):
        demuxer.demux(
            discs=[disc],
            selected_titles={disc: [title]},
            demux_dir=demux_dir,
            reporter=reporter,
        )

    triples = [(e.method, e.args, e.kwargs) for e in reporter.events]
    assert (
        "demux_title_substep",
        ("transcode",),
        (("has_progress", True),),
    ) in triples
    # No indexed form when only one file
    assert (
        "demux_title_substep",
        ("transcode 1/1",),
        (("has_progress", True),),
    ) not in triples


def test_rip_progress_with_none_fraction_is_ignored(tmp_path: Path) -> None:
    """ProgressSample(fraction=None) from the rip step must not be reported."""
    disc = _make_disc(tmp_path, "Y", DiscType.DVD)
    title = DiscTitle(number=1, duration_s=3600.0, raw_label="1)")
    demux_dir = tmp_path / "demux"
    demux_dir.mkdir()

    dvd_port = MagicMock()

    def _fake_demux(
        disc_path: Path,
        title_num: int,
        output_dir: Path,
        on_progress: OnProgress = None,
    ) -> list[Path]:
        out = output_dir / "x.mkv"
        out.write_bytes(b"\x00")
        if on_progress is not None:
            on_progress(ProgressSample(fraction=None))
            on_progress(ProgressSample(fraction=0.42))
        return [out]

    dvd_port.demux_title.side_effect = _fake_demux

    reporter = RecordingPlanReporter()
    demuxer = DiscDemuxer(
        bd_port=MagicMock(),
        dvd_port=dvd_port,
        mkvmerge_path=Path("mkvmerge"),
    )
    demuxer.demux(
        discs=[disc],
        selected_titles={disc: [title]},
        demux_dir=demux_dir,
        reporter=reporter,
    )
    progress_events = [e for e in reporter.events if e.method == "demux_title_progress"]
    assert len(progress_events) == 1
    assert progress_events[0].args == (0.42,)


def test_no_titles_for_disc_does_not_emit_cached(tmp_path: Path) -> None:
    """A disc with no selected titles emits neither cached nor start."""
    disc = _make_disc(tmp_path, "Empty", DiscType.BLURAY)
    demux_dir = tmp_path / "demux"
    demux_dir.mkdir()

    reporter = RecordingPlanReporter()
    demuxer = DiscDemuxer(
        bd_port=MagicMock(),
        dvd_port=MagicMock(),
        mkvmerge_path=Path("mkvmerge"),
    )
    result = demuxer.demux(
        discs=[disc],
        selected_titles={disc: []},
        demux_dir=demux_dir,
        reporter=reporter,
    )
    assert result == []
    assert reporter.events == []
