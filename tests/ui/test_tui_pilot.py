from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, Static

from furnace.core.audio_profile import AudioMetrics, AudioProfile, Verdict
from furnace.core.models import (
    CropRect,
    DiscTitle,
    DownmixMode,
    Movie,
    Track,
    TrackType,
)
from furnace.ui.tui import (
    CropConfirmScreen,
    FileSelection,
    FileSelectorScreen,
    FurnacePlanApp,
    LanguageSelectorScreen,
    PlaylistSelectorScreen,
    TrackSelection,
    TrackSelectorScreen,
    _fmt_audio_track,
    _fmt_subtitle_track,
    _PlanResult,
    build_language_map,
)
from tests.conftest import make_movie, make_track, make_video_info


def _audio_track(
    *,
    index: int = 1,
    channels: int = 6,
    channel_layout: str = "5.1(side)",
    is_default: bool = False,
    bitrate: int = 640_000,
    source_file: Path | None = None,
) -> Track:
    return make_track(
        index=index,
        track_type=TrackType.AUDIO,
        codec_name="ac3",
        channels=channels,
        channel_layout=channel_layout,
        is_default=is_default,
        bitrate=bitrate,
        title="Main",
        source_file=source_file,
    )


def _sub_track(*, index: int = 2, is_forced: bool = False) -> Track:
    return make_track(
        index=index,
        track_type=TrackType.SUBTITLE,
        codec_name="subrip",
        channels=None,
        channel_layout=None,
        is_default=False,
        is_forced=is_forced,
        bitrate=0,
        title="Subs",
    )


def _movie_with_audio_and_subs() -> Movie:
    return make_movie(
        video=make_video_info(),
        file_size=1_000_000_000,
        audio_tracks=[_audio_track()],
        subtitle_tracks=[_sub_track()],
    )


def _fake_stereo_profile() -> AudioProfile:
    metrics = AudioMetrics(
        channels=6,
        rms_l=-50.0,
        rms_r=-47.0,
        rms_c=-28.0,
        rms_lfe=-75.0,
        rms_ls=-47.0,
        rms_rs=-49.0,
        rms_lb=None,
        rms_rb=None,
        corr_lr=0.4,
        corr_ls_l=0.1,
        corr_rs_r=0.1,
        corr_ls_rs=0.2,
        corr_lb_ls=None,
        corr_rb_rs=None,
    )
    return AudioProfile(
        verdict=Verdict.FAKE,
        score=2,
        suggested=DownmixMode.STEREO,
        reasons=("LFE is dead",),
        metrics=metrics,
    )


class _HostApp(App[None]):
    def __init__(self, screen_factory: Any) -> None:
        super().__init__()
        self._screen_factory = screen_factory
        self.result: Any = "SENTINEL"

    def compose(self) -> ComposeResult:
        yield Static("host")

    async def on_mount(self) -> None:
        def _cb(r: Any) -> None:
            self.result = r
            self.exit()

        await self.push_screen(self._screen_factory(), _cb)


async def test_track_selector_audio_toggle_move_done() -> None:
    mv = _movie_with_audio_and_subs()
    tracks = [_audio_track(index=1), _audio_track(index=2, is_default=True)]

    app = _HostApp(lambda: TrackSelectorScreen(movie=mv, tracks=tracks, track_type=TrackType.AUDIO, preview_cb=None))
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TrackSelectorScreen)
        screen.action_move_down()
        screen.action_move_up()
        screen.action_move_up()
        await pilot.press("space")
        await pilot.press("d")
        await pilot.pause()
    assert isinstance(app.result, TrackSelection)
    assert {t.index for t in app.result.tracks} == {1, 2}


async def test_track_selector_subtitle_compose_and_done() -> None:
    mv = _movie_with_audio_and_subs()
    tracks = [_sub_track(index=3), _sub_track(index=4, is_forced=True)]
    app = _HostApp(lambda: TrackSelectorScreen(movie=mv, tracks=tracks, track_type=TrackType.SUBTITLE, preview_cb=None))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
    assert isinstance(app.result, TrackSelection)
    assert app.result.downmix == {}


async def test_track_selector_preview_with_callback() -> None:
    mv = _movie_with_audio_and_subs()
    tracks = [_audio_track()]
    seen: list[Track] = []

    def preview(t: Track) -> None:
        seen.append(t)

    app = _HostApp(lambda: TrackSelectorScreen(movie=mv, tracks=tracks, track_type=TrackType.AUDIO, preview_cb=preview))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("p")
        await pilot.press("d")
        await pilot.pause()
    assert len(seen) == 1


async def test_track_selector_preview_no_callback_noop() -> None:
    mv = _movie_with_audio_and_subs()
    tracks = [_audio_track()]
    app = _HostApp(lambda: TrackSelectorScreen(movie=mv, tracks=tracks, track_type=TrackType.AUDIO, preview_cb=None))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("p")
        await pilot.press("d")
        await pilot.pause()
    assert isinstance(app.result, TrackSelection)


async def test_track_selector_empty_tracks_guards() -> None:
    mv = _movie_with_audio_and_subs()
    previewed: list[Track] = []

    app = _HostApp(
        lambda: TrackSelectorScreen(movie=mv, tracks=[], track_type=TrackType.AUDIO, preview_cb=previewed.append)
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("space")
        await pilot.press("p")
        await pilot.press("s")
        await pilot.press("6")
        await pilot.press("c")
        await pilot.press("d")
        await pilot.pause()
    assert not previewed
    assert app.result == TrackSelection(tracks=[], downmix={})


async def test_track_selector_set_downmix_variants() -> None:
    mv = _movie_with_audio_and_subs()
    tracks = [
        _audio_track(index=1, channels=1, channel_layout="mono"),
        _audio_track(index=2, channels=2, channel_layout="stereo"),
        _audio_track(index=3, channels=6, channel_layout="5.1(side)"),
        _audio_track(index=4, channels=8, channel_layout="7.1"),
    ]
    for t in tracks:
        object.__setattr__(t, "is_default", True)
    app = _HostApp(lambda: TrackSelectorScreen(movie=mv, tracks=tracks, track_type=TrackType.AUDIO, preview_cb=None))
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TrackSelectorScreen)
        await pilot.press("s")
        await pilot.press("6")
        screen.action_move_down()
        await pilot.press("s")
        screen.action_move_down()
        await pilot.press("s")
        await pilot.press("s")
        await pilot.press("6")
        screen.action_move_down()
        await pilot.press("6")
        await pilot.press("d")
        await pilot.pause()
    assert isinstance(app.result, TrackSelection)
    key = (tracks[3].source_file, tracks[3].index)
    assert app.result.downmix.get(key) == DownmixMode.DOWN6


async def test_track_selector_set_downmix_ignored_for_subtitle() -> None:
    mv = _movie_with_audio_and_subs()
    tracks = [_sub_track()]
    app = _HostApp(lambda: TrackSelectorScreen(movie=mv, tracks=tracks, track_type=TrackType.SUBTITLE, preview_cb=None))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("s")
        await pilot.press("6")
        await pilot.press("c")
        await pilot.press("d")
        await pilot.pause()
    assert isinstance(app.result, TrackSelection)


async def test_track_selector_clear_downmix_removes_auto_applied() -> None:
    mv = _movie_with_audio_and_subs()
    t = _audio_track(index=1, channels=6, channel_layout="5.1(side)", is_default=True)
    t.audio_profile = _fake_stereo_profile()
    app = _HostApp(lambda: TrackSelectorScreen(movie=mv, tracks=[t], track_type=TrackType.AUDIO, preview_cb=None))
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TrackSelectorScreen)
        panel = screen.query_one("#detector-panel", Static)
        assert screen._downmix[0] == DownmixMode.STEREO
        before = str(panel.render())
        assert "downmix applied" in before
        assert "no downmix" not in before
        await pilot.press("c")
        assert screen._downmix[0] is None
        assert "no downmix applied" in str(panel.render())
        await pilot.press("d")
        await pilot.pause()
    assert isinstance(app.result, TrackSelection)
    assert app.result.downmix == {}
    assert len(app.result.tracks) == 1


async def test_track_selector_downmix_hint_mentions_clear() -> None:
    mv = _movie_with_audio_and_subs()
    app = _HostApp(
        lambda: TrackSelectorScreen(
            movie=mv,
            tracks=[_audio_track(index=1, is_default=True)],
            track_type=TrackType.AUDIO,
            preview_cb=None,
        )
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TrackSelectorScreen)
        hint = screen.query_one("#track-downmix-hint", Static)
        assert "C=clear" in str(hint.render())
        await pilot.press("d")
        await pilot.pause()


async def test_track_selector_set_downmix_channels_none() -> None:
    mv = _movie_with_audio_and_subs()
    t = _audio_track(index=1, channels=6)
    object.__setattr__(t, "channels", None)
    app = _HostApp(lambda: TrackSelectorScreen(movie=mv, tracks=[t], track_type=TrackType.AUDIO, preview_cb=None))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("s")
        await pilot.press("d")
        await pilot.pause()
    assert isinstance(app.result, TrackSelection)


async def test_track_selector_list_view_highlighted_updates_cursor() -> None:
    mv = _movie_with_audio_and_subs()
    tracks = [_audio_track(index=1), _audio_track(index=2)]
    app = _HostApp(lambda: TrackSelectorScreen(movie=mv, tracks=tracks, track_type=TrackType.AUDIO, preview_cb=None))
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TrackSelectorScreen)
        from textual.widgets import ListView

        class _FakeItem:
            id = "track-item-1"

        class _FakeEvent:
            item = _FakeItem()

        screen.on_list_view_highlighted(_FakeEvent())  # type: ignore[arg-type]
        assert screen._cursor == 1

        class _BadItem:
            id = "track-item-notanint"

        class _BadEvent:
            item = _BadItem()

        screen.on_list_view_highlighted(_BadEvent())  # type: ignore[arg-type]
        assert screen._cursor == 1

        class _OtherItem:
            id = "other-9"

        class _OtherEvent:
            item = _OtherItem()

        screen.on_list_view_highlighted(_OtherEvent())  # type: ignore[arg-type]
        assert screen._cursor == 1

        class _NoneEvent:
            item = None

        screen.on_list_view_highlighted(_NoneEvent())  # type: ignore[arg-type]
        assert screen._cursor == 1

        screen.on_click(object())

        await pilot.press("d")
        await pilot.pause()
    assert ListView is not None


async def test_track_selector_detector_panel_refreshes_on_highlight() -> None:
    real_metrics = AudioMetrics(
        channels=6,
        rms_l=-20.0,
        rms_r=-20.5,
        rms_c=-25.0,
        rms_lfe=-30.0,
        rms_ls=-22.0,
        rms_rs=-22.5,
        rms_lb=None,
        rms_rb=None,
        corr_lr=0.3,
        corr_ls_l=0.1,
        corr_rs_r=0.1,
        corr_ls_rs=0.2,
        corr_lb_ls=None,
        corr_rb_rs=None,
    )
    fake_metrics = AudioMetrics(
        channels=6,
        rms_l=-50.0,
        rms_r=-47.0,
        rms_c=-28.0,
        rms_lfe=-75.0,
        rms_ls=-47.0,
        rms_rs=-49.0,
        rms_lb=None,
        rms_rb=None,
        corr_lr=0.4,
        corr_ls_l=0.1,
        corr_rs_r=0.1,
        corr_ls_rs=0.2,
        corr_lb_ls=None,
        corr_rb_rs=None,
    )
    t0 = _audio_track(index=1)
    t0.audio_profile = AudioProfile(
        verdict=Verdict.REAL,
        score=0,
        suggested=None,
        reasons=(),
        metrics=real_metrics,
    )
    t1 = _audio_track(index=2)
    t1.audio_profile = AudioProfile(
        verdict=Verdict.FAKE,
        score=2,
        suggested=DownmixMode.STEREO,
        reasons=("LFE is dead",),
        metrics=fake_metrics,
    )

    mv = _movie_with_audio_and_subs()
    app = _HostApp(lambda: TrackSelectorScreen(movie=mv, tracks=[t0, t1], track_type=TrackType.AUDIO, preview_cb=None))
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TrackSelectorScreen)

        panel = screen.query_one("#detector-panel", Static)
        assert "real surround" in str(panel.render())
        assert "FAKE surround" not in str(panel.render())

        class _Item1:
            id = "track-item-1"

        class _Event1:
            item = _Item1()

        screen.on_list_view_highlighted(_Event1())  # type: ignore[arg-type]
        await pilot.pause()

        assert "FAKE surround" in str(panel.render())

        await pilot.press("d")
        await pilot.pause()


async def test_playlist_selector_toggle_and_done() -> None:
    playlists = [
        DiscTitle(number=1, duration_s=1200.0, raw_label="1: 20:00"),
        DiscTitle(number=2, duration_s=300.0, raw_label="2: 05:00"),
    ]
    app = _HostApp(lambda: PlaylistSelectorScreen(disc_label="Disc", playlists=playlists))
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, PlaylistSelectorScreen)
        screen.action_move_down()
        screen.action_move_up()
        screen.action_move_up()
        await pilot.press("space")
        await pilot.press("d")
        await pilot.pause()
    assert app.result == []


async def test_playlist_selector_empty_and_highlight() -> None:
    app = _HostApp(lambda: PlaylistSelectorScreen(disc_label="X", playlists=[]))
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, PlaylistSelectorScreen)
        await pilot.press("space")

        class _I:
            id = "pl-item-0"

        class _E:
            item = _I()

        screen.on_list_view_highlighted(_E())  # type: ignore[arg-type]

        class _BI:
            id = "pl-item-xx"

        class _BE:
            item = _BI()

        screen.on_list_view_highlighted(_BE())  # type: ignore[arg-type]

        class _OI:
            id = "other-0"

        class _OE:
            item = _OI()

        screen.on_list_view_highlighted(_OE())  # type: ignore[arg-type]

        class _NE:
            item = None

        screen.on_list_view_highlighted(_NE())  # type: ignore[arg-type]

        await pilot.press("d")
        await pilot.pause()
    assert app.result == []


async def test_file_selector_with_dvd_files_and_sar() -> None:
    p1 = Path("/demux/a.mkv")
    p2 = Path("/demux/b.mkv")
    files = [(p1, 3600.0, 1_000_000), (p2, 3600.0, 2_000_000)]
    seen_preview: list[tuple[Path, str | None]] = []

    def preview(path: Path, aspect: str | None) -> None:
        seen_preview.append((path, aspect))

    app = _HostApp(lambda: FileSelectorScreen(files=files, dvd_files={p1}, preview_cb=preview))
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, FileSelectorScreen)
        await pilot.press("s")
        await pilot.press("p")
        screen.action_move_down()
        await pilot.press("s")
        await pilot.press("p")
        screen.action_move_up()
        await pilot.press("d")
        await pilot.pause()
    assert isinstance(app.result, FileSelection)
    assert p1 in app.result.sar_override
    assert p2 not in app.result.sar_override
    assert (p1, "16:9") in seen_preview
    assert (p2, None) in seen_preview


async def test_file_selector_no_dvd_hint_and_toggle_item() -> None:
    p1 = Path("/demux/a.mkv")
    files = [(p1, 120.0, 500)]
    app = _HostApp(lambda: FileSelectorScreen(files=files))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("space")
        await pilot.press("d")
        await pilot.pause()
    assert isinstance(app.result, FileSelection)
    assert app.result.selected == []


async def test_file_selector_preview_no_callback_and_empty() -> None:
    p1 = Path("/demux/a.mkv")
    app = _HostApp(lambda: FileSelectorScreen(files=[(p1, 60.0, 100)]))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("p")
        await pilot.press("d")
        await pilot.pause()
    assert isinstance(app.result, FileSelection)


async def test_file_selector_empty_guards_and_highlight() -> None:
    app = _HostApp(lambda: FileSelectorScreen(files=[]))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("space")
        await pilot.press("s")
        await pilot.press("g")
        await pilot.press("p")
        screen = app.screen
        assert isinstance(screen, FileSelectorScreen)

        class _I:
            id = "file-item-0"

        class _E:
            item = _I()

        screen.on_list_view_highlighted(_E())  # type: ignore[arg-type]

        class _BI:
            id = "file-item-x"

        class _BE:
            item = _BI()

        screen.on_list_view_highlighted(_BE())  # type: ignore[arg-type]

        class _OI:
            id = "other-0"

        class _OE:
            item = _OI()

        screen.on_list_view_highlighted(_OE())  # type: ignore[arg-type]

        class _NE:
            item = None

        screen.on_list_view_highlighted(_NE())  # type: ignore[arg-type]

        await pilot.press("d")
        await pilot.pause()


async def test_file_selector_move_up_down_bounds() -> None:
    files = [(Path("/a"), 10.0, 1), (Path("/b"), 20.0, 2)]
    app = _HostApp(lambda: FileSelectorScreen(files=files))
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, FileSelectorScreen)
        screen.action_move_up()
        screen.action_move_down()
        screen.action_move_down()
        await pilot.press("d")
        await pilot.pause()


async def test_file_selector_grain_toggle_on_sd_file() -> None:
    p1 = Path("/demux/sd.mkv")
    files = [(p1, 3600.0, 1_000_000)]
    app = _HostApp(lambda: FileSelectorScreen(files=files, grain_files={p1}))
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, FileSelectorScreen)
        assert "GRAIN" not in screen._render_line(0)
        await pilot.press("g")
        assert "GRAIN" in screen._render_line(0)
        label = screen.query_one("#file-label-0", Static)
        assert "GRAIN" in str(label.render())
        await pilot.press("d")
        await pilot.pause()
    assert isinstance(app.result, FileSelection)
    assert app.result.grain == {p1: True}


async def test_file_selector_grain_defaults_prelit_and_untouched() -> None:
    p1 = Path("/demux/sd.mkv")
    files = [(p1, 3600.0, 1_000_000)]
    app = _HostApp(lambda: FileSelectorScreen(files=files, grain_files={p1}, grain_defaults={p1}))
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, FileSelectorScreen)
        assert "GRAIN" in screen._render_line(0)
        await pilot.press("d")
        await pilot.pause()
    assert isinstance(app.result, FileSelection)
    assert app.result.grain == {p1: True}


async def test_file_selector_grain_packing_only_selected_sd() -> None:
    p1 = Path("/demux/sd1.mkv")
    p2 = Path("/demux/sd2.mkv")
    p3 = Path("/demux/hd.mkv")
    files = [(p1, 3600.0, 1), (p2, 3600.0, 2), (p3, 3600.0, 3)]
    app = _HostApp(lambda: FileSelectorScreen(files=files, grain_files={p1, p2}, grain_defaults={p1, p2}))
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, FileSelectorScreen)
        assert "GRAIN" in screen._render_line(0)
        assert "GRAIN" in screen._render_line(1)
        assert "GRAIN" not in screen._render_line(2)
        screen.action_move_down()
        await pilot.press("space")
        screen.action_move_down()
        await pilot.press("g")
        assert "GRAIN" not in screen._render_line(2)
        await pilot.press("d")
        await pilot.pause()
    assert isinstance(app.result, FileSelection)
    assert app.result.grain == {p1: True}
    assert p2 not in app.result.selected
    assert p3 in app.result.selected


async def test_file_selector_grain_hint_shown_and_hidden() -> None:
    p1 = Path("/demux/sd.mkv")
    files = [(p1, 60.0, 100)]
    app = _HostApp(lambda: FileSelectorScreen(files=files, grain_files={p1}))
    async with app.run_test() as pilot:
        await pilot.pause()
        hint = app.screen.query_one("#file-hint", Static)
        assert "G=grain" in str(hint.render())
    app2 = _HostApp(lambda: FileSelectorScreen(files=files))
    async with app2.run_test() as pilot:
        await pilot.pause()
        hint = app2.screen.query_one("#file-hint", Static)
        assert "G=grain" not in str(hint.render())


async def test_crop_confirm_accept() -> None:
    crop = CropRect(w=1920, h=800, x=0, y=140)
    app = _HostApp(lambda: CropConfirmScreen(crop=crop, source_width=1920, source_height=1080))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("a")
        await pilot.pause()
    assert app.result == crop


async def test_crop_confirm_reject_escape() -> None:
    crop = CropRect(w=1920, h=800, x=0, y=140)
    app = _HostApp(lambda: CropConfirmScreen(crop=crop, source_width=1920, source_height=1080))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
    assert app.result is None


async def test_crop_confirm_reject_r_key() -> None:
    crop = CropRect(w=1920, h=800, x=0, y=140)
    app = _HostApp(lambda: CropConfirmScreen(crop=crop, source_width=1920, source_height=1080))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("r")
        await pilot.pause()
    assert app.result is None


async def test_crop_confirm_edit_valid() -> None:
    crop = CropRect(w=1920, h=800, x=0, y=140)
    app = _HostApp(lambda: CropConfirmScreen(crop=crop, source_width=1920, source_height=1080))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("e")
        screen = app.screen
        assert isinstance(screen, CropConfirmScreen)
        inp = screen.query_one("#crop-input", Input)
        inp.value = "1920:816:0:132"
        screen.action_accept()
        await pilot.pause()
    assert app.result == CropRect(w=1920, h=816, x=0, y=132)


async def test_crop_confirm_edit_invalid_shows_error_then_valid_submit() -> None:
    crop = CropRect(w=1920, h=800, x=0, y=140)
    app = _HostApp(lambda: CropConfirmScreen(crop=crop, source_width=1920, source_height=1080))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("e")
        screen = app.screen
        assert isinstance(screen, CropConfirmScreen)
        inp = screen.query_one("#crop-input", Input)
        inp.value = "garbage"
        screen.action_edit()
        assert app.result == "SENTINEL"
        inp.value = "1920:1000:0:40"
        await inp.action_submit()
        await pilot.pause()
    assert app.result == CropRect(w=1920, h=1000, x=0, y=40)


async def test_crop_confirm_input_submitted_non_crop_ignored() -> None:
    crop = CropRect(w=1920, h=800, x=0, y=140)
    app = _HostApp(lambda: CropConfirmScreen(crop=crop, source_width=1920, source_height=1080))
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, CropConfirmScreen)

        class _FakeInput:
            id = "other"
            value = "x"

        class _FakeSubmit:
            input = _FakeInput()
            value = "x"

        screen.on_input_submitted(_FakeSubmit())  # type: ignore[arg-type]
        await pilot.press("escape")
        await pilot.pause()
    assert app.result is None


async def test_language_selector_audio_with_movie() -> None:
    mv = _movie_with_audio_and_subs()
    t = _audio_track(channel_layout="5.1(side)")
    seen: list[Track] = []

    def preview(track: Track) -> None:
        seen.append(track)

    app = _HostApp(
        lambda: LanguageSelectorScreen(track=t, lang_list=["eng", "rus", "fra"], preview_cb=preview, movie=mv)
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, LanguageSelectorScreen)
        screen.action_move_down()
        screen.action_move_up()
        screen.action_move_down()
        await pilot.press("p")
        await pilot.press("d")
        await pilot.pause()
    assert app.result == "rus"
    assert len(seen) == 1


async def test_language_selector_subtitle_no_movie_no_preview() -> None:
    t = _sub_track()
    app = _HostApp(lambda: LanguageSelectorScreen(track=t, lang_list=["eng", "rus"], preview_cb=None, movie=None))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("p")
        await pilot.press("d")
        await pilot.pause()
    assert app.result == "eng"


async def test_language_selector_audio_no_channel_layout() -> None:
    mv = _movie_with_audio_and_subs()
    t = _audio_track(channel_layout="")
    object.__setattr__(t, "channel_layout", None)
    app = _HostApp(lambda: LanguageSelectorScreen(track=t, lang_list=["eng"], preview_cb=None, movie=mv))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
    assert app.result == "eng"


async def test_language_selector_list_view_highlighted() -> None:
    t = _sub_track()
    app = _HostApp(lambda: LanguageSelectorScreen(track=t, lang_list=["eng", "rus"], preview_cb=None, movie=None))
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, LanguageSelectorScreen)

        class _I:
            id = "lang-item-1"

        class _E:
            item = _I()

        screen.on_list_view_highlighted(_E())  # type: ignore[arg-type]
        assert screen._cursor == 1

        class _BI:
            id = "lang-item-xx"

        class _BE:
            item = _BI()

        screen.on_list_view_highlighted(_BE())  # type: ignore[arg-type]

        class _OI:
            id = "other-0"

        class _OE:
            item = _OI()

        screen.on_list_view_highlighted(_OE())  # type: ignore[arg-type]

        class _NE:
            item = None

        screen.on_list_view_highlighted(_NE())  # type: ignore[arg-type]

        await pilot.press("d")
        await pilot.pause()


def test_plan_result_defaults() -> None:
    r = _PlanResult()
    assert r.audio_tracks == []
    assert r.subtitle_tracks == []
    assert r.crop is None


async def test_furnace_plan_app_full_flow_without_crop() -> None:
    mv1 = _movie_with_audio_and_subs()
    mv2 = _movie_with_audio_and_subs()

    app = FurnacePlanApp(
        movies=[(mv1, Path("/out/a.mkv")), (mv2, Path("/out/b.mkv"))],
        preview_cb=None,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
    assert len(app.results) == 2
    assert all(r.crop is None for r in app.results)


async def test_furnace_plan_app_with_detected_crop_accepted() -> None:
    mv = _movie_with_audio_and_subs()
    crop = CropRect(w=1920, h=800, x=0, y=140)
    mv._detected_crop = crop  # type: ignore[attr-defined]

    app = FurnacePlanApp(movies=[(mv, Path("/out/a.mkv"))], preview_cb=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("a")
        await pilot.pause()
    assert len(app.results) == 1
    assert app.results[0].crop == crop


async def test_furnace_plan_app_dismiss_none_paths() -> None:
    mv = _movie_with_audio_and_subs()
    app = FurnacePlanApp(movies=[(mv, Path("/out/a.mkv"))], preview_cb=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.dismiss(None)
        await pilot.pause()
        app.screen.dismiss(None)
        await pilot.pause()
    assert len(app.results) == 1
    assert app.results[0].audio_tracks == []
    assert app.results[0].subtitle_tracks == []


def test_furnace_plan_app_empty_movies() -> None:
    app = FurnacePlanApp(movies=[], preview_cb=None)

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()

    import asyncio

    asyncio.run(_run())
    assert app.results == []


def test_fmt_audio_track_relabel_arrow_when_differs() -> None:
    t = _audio_track(index=1)
    out = _fmt_audio_track(t, selected=True, downmix=None, relabel_to="jpn")
    assert "eng->jpn" in out


def test_fmt_audio_track_relabel_plain_when_none() -> None:
    t = _audio_track(index=1)
    out = _fmt_audio_track(t, selected=True, downmix=None, relabel_to=None)
    assert "eng->" not in out
    assert "eng" in out


def test_fmt_audio_track_relabel_plain_when_equal() -> None:
    t = _audio_track(index=1)
    out = _fmt_audio_track(t, selected=True, downmix=None, relabel_to="eng")
    assert "eng->" not in out
    assert "eng" in out


def test_fmt_subtitle_track_relabel_arrow_and_plain() -> None:
    t = _sub_track(index=2)
    assert "eng->rus" in _fmt_subtitle_track(t, selected=True, relabel_to="rus")
    assert "eng->" not in _fmt_subtitle_track(t, selected=True, relabel_to=None)
    assert "eng->" not in _fmt_subtitle_track(t, selected=True, relabel_to="eng")


def test_build_language_map_includes_only_selected_non_none() -> None:
    t0 = _audio_track(index=1)
    t1 = _audio_track(index=2)
    t2 = _audio_track(index=3)
    tracks = [t0, t1, t2]
    selected = [True, True, False]
    overrides: list[str | None] = ["jpn", None, "rus"]
    result = build_language_map(tracks, selected, overrides)
    assert result == {(t0.source_file, t0.index): "jpn"}


def test_relabel_target_variants() -> None:
    mv = _movie_with_audio_and_subs()
    tracks = [_audio_track(index=1), _audio_track(index=2)]

    off = TrackSelectorScreen(
        movie=mv,
        tracks=tracks,
        track_type=TrackType.AUDIO,
        allow_relabel=False,
        lang_list=["jpn", "rus"],
    )
    assert off._relabel_target(0) is None

    empty = TrackSelectorScreen(
        movie=mv,
        tracks=tracks,
        track_type=TrackType.AUDIO,
        allow_relabel=True,
        lang_list=[],
    )
    assert empty._relabel_target(0) is None

    default_list = TrackSelectorScreen(
        movie=mv,
        tracks=tracks,
        track_type=TrackType.AUDIO,
        allow_relabel=True,
        lang_list=None,
    )
    assert default_list._relabel_target(0) is None

    on = TrackSelectorScreen(
        movie=mv,
        tracks=tracks,
        track_type=TrackType.AUDIO,
        allow_relabel=True,
        lang_list=["jpn", "rus"],
    )
    assert on._relabel_target(0) == "jpn"
    on._lang_override[0] = "rus"
    assert on._relabel_target(0) == "rus"


def test_check_action_hides_language_when_relabel_off() -> None:
    mv = _movie_with_audio_and_subs()
    tracks = [_audio_track(index=1)]

    off = TrackSelectorScreen(
        movie=mv,
        tracks=tracks,
        track_type=TrackType.AUDIO,
        allow_relabel=False,
    )
    assert off.check_action("set_language", ()) is False
    assert off.check_action("toggle_track", ()) is True

    on = TrackSelectorScreen(
        movie=mv,
        tracks=tracks,
        track_type=TrackType.AUDIO,
        allow_relabel=True,
        lang_list=["jpn"],
    )
    assert on.check_action("set_language", ()) is True


def test_action_set_language_guard_noop_when_relabel_off() -> None:
    mv = _movie_with_audio_and_subs()
    tracks = [_audio_track(index=1)]
    off = TrackSelectorScreen(
        movie=mv,
        tracks=tracks,
        track_type=TrackType.AUDIO,
        allow_relabel=False,
    )
    off.action_set_language()
    assert off._lang_override == [None]


async def test_track_selector_relabel_flow_sets_language() -> None:
    mv = _movie_with_audio_and_subs()
    track = _audio_track(index=1, is_default=True)
    app = _HostApp(
        lambda: TrackSelectorScreen(
            movie=mv,
            tracks=[track],
            track_type=TrackType.AUDIO,
            preview_cb=None,
            allow_relabel=True,
            lang_list=["jpn", "rus"],
        )
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TrackSelectorScreen)
        await pilot.press("l")
        await pilot.pause()
        lang_screen = app.screen
        assert isinstance(lang_screen, LanguageSelectorScreen)
        await pilot.press("d")
        await pilot.pause()
        label = screen.query_one("#track-label-0", Static)
        assert "eng->jpn" in str(label.render())
        await pilot.press("d")
        await pilot.pause()
    assert isinstance(app.result, TrackSelection)
    assert app.result.languages == {(track.source_file, track.index): "jpn"}


async def test_track_selector_relabel_cancel_keeps_language_unset() -> None:
    mv = _movie_with_audio_and_subs()
    track = _audio_track(index=1, is_default=True)
    app = _HostApp(
        lambda: TrackSelectorScreen(
            movie=mv,
            tracks=[track],
            track_type=TrackType.AUDIO,
            preview_cb=None,
            allow_relabel=True,
            lang_list=["jpn", "rus"],
        )
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TrackSelectorScreen)
        await pilot.press("l")
        await pilot.pause()
        lang_screen = app.screen
        assert isinstance(lang_screen, LanguageSelectorScreen)
        lang_screen.dismiss(None)
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
    assert isinstance(app.result, TrackSelection)
    assert app.result.languages == {}
    assert screen._lang_override == [None]


async def test_track_selector_relabel_off_l_key_is_noop() -> None:
    mv = _movie_with_audio_and_subs()
    track = _audio_track(index=1, is_default=True)
    app = _HostApp(
        lambda: TrackSelectorScreen(
            movie=mv,
            tracks=[track],
            track_type=TrackType.AUDIO,
            preview_cb=None,
            allow_relabel=False,
        )
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TrackSelectorScreen)
        await pilot.press("l")
        await pilot.pause()
        assert isinstance(app.screen, TrackSelectorScreen)
        await pilot.press("d")
        await pilot.pause()
    assert isinstance(app.result, TrackSelection)
    assert app.result.languages == {}


async def test_language_selector_shows_tagged_original() -> None:
    t = _audio_track(index=1)
    app = _HostApp(lambda: LanguageSelectorScreen(track=t, lang_list=["jpn", "rus"], preview_cb=None, movie=None))
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, LanguageSelectorScreen)
        hint = screen.query_one("#lang-hint", Static)
        assert "(tagged: eng)" in str(hint.render())
        await pilot.press("d")
        await pilot.pause()
    assert app.result == "jpn"


async def test_language_selector_omits_tagged_for_und() -> None:
    t = _sub_track(index=2)
    t.language = "und"
    app = _HostApp(lambda: LanguageSelectorScreen(track=t, lang_list=["jpn", "rus"], preview_cb=None, movie=None))
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, LanguageSelectorScreen)
        hint = screen.query_one("#lang-hint", Static)
        assert "(tagged:" not in str(hint.render())
        await pilot.press("d")
        await pilot.pause()
    assert app.result == "jpn"


_ = pytest
