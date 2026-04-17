"""Textual Pilot tests for furnace.ui.tui screens and FurnacePlanApp.

Each Screen subclass is hosted inside a tiny sentinel App and driven via
`App.run_test()` / `Pilot`.  The host captures the dismiss() result via
a callback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, Static

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
    _PlanResult,
)
from tests.conftest import make_movie, make_track, make_video_info

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


class _HostApp(App[None]):
    """Sentinel host that pushes a given screen on mount."""

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


# ---------------------------------------------------------------------------
# TrackSelectorScreen
# ---------------------------------------------------------------------------


async def test_track_selector_audio_toggle_move_done() -> None:
    mv = _movie_with_audio_and_subs()
    # Two tracks so move_down is meaningful
    tracks = [_audio_track(index=1), _audio_track(index=2, is_default=True)]

    app = _HostApp(
        lambda: TrackSelectorScreen(
            movie=mv, tracks=tracks, track_type=TrackType.AUDIO, preview_cb=None
        )
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TrackSelectorScreen)
        # Invoke the screen's move actions directly; ListView consumes arrow
        # keys before they reach the Screen binding, so we exercise the
        # screen-level actions this way.
        screen.action_move_down()  # cursor 0 -> 1
        screen.action_move_up()  # cursor 1 -> 0
        screen.action_move_up()  # clamp at 0
        await pilot.press("space")  # toggle track 0 on
        await pilot.press("d")
        await pilot.pause()
    assert isinstance(app.result, TrackSelection)
    # Track 0 selected manually; track 1 stays selected from is_default
    assert {t.index for t in app.result.tracks} == {1, 2}


async def test_track_selector_subtitle_compose_and_done() -> None:
    mv = _movie_with_audio_and_subs()
    tracks = [_sub_track(index=3), _sub_track(index=4, is_forced=True)]
    app = _HostApp(
        lambda: TrackSelectorScreen(
            movie=mv, tracks=tracks, track_type=TrackType.SUBTITLE, preview_cb=None
        )
    )
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

    app = _HostApp(
        lambda: TrackSelectorScreen(
            movie=mv, tracks=tracks, track_type=TrackType.AUDIO, preview_cb=preview
        )
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("p")
        await pilot.press("d")
        await pilot.pause()
    assert len(seen) == 1


async def test_track_selector_preview_no_callback_noop() -> None:
    mv = _movie_with_audio_and_subs()
    tracks = [_audio_track()]
    app = _HostApp(
        lambda: TrackSelectorScreen(
            movie=mv, tracks=tracks, track_type=TrackType.AUDIO, preview_cb=None
        )
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("p")  # no cb; branch only
        await pilot.press("d")
        await pilot.pause()
    assert isinstance(app.result, TrackSelection)


async def test_track_selector_empty_tracks_guards() -> None:
    """Hit the `not self._tracks` guards in toggle/preview/set_downmix."""
    mv = _movie_with_audio_and_subs()

    def preview(_t: Track) -> None:
        raise AssertionError("should not fire")

    app = _HostApp(
        lambda: TrackSelectorScreen(
            movie=mv, tracks=[], track_type=TrackType.AUDIO, preview_cb=preview
        )
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("space")  # toggle guarded
        await pilot.press("p")  # preview guarded
        await pilot.press("s")  # set_downmix guarded
        await pilot.press("6")
        await pilot.press("d")
        await pilot.pause()
    assert app.result == TrackSelection(tracks=[], downmix={})


async def test_track_selector_set_downmix_variants() -> None:
    """Exercise set_downmix branches: mono, stereo, 5.1, 7.1."""
    mv = _movie_with_audio_and_subs()
    tracks = [
        _audio_track(index=1, channels=1, channel_layout="mono"),
        _audio_track(index=2, channels=2, channel_layout="stereo"),
        _audio_track(index=3, channels=6, channel_layout="5.1(side)"),
        _audio_track(index=4, channels=8, channel_layout="7.1"),
    ]
    # mark all selected via is_default
    for t in tracks:
        object.__setattr__(t, "is_default", True)
    app = _HostApp(
        lambda: TrackSelectorScreen(
            movie=mv, tracks=tracks, track_type=TrackType.AUDIO, preview_cb=None
        )
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TrackSelectorScreen)
        # At cursor 0 (mono): both s and 6 hit the channels<=STEREO early-return
        await pilot.press("s")
        await pilot.press("6")
        # Move to stereo (cursor 1): still channels<=STEREO → early return
        screen.action_move_down()
        await pilot.press("s")
        # Move to 5.1 (cursor 2): s sets STEREO, pressing again clears it
        screen.action_move_down()
        await pilot.press("s")  # set STEREO
        await pilot.press("s")  # same mode -> None
        # 6 on 5.1 should early-return (channels <= SURROUND_5_1_CHANNELS)
        await pilot.press("6")
        # Move to 7.1 (cursor 3): 6 sets DOWN6
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
    app = _HostApp(
        lambda: TrackSelectorScreen(
            movie=mv, tracks=tracks, track_type=TrackType.SUBTITLE, preview_cb=None
        )
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("s")  # ignored: not AUDIO
        await pilot.press("6")
        await pilot.press("d")
        await pilot.pause()
    assert isinstance(app.result, TrackSelection)


async def test_track_selector_set_downmix_channels_none() -> None:
    """Track with channels=None should be ignored."""
    mv = _movie_with_audio_and_subs()
    t = _audio_track(index=1, channels=6)
    object.__setattr__(t, "channels", None)
    app = _HostApp(
        lambda: TrackSelectorScreen(
            movie=mv, tracks=[t], track_type=TrackType.AUDIO, preview_cb=None
        )
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("s")
        await pilot.press("d")
        await pilot.pause()
    assert isinstance(app.result, TrackSelection)


async def test_track_selector_list_view_highlighted_updates_cursor() -> None:
    """When ListView highlights a track-item-<n>, cursor follows."""
    mv = _movie_with_audio_and_subs()
    tracks = [_audio_track(index=1), _audio_track(index=2)]
    app = _HostApp(
        lambda: TrackSelectorScreen(
            movie=mv, tracks=tracks, track_type=TrackType.AUDIO, preview_cb=None
        )
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TrackSelectorScreen)
        # Simulate a Highlighted event with a fake item carrying the right id.
        from textual.widgets import ListView

        class _FakeItem:
            id = "track-item-1"

        class _FakeEvent:
            item = _FakeItem()

        screen.on_list_view_highlighted(_FakeEvent())  # type: ignore[arg-type]
        assert screen._cursor == 1

        # Bad id → cursor not changed
        class _BadItem:
            id = "track-item-notanint"

        class _BadEvent:
            item = _BadItem()

        screen.on_list_view_highlighted(_BadEvent())  # type: ignore[arg-type]
        assert screen._cursor == 1

        # Non-matching prefix → cursor not changed
        class _OtherItem:
            id = "other-9"

        class _OtherEvent:
            item = _OtherItem()

        screen.on_list_view_highlighted(_OtherEvent())  # type: ignore[arg-type]
        assert screen._cursor == 1

        # item is None → early return
        class _NoneEvent:
            item = None

        screen.on_list_view_highlighted(_NoneEvent())  # type: ignore[arg-type]
        assert screen._cursor == 1

        # on_click noop — just call it for coverage
        screen.on_click(object())

        await pilot.press("d")
        await pilot.pause()
    # reference ListView so import is used
    assert ListView is not None


# ---------------------------------------------------------------------------
# PlaylistSelectorScreen
# ---------------------------------------------------------------------------


async def test_playlist_selector_toggle_and_done() -> None:
    playlists = [
        DiscTitle(number=1, duration_s=1200.0, raw_label="1: 20:00"),
        DiscTitle(number=2, duration_s=300.0, raw_label="2: 05:00"),
    ]
    app = _HostApp(
        lambda: PlaylistSelectorScreen(disc_label="Disc", playlists=playlists)
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, PlaylistSelectorScreen)
        screen.action_move_down()
        screen.action_move_up()
        screen.action_move_up()
        await pilot.press("space")  # toggle playlist 0 off
        await pilot.press("d")
        await pilot.pause()
    # 0 was default-selected (>10m) but we toggled off; 1 stays off
    assert app.result == []


async def test_playlist_selector_empty_and_highlight() -> None:
    app = _HostApp(lambda: PlaylistSelectorScreen(disc_label="X", playlists=[]))
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, PlaylistSelectorScreen)
        await pilot.press("space")  # empty guard
        # Highlighted with valid id
        class _I:
            id = "pl-item-0"

        class _E:
            item = _I()

        screen.on_list_view_highlighted(_E())  # type: ignore[arg-type]
        # bad id
        class _BI:
            id = "pl-item-xx"

        class _BE:
            item = _BI()

        screen.on_list_view_highlighted(_BE())  # type: ignore[arg-type]
        # other prefix
        class _OI:
            id = "other-0"

        class _OE:
            item = _OI()

        screen.on_list_view_highlighted(_OE())  # type: ignore[arg-type]
        # None item
        class _NE:
            item = None

        screen.on_list_view_highlighted(_NE())  # type: ignore[arg-type]

        await pilot.press("d")
        await pilot.pause()
    assert app.result == []


# ---------------------------------------------------------------------------
# FileSelectorScreen
# ---------------------------------------------------------------------------


async def test_file_selector_with_dvd_files_and_sar() -> None:
    p1 = Path("/demux/a.mkv")
    p2 = Path("/demux/b.mkv")
    files = [(p1, 3600.0, 1_000_000), (p2, 3600.0, 2_000_000)]
    seen_preview: list[tuple[Path, str | None]] = []

    def preview(path: Path, aspect: str | None) -> None:
        seen_preview.append((path, aspect))

    app = _HostApp(
        lambda: FileSelectorScreen(
            files=files, dvd_files={p1}, preview_cb=preview
        )
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, FileSelectorScreen)
        # cursor=0, p1 is DVD: SAR toggle flips to True
        await pilot.press("s")
        await pilot.press("p")  # preview with aspect override
        # move down to p2 (not DVD)
        screen.action_move_down()
        await pilot.press("s")  # no-op: not a DVD file
        await pilot.press("p")  # preview without aspect
        screen.action_move_up()
        await pilot.press("d")
        await pilot.pause()
    assert isinstance(app.result, FileSelection)
    assert p1 in app.result.sar_override
    assert p2 not in app.result.sar_override
    # Two previews: one with "16:9", one with None
    assert (p1, "16:9") in seen_preview
    assert (p2, None) in seen_preview


async def test_file_selector_no_dvd_hint_and_toggle_item() -> None:
    p1 = Path("/demux/a.mkv")
    files = [(p1, 120.0, 500)]
    app = _HostApp(lambda: FileSelectorScreen(files=files))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("space")  # deselect
        await pilot.press("d")
        await pilot.pause()
    assert isinstance(app.result, FileSelection)
    assert app.result.selected == []


async def test_file_selector_preview_no_callback_and_empty() -> None:
    p1 = Path("/demux/a.mkv")
    app = _HostApp(lambda: FileSelectorScreen(files=[(p1, 60.0, 100)]))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("p")  # no cb branch
        await pilot.press("d")
        await pilot.pause()
    assert isinstance(app.result, FileSelection)


async def test_file_selector_empty_guards_and_highlight() -> None:
    app = _HostApp(lambda: FileSelectorScreen(files=[]))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("space")  # empty guard
        await pilot.press("s")  # empty guard
        await pilot.press("p")  # preview: files empty
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
        screen.action_move_up()  # clamped at 0
        screen.action_move_down()
        screen.action_move_down()  # clamped at top
        await pilot.press("d")
        await pilot.pause()


# ---------------------------------------------------------------------------
# CropConfirmScreen
# ---------------------------------------------------------------------------


async def test_crop_confirm_accept() -> None:
    crop = CropRect(w=1920, h=800, x=0, y=140)
    app = _HostApp(
        lambda: CropConfirmScreen(crop=crop, source_width=1920, source_height=1080)
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("a")
        await pilot.pause()
    assert app.result == crop


async def test_crop_confirm_reject_escape() -> None:
    crop = CropRect(w=1920, h=800, x=0, y=140)
    app = _HostApp(
        lambda: CropConfirmScreen(crop=crop, source_width=1920, source_height=1080)
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
    assert app.result is None


async def test_crop_confirm_reject_r_key() -> None:
    crop = CropRect(w=1920, h=800, x=0, y=140)
    app = _HostApp(
        lambda: CropConfirmScreen(crop=crop, source_width=1920, source_height=1080)
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("r")
        await pilot.pause()
    assert app.result is None


async def test_crop_confirm_edit_valid() -> None:
    crop = CropRect(w=1920, h=800, x=0, y=140)
    app = _HostApp(
        lambda: CropConfirmScreen(crop=crop, source_width=1920, source_height=1080)
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("e")  # enter edit mode
        screen = app.screen
        assert isinstance(screen, CropConfirmScreen)
        inp = screen.query_one("#crop-input", Input)
        inp.value = "1920:816:0:132"
        # In edit mode, action_accept path hits _confirm_edit.
        # Input captures 'a'; invoke the screen action directly for that branch.
        screen.action_accept()
        await pilot.pause()
    assert app.result == CropRect(w=1920, h=816, x=0, y=132)


async def test_crop_confirm_edit_invalid_shows_error_then_valid_submit() -> None:
    crop = CropRect(w=1920, h=800, x=0, y=140)
    app = _HostApp(
        lambda: CropConfirmScreen(crop=crop, source_width=1920, source_height=1080)
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("e")
        screen = app.screen
        assert isinstance(screen, CropConfirmScreen)
        inp = screen.query_one("#crop-input", Input)
        # Invalid first — triggers error branch
        inp.value = "garbage"
        screen.action_edit()  # _edit_mode already True → _confirm_edit (invalid)
        # Not dismissed yet (still SENTINEL)
        assert app.result == "SENTINEL"
        # Now submit a valid value via on_input_submitted (Enter key)
        inp.value = "1920:1000:0:40"
        await inp.action_submit()
        await pilot.pause()
    assert app.result == CropRect(w=1920, h=1000, x=0, y=40)


async def test_crop_confirm_input_submitted_non_crop_ignored() -> None:
    crop = CropRect(w=1920, h=800, x=0, y=140)
    app = _HostApp(
        lambda: CropConfirmScreen(crop=crop, source_width=1920, source_height=1080)
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, CropConfirmScreen)

        # Simulate a submit on a different input id — should be ignored
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


# ---------------------------------------------------------------------------
# LanguageSelectorScreen
# ---------------------------------------------------------------------------


async def test_language_selector_audio_with_movie() -> None:
    mv = _movie_with_audio_and_subs()
    t = _audio_track(channel_layout="5.1(side)")
    seen: list[Track] = []

    def preview(track: Track) -> None:
        seen.append(track)

    app = _HostApp(
        lambda: LanguageSelectorScreen(
            track=t, lang_list=["eng", "rus", "fra"], preview_cb=preview, movie=mv
        )
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, LanguageSelectorScreen)
        screen.action_move_down()  # cursor 1
        screen.action_move_up()  # cursor 0
        screen.action_move_down()  # cursor 1
        await pilot.press("p")  # preview
        await pilot.press("d")  # select
        await pilot.pause()
    assert app.result == "rus"
    assert len(seen) == 1


async def test_language_selector_subtitle_no_movie_no_preview() -> None:
    t = _sub_track()
    app = _HostApp(
        lambda: LanguageSelectorScreen(
            track=t, lang_list=["eng", "rus"], preview_cb=None, movie=None
        )
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("p")  # no-cb branch
        await pilot.press("d")
        await pilot.pause()
    assert app.result == "eng"


async def test_language_selector_audio_no_channel_layout() -> None:
    mv = _movie_with_audio_and_subs()
    t = _audio_track(channel_layout="")
    object.__setattr__(t, "channel_layout", None)
    app = _HostApp(
        lambda: LanguageSelectorScreen(
            track=t, lang_list=["eng"], preview_cb=None, movie=mv
        )
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
    assert app.result == "eng"


async def test_language_selector_list_view_highlighted() -> None:
    t = _sub_track()
    app = _HostApp(
        lambda: LanguageSelectorScreen(
            track=t, lang_list=["eng", "rus"], preview_cb=None, movie=None
        )
    )
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


# ---------------------------------------------------------------------------
# _PlanResult
# ---------------------------------------------------------------------------


def test_plan_result_defaults() -> None:
    r = _PlanResult()
    assert r.audio_tracks == []
    assert r.subtitle_tracks == []
    assert r.crop is None


# ---------------------------------------------------------------------------
# FurnacePlanApp
# ---------------------------------------------------------------------------


async def test_furnace_plan_app_full_flow_without_crop() -> None:
    """Run two movies (neither has _detected_crop) and drive them through."""
    mv1 = _movie_with_audio_and_subs()
    mv2 = _movie_with_audio_and_subs()

    app = FurnacePlanApp(
        movies=[(mv1, Path("/out/a.mkv")), (mv2, Path("/out/b.mkv"))],
        preview_cb=None,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        # Audio for mv1
        await pilot.press("d")
        await pilot.pause()
        # Subtitles for mv1
        await pilot.press("d")
        await pilot.pause()
        # Audio for mv2
        await pilot.press("d")
        await pilot.pause()
        # Subtitles for mv2
        await pilot.press("d")
        await pilot.pause()
    assert len(app.results) == 2
    assert all(r.crop is None for r in app.results)


async def test_furnace_plan_app_with_detected_crop_accepted() -> None:
    mv = _movie_with_audio_and_subs()
    crop = CropRect(w=1920, h=800, x=0, y=140)
    # sentinel attribute checked by _after_subtitles
    mv._detected_crop = crop  # type: ignore[attr-defined]

    app = FurnacePlanApp(movies=[(mv, Path("/out/a.mkv"))], preview_cb=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("d")  # audio done
        await pilot.pause()
        await pilot.press("d")  # subs done
        await pilot.pause()
        await pilot.press("a")  # accept crop
        await pilot.pause()
    assert len(app.results) == 1
    assert app.results[0].crop == crop


async def test_furnace_plan_app_dismiss_none_paths() -> None:
    """If a screen dismisses with None (no selection), _after_* handle it."""
    mv = _movie_with_audio_and_subs()
    app = FurnacePlanApp(movies=[(mv, Path("/out/a.mkv"))], preview_cb=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        # Programmatically dismiss with None to hit the 'else' branch of
        # `selected_audio.tracks if selected_audio else []`.
        app.screen.dismiss(None)
        await pilot.pause()
        # Now we're on the subtitle screen — dismiss None there too
        app.screen.dismiss(None)
        await pilot.pause()
    assert len(app.results) == 1
    assert app.results[0].audio_tracks == []
    assert app.results[0].subtitle_tracks == []


def test_furnace_plan_app_empty_movies() -> None:
    """Zero movies — on_mount should immediately exit."""
    app = FurnacePlanApp(movies=[], preview_cb=None)

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()

    import asyncio

    asyncio.run(_run())
    assert app.results == []


# ensure pytest imports pytest (avoid ruff F401)
_ = pytest
