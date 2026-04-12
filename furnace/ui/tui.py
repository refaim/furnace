from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, ListItem, ListView, Static

from furnace.core.models import (
    STEREO_CHANNELS,
    SURROUND_5_1_CHANNELS,
    CropRect,
    DiscTitle,
    DownmixMode,
    Movie,
    Track,
    TrackType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_size(n: int) -> str:
    """File size in MB."""
    mb = n / (1024 * 1024)
    return f"{mb:,.0f} MB"


def _fmt_duration(s: float) -> str:
    h = int(s) // 3600
    m = (int(s) % 3600) // 60
    sec = int(s) % 60
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def _fmt_audio_track(
    track: Track,
    *,
    selected: bool,
    downmix: DownmixMode | None = None,
) -> str:
    mark = "x" if selected else " "
    lang = (track.language or "und").ljust(4)
    codec = track.codec_name.upper()
    layout = ""
    if track.channel_layout:
        # Simplify channel layout: "5.1(side)" -> "5.1"
        layout = track.channel_layout.split("(")[0]
    codec_layout = f"{codec} {layout}".strip()
    bitrate = ""
    if track.bitrate:
        bitrate = f"{track.bitrate // 1000} kbps"
    title = f"'{track.title}'" if track.title else ""

    downmix_tag = ""
    if downmix == DownmixMode.STEREO:
        downmix_tag = "\\[-> 2.0]"
    elif downmix == DownmixMode.DOWN6:
        downmix_tag = "\\[-> 5.1]"

    parts = [p for p in [lang, codec_layout, bitrate, title, downmix_tag] if p]
    # Escape brackets so Rich doesn't interpret them as markup tags
    return f"\\[{mark}]  {'  '.join(parts)}"


def _fmt_subtitle_track(track: Track, *, selected: bool) -> str:
    mark = "x" if selected else " "
    lang = (track.language or "und").ljust(4)
    codec = track.codec_name.upper()
    forced = "\\[FORCED]" if track.is_forced else ""
    title = f"'{track.title}'" if track.title else ""
    parts = [p for p in [lang, codec, forced, title] if p]
    # Escape brackets so Rich doesn't interpret them as markup tags
    return f"\\[{mark}]  {'  '.join(parts)}"


# ---------------------------------------------------------------------------
# TrackSelection result
# ---------------------------------------------------------------------------


@dataclass
class TrackSelection:
    """Result from TrackSelectorScreen.

    `tracks` — tracks the user selected (same as the old list[Track] return).
    `downmix` — per-track downmix override keyed by (source_file, stream_index).
                Always empty for subtitle screens.
    """

    tracks: list[Track]
    downmix: dict[tuple[Path, int], DownmixMode]


# ---------------------------------------------------------------------------
# TrackSelectorScreen
# ---------------------------------------------------------------------------


class TrackSelectorScreen(Screen[TrackSelection]):
    """Screen for selecting audio or subtitle tracks.

    Returns a TrackSelection via dismiss().
    """

    BINDINGS = [
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("space", "toggle_track", "Toggle"),
        Binding("p", "preview_track", "Preview"),
        Binding("s", "set_downmix('stereo')", "Stereo", show=False),
        Binding("6", "set_downmix('down6')", "5.1", show=False),
        Binding("d", "done", "Done"),
    ]

    def __init__(
        self,
        movie: Movie,
        tracks: list[Track],
        track_type: TrackType,
        preview_cb: Callable[[Track], None] | None = None,
    ) -> None:
        super().__init__()
        self._movie = movie
        self._tracks = tracks
        self._track_type = track_type
        self._preview_cb = preview_cb
        # Pre-select tracks that are marked as default
        self._selected: list[bool] = [t.is_default for t in tracks]
        self._downmix: list[DownmixMode | None] = [None] * len(tracks)
        self._cursor: int = 0

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        v = self._movie.video
        duration = _fmt_duration(v.duration_s)
        size = _fmt_size(self._movie.file_size)
        codec = v.codec_name.upper()
        resolution = f"{v.width}x{v.height}"
        kind = "Audio" if self._track_type == TrackType.AUDIO else "Subtitles"
        filename = self._movie.main_file.name

        yield Header()
        yield Static(
            f"{filename}  |  {resolution}  {codec}  {duration}  {size}",
            id="track-header",
        )
        yield Static(f"Select {kind} tracks  (Space=toggle  P=preview  D=done)", id="track-hint")
        if self._track_type == TrackType.AUDIO:
            yield Static(
                "Downmix >2ch:  S=stereo  6=7.1->5.1",
                id="track-downmix-hint",
            )

        items: list[ListItem] = []
        for i, _track in enumerate(self._tracks):
            label = self._render_line(i)
            items.append(ListItem(Static(label, id=f"track-label-{i}"), id=f"track-item-{i}"))

        yield ListView(*items, id="track-list")

        yield Footer()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render_line(self, index: int) -> str:
        track = self._tracks[index]
        selected = self._selected[index]
        if self._track_type == TrackType.AUDIO:
            return _fmt_audio_track(track, selected=selected, downmix=self._downmix[index])
        return _fmt_subtitle_track(track, selected=selected)

    def _refresh_item(self, index: int) -> None:
        label_widget = self.query_one(f"#track-label-{index}", Static)
        label_widget.update(self._render_line(index))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_move_up(self) -> None:
        lv = self.query_one("#track-list", ListView)
        lv.action_cursor_up()
        self._cursor = max(0, self._cursor - 1)

    def action_move_down(self) -> None:
        lv = self.query_one("#track-list", ListView)
        lv.action_cursor_down()
        self._cursor = min(len(self._tracks) - 1, self._cursor + 1)

    def action_toggle_track(self) -> None:
        if not self._tracks:
            return
        self._selected[self._cursor] = not self._selected[self._cursor]
        self._refresh_item(self._cursor)

    def action_preview_track(self) -> None:
        if not self._tracks or self._preview_cb is None:
            return
        self._preview_cb(self._tracks[self._cursor])

    def action_set_downmix(self, mode: str) -> None:
        if not self._tracks or self._track_type != TrackType.AUDIO:
            return
        track = self._tracks[self._cursor]
        if track.channels is None or track.channels <= STEREO_CHANNELS:
            return
        new_mode = DownmixMode(mode)
        if new_mode == DownmixMode.DOWN6 and track.channels <= SURROUND_5_1_CHANNELS:
            return
        if self._downmix[self._cursor] == new_mode:
            self._downmix[self._cursor] = None
        else:
            self._downmix[self._cursor] = new_mode
        self._refresh_item(self._cursor)

    def action_done(self) -> None:
        selected_tracks = [t for t, sel in zip(self._tracks, self._selected, strict=True) if sel]
        downmix_map: dict[tuple[Path, int], DownmixMode] = {}
        for i, (track, mode) in enumerate(zip(self._tracks, self._downmix, strict=True)):
            if mode is not None and self._selected[i]:
                downmix_map[(Path(str(track.source_file)), track.index)] = mode
        self.dismiss(TrackSelection(tracks=selected_tracks, downmix=downmix_map))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is not None:
            item_id = event.item.id or ""
            if item_id.startswith("track-item-"):
                with contextlib.suppress(ValueError):
                    self._cursor = int(item_id.removeprefix("track-item-"))

    def on_click(self, event: object) -> None:
        """Handle click on the Done label."""
        # The Static with id btn-done triggers action_done via the D keybinding
        # No additional click handling needed — D key is the primary interaction


# ---------------------------------------------------------------------------
# PlaylistSelectorScreen
# ---------------------------------------------------------------------------


class PlaylistSelectorScreen(Screen[list[DiscTitle]]):
    """Screen for selecting disc playlists to demux.

    Pre-selects playlists > 10 minutes.
    """

    BINDINGS = [
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("space", "toggle_item", "Toggle"),
        Binding("d", "done", "Done"),
    ]

    MIN_DURATION_S = 600  # 10 minutes

    def __init__(self, disc_label: str, playlists: list[DiscTitle]) -> None:
        super().__init__()
        self._disc_label = disc_label
        self._playlists = playlists
        self._selected: list[bool] = [p.duration_s >= self.MIN_DURATION_S for p in playlists]
        self._cursor: int = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            f"Disc: {self._disc_label}  |  Select playlists to demux  (Space=toggle  D=done)",
            id="playlist-hint",
        )

        items: list[ListItem] = []
        for i in range(len(self._playlists)):
            label = self._render_line(i)
            items.append(ListItem(Static(label, id=f"pl-label-{i}"), id=f"pl-item-{i}"))

        yield ListView(*items, id="playlist-list")
        yield Footer()

    def _render_line(self, index: int) -> str:
        pl = self._playlists[index]
        mark = "x" if self._selected[index] else " "
        duration = _fmt_duration(pl.duration_s)
        return f"\\[{mark}]  {pl.raw_label}  ({duration})"

    def _refresh_item(self, index: int) -> None:
        label_widget = self.query_one(f"#pl-label-{index}", Static)
        label_widget.update(self._render_line(index))

    def action_move_up(self) -> None:
        lv = self.query_one("#playlist-list", ListView)
        lv.action_cursor_up()
        self._cursor = max(0, self._cursor - 1)

    def action_move_down(self) -> None:
        lv = self.query_one("#playlist-list", ListView)
        lv.action_cursor_down()
        self._cursor = min(len(self._playlists) - 1, self._cursor + 1)

    def action_toggle_item(self) -> None:
        if not self._playlists:
            return
        self._selected[self._cursor] = not self._selected[self._cursor]
        self._refresh_item(self._cursor)

    def action_done(self) -> None:
        result = [p for p, sel in zip(self._playlists, self._selected, strict=True) if sel]
        self.dismiss(result)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is not None:
            item_id = event.item.id or ""
            if item_id.startswith("pl-item-"):
                with contextlib.suppress(ValueError):
                    self._cursor = int(item_id.removeprefix("pl-item-"))


# ---------------------------------------------------------------------------
# FileSelection result
# ---------------------------------------------------------------------------


@dataclass
class FileSelection:
    """Result from FileSelectorScreen."""

    selected: list[Path]
    sar_override: set[Path]  # files with SAR override (64:45)


# ---------------------------------------------------------------------------
# FileSelectorScreen
# ---------------------------------------------------------------------------


class FileSelectorScreen(Screen[FileSelection]):
    """Screen for selecting demuxed MKV files to process.

    All files pre-selected. DVD files can be marked for SAR override.
    """

    BINDINGS = [
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("space", "toggle_item", "Toggle"),
        Binding("s", "toggle_sar", "SAR fix"),
        Binding("p", "preview", "Preview"),
        Binding("d", "done", "Done"),
    ]

    def __init__(
        self,
        files: list[tuple[Path, float, int]],  # (path, duration_s, size_bytes)
        dvd_files: set[Path] | None = None,  # which files are from DVD (SAR toggle available)
        preview_cb: Callable[[Path, str | None], None] | None = None,  # (path, aspect_override)
    ) -> None:
        super().__init__()
        self._files = files
        self._dvd_files = dvd_files or set()
        self._preview_cb = preview_cb
        self._selected: list[bool] = [True] * len(files)
        self._sar_override: list[bool] = [False] * len(files)
        self._cursor: int = 0

    def compose(self) -> ComposeResult:
        yield Header()
        has_dvd = any(f[0] in self._dvd_files for f in self._files)
        hint = "Select files  (Space=toggle  P=preview"
        if has_dvd:
            hint += "  S=SAR fix"
        hint += "  D=done)"
        yield Static(hint, id="file-hint")

        items: list[ListItem] = []
        for i in range(len(self._files)):
            label = self._render_line(i)
            items.append(ListItem(Static(label, id=f"file-label-{i}"), id=f"file-item-{i}"))

        yield ListView(*items, id="file-list")
        yield Footer()

    def _render_line(self, index: int) -> str:
        path, duration_s, size_bytes = self._files[index]
        mark = "x" if self._selected[index] else " "
        duration = _fmt_duration(duration_s)
        size = _fmt_size(size_bytes)
        sar_tag = "  SAR" if self._sar_override[index] else ""
        return f"\\[{mark}]  {path.name}  |  {duration}  {size}{sar_tag}"

    def _refresh_item(self, index: int) -> None:
        label_widget = self.query_one(f"#file-label-{index}", Static)
        label_widget.update(self._render_line(index))

    def action_move_up(self) -> None:
        lv = self.query_one("#file-list", ListView)
        lv.action_cursor_up()
        self._cursor = max(0, self._cursor - 1)

    def action_move_down(self) -> None:
        lv = self.query_one("#file-list", ListView)
        lv.action_cursor_down()
        self._cursor = min(len(self._files) - 1, self._cursor + 1)

    def action_toggle_item(self) -> None:
        if not self._files:
            return
        self._selected[self._cursor] = not self._selected[self._cursor]
        self._refresh_item(self._cursor)

    def action_toggle_sar(self) -> None:
        if not self._files:
            return
        path = self._files[self._cursor][0]
        if path not in self._dvd_files:
            return
        self._sar_override[self._cursor] = not self._sar_override[self._cursor]
        self._refresh_item(self._cursor)

    def action_preview(self) -> None:
        if not self._files or self._preview_cb is None:
            return
        path = self._files[self._cursor][0]
        aspect = "16:9" if self._sar_override[self._cursor] else None
        self._preview_cb(path, aspect)

    def action_done(self) -> None:
        selected = [f[0] for f, sel in zip(self._files, self._selected, strict=True) if sel]
        sar_set = {f[0] for f, sel, sar in zip(self._files, self._selected, self._sar_override, strict=True) if sel and sar}
        self.dismiss(FileSelection(selected=selected, sar_override=sar_set))

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is not None:
            item_id = event.item.id or ""
            if item_id.startswith("file-item-"):
                with contextlib.suppress(ValueError):
                    self._cursor = int(item_id.removeprefix("file-item-"))


# ---------------------------------------------------------------------------
# CropConfirmScreen
# ---------------------------------------------------------------------------


class CropConfirmScreen(Screen[CropRect | None]):
    """Dialog for confirming detected crop values.

    Dismisses with:
    - CropRect  — user accepted or entered custom values
    - None      — user rejected (no crop)
    """

    BINDINGS = [
        Binding("escape", "reject", "Reject"),
        Binding("a", "accept", "Accept"),
        Binding("r", "reject", "Reject"),
        Binding("e", "edit", "Edit"),
    ]

    def __init__(self, crop: CropRect, source_width: int, source_height: int) -> None:
        super().__init__()
        self._crop = crop
        self._source_width = source_width
        self._source_height = source_height
        self._edit_mode = False

    def compose(self) -> ComposeResult:
        src = f"{self._source_width}x{self._source_height}"
        crop_str = f"{self._crop.w}:{self._crop.h}:{self._crop.x}:{self._crop.y}"

        yield Header()
        yield Container(
            Static("Crop detected", id="crop-title"),
            Static(f"Source resolution : {src}", id="crop-source"),
            Static(f"Detected crop     : {crop_str}  (w:h:x:y)", id="crop-detected"),
            Static("", id="crop-error"),
            Input(
                placeholder="w:h:x:y  e.g. 1920:800:0:140",
                id="crop-input",
            ),
            Static("(A)ccept   (R)eject   (E)dit", id="crop-actions", classes="clickable-btn"),
            id="crop-dialog",
        )
        yield Footer()
        # Hide input initially
        self.query_one("#crop-input", Input).display = False

    def action_accept(self) -> None:
        if self._edit_mode:
            self._confirm_edit()
        else:
            self.dismiss(self._crop)

    def action_reject(self) -> None:
        self.dismiss(None)

    def action_edit(self) -> None:
        inp = self.query_one("#crop-input", Input)
        if not self._edit_mode:
            self._edit_mode = True
            inp.display = True
            inp.focus()
            self.query_one("#crop-actions", Static).update("(A)ccept/Confirm   (R)eject   editing...")
        else:
            self._confirm_edit()

    def _confirm_edit(self) -> None:
        inp = self.query_one("#crop-input", Input)
        error_widget = self.query_one("#crop-error", Static)
        raw = inp.value.strip()
        parts = raw.split(":")
        if len(parts) != len(["w", "h", "x", "y"]):
            error_widget.update("[red]Enter exactly 4 values: w:h:x:y[/red]")
            return
        try:
            w, h, x, y = (int(p) for p in parts)
        except ValueError:
            error_widget.update("[red]All values must be integers[/red]")
            return
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            error_widget.update("[red]w and h must be positive; x and y must be >= 0[/red]")
            return
        error_widget.update("")
        self.dismiss(CropRect(w=w, h=h, x=x, y=y))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "crop-input" and self._edit_mode:
            self._confirm_edit()


# ---------------------------------------------------------------------------
# LanguageSelectorScreen
# ---------------------------------------------------------------------------


class LanguageSelectorScreen(Screen[str]):
    """Screen for choosing a language for an 'und' track.

    Shows track info and a list of languages. User picks one with D.
    Returns the selected ISO 639-3 language code via dismiss().
    """

    BINDINGS = [
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("p", "preview_track", "Preview"),
        Binding("d", "select_lang", "Select"),
    ]

    def __init__(
        self,
        track: Track,
        lang_list: list[str],
        preview_cb: Callable[[Track], None] | None = None,
        movie: Movie | None = None,
    ) -> None:
        super().__init__()
        self._track = track
        self._lang_list = lang_list
        self._preview_cb = preview_cb
        self._movie = movie
        self._cursor: int = 0

    def compose(self) -> ComposeResult:
        yield Header()

        # File header (same style as TrackSelectorScreen)
        if self._movie is not None:
            v = self._movie.video
            filename = self._movie.main_file.name
            resolution = f"{v.width}x{v.height}"
            codec = v.codec_name.upper()
            yield Static(f"{filename}  |  {resolution}  {codec}", id="lang-header")

        # Track description
        t = self._track
        if t.track_type == TrackType.AUDIO:
            codec = t.codec_name.upper()
            layout = ""
            if t.channel_layout:
                layout = t.channel_layout.split("(")[0]
            desc = f"Audio: {codec} {layout}".strip()
        else:
            codec = t.codec_name.upper()
            desc = f"Subtitle: {codec}"

        yield Static(f"{desc}  |  Choose language  (P=preview  D=select)", id="lang-hint")

        items: list[ListItem] = []
        for i, lang in enumerate(self._lang_list):
            items.append(ListItem(Static(lang, id=f"lang-label-{i}"), id=f"lang-item-{i}"))

        yield ListView(*items, id="lang-list")
        yield Footer()

    def action_move_up(self) -> None:
        lv = self.query_one("#lang-list", ListView)
        lv.action_cursor_up()
        self._cursor = max(0, self._cursor - 1)

    def action_move_down(self) -> None:
        lv = self.query_one("#lang-list", ListView)
        lv.action_cursor_down()
        self._cursor = min(len(self._lang_list) - 1, self._cursor + 1)

    def action_preview_track(self) -> None:
        if self._preview_cb is not None:
            self._preview_cb(self._track)

    def action_select_lang(self) -> None:
        self.dismiss(self._lang_list[self._cursor])

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is not None:
            item_id = event.item.id or ""
            if item_id.startswith("lang-item-"):
                with contextlib.suppress(ValueError):
                    self._cursor = int(item_id.removeprefix("lang-item-"))


# ---------------------------------------------------------------------------
# FurnacePlanApp
# ---------------------------------------------------------------------------


class _PlanResult:
    """Intermediate container for one movie's selections."""

    def __init__(self) -> None:
        self.audio_tracks: list[Track] = []
        self.subtitle_tracks: list[Track] = []
        self.crop: CropRect | None = None


class FurnacePlanApp(App[list[_PlanResult]]):
    """Main Textual application for the planning phase.

    Iterates over each (Movie, output_path) pair, shows TrackSelectorScreen
    for audio, then for subtitles, then CropConfirmScreen.

    After all movies are processed, self.results holds a list of _PlanResult
    objects in the same order as the input list.
    """

    CSS = """
    #track-header {
        background: $surface;
        color: $text;
        padding: 0 1;
        height: 1;
    }
    #track-hint {
        color: $text-muted;
        padding: 0 1;
        height: 1;
    }
    #track-list {
        height: 1fr;
    }
    .clickable-btn {
        background: $surface;
        color: $text;
        padding: 0 1;
        height: 1;
        margin-top: 1;
    }
    #crop-dialog {
        width: 60;
        height: auto;
        border: dashed $primary;
        padding: 1 2;
        margin: 2 auto;
    }
    #crop-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #crop-error {
        height: 1;
        margin-top: 1;
    }
    #crop-input {
        margin-top: 1;
    }
    #crop-buttons {
        margin-top: 1;
        height: 3;
    }
    """

    def __init__(
        self,
        movies: list[tuple[Movie, Path]],
        preview_cb: Callable[[Track], None] | None = None,
    ) -> None:
        super().__init__()
        self._movies = movies
        self._preview_cb = preview_cb
        self._index: int = 0
        self.results: list[_PlanResult] = []

    def on_mount(self) -> None:
        self._process_next()

    def _process_next(self) -> None:
        if self._index >= len(self._movies):
            self.exit(self.results)
            return

        movie, _output_path = self._movies[self._index]
        result = _PlanResult()
        self.results.append(result)

        def _after_audio(selected_audio: TrackSelection | None) -> None:
            result.audio_tracks = selected_audio.tracks if selected_audio else []
            self.push_screen(
                TrackSelectorScreen(
                    movie=movie,
                    tracks=movie.subtitle_tracks,
                    track_type=TrackType.SUBTITLE,
                    preview_cb=self._preview_cb,
                ),
                _after_subtitles,
            )

        def _after_subtitles(selected_subs: TrackSelection | None) -> None:
            result.subtitle_tracks = selected_subs.tracks if selected_subs else []
            video = movie.video
            # Only show crop screen if there is a detectable crop to confirm;
            # callers may pre-populate a detected CropRect. Here we use a
            # sentinel: if movie has no crop hint we skip the screen.
            # The app does not run cropdetect itself; pass crop=None to skip.
            # To use the crop screen, callers should subclass or pass crop via
            # a wrapper. For now we always show the screen with a dummy crop
            # (callers must supply actual detected crop externally).
            detected_crop = getattr(movie, "_detected_crop", None)
            if detected_crop is not None:
                self.push_screen(
                    CropConfirmScreen(
                        crop=detected_crop,
                        source_width=video.width,
                        source_height=video.height,
                    ),
                    _after_crop,
                )
            else:
                _advance()

        def _after_crop(chosen_crop: CropRect | None) -> None:
            result.crop = chosen_crop
            _advance()

        def _advance() -> None:
            self._index += 1
            self._process_next()

        self.push_screen(
            TrackSelectorScreen(
                movie=movie,
                tracks=movie.audio_tracks,
                track_type=TrackType.AUDIO,
                preview_cb=self._preview_cb,
            ),
            _after_audio,
        )
