# TUI: Unified Keybindings, Fix Track Preview, App Titles

## Problems

1. **Enter doesn't work on LanguageSelectorScreen** — Textual's ListView intercepts Enter before the screen binding fires.
2. **Inconsistent keybindings** — TrackSelector uses Enter=preview/D=done, FileSelector uses P=preview/D=done, LanguageSelector uses Enter=select.
3. **Track preview broken** — `_select_tracks_tui` in cli.py passes `preview_cb=None` to TrackSelectorScreen. MpvAdapter has `preview_audio`/`preview_subtitle` but no bridge from Track object.
4. **No preview on LanguageSelectorScreen** — need P to preview the und track to hear/see the language.
5. **App titles show class names** — `_LangApp`, `_SelectorApp` etc. appear in the Textual Header instead of a proper title.

## Solution

### 1. Unified keybindings

**Checkbox screens** (TrackSelector, PlaylistSelector, FileSelector):
- Space = toggle
- P = preview (where applicable; PlaylistSelector has no preview)
- D = done
- Enter — remove from bindings

**Single-select screen** (LanguageSelector):
- D = select (confirm choice)
- P = preview
- Enter — remove from bindings

Update hint strings to match.

### 2. Fix track preview

In `_select_tracks_tui` (cli.py), create a preview callback with closure over `movie` and `mpv_adapter`:

```python
def _preview_track(track: Track) -> None:
    if track.track_type == TrackType.AUDIO:
        mpv_adapter.preview_audio(movie.main_file, track.source_file, track.index)
    else:
        mpv_adapter.preview_subtitle(movie.main_file, track.source_file, track.index)
```

Pass `preview_cb=_preview_track` instead of `None`.

### 3. Preview on LanguageSelectorScreen

Add `preview_cb: Callable[[Track], None] | None = None` parameter to `LanguageSelectorScreen.__init__`. Add P binding and `action_preview` method. Same pattern as TrackSelectorScreen.

In `_resolve_und_language_tui` (cli.py), create same `_preview_track` callback (closure over `movie` and `mpv_adapter`). This requires changing `_resolve_und_language_tui` signature to also receive `movie`.

Since the planner's `UndLanguageResolverFn` is `Callable[[Track, list[str]], str]` and doesn't pass Movie, the CLI callback needs Movie from somewhere. Two options:
- Change `UndLanguageResolverFn` to include Movie — changes the protocol.
- Capture Movie in the CLI closure — the planner calls `_resolve_und_language_tui(track, lang_list)` from inside `_build_job` which has `movie`. So change `UndLanguageResolverFn` to `Callable[[Movie, Track, list[str]], str]`.

Use the protocol change: `UndLanguageResolverFn = Callable[[Movie, Track, list[str]], str]`. This is cleaner than trying to hack closures in the planner.

### 4. App titles

All mini-apps in cli.py (`_SelectorApp`, `_LangApp`, `_PlaylistApp`, `_FileApp`) get `TITLE = "Furnace"`.

## Changes Summary

**furnace/ui/tui.py:**
- TrackSelectorScreen: remove Enter binding, change action_preview to P binding, update hint
- LanguageSelectorScreen: replace Enter with D=select, add P=preview with preview_cb param, update hint

**furnace/services/planner.py:**
- `UndLanguageResolverFn`: change to `Callable[[Movie, Track, list[str]], str]`
- `_resolve_und_languages`: pass movie to callback
- `_build_job`: pass movie when calling resolve

**furnace/cli.py:**
- `_select_tracks_tui`: create and pass `_preview_track` callback
- `_resolve_und_language_tui`: accept movie param, create and pass `_preview_track` callback
- All mini-apps: add `TITLE = "Furnace"`

**tests/services/test_planner_lang.py:**
- Update `_resolve_und_languages` test callbacks to accept Movie param
