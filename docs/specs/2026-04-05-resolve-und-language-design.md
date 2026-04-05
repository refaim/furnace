# Resolve Unknown Track Language from CLI Filters

## Problem

When a track has no language tag (ffprobe returns nothing, eac3to/MakeMKV don't include `[lang]`), the analyzer assigns `"und"`. The planner's language filters always pass `und` tracks through, so they end up in the final plan with `language="und"`. Players (Plex, Jellyfin, TVs) can't auto-select or display these tracks correctly.

## Solution

After the user selects tracks (via `auto_select_from_candidates` / TUI), but before building instructions, resolve `und` languages:

- If the corresponding CLI filter has exactly 1 language — auto-assign it.
- If >1 language — show a new `LanguageSelectorScreen` for each `und` track.
- After assignment — re-sort tracks by filter order and reassign `is_default`.

## Flow in PlannerService.create_plan()

```
filter_by_lang
  -> auto_select_from_candidates (-> TUI track selector if ambiguous)
  -> resolve_und_languages (-> TUI language selector if ambiguous)
  -> re-sort + reassign is_default
  -> build_instructions
```

## Planner Changes

### New callback

`resolve_und_language(track: Track, lang_list: list[str]) -> str`

Passed into `create_plan()` alongside existing `select_tracks` callback. Called for each `und` track among already-selected tracks, only when `len(lang_list) > 1`.

### New method: `_resolve_und_languages`

```python
def _resolve_und_languages(
    self,
    tracks: list[Track],
    lang_filter: list[str],
    resolve_cb: Callable[[Track, list[str]], str],
) -> list[Track]:
```

- Finds tracks with `language == "und"` in the selected list.
- If none — returns tracks unchanged.
- If `len(lang_filter) == 1` — assigns `lang_filter[0]` to all `und` tracks.
- If `len(lang_filter) > 1` — calls `resolve_cb(track, lang_filter)` for each `und` track.
- Returns the list (mutated in place is fine, tracks are dataclasses).

### New method: `_sort_and_set_default`

```python
def _sort_and_set_default(
    self,
    tracks: list[Track],
    lang_filter: list[str],
) -> list[Track]:
```

Extracted from the existing sort logic in `_filter_*_by_lang`:
- Sort by `lang_filter` order.
- Set `is_default = True` on first track, `False` on the rest.

Used both after `_resolve_und_languages` and inside the existing filter methods (to deduplicate).

## TUI Changes

### New screen: `LanguageSelectorScreen`

In `furnace/ui/tui.py`. Minimal single-selection screen.

**Display:**
- Header: track info (type, codec, channels/format — enough to identify the track).
- `ListView` with one item per language from the filter.
- Navigation: arrows + Enter to confirm.

**Returns:** `str` — selected ISO 639-3 language code.

### New callback in cli.py

```python
def _resolve_und_language_tui(track: Track, lang_list: list[str]) -> str:
```

Same pattern as `_select_tracks_tui`: creates Textual App, pushes `LanguageSelectorScreen`, waits for dismiss, returns result.

## Edge Cases

- **No `und` tracks among selected** — no-op, flow is transparent.
- **Multiple `und` tracks** — callback called for each separately; user can assign different languages.
- **Single language in filter** — auto-assign, no TUI prompt.
- **External subtitle without language suffix** (`.srt` not `.rus.srt`) — already a `Track` with `language="und"`, handled the same way.
