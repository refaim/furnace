# Resolve Unknown Track Language — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a track has `language="und"`, let the user assign a language from the CLI filter list during planning — auto-assign if only one language in the filter, show TUI selector if multiple.

**Architecture:** New `_resolve_und_languages` and `_sort_and_set_default` methods in `PlannerService`. New `LanguageSelectorScreen` in TUI. New callback wired through `cli.py`, same pattern as the existing `track_selector`.

**Tech Stack:** Python, Textual (TUI), pytest

---

### Task 1: Extract `_sort_and_set_default` from existing filter methods

**Files:**
- Modify: `furnace/services/planner.py:214-233`
- Test: `tests/services/test_planner_lang.py`

- [ ] **Step 1: Write test for `_sort_and_set_default`**

Add to `tests/services/test_planner_lang.py`:

```python
class TestSortAndSetDefault:
    def test_sorts_by_lang_filter_order(self):
        tracks = [_audio_track("eng", index=0), _audio_track("rus", index=1), _audio_track("jpn", index=2)]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._sort_and_set_default(tracks, ["jpn", "rus", "eng"])
        assert [t.language for t in result] == ["jpn", "rus", "eng"]

    def test_first_track_is_default(self):
        tracks = [_audio_track("eng", index=0), _audio_track("rus", index=1)]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._sort_and_set_default(tracks, ["rus", "eng"])
        assert result[0].is_default is True
        assert result[1].is_default is False

    def test_empty_list(self):
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._sort_and_set_default([], ["rus"])
        assert result == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/services/test_planner_lang.py::TestSortAndSetDefault -v`
Expected: FAIL — `PlannerService` has no `_sort_and_set_default` method.

- [ ] **Step 3: Implement `_sort_and_set_default`**

Add to `furnace/services/planner.py` in `PlannerService` class (after `_filter_sub_tracks_by_lang`):

```python
def _sort_and_set_default(
    self,
    tracks: list[Track],
    lang_filter: list[str],
) -> list[Track]:
    """Sort tracks by lang_filter order and set is_default on the first."""
    if not tracks:
        return tracks
    lang_order = {lang: i for i, lang in enumerate(lang_filter)}
    tracks.sort(key=lambda t: lang_order.get(t.language, len(lang_filter)))
    for i, t in enumerate(tracks):
        t.is_default = i == 0
    return tracks
```

- [ ] **Step 4: Refactor `_filter_audio_tracks_by_lang` and `_filter_sub_tracks_by_lang` to use `_sort_and_set_default`**

Replace the sort logic in both filter methods. `_filter_audio_tracks_by_lang` becomes:

```python
def _filter_audio_tracks_by_lang(
    self, tracks: list[Track], lang_filter: list[str],
) -> list[Track]:
    """Filter audio tracks: keep matching languages + 'und', sort by lang_filter order."""
    filtered = [t for t in tracks if t.language in lang_filter or t.language == "und"]
    return self._sort_and_set_default(filtered, lang_filter)
```

`_filter_sub_tracks_by_lang` becomes:

```python
def _filter_sub_tracks_by_lang(
    self, tracks: list[Track], lang_filter: list[str],
) -> list[Track]:
    """Filter subtitle tracks: keep matching languages + 'und', discard forced, sort by lang_filter order."""
    filtered = [
        t for t in tracks
        if not t.is_forced and (t.language in lang_filter or t.language == "und")
    ]
    return self._sort_and_set_default(filtered, lang_filter)
```

- [ ] **Step 5: Run all existing lang filter tests to verify no regression**

Run: `uv run pytest tests/services/test_planner_lang.py -v`
Expected: ALL PASS (existing tests + new `TestSortAndSetDefault` tests).

- [ ] **Step 6: Run quality gates**

Run: `uv run ruff check furnace/ && uv run mypy furnace/ --strict`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add furnace/services/planner.py tests/services/test_planner_lang.py
git commit -m "refactor: extract _sort_and_set_default from language filters"
```

---

### Task 2: Implement `_resolve_und_languages` in PlannerService

**Files:**
- Modify: `furnace/services/planner.py:40-50` (constructor + type alias), `furnace/services/planner.py:90-163` (`_build_job`)
- Test: `tests/services/test_planner_lang.py`

- [ ] **Step 1: Write tests for `_resolve_und_languages`**

Add to `tests/services/test_planner_lang.py`:

```python
class TestResolveUndLanguages:
    def test_no_und_tracks_unchanged(self):
        tracks = [_audio_track("jpn", index=0), _audio_track("eng", index=1)]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        cb = MagicMock()
        result = planner._resolve_und_languages(tracks, ["jpn", "eng"], cb)
        cb.assert_not_called()
        assert [t.language for t in result] == ["jpn", "eng"]

    def test_single_lang_auto_assigns(self):
        tracks = [_audio_track("jpn", index=0), _audio_track("und", index=1)]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        cb = MagicMock()
        result = planner._resolve_und_languages(tracks, ["jpn"], cb)
        cb.assert_not_called()
        assert [t.language for t in result] == ["jpn", "jpn"]

    def test_multiple_langs_calls_callback(self):
        tracks = [_audio_track("jpn", index=0), _audio_track("und", index=1)]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        cb = MagicMock(return_value="eng")
        result = planner._resolve_und_languages(tracks, ["jpn", "eng"], cb)
        cb.assert_called_once_with(tracks[1], ["jpn", "eng"])
        assert [t.language for t in result] == ["jpn", "eng"]

    def test_multiple_und_tracks_each_gets_callback(self):
        tracks = [_audio_track("und", index=0), _audio_track("und", index=1)]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        cb = MagicMock(side_effect=["rus", "eng"])
        result = planner._resolve_und_languages(tracks, ["rus", "eng"], cb)
        assert cb.call_count == 2
        assert [t.language for t in result] == ["rus", "eng"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/services/test_planner_lang.py::TestResolveUndLanguages -v`
Expected: FAIL — no `_resolve_und_languages` method.

- [ ] **Step 3: Implement `_resolve_und_languages`**

Add to `furnace/services/planner.py` in `PlannerService` class (after `_sort_and_set_default`):

```python
def _resolve_und_languages(
    self,
    tracks: list[Track],
    lang_filter: list[str],
    resolve_cb: Callable[[Track, list[str]], str],
) -> list[Track]:
    """Assign real languages to 'und' tracks from lang_filter.

    - No und tracks: return unchanged.
    - Single lang in filter: auto-assign to all und tracks.
    - Multiple langs: call resolve_cb for each und track.
    """
    und_tracks = [t for t in tracks if t.language == "und"]
    if not und_tracks:
        return tracks
    if len(lang_filter) == 1:
        for t in und_tracks:
            t.language = lang_filter[0]
    else:
        for t in und_tracks:
            t.language = resolve_cb(t, lang_filter)
    return tracks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/services/test_planner_lang.py::TestResolveUndLanguages -v`
Expected: ALL PASS.

- [ ] **Step 5: Add `UndLanguageResolverFn` type alias and `und_resolver` parameter**

In `furnace/services/planner.py`, add the type alias after the existing `TrackSelectorFn` (line 37):

```python
# Callback type: (track, lang_list) -> chosen_language
UndLanguageResolverFn = Callable[[Track, list[str]], str]
```

Update `__init__` to accept the new callback:

```python
def __init__(
    self,
    prober: Prober,
    previewer: Previewer | None,
    track_selector: TrackSelectorFn | None = None,
    und_resolver: UndLanguageResolverFn | None = None,
) -> None:
    self._prober = prober
    self._previewer = previewer
    self._track_selector = track_selector
    self._und_resolver = und_resolver
```

- [ ] **Step 6: Wire `_resolve_und_languages` into `_build_job`**

In `_build_job`, after `selected_audio` is determined (after line 146) and before "Build audio instructions" (line 166), add:

```python
# Resolve und languages for selected audio
if self._und_resolver is not None:
    selected_audio = self._resolve_und_languages(selected_audio, audio_lang_filter, self._und_resolver)
    selected_audio = self._sort_and_set_default(selected_audio, audio_lang_filter)
```

Similarly, after `selected_subs` is determined (after line 163) and before "Build subtitle instructions" (line 173), add:

```python
# Resolve und languages for selected subs
if self._und_resolver is not None:
    selected_subs = self._resolve_und_languages(selected_subs, sub_lang_filter, self._und_resolver)
    selected_subs = self._sort_and_set_default(selected_subs, sub_lang_filter)
```

- [ ] **Step 7: Run all tests**

Run: `uv run pytest tests/services/test_planner_lang.py -v`
Expected: ALL PASS.

- [ ] **Step 8: Run quality gates**

Run: `uv run ruff check furnace/ && uv run mypy furnace/ --strict`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add furnace/services/planner.py tests/services/test_planner_lang.py
git commit -m "feat: resolve und track languages from CLI filter in planner"
```

---

### Task 3: Add `LanguageSelectorScreen` TUI

**Files:**
- Modify: `furnace/ui/tui.py`

- [ ] **Step 1: Implement `LanguageSelectorScreen`**

Add to `furnace/ui/tui.py` before the `FurnacePlanApp` class. Import `Track` is already present.

```python
# ---------------------------------------------------------------------------
# LanguageSelectorScreen
# ---------------------------------------------------------------------------

class LanguageSelectorScreen(Screen[str]):
    """Screen for choosing a language for an 'und' track.

    Shows track info and a list of languages. User picks one with Enter.
    Returns the selected ISO 639-3 language code via dismiss().
    """

    BINDINGS = [
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("enter", "select_lang", "Select"),
    ]

    def __init__(self, track: Track, lang_list: list[str]) -> None:
        super().__init__()
        self._track = track
        self._lang_list = lang_list
        self._cursor: int = 0

    def compose(self) -> ComposeResult:
        yield Header()

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

        yield Static(f"{desc}  |  Choose language  (Enter=select)", id="lang-hint")

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

    def action_select_lang(self) -> None:
        self.dismiss(self._lang_list[self._cursor])

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is not None:
            item_id = event.item.id or ""
            if item_id.startswith("lang-item-"):
                try:
                    self._cursor = int(item_id.removeprefix("lang-item-"))
                except ValueError:
                    pass
```

- [ ] **Step 2: Run quality gates**

Run: `uv run ruff check furnace/ui/tui.py && uv run mypy furnace/ui/tui.py --strict`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add furnace/ui/tui.py
git commit -m "feat: add LanguageSelectorScreen for und track language choice"
```

---

### Task 4: Wire TUI callback in `cli.py`

**Files:**
- Modify: `furnace/cli.py:91` (import), `furnace/cli.py:276-310` (callback + planner creation)

- [ ] **Step 1: Add `_resolve_und_language_tui` callback**

In `furnace/cli.py`, after the `_select_tracks_tui` function (after line 304), add:

```python
def _resolve_und_language_tui(track: Track, lang_list: list[str]) -> str:
    """Run Textual LanguageSelectorScreen synchronously for user to pick a language."""
    from textual.app import App, ComposeResult
    from textual.widgets import Header

    chosen: str = lang_list[0]  # fallback

    class _LangApp(App[str]):
        def compose(self) -> ComposeResult:
            yield Header()

        def on_mount(self) -> None:
            def _on_dismiss(result: str | None) -> None:
                nonlocal chosen
                chosen = result or lang_list[0]
                self.exit(chosen)

            self.push_screen(
                LanguageSelectorScreen(track=track, lang_list=lang_list),
                _on_dismiss,
            )

    _LangApp().run()
    return chosen
```

- [ ] **Step 2: Update the import line**

On line 91, update the import from `furnace.ui.tui` to also import `LanguageSelectorScreen`:

```python
from .ui.tui import FileSelection, FileSelectorScreen, LanguageSelectorScreen, PlaylistSelectorScreen, TrackSelectorScreen
```

- [ ] **Step 3: Pass `und_resolver` to `PlannerService`**

Update the `PlannerService` construction (around line 306) to pass the new callback:

```python
planner = PlannerService(
    prober=ffmpeg_adapter,
    previewer=mpv_adapter,
    track_selector=_select_tracks_tui if not dry_run else None,
    und_resolver=_resolve_und_language_tui if not dry_run else None,
)
```

- [ ] **Step 4: Run quality gates**

Run: `uv run ruff check furnace/ && uv run mypy furnace/ --strict`
Expected: clean.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: ALL PASS.

- [ ] **Step 6: Commit**

```bash
git add furnace/cli.py
git commit -m "feat: wire LanguageSelectorScreen into CLI plan command"
```
