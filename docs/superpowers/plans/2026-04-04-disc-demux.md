# DVD/Blu-ray Demux & Language Filtering — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add disc demux (DVD/BD via eac3to) as a pre-scan step in `furnace plan`, and replace `--lang` with mandatory `--audio-lang`/`--sub-lang` filters that also discard forced subs.

**Architecture:** New `DiscDemuxer` service detects disc structures and orchestrates demux via a new `DiscDemuxerPort` protocol implemented by the existing `Eac3toAdapter`. The planner's `_filter_tracks_by_lang` is split into separate audio/sub language lists. Two new TUI screens handle playlist selection and demuxed-file selection.

**Tech Stack:** Python 3.13, Textual (TUI), eac3to (external tool), pytest, mypy --strict, ruff

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `furnace/core/models.py` | Add `DiscType`, `DiscSource`, `DiscPlaylist`; add `demux_dir` to `Plan` |
| Modify | `furnace/core/ports.py` | Add `DiscDemuxerPort` protocol |
| Modify | `furnace/adapters/eac3to.py` | Refactor `_run()`, add `list_playlists`, `demux_to_mkv` |
| Create | `furnace/services/disc_demuxer.py` | `DiscDemuxer` service (detect + demux orchestration) |
| Modify | `furnace/services/planner.py` | Split `lang_filter` into `audio_lang_filter` / `sub_lang_filter`; add `und` inclusion; drop forced subs |
| Modify | `furnace/services/scanner.py` | Exclude `.furnace_demux/` from recursive scan |
| Modify | `furnace/plan.py` | Bump `PLAN_VERSION` to `"2"`, handle `demux_dir` in save/load |
| Modify | `furnace/cli.py` | Replace `--lang` with `-al`/`-sl`; wire disc demux + TUI screens; cleanup in `run` |
| Modify | `furnace/ui/tui.py` | Add `PlaylistSelectorScreen`, `FileSelectorScreen` |
| Modify | `furnace/__init__.py` | Bump `VERSION` to `"1.4.0"` |
| Create | `tests/test_disc_demux.py` | Unit tests for `DiscDemuxer.detect` |
| Create | `tests/test_eac3to_playlist_parse.py` | Unit tests for eac3to playlist output parsing |
| Modify | `tests/test_plan.py` | Update `make_plan` for `demux_dir`, add roundtrip test for v2 |
| Create | `tests/services/test_planner_lang.py` | Unit tests for split language filtering + forced sub discard |

---

### Task 1: Core Models — `DiscType`, `DiscSource`, `DiscPlaylist`, Plan.demux_dir

**Files:**
- Modify: `furnace/core/models.py`

- [ ] **Step 1: Add disc-related models at the end of the enums section (after `ColorSpace`)**

In `furnace/core/models.py`, after the `ColorSpace` enum (line 76), add:

```python
class DiscType(enum.Enum):
    DVD = "dvd"
    BLURAY = "bluray"
```

After the `ScanResult` dataclass (line 102), add:

```python
@dataclass(frozen=True)
class DiscSource:
    """A detected disc structure in the source directory."""
    path: Path          # path to VIDEO_TS/ or BDMV/ directory
    disc_type: DiscType


@dataclass(frozen=True)
class DiscPlaylist:
    """One playlist/VTS entry from eac3to listing."""
    number: int         # playlist number in eac3to output
    duration_s: float   # duration in seconds
    raw_label: str      # original line from eac3to
```

- [ ] **Step 2: Add `demux_dir` field to `Plan`**

In the `Plan` dataclass (line 242), add `demux_dir` before `jobs`:

```python
@dataclass
class Plan:
    """Весь план -- сериализуется в JSON."""
    version: str                       # "2"
    furnace_version: str
    created_at: str
    source: str
    destination: str
    vmaf_enabled: bool
    demux_dir: str | None = None       # path to .furnace_demux/ or None
    jobs: list[Job] = field(default_factory=list)
```

- [ ] **Step 3: Run quality gates**

```bash
uv run ruff check furnace/core/models.py
uv run mypy furnace/core/models.py --strict
```

Expected: both pass clean.

- [ ] **Step 4: Commit**

```bash
git add furnace/core/models.py
git commit -m "feat: add DiscType, DiscSource, DiscPlaylist models and Plan.demux_dir"
```

---

### Task 2: Plan Serialization — Version Bump & `demux_dir`

**Files:**
- Modify: `furnace/plan.py`
- Modify: `tests/test_plan.py`

- [ ] **Step 1: Update test helpers and add new roundtrip test**

In `tests/test_plan.py`, update `make_plan` to include `demux_dir` and accept v2:

```python
def make_plan(jobs: list[Job] | None = None, demux_dir: str | None = None) -> Plan:
    return Plan(
        version="2",
        furnace_version="0.1.0",
        created_at="2026-04-01T00:00:00",
        source="/src",
        destination="/out",
        vmaf_enabled=False,
        demux_dir=demux_dir,
        jobs=[make_job()] if jobs is None else jobs,
    )
```

Add a new test class:

```python
class TestPlanDemuxDir:
    def test_roundtrip_with_demux_dir(self, tmp_path):
        """Plan with demux_dir survives roundtrip."""
        plan = make_plan(demux_dir="/src/.furnace_demux")
        plan_path = tmp_path / "plan.json"

        save_plan(plan, plan_path)
        loaded = load_plan(plan_path)

        assert loaded.demux_dir == "/src/.furnace_demux"

    def test_roundtrip_without_demux_dir(self, tmp_path):
        """Plan without demux_dir (None) survives roundtrip."""
        plan = make_plan(demux_dir=None)
        plan_path = tmp_path / "plan.json"

        save_plan(plan, plan_path)
        loaded = load_plan(plan_path)

        assert loaded.demux_dir is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_plan.py -v
```

Expected: FAIL — `make_plan` builds v2 but `load_plan` expects v1; `demux_dir` not handled.

- [ ] **Step 3: Update `furnace/plan.py`**

Change `PLAN_VERSION`:

```python
PLAN_VERSION = "2"
```

Update `load_plan` to read `demux_dir`:

```python
def load_plan(path: Path) -> Plan:
    """Read JSON, validate version, reconstruct all nested dataclasses."""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    version = raw.get("version")
    if version != PLAN_VERSION:
        raise ValueError(f"Unsupported plan version: {version!r} (expected {PLAN_VERSION!r})")

    return Plan(
        version=raw["version"],
        furnace_version=raw["furnace_version"],
        created_at=raw["created_at"],
        source=raw["source"],
        destination=raw["destination"],
        vmaf_enabled=raw["vmaf_enabled"],
        demux_dir=raw.get("demux_dir"),
        jobs=[_load_job(j) for j in raw.get("jobs", [])],
    )
```

- [ ] **Step 4: Update the version check test to use "2"**

In `tests/test_plan.py`, update `TestPlanVersionValidation.test_correct_version_loads`:

```python
def test_correct_version_loads(self, tmp_path):
    """Plan with version '2' loads without error."""
    plan = make_plan()
    plan_path = tmp_path / "plan.json"
    save_plan(plan, plan_path)
    loaded = load_plan(plan_path)
    assert loaded.version == "2"
```

And `test_wrong_version_raises` — it already tests version "99" so it still works.

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_plan.py -v
```

Expected: all PASS.

- [ ] **Step 6: Run quality gates**

```bash
uv run ruff check furnace/plan.py
uv run mypy furnace/plan.py --strict
```

Expected: both pass clean.

- [ ] **Step 7: Commit**

```bash
git add furnace/plan.py tests/test_plan.py
git commit -m "feat: bump plan version to 2, add demux_dir serialization"
```

---

### Task 3: Eac3to Adapter Refactor — Extract `_run()`, Add Disc Methods

**Files:**
- Modify: `furnace/adapters/eac3to.py`
- Create: `tests/test_eac3to_playlist_parse.py`

- [ ] **Step 1: Write tests for playlist parsing**

Create `tests/test_eac3to_playlist_parse.py`:

```python
from __future__ import annotations

import pytest

from furnace.adapters.eac3to import Eac3toAdapter


class TestParsePlaylistOutput:
    """Test parsing of eac3to playlist listing output."""

    def test_parse_bluray_playlists(self):
        """Parse typical Blu-ray eac3to output."""
        output = (
            "M2TS, 1 video track, 3 audio tracks, 2 subtitle tracks, 1:45:23\n"
            "\n"
            "1) 00800.mpls, 1:45:23\n"
            "2) 00801.mpls, 0:02:15\n"
            "3) 00802.mpls, 0:31:10\n"
        )
        result = Eac3toAdapter._parse_playlist_output(output)
        assert len(result) == 3
        assert result[0].number == 1
        assert result[0].duration_s == pytest.approx(6323.0)
        assert "00800.mpls" in result[0].raw_label
        assert result[1].number == 2
        assert result[1].duration_s == pytest.approx(135.0)
        assert result[2].number == 3
        assert result[2].duration_s == pytest.approx(1870.0)

    def test_parse_dvd_playlists(self):
        """Parse typical DVD eac3to output."""
        output = (
            "1) 01 - Title 1, 1:32:05\n"
            "2) 02 - Title 2, 0:05:30\n"
        )
        result = Eac3toAdapter._parse_playlist_output(output)
        assert len(result) == 2
        assert result[0].number == 1
        assert result[0].duration_s == pytest.approx(5525.0)
        assert result[1].number == 2
        assert result[1].duration_s == pytest.approx(330.0)

    def test_parse_empty_output(self):
        """Empty output returns empty list."""
        result = Eac3toAdapter._parse_playlist_output("")
        assert result == []

    def test_parse_lines_without_playlist_numbers(self):
        """Lines without N) prefix are ignored."""
        output = (
            "M2TS, 1 video track\n"
            "\n"
            "1) 00800.mpls, 1:00:00\n"
        )
        result = Eac3toAdapter._parse_playlist_output(output)
        assert len(result) == 1

    def test_parse_duration_hours_minutes_seconds(self):
        """Various duration formats are handled."""
        output = "1) test, 2:03:45\n"
        result = Eac3toAdapter._parse_playlist_output(output)
        assert result[0].duration_s == pytest.approx(2 * 3600 + 3 * 60 + 45)

    def test_parse_duration_minutes_seconds(self):
        """Duration without hours."""
        output = "1) test, 5:30\n"
        result = Eac3toAdapter._parse_playlist_output(output)
        assert result[0].duration_s == pytest.approx(330.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_eac3to_playlist_parse.py -v
```

Expected: FAIL — `_parse_playlist_output` does not exist.

- [ ] **Step 3: Refactor eac3to adapter and add new methods**

Replace the entire content of `furnace/adapters/eac3to.py`:

```python
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

from furnace.core.models import DiscPlaylist

from ._subprocess import OutputCallback, run_tool

logger = logging.getLogger(__name__)

# Regex: "1) 00800.mpls, 1:45:23" or "1) 01 - Title 1, 1:32:05"
_PLAYLIST_RE = re.compile(r"^(\d+)\)\s+(.+),\s+(\d+:\d{2}(?::\d{2})?)$")


class Eac3toAdapter:
    """Implements AudioDecoder and DiscDemuxerPort via eac3to."""

    def __init__(
        self,
        eac3to_path: Path,
        on_output: OutputCallback = None,
        log_dir: Path | None = None,
    ) -> None:
        self._eac3to = eac3to_path
        self._on_output = on_output
        self._log_dir = log_dir

    def set_log_dir(self, log_dir: Path | None) -> None:
        self._log_dir = log_dir

    def _log_path(self, label: str) -> Path | None:
        if self._log_dir is None:
            return None
        return self._log_dir / f"eac3to_{label}.log"

    @staticmethod
    def _delay_arg(delay_ms: int) -> list[str]:
        """Return eac3to delay argument as list, or empty list if delay is 0."""
        if delay_ms == 0:
            return []
        if delay_ms > 0:
            return [f"+{delay_ms}ms"]
        return [f"{delay_ms}ms"]

    def _run(
        self,
        args: list[str],
        log_label: str,
        on_output: OutputCallback = None,
    ) -> tuple[int, str]:
        """Common eac3to invocation with logging and progress.

        Returns (return_code, combined_output).
        """
        cmd = [str(self._eac3to), *args, "-progressnumbers"]
        rc, output = run_tool(
            cmd,
            on_output=on_output or self._on_output,
            log_path=self._log_path(log_label),
        )
        return rc, output

    # -- AudioDecoder protocol -------------------------------------------------

    def denormalize(self, input_path: Path, output_path: Path, delay_ms: int) -> int:
        """eac3to input output -removeDialnorm [+Xms/-Xms]"""
        rc, _output = self._run(
            [str(input_path), str(output_path), "-removeDialnorm",
             *self._delay_arg(delay_ms)],
            "denorm",
        )
        return rc

    def decode_lossless(self, input_path: Path, output_path: Path, delay_ms: int) -> int:
        """eac3to input output.wav -removeDialnorm [+Xms/-Xms]"""
        rc, _output = self._run(
            [str(input_path), str(output_path), "-removeDialnorm",
             *self._delay_arg(delay_ms)],
            "decode",
        )
        return rc

    # -- DiscDemuxerPort protocol ----------------------------------------------

    def list_playlists(self, disc_path: Path) -> list[DiscPlaylist]:
        """Run eac3to on disc path, parse playlist listing."""
        cmd = [str(self._eac3to), str(disc_path)]
        rc, output = run_tool(cmd, log_path=self._log_path("list_playlists"))
        if rc != 0:
            raise RuntimeError(
                f"eac3to listing failed for {disc_path} (rc={rc})"
            )
        return self._parse_playlist_output(output)

    def demux_to_mkv(
        self,
        disc_path: Path,
        playlist_num: int,
        output_mkv: Path,
        on_progress: Callable[[str], None] | None = None,
    ) -> int:
        """Demux one playlist to MKV."""
        rc, _output = self._run(
            [str(disc_path), f"{playlist_num})", str(output_mkv)],
            f"demux_p{playlist_num}",
            on_output=on_progress,
        )
        return rc

    # -- Parsing ---------------------------------------------------------------

    @staticmethod
    def _parse_playlist_output(output: str) -> list[DiscPlaylist]:
        """Parse eac3to listing output into DiscPlaylist objects.

        Expected line format: "N) description, H:MM:SS" or "N) description, M:SS"
        """
        results: list[DiscPlaylist] = []
        for line in output.splitlines():
            line = line.strip()
            m = _PLAYLIST_RE.match(line)
            if not m:
                continue
            number = int(m.group(1))
            label = m.group(2).strip()
            duration_str = m.group(3)
            duration_s = _parse_duration(duration_str)
            results.append(DiscPlaylist(
                number=number,
                duration_s=duration_s,
                raw_label=f"{number}) {label}, {duration_str}",
            ))
        return results


def _parse_duration(s: str) -> float:
    """Parse 'H:MM:SS' or 'M:SS' into total seconds."""
    parts = s.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return 0.0
```

- [ ] **Step 4: Run parsing tests to verify they pass**

```bash
uv run pytest tests/test_eac3to_playlist_parse.py -v
```

Expected: all PASS.

- [ ] **Step 5: Run quality gates**

```bash
uv run ruff check furnace/adapters/eac3to.py
uv run mypy furnace/adapters/eac3to.py --strict
```

Expected: both pass clean.

- [ ] **Step 6: Commit**

```bash
git add furnace/adapters/eac3to.py tests/test_eac3to_playlist_parse.py
git commit -m "refactor: extract eac3to _run(), add list_playlists and demux_to_mkv"
```

---

### Task 4: `DiscDemuxerPort` Protocol

**Files:**
- Modify: `furnace/core/ports.py`

- [ ] **Step 1: Add `DiscDemuxerPort` protocol**

At the end of `furnace/core/ports.py`, add:

```python
from .models import CropRect, DiscPlaylist, VideoParams
```

(Update the existing import line to include `DiscPlaylist`.)

Then add the protocol:

```python
@runtime_checkable
class DiscDemuxerPort(Protocol):
    """Demux disc structures (DVD/Blu-ray) via eac3to or similar."""

    def list_playlists(self, disc_path: Path) -> list[DiscPlaylist]:
        """List playlists/VTS from a disc structure. Returns parsed entries."""
        ...

    def demux_to_mkv(
        self,
        disc_path: Path,
        playlist_num: int,
        output_mkv: Path,
        on_progress: Callable[[str], None] | None = None,
    ) -> int:
        """Demux one playlist to MKV. Returns return code (0 = ok)."""
        ...
```

- [ ] **Step 2: Run quality gates**

```bash
uv run ruff check furnace/core/ports.py
uv run mypy furnace/core/ports.py --strict
```

Expected: both pass clean.

- [ ] **Step 3: Commit**

```bash
git add furnace/core/ports.py
git commit -m "feat: add DiscDemuxerPort protocol"
```

---

### Task 5: `DiscDemuxer` Service

**Files:**
- Create: `furnace/services/disc_demuxer.py`
- Create: `tests/test_disc_demux.py`

- [ ] **Step 1: Write tests for disc detection**

Create `tests/test_disc_demux.py`:

```python
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from furnace.core.models import DiscType
from furnace.services.disc_demuxer import DiscDemuxer


class TestDiscDetection:
    def test_detect_bluray(self, tmp_path: Path) -> None:
        """Detects BDMV directory as Blu-ray."""
        bdmv = tmp_path / "movie" / "BDMV"
        bdmv.mkdir(parents=True)
        demuxer = DiscDemuxer(disc_demuxer_port=MagicMock())
        discs = demuxer.detect(tmp_path)
        assert len(discs) == 1
        assert discs[0].disc_type == DiscType.BLURAY
        assert discs[0].path == bdmv

    def test_detect_dvd(self, tmp_path: Path) -> None:
        """Detects VIDEO_TS directory as DVD."""
        video_ts = tmp_path / "movie" / "VIDEO_TS"
        video_ts.mkdir(parents=True)
        demuxer = DiscDemuxer(disc_demuxer_port=MagicMock())
        discs = demuxer.detect(tmp_path)
        assert len(discs) == 1
        assert discs[0].disc_type == DiscType.DVD
        assert discs[0].path == video_ts

    def test_detect_multiple_discs(self, tmp_path: Path) -> None:
        """Detects multiple disc structures."""
        (tmp_path / "bd" / "BDMV").mkdir(parents=True)
        (tmp_path / "dvd" / "VIDEO_TS").mkdir(parents=True)
        demuxer = DiscDemuxer(disc_demuxer_port=MagicMock())
        discs = demuxer.detect(tmp_path)
        assert len(discs) == 2
        types = {d.disc_type for d in discs}
        assert types == {DiscType.DVD, DiscType.BLURAY}

    def test_detect_no_discs(self, tmp_path: Path) -> None:
        """Returns empty list when no disc structures found."""
        (tmp_path / "movie.mkv").touch()
        demuxer = DiscDemuxer(disc_demuxer_port=MagicMock())
        discs = demuxer.detect(tmp_path)
        assert discs == []

    def test_detect_recursive(self, tmp_path: Path) -> None:
        """Finds disc structures at arbitrary depth."""
        deep = tmp_path / "a" / "b" / "c" / "BDMV"
        deep.mkdir(parents=True)
        demuxer = DiscDemuxer(disc_demuxer_port=MagicMock())
        discs = demuxer.detect(tmp_path)
        assert len(discs) == 1
        assert discs[0].path == deep

    def test_detect_ignores_furnace_demux_dir(self, tmp_path: Path) -> None:
        """Ignores .furnace_demux directory."""
        demux_dir = tmp_path / ".furnace_demux"
        demux_dir.mkdir()
        (demux_dir / "BDMV").mkdir()  # should be ignored
        (tmp_path / "real" / "BDMV").mkdir(parents=True)
        demuxer = DiscDemuxer(disc_demuxer_port=MagicMock())
        discs = demuxer.detect(tmp_path)
        assert len(discs) == 1
        assert ".furnace_demux" not in str(discs[0].path)


class TestDemuxDirManagement:
    def test_skip_already_demuxed(self, tmp_path: Path) -> None:
        """MKV with .done marker is not re-demuxed."""
        demux_dir = tmp_path / ".furnace_demux"
        demux_dir.mkdir()
        mkv = demux_dir / "movie_playlist_1.mkv"
        mkv.write_bytes(b"x" * 1000)
        (demux_dir / "movie_playlist_1.mkv.done").touch()

        port = MagicMock()
        demuxer = DiscDemuxer(disc_demuxer_port=port)

        from furnace.core.models import DiscPlaylist, DiscSource, DiscType
        disc = DiscSource(path=tmp_path / "BDMV", disc_type=DiscType.BLURAY)
        playlist = DiscPlaylist(number=1, duration_s=6000.0, raw_label="1) test, 1:40:00")

        result = demuxer.demux(
            discs=[disc],
            selected_playlists={disc: [playlist]},
            demux_dir=demux_dir,
        )
        # Port should NOT have been called
        port.demux_to_mkv.assert_not_called()
        assert len(result) == 1
        assert result[0] == mkv

    def test_delete_partial_and_redemux(self, tmp_path: Path) -> None:
        """MKV without .done marker is deleted and re-demuxed."""
        demux_dir = tmp_path / ".furnace_demux"
        demux_dir.mkdir()
        mkv = demux_dir / "BDMV_playlist_1.mkv"
        mkv.write_bytes(b"partial data")
        # No .done marker

        port = MagicMock()
        port.demux_to_mkv.return_value = 0

        demuxer = DiscDemuxer(disc_demuxer_port=port)

        from furnace.core.models import DiscPlaylist, DiscSource, DiscType
        disc = DiscSource(path=tmp_path / "BDMV", disc_type=DiscType.BLURAY)
        playlist = DiscPlaylist(number=1, duration_s=6000.0, raw_label="1) test, 1:40:00")

        result = demuxer.demux(
            discs=[disc],
            selected_playlists={disc: [playlist]},
            demux_dir=demux_dir,
        )
        port.demux_to_mkv.assert_called_once()
        assert len(result) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_disc_demux.py -v
```

Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `DiscDemuxer` service**

Create `furnace/services/disc_demuxer.py`:

```python
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from ..core.models import DiscPlaylist, DiscSource, DiscType
from ..core.ports import DiscDemuxerPort

logger = logging.getLogger(__name__)

_DISC_DIR_NAMES: dict[str, DiscType] = {
    "VIDEO_TS": DiscType.DVD,
    "BDMV": DiscType.BLURAY,
}


class DiscDemuxer:
    """Detect disc structures and orchestrate demux via eac3to."""

    def __init__(self, disc_demuxer_port: DiscDemuxerPort) -> None:
        self._port = disc_demuxer_port

    def detect(self, source: Path) -> list[DiscSource]:
        """Recursively search source for VIDEO_TS/ and BDMV/ directories."""
        results: list[DiscSource] = []
        for path in sorted(source.rglob("*")):
            if not path.is_dir():
                continue
            # Skip .furnace_demux directory tree
            if ".furnace_demux" in path.parts:
                continue
            disc_type = _DISC_DIR_NAMES.get(path.name)
            if disc_type is not None:
                results.append(DiscSource(path=path, disc_type=disc_type))
                logger.info("Detected %s at %s", disc_type.value.upper(), path)
        return results

    def demux(
        self,
        discs: list[DiscSource],
        selected_playlists: dict[DiscSource, list[DiscPlaylist]],
        demux_dir: Path,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[Path]:
        """Demux selected playlists to MKV files.

        - Skips playlists with existing .done marker.
        - Deletes MKV without .done marker (partial) and re-demuxes.
        - Raises RuntimeError on demux failure.

        Returns list of paths to demuxed MKV files.
        """
        demux_dir.mkdir(parents=True, exist_ok=True)
        result_paths: list[Path] = []

        for disc in discs:
            playlists = selected_playlists.get(disc, [])
            disc_label = disc.path.parent.name
            for playlist in playlists:
                mkv_name = f"{disc_label}_playlist_{playlist.number}.mkv"
                mkv_path = demux_dir / mkv_name
                done_marker = demux_dir / f"{mkv_name}.done"

                if done_marker.exists() and mkv_path.exists():
                    logger.info("Already demuxed, skipping: %s", mkv_name)
                    result_paths.append(mkv_path)
                    continue

                # Delete partial file if exists without done marker
                if mkv_path.exists():
                    logger.warning("Deleting partial demux: %s", mkv_name)
                    mkv_path.unlink()
                if done_marker.exists():
                    done_marker.unlink()

                logger.info(
                    "Demuxing playlist %d from %s -> %s",
                    playlist.number, disc.path, mkv_name,
                )
                rc = self._port.demux_to_mkv(
                    disc.path, playlist.number, mkv_path, on_progress,
                )
                if rc != 0:
                    raise RuntimeError(
                        f"Demux failed for {disc.path} playlist {playlist.number} (rc={rc})"
                    )

                # Write success marker
                done_marker.touch()
                result_paths.append(mkv_path)

        return result_paths
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_disc_demux.py -v
```

Expected: all PASS.

- [ ] **Step 5: Run quality gates**

```bash
uv run ruff check furnace/services/disc_demuxer.py
uv run mypy furnace/services/disc_demuxer.py --strict
```

Expected: both pass clean.

- [ ] **Step 6: Commit**

```bash
git add furnace/services/disc_demuxer.py tests/test_disc_demux.py
git commit -m "feat: add DiscDemuxer service with detect and demux"
```

---

### Task 6: Language Filtering — Split `--lang` Into `--audio-lang`/`--sub-lang`

**Files:**
- Modify: `furnace/services/planner.py`
- Create: `tests/services/test_planner_lang.py`

- [ ] **Step 1: Write tests for split language filtering**

Create `tests/services/test_planner_lang.py`:

```python
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from furnace.core.models import (
    AudioCodecId,
    SubtitleCodecId,
    Track,
    TrackType,
)
from furnace.services.planner import PlannerService


def _audio_track(language: str, index: int = 0) -> Track:
    return Track(
        index=index,
        track_type=TrackType.AUDIO,
        codec_name="aac",
        codec_id=AudioCodecId.AAC_LC,
        language=language,
        title="",
        is_default=False,
        is_forced=False,
        source_file=Path("/src/movie.mkv"),
        channels=2,
        bitrate=192000,
    )


def _sub_track(language: str, index: int = 0, is_forced: bool = False) -> Track:
    return Track(
        index=index,
        track_type=TrackType.SUBTITLE,
        codec_name="subrip",
        codec_id=SubtitleCodecId.SRT,
        language=language,
        title="",
        is_default=False,
        is_forced=is_forced,
        source_file=Path("/src/movie.mkv"),
    )


class TestAudioLangFilter:
    def test_filters_by_audio_lang(self):
        """Only tracks matching audio_lang_filter are included."""
        tracks = [_audio_track("jpn"), _audio_track("eng"), _audio_track("rus")]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_audio_tracks_by_lang(tracks, ["jpn"])
        assert [t.language for t in result] == ["jpn"]

    def test_und_always_included(self):
        """Tracks with language 'und' are always included."""
        tracks = [_audio_track("jpn"), _audio_track("und")]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_audio_tracks_by_lang(tracks, ["jpn"])
        assert [t.language for t in result] == ["jpn", "und"]

    def test_multiple_langs(self):
        """Multiple languages filter correctly."""
        tracks = [_audio_track("jpn"), _audio_track("eng"), _audio_track("rus")]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_audio_tracks_by_lang(tracks, ["jpn", "eng"])
        assert [t.language for t in result] == ["jpn", "eng"]


class TestSubLangFilter:
    def test_filters_by_sub_lang(self):
        """Only tracks matching sub_lang_filter are included."""
        tracks = [_sub_track("rus"), _sub_track("eng"), _sub_track("jpn")]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_sub_tracks_by_lang(tracks, ["rus", "eng"])
        assert [t.language for t in result] == ["rus", "eng"]

    def test_forced_subs_discarded(self):
        """Forced subtitles are discarded regardless of language."""
        tracks = [_sub_track("rus"), _sub_track("eng", is_forced=True)]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_sub_tracks_by_lang(tracks, ["rus", "eng"])
        assert [t.language for t in result] == ["rus"]

    def test_und_always_included(self):
        """Tracks with language 'und' are always included."""
        tracks = [_sub_track("rus"), _sub_track("und")]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_sub_tracks_by_lang(tracks, ["rus"])
        assert [t.language for t in result] == ["rus", "und"]

    def test_forced_und_discarded(self):
        """Forced 'und' tracks are also discarded."""
        tracks = [_sub_track("rus"), _sub_track("und", is_forced=True)]
        planner = PlannerService(prober=MagicMock(), previewer=None)
        result = planner._filter_sub_tracks_by_lang(tracks, ["rus"])
        assert [t.language for t in result] == ["rus"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/services/test_planner_lang.py -v
```

Expected: FAIL — methods `_filter_audio_tracks_by_lang` and `_filter_sub_tracks_by_lang` don't exist.

- [ ] **Step 3: Update `PlannerService` to use split language filters**

In `furnace/services/planner.py`:

**Change `create_plan` signature** — replace `lang_filter: list[str] | None` with two separate filters:

```python
def create_plan(
    self,
    movies: list[tuple[Movie, Path]],
    audio_lang_filter: list[str],
    sub_lang_filter: list[str],
    vmaf_enabled: bool,
    dry_run: bool,
) -> Plan:
```

**Update the call to `_build_job`:**

```python
job = self._build_job(movie, output_path, audio_lang_filter, sub_lang_filter, vmaf_enabled, dry_run)
```

**Change `_build_job` signature:**

```python
def _build_job(
    self,
    movie: Movie,
    output_path: Path,
    audio_lang_filter: list[str],
    sub_lang_filter: list[str],
    vmaf_enabled: bool,
    dry_run: bool,
) -> Job | None:
```

**Replace the audio/sub selection blocks** in `_build_job`. Change:

```python
# Auto-select audio tracks
selected_audio = self._auto_select_tracks(movie.audio_tracks, lang_filter)
if selected_audio is None:
    candidates = self._filter_tracks_by_lang(movie.audio_tracks, lang_filter)
```

To:

```python
# Auto-select audio tracks
audio_candidates = self._filter_audio_tracks_by_lang(movie.audio_tracks, audio_lang_filter)
selected_audio = self._auto_select_from_candidates(audio_candidates)
if selected_audio is None:
    if self._track_selector is not None:
```

(And use `audio_candidates` as the candidates passed to `self._track_selector`.)

Similarly for subtitles, replace:

```python
selected_subs = self._auto_select_tracks(movie.subtitle_tracks, lang_filter)
if selected_subs is None:
    candidates = self._filter_tracks_by_lang(movie.subtitle_tracks, lang_filter)
```

With:

```python
sub_candidates = self._filter_sub_tracks_by_lang(movie.subtitle_tracks, sub_lang_filter)
selected_subs = self._auto_select_from_candidates(sub_candidates)
if selected_subs is None:
    if self._track_selector is not None:
```

(And use `sub_candidates` as candidates.)

**Replace `_filter_tracks_by_lang` and `_auto_select_tracks`** with three new methods:

```python
def _filter_audio_tracks_by_lang(
    self, tracks: list[Track], lang_filter: list[str],
) -> list[Track]:
    """Filter audio tracks: keep only matching languages + 'und'."""
    return [t for t in tracks if t.language in lang_filter or t.language == "und"]

def _filter_sub_tracks_by_lang(
    self, tracks: list[Track], lang_filter: list[str],
) -> list[Track]:
    """Filter subtitle tracks: keep matching languages + 'und', discard forced."""
    return [
        t for t in tracks
        if not t.is_forced and (t.language in lang_filter or t.language == "und")
    ]

def _auto_select_from_candidates(
    self, candidates: list[Track],
) -> list[Track] | None:
    """If exactly one track per language -> auto-select.
    If multiple tracks for any language -> return None (caller shows TUI).
    """
    if not candidates:
        return candidates

    lang_groups: dict[str, list[Track]] = {}
    for track in candidates:
        lang_groups.setdefault(track.language, []).append(track)

    for group in lang_groups.values():
        if len(group) > 1:
            return None

    return candidates
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/services/test_planner_lang.py -v
```

Expected: all PASS.

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest tests/ -q
```

Expected: some tests may fail due to `create_plan` signature change — those will be fixed in the CLI task. The planner lang tests should pass.

- [ ] **Step 6: Run quality gates on planner**

```bash
uv run ruff check furnace/services/planner.py
uv run mypy furnace/services/planner.py --strict
```

Expected: both pass clean.

- [ ] **Step 7: Commit**

```bash
git add furnace/services/planner.py tests/services/test_planner_lang.py
git commit -m "feat: split lang filter into audio_lang/sub_lang, discard forced subs"
```

---

### Task 7: Scanner — Exclude `.furnace_demux/`

**Files:**
- Modify: `furnace/services/scanner.py`
- Modify: `tests/services/test_scanner.py`

- [ ] **Step 1: Add test**

In `tests/services/test_scanner.py`, add a new test class:

```python
class TestScannerIgnoresDemuxDir:
    def test_scanner_ignores_furnace_demux(self, tmp_path: Path) -> None:
        """Scanner skips .furnace_demux directory."""
        # Regular file
        movie = tmp_path / "movie.mkv"
        movie.touch()
        # Demuxed file inside .furnace_demux
        demux_dir = tmp_path / ".furnace_demux"
        demux_dir.mkdir()
        demuxed = demux_dir / "demuxed.mkv"
        demuxed.touch()

        dest = tmp_path / "output"
        dest.mkdir()

        scanner = make_scanner()
        results = scanner.scan(tmp_path, dest)

        found_files = [r.main_file for r in results]
        assert movie in found_files
        assert demuxed not in found_files
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/services/test_scanner.py::TestScannerIgnoresDemuxDir -v
```

Expected: FAIL — scanner currently picks up `.furnace_demux` contents.

- [ ] **Step 3: Update Scanner to skip `.furnace_demux`**

In `furnace/services/scanner.py`, in the `scan` method, add a check inside the `for path in sorted(source.rglob("*")):` loop, right after `if not path.is_file(): continue`:

```python
# Skip .furnace_demux directory
if ".furnace_demux" in path.parts:
    continue
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/services/test_scanner.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add furnace/services/scanner.py tests/services/test_scanner.py
git commit -m "fix: scanner excludes .furnace_demux directory"
```

---

### Task 8: TUI — Playlist Selector & File Selector Screens

**Files:**
- Modify: `furnace/ui/tui.py`

- [ ] **Step 1: Add `PlaylistSelectorScreen`**

In `furnace/ui/tui.py`, after the `TrackSelectorScreen` class and before `CropConfirmScreen`, add:

```python
# ---------------------------------------------------------------------------
# PlaylistSelectorScreen
# ---------------------------------------------------------------------------

class PlaylistSelectorScreen(Screen[list["DiscPlaylist"]]):
    """Screen for selecting disc playlists to demux.

    Pre-selects playlists > 10 minutes.
    Returns list of selected DiscPlaylist objects via dismiss().
    """

    BINDINGS = [
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("space", "toggle_item", "Toggle"),
        Binding("d", "done", "Done"),
    ]

    MIN_DURATION_S = 600  # 10 minutes

    def __init__(self, disc_label: str, playlists: list["DiscPlaylist"]) -> None:
        super().__init__()
        self._disc_label = disc_label
        self._playlists = playlists
        self._selected: list[bool] = [
            p.duration_s >= self.MIN_DURATION_S for p in playlists
        ]
        self._cursor: int = 0

    def compose(self) -> ComposeResult:
        from furnace.core.models import DiscPlaylist

        yield Header()
        yield Static(
            f"Disc: {self._disc_label}  |  Select playlists to demux  (Space=toggle  D=done)",
            id="playlist-hint",
        )

        items: list[ListItem] = []
        for i, pl in enumerate(self._playlists):
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
        result = [p for p, sel in zip(self._playlists, self._selected) if sel]
        self.dismiss(result)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is not None:
            item_id = event.item.id or ""
            if item_id.startswith("pl-item-"):
                try:
                    self._cursor = int(item_id.removeprefix("pl-item-"))
                except ValueError:
                    pass
```

- [ ] **Step 2: Add `FileSelectorScreen`**

After `PlaylistSelectorScreen`, add:

```python
# ---------------------------------------------------------------------------
# FileSelectorScreen
# ---------------------------------------------------------------------------

class FileSelectorScreen(Screen[list[Path]]):
    """Screen for selecting demuxed MKV files to process.

    All files pre-selected. User can toggle and preview via mpv.
    Returns list of selected Path objects via dismiss().
    """

    BINDINGS = [
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("space", "toggle_item", "Toggle"),
        Binding("p", "preview", "Preview"),
        Binding("d", "done", "Done"),
    ]

    def __init__(
        self,
        files: list[tuple[Path, float, int]],  # (path, duration_s, size_bytes)
        preview_cb: Callable[[Path], None] | None = None,
    ) -> None:
        super().__init__()
        self._files = files
        self._preview_cb = preview_cb
        self._selected: list[bool] = [True] * len(files)
        self._cursor: int = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            "Select files to process  (Space=toggle  P=preview  D=done)",
            id="file-hint",
        )

        items: list[ListItem] = []
        for i, (path, duration_s, size_bytes) in enumerate(self._files):
            label = self._render_line(i)
            items.append(ListItem(Static(label, id=f"file-label-{i}"), id=f"file-item-{i}"))

        yield ListView(*items, id="file-list")
        yield Footer()

    def _render_line(self, index: int) -> str:
        path, duration_s, size_bytes = self._files[index]
        mark = "x" if self._selected[index] else " "
        duration = _fmt_duration(duration_s)
        size = _fmt_size(size_bytes)
        return f"\\[{mark}]  {path.name}  |  {duration}  {size}"

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

    def action_preview(self) -> None:
        if not self._files or self._preview_cb is None:
            return
        path = self._files[self._cursor][0]
        self._preview_cb(path)

    def action_done(self) -> None:
        result = [f[0] for f, sel in zip(self._files, self._selected) if sel]
        self.dismiss(result)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is not None:
            item_id = event.item.id or ""
            if item_id.startswith("file-item-"):
                try:
                    self._cursor = int(item_id.removeprefix("file-item-"))
                except ValueError:
                    pass
```

- [ ] **Step 3: Run quality gates**

```bash
uv run ruff check furnace/ui/tui.py
uv run mypy furnace/ui/tui.py --strict
```

Expected: both pass clean.

- [ ] **Step 4: Commit**

```bash
git add furnace/ui/tui.py
git commit -m "feat: add PlaylistSelectorScreen and FileSelectorScreen TUI screens"
```

---

### Task 9: CLI — Wire Everything Together

**Files:**
- Modify: `furnace/cli.py`

- [ ] **Step 1: Update `plan` command signature**

Replace the current `plan` function arguments:

```python
@app.command()
def plan(
    source: Path = typer.Argument(..., help="Video file or directory"),
    output: Path = typer.Option(..., "-o", help="Output directory"),
    audio_lang: list[str] = typer.Option(..., "--audio-lang", "-al", help="Audio language filter (e.g. jpn)"),
    sub_lang: list[str] = typer.Option(..., "--sub-lang", "-sl", help="Subtitle language filter (e.g. rus eng)"),
    names: Path | None = typer.Option(None, "--names", help="Rename map file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan without saving"),
    vmaf: bool = typer.Option(False, "--vmaf", help="Enable VMAF"),
    config: Path | None = typer.Option(None, "--config", help="Path to config file"),
) -> None:
```

- [ ] **Step 2: Add disc demux phase before scanning**

After the config/logging setup (after step 3 in existing code, before step 5 `Scanner.scan()`), add the disc demux phase:

```python
    # --- Disc demux phase ---
    from .adapters.eac3to import Eac3toAdapter as _Eac3toAdapterPlan
    from .services.disc_demuxer import DiscDemuxer

    eac3to_adapter = _Eac3toAdapterPlan(cfg.eac3to)
    disc_demuxer = DiscDemuxer(disc_demuxer_port=eac3to_adapter)
    discs = disc_demuxer.detect(source)

    demux_dir: Path | None = None
    demuxed_mkv_files: list[Path] = []

    if discs and not dry_run:
        demux_dir = source / ".furnace_demux"

        # For each disc, list playlists and let user choose
        from .ui.tui import PlaylistSelectorScreen, FileSelectorScreen
        from textual.app import App, ComposeResult
        from textual.widgets import Header

        all_selected_playlists: dict[object, list[object]] = {}
        for disc in discs:
            playlists = eac3to_adapter.list_playlists(disc.path)
            if not playlists:
                logger.warning("No playlists found in %s", disc.path)
                continue

            disc_label = f"{disc.disc_type.value.upper()}: {disc.path.parent.name}"

            selected: list[object] = []

            class _PlaylistApp(App[list[object]]):
                def compose(self) -> ComposeResult:
                    yield Header()

                def on_mount(self_app) -> None:
                    def _on_dismiss(result: list[object] | None) -> None:
                        nonlocal selected
                        selected = result or []
                        self_app.exit(selected)

                    self_app.push_screen(
                        PlaylistSelectorScreen(disc_label=disc_label, playlists=playlists),
                        _on_dismiss,
                    )

            _PlaylistApp().run()

            if selected:
                all_selected_playlists[disc] = selected

        # Demux selected playlists
        if all_selected_playlists:
            demuxed_mkv_files = disc_demuxer.demux(
                discs=discs,
                selected_playlists=all_selected_playlists,
                demux_dir=demux_dir,
            )

        # If multiple demuxed files, let user choose which to process
        if len(demuxed_mkv_files) > 1:
            # Probe each for duration and size
            file_info: list[tuple[Path, float, int]] = []
            for mkv in demuxed_mkv_files:
                try:
                    probe_data = ffmpeg_adapter.probe(mkv)
                    dur = float(probe_data.get("format", {}).get("duration", 0))
                except Exception:
                    dur = 0.0
                size = mkv.stat().st_size if mkv.exists() else 0
                file_info.append((mkv, dur, size))

            chosen: list[Path] = []

            class _FileApp(App[list[Path]]):
                def compose(self) -> ComposeResult:
                    yield Header()

                def on_mount(self_app) -> None:
                    def _on_dismiss(result: list[Path] | None) -> None:
                        nonlocal chosen
                        chosen = result or []
                        self_app.exit(chosen)

                    self_app.push_screen(
                        FileSelectorScreen(
                            files=file_info,
                            preview_cb=lambda p: MpvAdapter(cfg.mpv).preview_file(p),
                        ),
                        _on_dismiss,
                    )

            _FileApp().run()
            demuxed_mkv_files = chosen
```

- [ ] **Step 3: Integrate demuxed files with scanner results**

After the existing `Scanner.scan()` call, add logic to create `ScanResult` entries for demuxed files:

```python
    # Add demuxed MKV files as extra scan results
    for mkv_path in demuxed_mkv_files:
        demux_output = Scanner.build_output_path(mkv_path, source, output, names_map)
        scan_results.append(ScanResult(
            main_file=mkv_path,
            satellite_files=[],
            output_path=demux_output,
        ))
```

- [ ] **Step 4: Update `create_plan` call**

Replace:

```python
    plan_obj = planner.create_plan(
        movies=movies_with_paths,
        lang_filter=lang,
        vmaf_enabled=vmaf,
        dry_run=dry_run,
    )
```

With:

```python
    plan_obj = planner.create_plan(
        movies=movies_with_paths,
        audio_lang_filter=audio_lang,
        sub_lang_filter=sub_lang,
        vmaf_enabled=vmaf,
        dry_run=dry_run,
    )
```

- [ ] **Step 5: Set `demux_dir` on the plan**

After `planner.create_plan(...)`:

```python
    if demux_dir is not None:
        plan_obj.demux_dir = str(demux_dir)
```

- [ ] **Step 6: Update `run` command — cleanup `demux_dir`**

In the `run` command, after `printer.print_report(plan_obj, console)` (at the end, before `logger.debug("run command finished")`), add:

```python
    # Cleanup demux directory after successful run
    if plan_obj.demux_dir:
        demux_path = Path(plan_obj.demux_dir)
        if demux_path.exists():
            all_done = all(j.status == JobStatus.DONE for j in plan_obj.jobs)
            if all_done:
                import shutil
                shutil.rmtree(demux_path, ignore_errors=True)
                logger.info("Cleaned up demux directory: %s", demux_path)
```

Add `JobStatus` import at the top of the `run` function body:

```python
from .core.models import JobStatus
```

- [ ] **Step 7: Update logging line in `plan` command**

Replace the existing `logger.debug(...)` that logs `lang=` with:

```python
    logger.debug(
        "plan command started: source=%s output=%s audio_lang=%s sub_lang=%s names=%s dry_run=%s vmaf=%s",
        source, output, audio_lang, sub_lang, names, dry_run, vmaf,
    )
```

- [ ] **Step 8: Add `preview_file` to MpvAdapter**

In `furnace/adapters/mpv.py`, add a method after `preview_subtitle`:

```python
    def preview_file(self, path: Path) -> None:
        """mpv file. Blocks until mpv closes."""
        cmd = [str(self._mpv), str(path)]
        logger.info("mpv preview_file cmd: %s", " ".join(cmd))
        subprocess.run(cmd)
```

- [ ] **Step 9: Run quality gates**

```bash
uv run ruff check furnace/cli.py furnace/adapters/mpv.py
uv run mypy furnace/cli.py furnace/adapters/mpv.py --strict
```

Expected: both pass clean.

- [ ] **Step 10: Run full test suite**

```bash
uv run pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 11: Commit**

```bash
git add furnace/cli.py furnace/adapters/mpv.py
git commit -m "feat: wire disc demux and split lang filters into CLI"
```

---

### Task 10: Version Bump & Final Validation

**Files:**
- Modify: `furnace/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Bump version**

In `furnace/__init__.py`:

```python
VERSION = "1.4.0"
```

In `pyproject.toml`, update the `version` field to `"1.4.0"`.

- [ ] **Step 2: Run full quality gates**

```bash
uv run ruff check furnace/
uv run mypy furnace/ --strict
uv run pytest tests/ -q
```

Expected: all three pass clean.

- [ ] **Step 3: Commit**

```bash
git add furnace/__init__.py pyproject.toml
git commit -m "bump: version 1.4.0 — disc demux, split language filters"
```
