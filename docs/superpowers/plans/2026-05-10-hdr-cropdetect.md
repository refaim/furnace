# HDR-Aware Cropdetect Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `FFmpegAdapter.detect_crop` reliably find black bars on PQ (`smpte2084`) and HLG (`arib-std-b67`) HDR sources by inserting a `zscale` tonemap chain before `cropdetect=24:16:0` whenever the planner reports an HDR transfer. SDR detection stays bit-for-bit unchanged.

**Architecture:** Add a 5-line helper `hdr_transfer_for_cropdetect` in `core/detect.py`. Add an `hdr_transfer: str | None` keyword to the `Prober.detect_crop` Protocol and the `FFmpegAdapter` implementation. When non-None, prepend a two-stage zscale chain (`PQ/HLG -> linear -> bt709`) plus `format=yuv420p` so the existing 8-bit `limit=24` keeps its meaning. Planner passes the helper's output unconditionally — no other call sites change.

**Tech Stack:** Python 3.13, ffmpeg with libzimg (`zscale` filter), pytest (mocked `subprocess.run`), mypy strict, ruff, 100 % line+branch coverage via `make check`.

**Source spec:** `docs/superpowers/specs/2026-05-10-hdr-cropdetect-design.md`

**Project rules in force (override conflicting skill defaults):**
- TDD strict — failing test before any production code, every task.
- 100 % line + branch coverage on touched code; verify before claiming done.
- **No intermediate commits.** All tasks land in a single commit at the end (Task 6).
- Tools only via `make lint` / `make typecheck` / `make test` / `make check`. Never call `uv run ruff/mypy/pytest` directly.
- Subagents pass `model: "opus"`. No worktrees.

---

## File Map

| File | Change | Responsibility |
|------|--------|----------------|
| `furnace/core/detect.py` | modify (add helper) | Pure-function classifier: maps `color_transfer` → HDR transfer string for cropdetect, or `None`. |
| `furnace/core/ports.py` | modify (line 21–34) | Add `hdr_transfer: str \| None = None` keyword to `Prober.detect_crop` Protocol. |
| `furnace/adapters/ffmpeg.py` | modify (line 173–253) | Build `-vf` argument from segments; inject zscale chain when `hdr_transfer is not None`. |
| `furnace/services/planner.py` | modify (line 182–188) | Compute `hdr_transfer` via helper, pass to `detect_crop`. |
| `furnace/__init__.py` | modify (single line) | Bump `VERSION` to `"1.14.3"`. |
| `pyproject.toml` | modify (line 3) | Bump `version` to `"1.14.3"`. |
| `tests/core/test_detect.py` | modify (append cases) | Cover `hdr_transfer_for_cropdetect` for PQ, HLG, BT.709, SMPTE 170M, None. |
| `tests/core/test_ports.py` | modify | Update `_MinimalProber.detect_crop` signature to include `hdr_transfer`; add Protocol-shape assertion. |
| `tests/adapters/test_ffmpeg_cropdetect_hdr.py` | create | New test module: assert constructed `-vf` for SDR / HDR×PQ / HDR×HLG / interlaced × HDR matrix. |
| `tests/adapters/test_ffmpeg_cropdetect_progress.py` | unchanged | Existing SDR regression checks pass without modification. |
| `tests/services/test_planner_crop_detect.py` | modify (append class) | Assert planner passes the right `hdr_transfer` kwarg for SDR / PQ / HLG movies. |

---

## Task 1: Add `hdr_transfer_for_cropdetect` helper (`core/detect.py`)

**Files:**
- Modify: `furnace/core/detect.py` (append at end of module, before any trailing functions)
- Test: `tests/core/test_detect.py` (append a new test class)

- [ ] **Step 1.1: Write the failing test**

Append to `tests/core/test_detect.py` (after the existing imports & helpers, e.g. at the very end of the file):

```python
import pytest as _pytest_for_hdr  # noqa: E402  -- already imported above; alias only if needed

from furnace.core.detect import hdr_transfer_for_cropdetect  # noqa: E402


class TestHdrTransferForCropdetect:
    @_pytest_for_hdr.mark.parametrize(
        ("color_transfer", "expected"),
        [
            ("smpte2084", "smpte2084"),
            ("arib-std-b67", "arib-std-b67"),
            ("bt709", None),
            ("smpte170m", None),
            (None, None),
        ],
    )
    def test_hdr_transfer_for_cropdetect(
        self, color_transfer: str | None, expected: str | None
    ) -> None:
        assert hdr_transfer_for_cropdetect(color_transfer) == expected
```

(If `pytest` is already imported at the top of the file as `import pytest`, drop the aliased import and use `@pytest.mark.parametrize` directly. Inspect the file before writing — keep imports clean.)

- [ ] **Step 1.2: Run the test, expect import failure**

Run: `make test ARGS="tests/core/test_detect.py::TestHdrTransferForCropdetect -v"`

Expected: `ImportError: cannot import name 'hdr_transfer_for_cropdetect'`.

(If the project's `Makefile` doesn't expose `ARGS`, use `make test` and let it run the full suite — the new test will still surface as the only failure.)

- [ ] **Step 1.3: Implement the helper**

Edit `furnace/core/detect.py`. Add this near the other small utility functions (e.g. directly after `is_dvd_resolution`, around line 219):

```python
_HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})


def hdr_transfer_for_cropdetect(color_transfer: str | None) -> str | None:
    """Return the transfer string when cropdetect needs HDR tonemapping.

    Maps PQ ('smpte2084') and HLG ('arib-std-b67') through unchanged so the
    adapter can plug them straight into ``zscale=tin=...``. Anything else
    (including ``None``) returns ``None`` -- SDR path unchanged.
    """
    return color_transfer if color_transfer in _HDR_TRANSFERS else None
```

- [ ] **Step 1.4: Run the test, expect pass**

Run: `make test`

Expected: the new parametrised test passes (5 sub-cases). Existing tests stay green.

---

## Task 2: Add `hdr_transfer` keyword to `Prober.detect_crop` Protocol (`core/ports.py`)

**Files:**
- Modify: `furnace/core/ports.py` (lines 21–34)
- Test: `tests/core/test_ports.py` (modify `_MinimalProber.detect_crop`; add signature assertion)

- [ ] **Step 2.1: Write the failing test**

Add to `tests/core/test_ports.py` (alongside the existing Protocol-shape tests, e.g. after `test_prober_profile_audio_track_signature`):

```python
def test_prober_detect_crop_signature_includes_hdr_transfer() -> None:
    sig = inspect.signature(Prober.detect_crop)
    params = sig.parameters
    assert list(params) == [
        "self",
        "path",
        "duration_s",
        "interlaced",
        "is_dvd",
        "hdr_transfer",
        "on_progress",
    ]
    assert params["hdr_transfer"].default is None

    hints = typing.get_type_hints(Prober.detect_crop)
    assert hints["hdr_transfer"] == str | None
```

Also widen the existing `_MinimalProber.detect_crop` so the Protocol-conformance test (`test_minimal_prober_satisfies_runtime_checkable_protocol`) keeps passing once the Protocol is widened. Replace the current method with:

```python
    def detect_crop(
        self,
        path: Path,  # noqa: ARG002
        duration_s: float,  # noqa: ARG002
        *,
        interlaced: bool = False,  # noqa: ARG002
        is_dvd: bool = False,  # noqa: ARG002
        hdr_transfer: str | None = None,  # noqa: ARG002
    ) -> CropRect | None:
        return None
```

And extend `test_minimal_prober_method_surface` so the new keyword is exercised (keeps the stub at 100 % coverage):

```python
    assert stub.detect_crop(
        Path("/dev/null"), 60.0, interlaced=True, is_dvd=True, hdr_transfer="smpte2084",
    ) is None
```

- [ ] **Step 2.2: Run the tests, expect failure**

Run: `make test`

Expected: `test_prober_detect_crop_signature_includes_hdr_transfer` fails because the real `Prober.detect_crop` Protocol still has the old 5-parameter signature (no `hdr_transfer`).

- [ ] **Step 2.3: Update the Protocol**

Edit `furnace/core/ports.py`, replace the `detect_crop` method on `Prober` (currently lines 21–34) with:

```python
    def detect_crop(
        self,
        path: Path,
        duration_s: float,
        *,
        interlaced: bool = False,
        is_dvd: bool = False,
        hdr_transfer: str | None = None,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> CropRect | None:
        """Run cropdetect, return detected values (before alignment).

        ``hdr_transfer`` is the source's color transfer string ('smpte2084'
        or 'arib-std-b67') when the input needs HDR tonemapping before
        cropdetect; ``None`` for SDR.

        ``on_progress`` is called after each sample point.
        """
        ...
```

- [ ] **Step 2.4: Run the tests, expect pass**

Run: `make test`

Expected: both the new signature test and `test_minimal_prober_satisfies_runtime_checkable_protocol` pass. The whole suite remains green.

---

## Task 3: Inject HDR `zscale` chain in `FFmpegAdapter.detect_crop` (`adapters/ffmpeg.py`)

**Files:**
- Modify: `furnace/adapters/ffmpeg.py` (function `detect_crop`, lines 173–253)
- Create: `tests/adapters/test_ffmpeg_cropdetect_hdr.py`

- [ ] **Step 3.1: Write the failing tests**

Create `tests/adapters/test_ffmpeg_cropdetect_hdr.py` with the full 5-case matrix. The trick: capture the command list passed to `subprocess.run`, then locate `-vf` in it.

```python
"""Tests for the HDR-aware filter chain built by ``FFmpegAdapter.detect_crop``.

Mocks ``subprocess.run`` so no ffmpeg binary is invoked — we just inspect the
`-vf` argument the adapter constructs.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from furnace.adapters.ffmpeg import FFmpegAdapter

_SDR_VF = "cropdetect=24:16:0"
_SDR_VF_INTERLACED = "yadif,cropdetect=24:16:0"

_PQ_CHAIN = (
    "zscale=tin=smpte2084:min=2020_ncl:pin=2020:t=linear:npl=100,"
    "zscale=tin=linear:min=2020_ncl:pin=2020:t=bt709:m=bt709:p=bt709:r=tv,"
    "format=yuv420p,"
    "cropdetect=24:16:0"
)
_HLG_CHAIN = (
    "zscale=tin=arib-std-b67:min=2020_ncl:pin=2020:t=linear:npl=100,"
    "zscale=tin=linear:min=2020_ncl:pin=2020:t=bt709:m=bt709:p=bt709:r=tv,"
    "format=yuv420p,"
    "cropdetect=24:16:0"
)
_PQ_CHAIN_INTERLACED = "yadif," + _PQ_CHAIN


def _captured_vf(call_args_list: list, point_idx: int = 0) -> str:
    """Pluck the value passed after `-vf` in the call's positional cmd list."""
    cmd = call_args_list[point_idx].args[0]
    vf_idx = cmd.index("-vf")
    return cmd[vf_idx + 1]


@pytest.mark.parametrize(
    ("interlaced", "hdr_transfer", "expected_vf"),
    [
        (False, None, _SDR_VF),
        (True, None, _SDR_VF_INTERLACED),
        (False, "smpte2084", _PQ_CHAIN),
        (False, "arib-std-b67", _HLG_CHAIN),
        (True, "smpte2084", _PQ_CHAIN_INTERLACED),
    ],
)
def test_detect_crop_filter_chain(
    interlaced: bool, hdr_transfer: str | None, expected_vf: str,
) -> None:
    adapter = FFmpegAdapter(Path("ffmpeg"), Path("ffprobe"))
    fake_result = MagicMock()
    fake_result.stderr = "[Parsed_cropdetect_0 @ 0x0] crop=3840:1600:0:280\n"
    fake_result.returncode = 0
    with patch(
        "furnace.adapters.ffmpeg.subprocess.run", return_value=fake_result,
    ) as mock_run:
        adapter.detect_crop(
            Path("x.mkv"),
            duration_s=1000.0,
            interlaced=interlaced,
            is_dvd=False,
            hdr_transfer=hdr_transfer,
        )
    # All 10 sample-point invocations should use the same -vf.
    assert mock_run.call_count == 10
    for i in range(10):
        assert _captured_vf(mock_run.call_args_list, i) == expected_vf
```

- [ ] **Step 3.2: Run the new test file, expect failure**

Run: `make test`

Expected: every parametrised case in the new file fails — the adapter still emits the SDR-only chain and doesn't accept `hdr_transfer` as a keyword. (Cases with `hdr_transfer=None` may even fail with `TypeError: unexpected keyword 'hdr_transfer'`.)

- [ ] **Step 3.3: Update `detect_crop`**

Edit `furnace/adapters/ffmpeg.py`. Replace the `vf = ...` line (around line 191) and the surrounding signature/body. The full updated method:

```python
    def detect_crop(
        self,
        path: Path,
        duration_s: float,
        *,
        interlaced: bool = False,
        is_dvd: bool = False,
        hdr_transfer: str | None = None,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> CropRect | None:
        """Run cropdetect at multiple points across the timeline.

        Returns the median crop of the dominant cluster only if the cluster
        contains >50 % of samples.  Returns None otherwise.

        ``hdr_transfer`` is the source's color transfer ('smpte2084' or
        'arib-std-b67') when the input needs HDR tonemapping before
        cropdetect (PQ/HLG -> linear -> bt709, then ``format=yuv420p`` so
        the SDR ``limit=24`` keeps its intended meaning -- cropdetect does
        NOT auto-scale ``limit`` to bit depth). DV Profile 5 (single-layer
        dvhe.05) is also tagged as smpte2084 in container metadata; zscale
        mis-handles its IPT-PQ-C2 colors but luma magnitude near zero is
        identical to YCbCr black, so cropdetect still returns the right
        geometry.

        ``on_progress`` is called after each sample point with a fraction
        (``points_done / total_points``).
        """
        points = self._CROP_SAMPLE_POINTS_DVD if is_dvd else self._CROP_SAMPLE_POINTS

        parts: list[str] = []
        if interlaced:
            parts.append("yadif")
        if hdr_transfer is not None:
            parts.append(
                f"zscale=tin={hdr_transfer}:min=2020_ncl:pin=2020:t=linear:npl=100"
            )
            parts.append(
                "zscale=tin=linear:min=2020_ncl:pin=2020:"
                "t=bt709:m=bt709:p=bt709:r=tv"
            )
            parts.append("format=yuv420p")
        parts.append("cropdetect=24:16:0")
        vf = ",".join(parts)

        crop_values: list[CropRect] = []

        for i, pct in enumerate(points, start=1):
            seek = duration_s * pct
            cmd = [
                str(self._ffmpeg),
                "-hide_banner",
                "-ss",
                f"{seek:.2f}",
                "-i",
                str(path),
                "-t",
                "2",
                "-vf",
                vf,
                "-f",
                "null",
                "-",
            ]
            logger.debug("detect_crop cmd: %s", cmd)
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace", check=False,
            )
            last_crop: str | None = None
            for line in result.stderr.splitlines():
                m = re.search(r"crop=(\d+:\d+:\d+:\d+)", line)
                if m:
                    last_crop = m.group(1)
            if last_crop is not None:
                parts_crop = last_crop.split(":")
                # Regex `crop=(\d+:\d+:\d+:\d+)` structurally guarantees 4 parts.
                crop_values.append(
                    CropRect(
                        w=int(parts_crop[0]),
                        h=int(parts_crop[1]),
                        x=int(parts_crop[2]),
                        y=int(parts_crop[3]),
                    )
                )

            if on_progress is not None:
                on_progress(ProgressSample(fraction=i / len(points)))

        if not crop_values:
            return None

        median_crop, cluster_size = cluster_crop_values(crop_values)
        if cluster_size <= len(crop_values) // 2:
            logger.info(
                "Crop not reliable: cluster %d:%d:%d:%d has %d/%d samples",
                median_crop.w,
                median_crop.h,
                median_crop.x,
                median_crop.y,
                cluster_size,
                len(crop_values),
            )
            return None

        return median_crop
```

Note: the inner `parts` variable from the old `crop=...` parser was renamed to `parts_crop` to avoid shadowing the new outer `parts` list.

- [ ] **Step 3.4: Run the tests, expect pass**

Run: `make test`

Expected: all five `test_detect_crop_filter_chain` cases pass; the existing `test_ffmpeg_cropdetect_progress.py` tests still pass (regression check for SDR — they pass `hdr_transfer` implicitly as the new default `None`).

---

## Task 4: Wire planner to compute and pass `hdr_transfer` (`services/planner.py`)

**Files:**
- Modify: `furnace/services/planner.py` (imports near line 11; `detect_crop` call at line 182–188)
- Test: `tests/services/test_planner_crop_detect.py` (append a new test class)

- [ ] **Step 4.1: Write the failing tests**

Append to `tests/services/test_planner_crop_detect.py` (after the existing `TestCropDetectNonDryRun` class):

```python
import pytest


class TestCropDetectHdrTransferKwarg:
    """Planner must pass the right `hdr_transfer` kwarg to detect_crop."""

    @pytest.mark.parametrize(
        ("color_transfer", "expected_kwarg"),
        [
            ("bt709", None),
            ("smpte170m", None),
            ("smpte2084", "smpte2084"),
            ("arib-std-b67", "arib-std-b67"),
            (None, None),
        ],
    )
    def test_planner_passes_hdr_transfer(
        self, tmp_path: Path,
        color_transfer: str | None, expected_kwarg: str | None,
    ) -> None:
        main = tmp_path / "movie.mkv"
        main.write_bytes(b"")
        movie = make_movie(
            main_file=main,
            video=make_video_info(
                width=3840, height=2160, source_file=main,
                bitrate=20_000_000, color_transfer=color_transfer,
            ),
            audio_tracks=[
                make_track(
                    index=1, track_type=TrackType.AUDIO,
                    codec_name="aac", codec_id=AudioCodecId.AAC_LC,
                    language="eng", is_default=True, source_file=main,
                    channels=2, bitrate=192_000,
                ),
            ],
        )
        prober = MagicMock()
        prober.detect_crop.return_value = None
        planner = PlannerService(prober=prober, previewer=None)
        planner.create_plan(
            [(movie, tmp_path / "out.mkv")],
            audio_lang_filter=["eng"],
            sub_lang_filter=["eng"],
            vmaf_enabled=False,
            dry_run=False,
        )
        prober.detect_crop.assert_called_once()
        _, kwargs = prober.detect_crop.call_args
        assert kwargs["hdr_transfer"] == expected_kwarg
```

You'll also need `make_movie` and `make_video_info` imported at the top — they're already in scope via the existing file's `from tests.conftest import make_movie, make_track, make_video_info`. Confirm before adding duplicates.

- [ ] **Step 4.2: Run the tests, expect failure**

Run: `make test`

Expected: every parametrised case fails — `kwargs["hdr_transfer"]` raises `KeyError` because the planner doesn't pass that keyword yet.

- [ ] **Step 4.3: Update the planner**

Edit `furnace/services/planner.py`.

(a) Add `hdr_transfer_for_cropdetect` to the import from `furnace.core.detect` (currently line 11):

```python
from furnace.core.detect import (
    detect_video_system,
    hdr_transfer_for_cropdetect,
    is_dvd_resolution,
    resolve_color_metadata,
)
```

(b) Modify the `detect_crop` call (currently lines 182–188) to pass `hdr_transfer`:

```python
                raw_crop = self._prober.detect_crop(
                    movie.main_file,
                    movie.video.duration_s,
                    interlaced=movie.video.interlaced,
                    is_dvd=is_dvd,
                    hdr_transfer=hdr_transfer_for_cropdetect(
                        movie.video.color_transfer,
                    ),
                    on_progress=self._on_crop_progress,
                )
```

- [ ] **Step 4.4: Run the tests, expect pass**

Run: `make test`

Expected: all 5 parametrised cases in `TestCropDetectHdrTransferKwarg` pass; `TestCropDetectNonDryRun` cases continue to pass (they don't assert on the new kwarg).

---

## Task 5: Bump version to 1.14.3

**Files:**
- Modify: `furnace/__init__.py` (line 1)
- Modify: `pyproject.toml` (line 3)

- [ ] **Step 5.1: Bump `furnace/__init__.py`**

Change `VERSION = "1.14.2"` → `VERSION = "1.14.3"`.

- [ ] **Step 5.2: Bump `pyproject.toml`**

Change `version = "1.14.2"` → `version = "1.14.3"`.

- [ ] **Step 5.3: Sanity-check both files agree**

Run: `grep -E '^(VERSION|version)' furnace/__init__.py pyproject.toml`

Expected output:
```
furnace/__init__.py:VERSION = "1.14.3"
pyproject.toml:version = "1.14.3"
```

---

## Task 6: Verify the whole suite, then commit

**Files:** none new.

- [ ] **Step 6.1: Run the full quality gate**

Run: `make check`

Expected: ruff clean, mypy strict clean, pytest green with 100 % line + branch coverage on `furnace/` and `tests/`.

If coverage drops below 100 %, look at the report to identify which new branch isn't exercised. Common culprits:
- The `if interlaced:` branch in `detect_crop`'s chain builder when only HDR-progressive cases are tested.
- The `if hdr_transfer is not None:` branch when only SDR cases are tested.

The Task 3 test matrix already covers all four combinations (interlaced × hdr_transfer ∈ {None, "smpte2084"}); if anything is uncovered, add the missing variant.

- [ ] **Step 6.2: Verify spec / plan present and not committed yet**

Run: `git status --porcelain`

Expected: shows changes in the source files plus the spec and plan markdown files (if not already committed).

- [ ] **Step 6.3: Stage explicit paths and commit**

Stage *only* the files this work changed (avoid `git add -A` per project rule):

```bash
git add \
  furnace/__init__.py \
  furnace/core/detect.py \
  furnace/core/ports.py \
  furnace/adapters/ffmpeg.py \
  furnace/services/planner.py \
  pyproject.toml \
  tests/core/test_detect.py \
  tests/core/test_ports.py \
  tests/adapters/test_ffmpeg_cropdetect_hdr.py \
  tests/services/test_planner_crop_detect.py \
  docs/superpowers/specs/2026-05-10-hdr-cropdetect-design.md \
  docs/superpowers/plans/2026-05-10-hdr-cropdetect.md
```

Then commit (single commit per project rule):

```bash
git commit -m "$(cat <<'EOF'
Bump to 1.14.3: HDR-aware cropdetect

Insert a zscale tonemap chain (PQ/HLG -> linear -> bt709 + format=yuv420p)
before cropdetect=24:16:0 whenever the source's color_transfer is
smpte2084 or arib-std-b67. SDR detection is byte-for-byte unchanged.

Fixes cropdetect returning no crop / full-frame on HDR sources where
PQ-amplified shadow noise pushes "black" luma above the SDR-calibrated
limit threshold.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6.4: Verify the commit landed**

Run: `git log -1 --stat`

Expected: one new commit on top of `78f87cc`, listing exactly the files staged above.

---

## Manual validation (out of band, after merge)

Not part of the automated suite. Run these once the commit is on `master` to confirm the fix works on real files:

1. `furnace plan` on `Zhili.byli.2017.WEB-DL.IVI.HEVC.HDR.2160p-SOFCJ.mkv` → log line `crop detected … (source 3840x2160)` with `h < 2160`.
2. `furnace plan` on `Звездный десант.1997.UHD.Blu-Ray.Remux.2160p.mkv` → ~`3840x1606` crop (2.39:1 inside 16:9).
3. `furnace plan` on any SDR file from the user's library → identical crop result to before this commit (regression check).

If any of these reports the wrong crop or a `cropdetect unable to determine crop` warning, capture the ffmpeg log line for the bad sample point and reopen.
