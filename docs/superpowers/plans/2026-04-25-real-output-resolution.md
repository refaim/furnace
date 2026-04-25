# Real Output Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the plan-phase log and the run-phase TUI display the real encoded output dimensions of every video job, computed in a single source of truth, and remove the misleading `sar=` field from the MKV encoder-settings tag.

**Architecture:** Introduce a pure helper `final_output_dimensions(vp: VideoParams) -> tuple[int, int]` in `furnace/core/quality.py`. The function composes existing primitives (`correct_sar`, `align_dimensions`) into the canonical pipeline `crop → SAR-correction → mod-8 HEVC alignment`. Four call sites delegate to it: the plan summary line, the run-phase target text, the NVEncC `--output-res` emission, and (with the misleading `sar=` field removed) the encoder-settings string.

**Tech Stack:** Python 3.12, pytest, `make check` (ruff + mypy strict + pytest with 100 % line+branch coverage on `furnace/` and `tests/`). No new dependencies.

**Project conventions (apply throughout):**

- TDD strict — failing test first, then implementation. No exceptions.
- 100 % line and branch coverage on every new or touched line. Run `make check` before declaring a task done.
- All linters/tests via the Makefile: `make check`, `make lint`, `make typecheck`, `make test`. Never call `uv run ruff/mypy/pytest` directly.
- **No intermediate commits.** Tasks 1–6 do not commit. The single commit happens only when the user explicitly authorises it after the whole plan is green. Task 7 prepares the commit but does not create it without authorisation.
- Hexagonal rules hold: core stays pure (no I/O); adapters import core, never the other way round.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `furnace/core/quality.py` | Modify | Add `final_output_dimensions()` next to `correct_sar` and `align_dimensions`. |
| `tests/core/test_quality.py` | Modify | Add `TestFinalOutputDimensions` class with 7 cases covering every branch. |
| `furnace/services/planner.py` | Modify | `_format_plan_summary` delegates `dst_w/dst_h` to the helper. |
| `tests/services/test_planner_reports.py` | Modify | Update existing assertions; add an anamorphic-DVD case. |
| `furnace/ui/run_tui.py` | Modify | `_build_target_text` delegates resolution to the helper; drop the inline `correct_sar` call. |
| `tests/ui/test_formatters.py` | Modify | Update the two anamorphic assertions (1024×480 and 1001×464 stay valid only if mod-8; verify and adjust). |
| `furnace/adapters/nvencc.py` | Modify | `_build_encoder_settings`: remove the `sar=` field. `_build_encode_cmd`: consolidate the two `--output-res` branches via the helper. |
| `tests/adapters/test_nvencc_cmd.py` | Modify | Replace `TestNVEncCSarInSettings` (the `sar=` assertions are inverted now); tighten `TestNVEncCSar` to assert exact mod-8 dims. |
| `furnace/__init__.py` | Modify | `VERSION = "1.14.2"`. |
| `pyproject.toml` | Modify | `version = "1.14.2"`. |

---

## Task 1: Add `final_output_dimensions()` helper (TDD)

**Files:**
- Modify: `furnace/core/quality.py:55-65` (append new function below `correct_sar`)
- Test: `tests/core/test_quality.py` (append a new `TestFinalOutputDimensions` class)

The helper is a pure composition of `correct_sar` (already present) and `align_dimensions` (already present). The signature accepts a `VideoParams` so callers do not have to unpack five fields by hand.

- [ ] **Step 1.1: Write the failing tests**

Append to `tests/core/test_quality.py` (after the existing `TestCorrectSar` block, line 227+). The fixture `_vp` builds a minimal `VideoParams` with only the fields the helper reads — every other field is irrelevant but required by the dataclass.

```python
# ---------------------------------------------------------------------------
# test_final_output_dimensions
# ---------------------------------------------------------------------------

from furnace.core.models import VideoParams
from furnace.core.quality import final_output_dimensions


def _vp(
    *,
    source_width: int,
    source_height: int,
    crop: CropRect | None = None,
    sar_num: int = 1,
    sar_den: int = 1,
) -> VideoParams:
    """Minimal VideoParams stub — only the fields final_output_dimensions reads."""
    return VideoParams(
        cq=22,
        crop=crop,
        deinterlace=False,
        color_matrix="bt709",
        color_range="tv",
        color_transfer="bt709",
        color_primaries="bt709",
        hdr=None,
        gop=120,
        fps_num=24, fps_den=1,
        source_width=source_width,
        source_height=source_height,
        source_codec="h264",
        source_bitrate=10_000_000,
        sar_num=sar_num,
        sar_den=sar_den,
        dv_mode=None,
    )


class TestFinalOutputDimensions:
    """`final_output_dimensions` is the single source of truth for the encoded
    output (width, height): crop -> SAR correction -> mod-8 HEVC alignment."""

    def test_no_crop_square_sar_mod8_passthrough(self) -> None:
        """1920x1080, square SAR, no crop -> unchanged."""
        vp = _vp(source_width=1920, source_height=1080)
        assert final_output_dimensions(vp) == (1920, 1080)

    def test_no_crop_square_sar_non_mod8_aligned(self) -> None:
        """1916x802, square SAR, no crop -> mod-8 trim to 1912x800."""
        vp = _vp(source_width=1916, source_height=802)
        assert final_output_dimensions(vp) == (1912, 800)

    def test_crop_square_sar_mod8_passthrough(self) -> None:
        """Crop 1920x800, square SAR -> unchanged (already mod-8)."""
        vp = _vp(
            source_width=1920, source_height=1080,
            crop=CropRect(w=1920, h=800, x=0, y=140),
        )
        assert final_output_dimensions(vp) == (1920, 800)

    def test_crop_square_sar_non_mod8_aligned(self) -> None:
        """Crop 1916x802, square SAR -> mod-8 trim to 1912x800."""
        vp = _vp(
            source_width=1920, source_height=1080,
            crop=CropRect(w=1916, h=802, x=2, y=139),
        )
        assert final_output_dimensions(vp) == (1912, 800)

    def test_no_crop_pal_dvd_anamorphic(self) -> None:
        """720x576 SAR 16:15 -> displayed 768x576 (already mod-8)."""
        vp = _vp(source_width=720, source_height=576, sar_num=16, sar_den=15)
        assert final_output_dimensions(vp) == (768, 576)

    def test_crop_pal_dvd_anamorphic_bug_case(self) -> None:
        """The motivating bug: 720x576 SAR 16:15 + crop 704x400.
        Pipeline: 704 * 16/15 = 750.93 -> 751, then mod-8 -> 744. Height stays 400."""
        vp = _vp(
            source_width=720, source_height=576,
            sar_num=16, sar_den=15,
            crop=CropRect(w=704, h=400, x=8, y=88),
        )
        assert final_output_dimensions(vp) == (744, 400)

    def test_anamorphic_height_grows(self) -> None:
        """SAR < 1 stretches height. 1024x576 SAR 4:5 -> 1024 x round(576*5/4)=720
        (mod-8 already)."""
        vp = _vp(source_width=1024, source_height=576, sar_num=4, sar_den=5)
        assert final_output_dimensions(vp) == (1024, 720)
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `make test`

Expected: 7 failures with `ImportError: cannot import name 'final_output_dimensions' from 'furnace.core.quality'`.

- [ ] **Step 1.3: Implement the helper**

`core/quality.py` already imports `CropRect` from `.models` at the top of the file (line 6). Extend that line to also import `VideoParams`:

```python
from .models import CropRect, VideoParams
```

Then append after `correct_sar` (after line 64):

```python
def final_output_dimensions(vp: VideoParams) -> tuple[int, int]:
    """Return the actual (width, height) that will be encoded in the HEVC track.

    Pipeline: crop (if set) -> SAR correction (if non-square) -> mod-8 HEVC
    CU alignment. This is the single source of truth — UI labels, plan-log
    summaries and the NVEncC ``--output-res`` flag all derive from here.
    """
    cur_w = vp.crop.w if vp.crop is not None else vp.source_width
    cur_h = vp.crop.h if vp.crop is not None else vp.source_height
    if vp.sar_num != vp.sar_den:
        cur_w, cur_h = correct_sar(cur_w, cur_h, vp.sar_num, vp.sar_den)
    aligned = align_dimensions(cur_w, cur_h)
    return aligned.w, aligned.h
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `make test`

Expected: all 7 new tests PASS, all existing tests still PASS.

- [ ] **Step 1.5: Run full quality gate**

Run: `make check`

Expected: ruff clean, mypy clean, pytest 100 % line+branch on `furnace/` and `tests/`. If coverage drops, add a missing branch test (e.g. for `sar_num == sar_den` short-circuit — covered by Test 1 already; for `crop=None` branch — covered by Test 1; for `sar_num != sar_den` — covered by Test 5).

---

## Task 2: Plan-phase log uses the helper (TDD)

**Files:**
- Modify: `furnace/services/planner.py:41-62` (the `_format_plan_summary` function)
- Test: `tests/services/test_planner_reports.py:170-219` (existing 3 tests) and append a new anamorphic case.

- [ ] **Step 2.1: Add a failing test for the anamorphic case**

Append to `tests/services/test_planner_reports.py` (after `test_format_plan_summary_includes_deinterlace_flag`, around line 219):

```python
def test_format_plan_summary_anamorphic_dvd_uses_real_output_dims() -> None:
    """PAL DVD anamorphic source: 720x576 SAR 16:15 + crop 704x400.
    The summary must reflect the actual encoded output 744x400, not the
    raw crop dims or the SAR-corrected-but-unaligned 751x400."""
    pal_dvd_video = replace(
        _make_video_info(),
        width=720, height=576,
        sar_num=16, sar_den=15,
    )
    movie = replace(_make_movie(), video=pal_dvd_video)
    prober = MagicMock()
    prober.detect_crop.return_value = CropRect(w=704, h=400, x=8, y=88)
    planner = PlannerService(prober=prober, previewer=None)
    plan = planner.create_plan(
        movies=[(movie, Path("/out/x.mkv"))],
        audio_lang_filter=["eng"],
        sub_lang_filter=[],
        vmaf_enabled=False,
        dry_run=False,
    )
    summary = _format_plan_summary(movie, plan.jobs[0])
    assert "720x576 to 744x400" in summary
```

If `_make_video_info` does not already accept `sar_num`/`sar_den` overrides via `replace`, verify the `VideoInfo` dataclass exposes those fields (it does — see `core/models.py:216-217`). No fixture change needed.

- [ ] **Step 2.2: Run the new test to verify it fails**

Run: `uv run pytest tests/services/test_planner_reports.py::test_format_plan_summary_anamorphic_dvd_uses_real_output_dims -v` — actually, do NOT run pytest directly. Run the whole suite:

`make test`

Expected: the new test FAILS with `assert '720x576 to 744x400' in 'cq 22, 720x576 to 704x400'` (or equivalent). The other three `_format_plan_summary` tests still pass.

- [ ] **Step 2.3: Refactor `_format_plan_summary` to use the helper**

Edit `furnace/services/planner.py:41-62`. Replace the body:

```python
def _format_plan_summary(movie: Movie, job: Job) -> str:
    """One-line per-movie summary shown after Plan completes for that movie.

    Format: ``cq <CQ>, <SrcW>x<SrcH> to <DstW>x<DstH>[, deinterlace]``

    The ``DstWxDstH`` part is the *actual* encoded output (crop -> SAR ->
    mod-8 HEVC alignment), via :func:`final_output_dimensions`. The
    resolution separator is the word ``to`` (not ``->``) so it doesn't
    collide with the reporter's ``label -> status`` arrow.
    """
    src_w = movie.video.width
    src_h = movie.video.height
    dst_w, dst_h = final_output_dimensions(job.video_params)
    parts = [
        f"cq {job.video_params.cq}",
        f"{src_w}x{src_h} to {dst_w}x{dst_h}",
    ]
    if job.video_params.deinterlace:
        parts.append("deinterlace")
    return ", ".join(parts)
```

Update the imports in `planner.py:35`:

```python
from furnace.core.quality import calculate_gop, final_output_dimensions, interpolate_cq
```

- [ ] **Step 2.4: Run tests to verify all pass**

Run: `make test`

Expected: the new anamorphic test PASSES. The three pre-existing tests (`test_format_plan_summary_no_crop_no_deinterlace`, `..._with_crop_uses_cropped_dims`, `..._includes_deinterlace_flag`) also still pass — none of them use a non-square SAR, so their assertions are unchanged.

- [ ] **Step 2.5: Run quality gate**

Run: `make check`

Expected: clean.

---

## Task 3: Run-phase TUI uses the helper (TDD)

**Files:**
- Modify: `furnace/ui/run_tui.py:201-211` (the `_build_target_text` function, video line)
- Test: `tests/ui/test_formatters.py:379-386` and `:440-448` (the SAR-related cases)

- [ ] **Step 3.1: Update the failing tests**

The existing `test_sar_corrects_resolution` (line 379) asserts `"1024x480"` for source 720×480 SAR 64:45. Verify: 720 × 64/45 = 1024 (exact, mod-8). After alignment: 1024×480. The assertion stays valid — no edit needed for that test.

The existing `test_crop_with_sar` (line 440) asserts `"1001x464"` for crop 704×464 SAR 64:45. Compute: 704 × 64/45 = 1001.24 → 1001 (round). 1001 % 8 = 1 → mod-8 trim to 1000. Height 464 % 8 = 0. Expected after refactor: `1000x464`, not `1001x464`. The assertion needs updating.

Edit `tests/ui/test_formatters.py:440-448`:

```python
    def test_crop_with_sar(self) -> None:
        """Crop + anamorphic SAR: actual encoded dims include mod-8 alignment.
        704 * 64/45 = 1001.24 -> 1001 -> mod-8 -> 1000. Height 464 stays."""
        crop = CropRect(w=704, h=464, x=8, y=8)
        vp = make_video_params(
            source_width=720, source_height=480, cq=22,
            sar_num=64, sar_den=45, crop=crop,
        )
        job = make_job(video_params=vp)
        text = _build_target_text(job)
        assert "1000x464" in text
```

Add a new test for the motivating bug case (PAL DVD 16:15 SAR + crop 704×400):

```python
    def test_crop_with_pal_dvd_anamorphic_sar(self) -> None:
        """The motivating bug: 720x576 SAR 16:15 + crop 704x400 -> 744x400."""
        crop = CropRect(w=704, h=400, x=8, y=88)
        vp = make_video_params(
            source_width=720, source_height=576, cq=22,
            sar_num=16, sar_den=15, crop=crop,
        )
        job = make_job(video_params=vp)
        text = _build_target_text(job)
        assert "744x400" in text
```

- [ ] **Step 3.2: Run tests to verify the SAR+crop test fails**

Run: `make test`

Expected: `test_crop_with_sar` and `test_crop_with_pal_dvd_anamorphic_sar` FAIL because the current implementation prints `1001x464` and `751x400` respectively (no mod-8 alignment).

- [ ] **Step 3.3: Refactor `_build_target_text`**

Edit `furnace/ui/run_tui.py:201-221`. Replace the video-line block:

```python
def _build_target_text(job: Job) -> str:
    """Build target info block from Job data."""
    lines: list[str] = []

    vp = job.video_params
    final_w, final_h = final_output_dimensions(vp)
    res = f"{final_w}x{final_h}"
    lines.append(f"Video: HEVC {res} CQ{vp.cq}")
    ...
```

Update imports (top of `run_tui.py`, search for the existing `correct_sar` import and replace it):

```python
from furnace.core.quality import final_output_dimensions
```

Drop the `correct_sar` import — the helper now owns that call.

- [ ] **Step 3.4: Run tests to verify all pass**

Run: `make test`

Expected: every `TestBuildTargetText` test passes including the two updated ones.

- [ ] **Step 3.5: Run quality gate**

Run: `make check`

Expected: clean.

---

## Task 4: Remove the `sar=` field from encoder settings (TDD)

**Files:**
- Modify: `furnace/adapters/nvencc.py:177-181` (the `_build_encoder_settings` function)
- Test: `tests/adapters/test_nvencc_cmd.py:456-469` (the `TestNVEncCSarInSettings` class)

- [ ] **Step 4.1: Invert the failing tests**

The existing `TestNVEncCSarInSettings` class asserts the opposite of what we want. Replace lines 456–469 in `tests/adapters/test_nvencc_cmd.py`:

```python
class TestNVEncCSarInSettings:
    """The encoder_settings string never includes a `sar=` field — actual
    encoded dims are already in the MKV video-track metadata, the field
    used to be misleading because it omitted mod-8 alignment."""

    def test_sar_field_absent_with_anamorphic(self) -> None:
        adapter = _adapter()
        vp = _make_vp(sar_num=64, sar_den=45)
        settings = adapter._build_encoder_settings(vp)
        assert "sar=" not in settings

    def test_sar_field_absent_with_square_pixels(self) -> None:
        adapter = _adapter()
        vp = _make_vp(sar_num=1, sar_den=1)
        settings = adapter._build_encoder_settings(vp)
        assert "sar=" not in settings
```

- [ ] **Step 4.2: Run tests to verify the anamorphic case fails**

Run: `make test`

Expected: `test_sar_field_absent_with_anamorphic` FAILS (current code emits `sar=1024x480`). `test_sar_field_absent_with_square_pixels` PASSES (square SAR was already skipped by the `if vp.sar_num != vp.sar_den:` guard).

- [ ] **Step 4.3: Remove the `sar=` block**

Edit `furnace/adapters/nvencc.py:177-181`. Delete the entire `if vp.sar_num != vp.sar_den:` block:

```python
        # (deleted: the sar=… field omitted mod-8 alignment and was misleading;
        # actual encoded dimensions are in the MKV video track metadata)
```

Resulting fragment (after the `crop=` append, before the `dv_mode` append):

```python
        if vp.crop is not None:
            left, top, right, bottom = _convert_crop(vp.crop, vp.source_width, vp.source_height)
            parts.append(f"crop={top}:{bottom}:{left}:{right}")

        if vp.dv_mode is not None:
            parts.append("dolby-vision=8.1")
```

Drop the now-unused inline import (do not remove the module-level `correct_sar` import yet — Task 5 still uses it via the helper indirectly? Verify: after Task 5 the adapter no longer calls `correct_sar` directly because the helper is called. So you may remove the `correct_sar` import once Task 5 is done — handle that in Task 5).

For now, leave the imports alone; Task 5 cleans them up.

- [ ] **Step 4.4: Run tests to verify all pass**

Run: `make test`

Expected: both `TestNVEncCSarInSettings` tests pass; every other settings test still passes (the `sar=` substring assertion was the only one impacted).

- [ ] **Step 4.5: Run quality gate**

Run: `make check`

Expected: clean. If `correct_sar` is now unused at the module level, mypy/ruff will flag the dead import — Task 5 removes it.

---

## Task 5: Consolidate `--output-res` in `_build_encode_cmd` via the helper (TDD)

**Files:**
- Modify: `furnace/adapters/nvencc.py:235-262` (the `--- Crop ---` and `--- SAR correction ---` blocks of `_build_encode_cmd`)
- Modify: `furnace/adapters/nvencc.py:18` (imports — drop `correct_sar`, `align_dimensions`; add `final_output_dimensions`)
- Test: `tests/adapters/test_nvencc_cmd.py` (`TestNVEncCSar` and `TestNVEncCCrop`)

- [ ] **Step 5.1: Add a failing test for the motivating bug case**

Append to `tests/adapters/test_nvencc_cmd.py` inside `TestNVEncCSar` (around line 256):

```python
    def test_pal_dvd_anamorphic_with_crop_emits_aligned_output_res(self) -> None:
        """Bug case: 720x576 SAR 16:15 + crop 704x400.
        Pipeline yields 744x400; --output-res must reflect that, not 751x400."""
        vp = _make_vp(
            sar_num=16, sar_den=15,
            crop=CropRect(w=704, h=400, x=8, y=88),
        )
        # Override the default 4K source dims set by _make_vp.
        vp.source_width = 720
        vp.source_height = 576
        cmd = _cmd(vp)
        idx = cmd.index("--output-res")
        assert cmd[idx + 1] == "744x400"
        # SAR correction is still applied at NVEncC level.
        idx = cmd.index("--sar")
        assert cmd[idx + 1] == "1:1"
        assert "--vpp-resize" in cmd
        idx = cmd.index("--vpp-resize")
        assert cmd[idx + 1] == "spline64"
```

The existing `test_sar_resolution_calculation` (line 245) already passes because `3840 × 4/3 = 5120` is mod-8. Leave it alone.

The existing `test_crop_with_alignment` (line 168) asserts only that `--output-res` is present when crop produces non-mod-8 dims. After the refactor it must still hold — leave it alone, but add a tighter assertion:

Edit `tests/adapters/test_nvencc_cmd.py:168-175`:

```python
    def test_crop_with_alignment(self) -> None:
        """Crop that produces non-mod-8 dims -> --output-res emits the
        mod-8-aligned dims, not the raw crop dims."""
        # CropRect 3830x2150 -> mod-8 -> 3824x2144.
        vp = _make_vp(crop=CropRect(w=3830, h=2150, x=3, y=5))
        cmd = _cmd(vp)
        assert "--crop" in cmd
        idx = cmd.index("--output-res")
        assert cmd[idx + 1] == "3824x2144"
        # No SAR correction here — square pixels, so no --sar/--vpp-resize.
        assert "--sar" not in cmd
        assert "--vpp-resize" not in cmd
```

- [ ] **Step 5.2: Run tests to verify the new assertion fails**

Run: `make test`

Expected: `test_pal_dvd_anamorphic_with_crop_emits_aligned_output_res` FAILS — the current code emits `--output-res 751x400` first (wait — the current crop branch emits when `aligned != raw`, then the SAR branch emits a different value; `cmd.index("--output-res")` returns the *first* occurrence). Verify the failure mode by reading the test output before patching.

`test_crop_with_alignment` may also FAIL if the previous behaviour produced different aligned dims — verify before patching.

- [ ] **Step 5.3: Refactor `_build_encode_cmd`**

Edit `furnace/adapters/nvencc.py:235-262`. Replace both `--- Crop ---` and `--- SAR correction ---` blocks. Keep the `cmd += ["--crop", ...]` line in the crop block; move `--output-res` emission out into a single, post-deinterlace block fed by the helper.

Old (lines 235–262):

```python
        # --- Crop ---
        if vp.crop is not None:
            left, top, right, bottom = _convert_crop(
                vp.crop,
                vp.source_width,
                vp.source_height,
            )
            cmd += ["--crop", f"{left},{top},{right},{bottom}"]
            # Align final dimensions to mod-8 for HEVC CU
            final_w = vp.crop.w
            final_h = vp.crop.h
            aligned = align_dimensions(final_w, final_h)
            if aligned.w != final_w or aligned.h != final_h:
                cmd += ["--output-res", f"{aligned.w}x{aligned.h}"]

        # --- Deinterlace ---
        if vp.deinterlace:
            cmd += ["--vpp-nnedi", "nns=64,nsize=32x6,quality=slow"]

        # --- SAR correction ---
        if vp.sar_num != vp.sar_den:
            cur_w = vp.crop.w if vp.crop is not None else vp.source_width
            cur_h = vp.crop.h if vp.crop is not None else vp.source_height
            display_w, display_h = correct_sar(cur_w, cur_h, vp.sar_num, vp.sar_den)
            aligned = align_dimensions(display_w, display_h)
            cmd += ["--output-res", f"{aligned.w}x{aligned.h}"]
            cmd += ["--vpp-resize", "spline64"]
            cmd += ["--sar", "1:1"]
```

New:

```python
        # --- Crop ---
        if vp.crop is not None:
            left, top, right, bottom = _convert_crop(
                vp.crop,
                vp.source_width,
                vp.source_height,
            )
            cmd += ["--crop", f"{left},{top},{right},{bottom}"]

        # --- Deinterlace ---
        if vp.deinterlace:
            cmd += ["--vpp-nnedi", "nns=64,nsize=32x6,quality=slow"]

        # --- Output resolution (single source of truth: helper) ---
        final_w, final_h = final_output_dimensions(vp)
        pre_resize_w = vp.crop.w if vp.crop is not None else vp.source_width
        pre_resize_h = vp.crop.h if vp.crop is not None else vp.source_height
        if (final_w, final_h) != (pre_resize_w, pre_resize_h):
            cmd += ["--output-res", f"{final_w}x{final_h}"]

        # --- SAR correction (anamorphic only) ---
        # When SAR is non-square, NVEncC must rescale via a high-quality
        # filter and override the sample aspect ratio of the encoded stream
        # back to 1:1. For pure mod-8 trims (square SAR) the default resize
        # is fine — no spline64 / sar override needed.
        if vp.sar_num != vp.sar_den:
            cmd += ["--vpp-resize", "spline64"]
            cmd += ["--sar", "1:1"]
```

Update imports at `furnace/adapters/nvencc.py:18`:

```python
from furnace.core.quality import final_output_dimensions
```

Remove the now-unused `align_dimensions` and `correct_sar` imports from this line. (`correct_sar` was already used only in the deleted `sar=` block from Task 4 and the old SAR branch above. `align_dimensions` was only used in the old crop and SAR branches.)

- [ ] **Step 5.4: Run tests to verify all pass**

Run: `make test`

Expected: every `TestNVEncCSar`, `TestNVEncCCrop` test passes including the two new/updated ones. Existing tests for non-anamorphic, mod-8 crop cases still pass because the helper returns the post-crop dims unchanged and `--output-res` is not emitted (matches current behaviour).

- [ ] **Step 5.5: Run quality gate**

Run: `make check`

Expected: clean. Coverage on the new branches:
- `final_w, final_h != pre_resize` → covered by `test_pal_dvd_anamorphic_with_crop_emits_aligned_output_res`, `test_sar_resolution_calculation`, `test_crop_with_alignment`.
- `final_w, final_h == pre_resize` → covered by every test that does *not* assert `--output-res`, e.g. `test_no_crop_when_none`, `test_hevc_codec_present`.
- `vp.sar_num != vp.sar_den` → covered by `test_sar_correction_applied`, the bug case test.
- `vp.sar_num == vp.sar_den` → covered by `test_sar_not_applied_when_square`.

---

## Task 6: Verify behavioural parity end-to-end

**Files:** none modified — read-only validation pass before the version bump.

- [ ] **Step 6.1: Run the full quality gate one more time**

Run: `make check`

Expected: ruff clean, mypy clean, pytest 100 % line+branch on `furnace/` and `tests/`.

- [ ] **Step 6.2: Sanity-check the four call sites by reading them**

Read each of the following and verify the helper is the only place that combines `crop`, `sar_*`, `align_dimensions`:

- `furnace/services/planner.py::_format_plan_summary` — calls `final_output_dimensions(job.video_params)`
- `furnace/ui/run_tui.py::_build_target_text` — calls `final_output_dimensions(vp)`
- `furnace/adapters/nvencc.py::_build_encoder_settings` — does **not** mention SAR/alignment at all
- `furnace/adapters/nvencc.py::_build_encode_cmd` — calls `final_output_dimensions(vp)` for `--output-res`; the `--sar 1:1`/`--vpp-resize` flags remain conditional on `vp.sar_num != vp.sar_den`

If any other file still imports `correct_sar` or `align_dimensions` from `core.quality` *outside* of `quality.py` itself, that's a leftover — remove the dead import. (`correct_sar` is still kept as a public function for the helper's internal use; do not delete the function itself.)

- [ ] **Step 6.3: Sanity-check the unchanged hexagonal boundary**

`grep "from furnace.adapters" furnace/core/` and `grep "from furnace.services" furnace/core/` — both should return nothing. The helper lives in `core/`, depends only on existing `core/` primitives, and has no I/O.

---

## Task 7: Version bump and final commit (gated on user authorisation)

**Files:**
- Modify: `furnace/__init__.py:1`
- Modify: `pyproject.toml` (the `version = "..."` line)

- [ ] **Step 7.1: Bump version in `furnace/__init__.py`**

```python
VERSION = "1.14.2"
```

- [ ] **Step 7.2: Bump version in `pyproject.toml`**

```toml
version = "1.14.2"
```

- [ ] **Step 7.3: Run quality gate**

Run: `make check`

Expected: clean.

- [ ] **Step 7.4: Stage the changes (do NOT commit yet)**

Run:

```bash
git add furnace/__init__.py pyproject.toml \
        furnace/core/quality.py \
        furnace/services/planner.py \
        furnace/ui/run_tui.py \
        furnace/adapters/nvencc.py \
        tests/core/test_quality.py \
        tests/services/test_planner_reports.py \
        tests/ui/test_formatters.py \
        tests/adapters/test_nvencc_cmd.py \
        docs/superpowers/specs/2026-04-25-real-output-resolution-design.md \
        docs/superpowers/plans/2026-04-25-real-output-resolution.md
git status
```

Expected: every listed file is staged and `git status` shows no unstaged code changes outside this set.

- [ ] **Step 7.5: Pause for explicit commit authorisation**

**Do not run `git commit` until the user explicitly authorises it.** Per the project's `feedback_no_step_commits` memory: "Never commit anything (spec, code, docs) until user explicitly says to."

When authorised, the commit message:

```
Bump to 1.14.2: real output resolution everywhere

Plan-phase log and run-phase TUI now display the actual encoded
resolution (crop -> SAR-correction -> mod-8 HEVC alignment), computed
in a single helper `final_output_dimensions(vp)` in core/quality.py.
The misleading `sar=` field is removed from the MKV ENCODER_SETTINGS
tag — actual dimensions are already in the encoded video track.
```

---

## Self-Review Notes

**Spec coverage:** every requirement of `docs/superpowers/specs/2026-04-25-real-output-resolution-design.md` maps to a task — helper (T1), planner (T2), run-TUI (T3), encoder-settings field removal (T4), `_build_encode_cmd` consolidation (T5), version bump (T7). The spec's "behaviour preservation" corner case (non-mod-8 source with no crop, square SAR) is exercised by `test_no_crop_square_sar_non_mod8_aligned` in T1 and is validated indirectly by full coverage on the helper.

**No placeholders:** every step contains the exact code or command an engineer needs.

**Type consistency:** `final_output_dimensions(vp: VideoParams) -> tuple[int, int]` is used unchanged in every reference.

**Coverage strategy:** the helper has 7 unit tests covering each branch; consumers are tested through their own existing test suites (planner reports, formatters, nvencc cmd) with at least one anamorphic-DVD assertion per consumer to prove the integration.
