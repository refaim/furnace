# Real Output Resolution: Single Source of Truth

## Problem

Furnace currently displays inconsistent video output dimensions across plan
logs, run-phase TUI, the MKV `ENCODER_SETTINGS` tag, and the actual NVEncC
`--output-res` flag. On anamorphic DVD sources (PAL DVD with SAR 16:15) the
displayed resolution does not match the resolution that is actually encoded.

Concrete example — `Bury Me Behind the Baseboard (2009)`:

- Source: 720×576 MPEG-2, SAR 16:15
- Crop:   704×400
- Plan log shows:                  `cq 22, 720x576 to 704x400`  ← wrong, pre-SAR
- Run-phase TUI shows:             `HEVC 751x400 CQ22`           ← wrong, missing mod-8 alignment
- MKV `ENCODER_SETTINGS` tag has:  `sar=751x400`                 ← also wrong, missing alignment
- Actual NVEncC `--output-res`:    `744x400`                     ← the truth

The dimension formula (crop → SAR-correction → mod-8 alignment) is duplicated
across four call sites and three of them disagree with what the encoder
actually does.

## Goal

1. Plan-phase log and run-phase TUI display the **real encoded output dimensions**.
2. **One** computation site for output dimensions — every call site delegates to it.
3. Remove the misleading `sar=…` field from the MKV `ENCODER_SETTINGS` tag.

## Design

### New helper

In `furnace/core/quality.py`:

```python
def final_output_dimensions(vp: VideoParams) -> tuple[int, int]:
    """Crop -> SAR-correction -> mod-8 HEVC CU alignment.

    Returns the actual width/height that will be written to the encoded
    HEVC track. This is the single source of truth — UI, logs and the
    NVEncC --output-res flag all derive from here.
    """
    cur_w = vp.crop.w if vp.crop is not None else vp.source_width
    cur_h = vp.crop.h if vp.crop is not None else vp.source_height
    if vp.sar_num != vp.sar_den:
        cur_w, cur_h = correct_sar(cur_w, cur_h, vp.sar_num, vp.sar_den)
    aligned = align_dimensions(cur_w, cur_h)
    return aligned.w, aligned.h
```

Pure function, no I/O, sits next to `correct_sar` and `align_dimensions`.

### Call site changes

1. **`services/planner.py::_format_plan_summary`**
   Replace the manual `dst_w/dst_h` calculation with
   `final_output_dimensions(job.video_params)`.
   New format on the example: `cq 22, 720x576 to 744x400`.

2. **`ui/run_tui.py::_build_target_text`**
   Replace the inline `correct_sar` call with `final_output_dimensions(vp)`.
   New format: `Video: HEVC 744x400 CQ22`.

3. **`adapters/nvencc.py::_build_encoder_settings`**
   **Remove** the `sar=…` part entirely. The actual encoded dimensions are
   already in the MKV video-track metadata; this string field is misleading.
   Old: `… / sar=751x400`
   New: omit.

4. **`adapters/nvencc.py::_build_encode_cmd`**
   Consolidate the two branches that emit `--output-res` into one. Use
   `final_output_dimensions(vp)`. Emit `--output-res` whenever the final
   dimensions differ from the post-crop dimensions (i.e. when a resize is
   needed). Keep `--vpp-resize spline64 --sar 1:1` conditional on
   `vp.sar_num != vp.sar_den` — high-quality resampling is reserved for
   anamorphic correction; mod-8 trims do not need it.

### Behaviour preservation

The refactor is behaviour-preserving for every realistic case. There is one
corner case where behaviour changes:

- Source with non-mod-8 dimensions, no crop, square SAR.

Currently NVEncC does not emit `--output-res` and encodes at the non-mod-8
source dims. After the refactor it will emit `--output-res` with the mod-8
aligned dims. This case does not occur in practice (disc sources are always
standard sizes 1920×1080, 720×576, etc., all mod-8) and the new behaviour
is more correct for HEVC CU alignment.

### Plan JSON

No change. JSON keeps `source_width`, `source_height`, `crop`, `sar_num`,
`sar_den` as raw fields. `final_output_dimensions` is a derivation, computed
on demand from those raw fields.

## Tests (TDD, 100% line + branch)

### New unit tests in `tests/core/test_quality.py`

`final_output_dimensions` covering:

- crop=None, square SAR, mod-8 source → returns source unchanged
- crop=None, square SAR, non-mod-8 source → returns mod-8-aligned source
- crop set, square SAR, mod-8 crop → returns crop unchanged
- crop set, square SAR, non-mod-8 crop → returns mod-8-aligned crop
- crop=None, anamorphic SAR (PAL 720×576 SAR 16:15) → returns 768×576
- crop set, anamorphic SAR (the bug case: 704×400 SAR 16:15) → returns 744×400
- anamorphic SAR with `sar_num < sar_den` (height grows, e.g. 1024×576 SAR 4:5)

### Updates to existing tests

- `tests/services/test_planner_reports.py` — update assertions on
  `_format_plan_summary` (3 existing tests for the format) and add a new
  test for the anamorphic-DVD case.
- `tests/ui/test_formatters.py` — update assertions on `_build_target_text`
  for the anamorphic case (751 → 744).
- `tests/adapters/test_nvencc_cmd.py` — assert the consolidated
  `--output-res` logic; assert `sar=` is no longer in the
  `_build_encoder_settings` output.
- `tests/adapters/test_mkvpropedit_tags.py` — update if any test asserts
  the `sar=` substring in the tag string.

## Versioning

Bump to **1.14.2** (patch): fixes wrong reported output resolution,
behaviour-preserving refactor, removes the misleading `sar=` field from
the MKV encoder-settings tag. (1.14.1 was already taken by an unrelated
polish commit.)

Update both:

- `furnace/__init__.py` → `VERSION = "1.14.2"`
- `pyproject.toml` → `version = "1.14.2"`

## Out of scope

- Plan JSON schema changes.
- Changes to NVEncC encode quality settings (`--vpp-resize`, `--sar`,
  preset, tune, AQ, etc).
- Validation that source dimensions are mod-2 (assumed on input).
