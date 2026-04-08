# Fix HDR Metadata Pipeline

**Date:** 2026-04-08
**Status:** Approved

## Problem

HDR metadata is lost during encoding. The output file is missing:
- Transfer characteristics (PQ) — affects all files, not just HDR
- Mastering display metadata (SMPTE ST 2086)
- Content light level (MaxCLL / MaxFALL)
- Matrix coefficients may be wrong for PAL sources (hardcoded to smpte170m)

## Root Causes

### Bug 1: Wrong ffprobe key for transfer characteristics

`analyzer.py:190` reads `stream.get("color_trc")` but ffprobe outputs the field as `color_transfer`. Result: `color_transfer` is **always null** in the plan for every file. Tests use the same wrong key in mocks, so they pass.

### Bug 2: `side_data_list` empty from `-show_streams`

For HEVC (and potentially other codecs), mastering display and content light level are stored in SEI NAL units. `ffprobe -show_streams` doesn't extract them. Need to read the first frame via `-show_frames -read_intervals "%+#1"` to get this data. For HDR10 these are static metadata — identical across all frames, so first frame is sufficient.

### Bug 3: Fraction format in frame side_data

ffprobe frame-level side_data returns mastering display values as fractions (e.g., `"8500/50000"`). The `detect_hdr` function passes them as-is, but NVEncC expects numerators only: `G(8500,39850)`.

### Bug 4: No fallback inference for color metadata

When ffprobe doesn't report `color_transfer`, `color_primaries`, or `color_space` (matrix), these are passed as null — no corresponding NVEncC flags. The values can be deterministically inferred from known fields per ITU standards.

### Bug 5: Matrix hardcoded for BT.601

`_COLORMATRIX_MAP` maps `ColorSpace.BT601` → `"smpte170m"` always, even for PAL sources where it should be `"bt470bg"` (mathematically identical but different identifier).

## Design

### 1. `VideoSystem` enum and `detect_video_system`

**File:** `furnace/core/detect.py`

```python
class VideoSystem(Enum):
    PAL = "pal"
    NTSC = "ntsc"
    HD = "hd"

def detect_video_system(height: int) -> VideoSystem:
    """Determine video system from frame height.
    
    PAL:  576, 288
    NTSC: 480, 486, 240
    HD:   >= 720
    Other SD: ValueError
    """
```

Standard heights only. Unknown SD heights raise `ValueError` — to be handled as encountered.

### 2. `resolve_color_metadata`

**File:** `furnace/core/detect.py`

```python
@dataclass(frozen=True)
class ResolvedColor:
    matrix: str
    transfer: str
    primaries: str

def resolve_color_metadata(
    matrix_raw: str | None,      # ffprobe color_space (= matrix coefficients)
    transfer_raw: str | None,    # ffprobe color_transfer
    primaries_raw: str | None,   # ffprobe color_primaries
    system: VideoSystem,
    has_hdr: bool,
) -> ResolvedColor:
```

Pure function, no I/O. Logic:

**Step 1 — determine family** (internal, not exposed):
- `matrix_raw` in `(bt2020nc, bt2020c)` → bt2020
- `matrix_raw` = `bt709` → bt709
- `matrix_raw` in `(bt470bg, smpte170m)` → bt601
- `matrix_raw` is None + `has_hdr` → bt2020
- `matrix_raw` is None + `system=HD` → bt709
- `matrix_raw` is None + `system=PAL/NTSC` → bt601

**Step 2 — resolve matrix:**
- `matrix_raw` not None → passthrough
- bt2020 → `bt2020nc`
- bt709 → `bt709`
- bt601 + PAL → `bt470bg`
- bt601 + NTSC → `smpte170m`

**Step 3 — resolve primaries:**
- `primaries_raw` not None → passthrough
- bt2020 → `bt2020`
- bt709 → `bt709`
- bt601 + PAL → `bt470bg`
- bt601 + NTSC → `smpte170m`

**Step 4 — resolve transfer:**
- `transfer_raw` not None → passthrough
- bt2020 + has_hdr → `smpte2084`
- bt2020 + no hdr → `bt709`
- bt709 → `bt709`
- bt601: infer from **resolved primaries** (not system):
  - `bt470bg` → `bt470bg`
  - `smpte170m` → `smpte170m`
  - `bt470m` → `bt470m`
  - `bt709` → `bt709`
  - fallback PAL → `bt470bg`, NTSC → `smpte170m`

**Unhandled cases:** If any step encounters an input combination not covered by the logic above (e.g., unknown `matrix_raw` string not in recognized set), raise `ValueError` with a descriptive message. Better to fail loudly and add the case explicitly than silently produce wrong metadata.

### 3. Fix ffprobe key

**File:** `furnace/services/analyzer.py:190`

Change `stream.get("color_trc")` to `stream.get("color_transfer")`.

**File:** `tests/services/test_analyzer.py` — fix mock key.

### 4. HDR side_data from first frame

**File:** `furnace/adapters/ffmpeg.py`

New method:
```python
def probe_hdr_side_data(self, path: Path) -> list[dict[str, Any]]:
    """Read side_data_list from the first video frame.
    
    ffprobe -v quiet -print_format json -select_streams v:0
      -show_frames -read_intervals "%+#1" path
    Returns frames[0]["side_data_list"] or []
    """
```

**File:** `furnace/core/ports.py` — add `probe_hdr_side_data` to Prober protocol.

**File:** `furnace/services/analyzer.py` — in `_parse_video_info()`, if `stream.get("side_data_list")` is empty, call `probe_hdr_side_data()` and pass result to `detect_hdr()`.

### 5. Parse fractions in mastering display

**File:** `furnace/core/detect.py`

Helper `_fraction_numerator(val: str) -> str`:
- `"8500/50000"` → `"8500"`
- `"8500"` → `"8500"`

Apply to all mastering display fields in `detect_hdr()`.

### 6. Planner + NVEncC integration

**File:** `furnace/services/planner.py`

In `_build_video_params()`:
- Call `detect_video_system(video.height)` and `resolve_color_metadata()`
- Use `ResolvedColor` fields for `VideoParams`

**File:** `furnace/core/models.py`

`VideoParams` changes:
- Replace `color_space: ColorSpace` with `color_matrix: str`
- Keep `color_transfer: str | None` → becomes `color_transfer: str` (always resolved)
- Keep `color_primaries: str | None` → becomes `color_primaries: str` (always resolved)

**File:** `furnace/adapters/nvencc.py`

- Use `vp.color_matrix` directly instead of `_COLORMATRIX_MAP[vp.color_space]`
- Remove `_COLORMATRIX_MAP`

**File:** `furnace/core/quality.py`

- Remove `determine_color_space()` (absorbed into `resolve_color_metadata`)

### 7. Remove `ColorSpace` enum

After all callers migrated:
- Remove `ColorSpace` from `furnace/core/models.py`
- Remove from `VideoParams`, `VideoInfo`
- Update plan serialization in `furnace/plan.py`
- Update all tests

## What does NOT change

- **Executor** — already passes `color_transfer` and parses `content_light` to video_meta
- **mkvmerge container-level duplication** — separate task (VUI in bitstream is sufficient)

## Test plan

### `tests/core/test_detect_video_system.py`
- Standard PAL heights: 576, 288 → PAL
- Standard NTSC heights: 480, 486, 240 → NTSC
- HD heights: 720, 1080, 2160 → HD
- Non-standard SD height → ValueError

### `tests/core/test_color_resolve.py`
243 test cases generated by `scripts/gen_color_tests.py`, covering:
- All passthrough scenarios (known values preserved as-is)
- All inference scenarios (None values filled from family/system)
- Cross-field consistency (transfer inferred from resolved primaries, not raw system)

See approved test case tables in conversation history.

### Existing tests to update
- `tests/services/test_analyzer.py` — fix `color_trc` → `color_transfer` in mocks
- `tests/test_nvencc_cmd.py` — use `color_matrix` instead of `color_space`
- `tests/test_plan.py` — update serialization for new field names
- `tests/core/test_detect.py` — add fraction parsing tests for `detect_hdr`

## Implementation order (TDD)

1. Write tests for `detect_video_system` → implement
2. Write 243 tests for `resolve_color_metadata` → implement
3. Fix ffprobe key `color_trc` → `color_transfer` + fix test mocks
4. Add `probe_hdr_side_data` + fraction parsing in `detect_hdr`
5. Integrate into planner: replace `determine_color_space` + `ColorSpace` enum
6. Update NVEncC adapter to use `color_matrix` directly
7. Update plan serialization
8. Run full quality gates
