# NTSC Telecine Guard — Design Note

**Status:** deferred (not implemented)
**Date:** 2026-04-18
**Scope:** defensive guard — fail fast on NTSC interlaced sources the pipeline
cannot correctly process

## Problem

Furnace currently detects interlaced content via `field_order` (from ffprobe)
and `idet` ratio, then routes it through NVEncC's NNEDI3 deinterlace
(`furnace/core/detect.py:273`, `furnace/adapters/nvencc.py:170`). This works
for true interlaced video (PAL 50i TV, NTSC 60i TV).

It silently produces poor output on **hard-telecined NTSC DVDs** — 24p film
stored at 29.97 fps with 3:2 pulldown baked into the fields. On such sources:

- ffprobe reports `fps ≈ 29.97` and `field_order ∈ {tt, bb}`, `idet` confirms
  interlace → current logic triggers NNEDI3 deinterlace.
- NNEDI3 interpolates new fields for every frame, yielding 29.97p output with
  **3:2 judder** from duplicated film frames instead of recovering the
  original 23.976p progressive stream.
- Correct handling requires **Inverse Telecine (IVTC)** — detecting the 3:2
  pattern and reassembling the original film frames. Not implemented.

The same ambiguity applies to true 30i NTSC content: without pattern analysis
we cannot distinguish it from hard telecine, and neither is handled well by
plain NNEDI3.

## Why deferred

The current user library is exclusively **PAL 25 fps and film 23.976 fps**.
NTSC interlaced sources do not appear in practice. Building IVTC support now
would be speculative — neither NVEncC's `--vpp-afs` nor an ffmpeg
`fieldmatch,decimate` pre-pass is justified until a real NTSC rip shows up.

## Proposed guard (when implemented)

Add a pure function in `furnace/core/detect.py`, called from
`furnace/services/analyzer.py` right after `should_deinterlace()` returns
True:

```python
def check_unsupported_telecine(
    field_order: str | None, fps: float, idet_ratio: float
) -> None:
    """Raise ValueError on NTSC interlaced sources — likely hard telecine or
    true 30i, both require IVTC which is not implemented.
    """
    if field_order not in {"tt", "bb"}:
        return
    if idet_ratio <= 0.05:
        return
    if 29.0 < fps < 30.5:
        raise ValueError(
            "Unsupported source: NTSC interlaced (29.97 fps, "
            f"field_order={field_order}). Likely hard-telecined film or true "
            "30i — requires IVTC (not implemented). Re-rip as progressive "
            "23.976p or use a different source."
        )
```

Consistent with the existing policy in `CLAUDE.md`: *"Unknown codecs or
unrecognized color matrix values: raise ValueError, don't silently degrade."*

## Allowed fps × interlace matrix

For reference, what the pipeline supports today:

| fps         | progressive | interlaced                |
|-------------|-------------|---------------------------|
| 23.976 / 24 | yes         | invalid (would be guarded) |
| 25          | yes         | yes (PAL 50i via NNEDI3)  |
| 29.97 / 30  | yes         | **guarded — raise**       |
| 50          | yes         | yes (true 50i)            |
| 59.94 / 60  | yes         | yes (true 60i)            |

## Out of scope

- IVTC implementation itself (NVEncC `--vpp-afs` or ffmpeg
  `fieldmatch,decimate` chain).
- Detection of soft telecine (flagged pulldown in MPEG-2 streams) — those
  decode as 23.976p progressive at the rip stage, no action needed.
- PAL 2:2 hard telecine — vanishingly rare, not worth handling.

## Trigger for implementation

Add the guard when one of the following happens:

1. A user NTSC DVD rip appears and the pipeline misbehaves.
2. Library scope expands to include NTSC sources.

Until then: documented but not coded.
