# HEVC → AV1 — design

Date: 2026-06-27
Status: draft (pending spec review)

## Purpose

Replace HEVC with AV1 as Furnace's sole video output codec, driven by NVENC
hardware AV1 (NVEncC), while **keeping** Furnace's full HDR feature set —
HDR10 static mastering metadata **and** Dolby Vision. The approach mirrors the
sibling tool `crucible` (which migrated HEVC→AV1 on the same NVEncC/QVBR
toolchain) but, unlike crucible, retains DV/HDR10 because Furnace is a library
encoder where that metadata matters.

This is an **in-place codec swap**: HEVC is removed entirely, AV1 becomes the
only target, and no codec-abstraction layer is introduced (Furnace has none
today and a second codec is not a goal — YAGNI).

## Decisions (confirmed)

- **Approach A — in-place replacement.** HEVC gone, AV1 only, no codec enum /
  `CodecSpec` abstraction.
- **Encoder calibration adopted from crucible**: NVEncC `--preset P4` + QVBR
  anchors **35 / 35 / 36 / 38 / 41** (SD / 720p / 1080p / 1440p / 4K),
  calibrated to ≈ VMAF 94 on the same RTX 5060 Ti the user runs. Preset and
  anchors are coupled — they move together.
- **Version → 2.0.0 (MAJOR).** Removing HEVC and changing the output codec is a
  breaking change: prior plans carry HEVC-tuned `cq` values that become
  semantically invalid against AV1 QVBR.

## Feasibility (the one real risk, now cleared)

DV-over-AV1 through NVEncC is a working, documented path in 2025–2026:

- **NVEncC supports DV on AV1.** `--dolby-vision-rpu` and
  `--dolby-vision-profile` are tagged `[HEVC, AV1]` in the official options
  reference. Requires **NVEncC ≥ 8.00** (profile `10.x` values added in 8.00b4;
  the AV1 RPU-interleaving bug fixed in 8.00b5). The user's crucible runs
  NVEncC 9.19 — satisfied.
- **Profile changes 8.1 → 10.1.** AV1 DV is Profile 10; `10.1` is the
  HDR10-cross-compatible single-layer (BL+RPU) variant — the AV1 analogue of
  HEVC 8.1. The base layer plays as HDR10 on non-DV displays; the RPU adds DV on
  capable ones. Passing HEVC's `8.1` to an AV1 encode yields a mis-tagged /
  broken DV file.
- **RPU extraction is unchanged.** The RPU is codec-independent. Furnace's
  existing extraction (ffmpeg `-f hevc` from the HEVC *source* →
  `dovi_tool extract-rpu`) stays exactly as-is; the resulting `RPU.bin` is fed
  to the AV1 encode. `dovi_tool`'s CLI does **not** process AV1 and there is **no
  AV1 `inject-rpu`** — but Furnace never needed inject: NVEncC embeds the RPU at
  encode time as an ITU-T T.35 metadata OBU.
- **Container stays MKV.** mkvmerge reads DV from AV1 IVF/OBU streams since
  v79 (complete v81) and writes the `dvvC` BlockAdditionMapping. Per rigaya
  issue #663, NVEncC writing **direct to MKV can drop the DV metadata**, while a
  **raw elementary-stream path preserves the RPU** (it rides in-stream as a T.35
  OBU). **Correction to an earlier assumption:** Furnace today does *not* write a
  raw elementary stream — the executor names the encode output `video.mkv`
  (`executor.py:341`) and NVEncC self-muxes to MKV; the raw `.hevc` only appears
  in test fixtures. For HEVC this still yields working DV because the RPU lives
  in-stream as HEVC NAL units and the subsequent mkvmerge re-mux restores the
  `dvvC` signalling. For AV1 that guarantee is weaker (NVEncC's MKV muxer may
  drop the metadata OBU on OBU re-packetisation), so **the AV1 encode output is
  switched to a raw OBU elementary stream** (`video.obu`) which mkvmerge then
  muxes — the research-backed reliable path. See "Container / output" below.
- **HDR10 static metadata is codec-agnostic.** `--master-display` / `--max-cll`
  are tagged `[HEVC, AV1]`; no change needed.
- **Hardware.** NVENC AV1 encode requires RTX 40-series (Ada) or 50-series
  (Blackwell). RTX 30-series and older cannot. The user's RTX 5060 Ti qualifies.

Accepted caveat (the user explicitly chose to keep DV): DV-over-AV1 player/TV
support is narrower than DV HEVC, but Profile 10.1 degrades gracefully to HDR10.

## Encoder adapter — NVEncC command (`adapters/nvencc.py`)

Deltas to `_build_encode_cmd` (currently `nvencc.py:186-299`). Everything not
listed is unchanged (crop, deinterlace, output-res, SAR, color flags, VMAF).

| Concern | HEVC (now) | AV1 (target) |
|---|---|---|
| Codec / profile / depth / tier | `-c hevc --profile main10 --output-depth 10 --tier high` | `-c av1 --profile main --output-depth 10` (no `--tier`) |
| Preset | `--preset P5` | `--preset P4` |
| Tune | `--tune uhq` | `--tune uhq` (unchanged; valid for AV1) |
| Rate control | `--qvbr {cq}` | `--qvbr {cq}` (unchanged flag; new anchor values) |
| AQ | `--aq --aq-temporal` | unchanged |
| Lookahead | `--lookahead 32 --lookahead-level 3` | `--lookahead 32` (**drop** `--lookahead-level`) |
| Multipass | `--multipass 2pass-quarter` | unchanged |
| GOP | `--gop-len {gop} --strict-gop --repeat-headers` | unchanged |
| HDR10 | `--max-cll … --master-display …` | unchanged |
| Dolby Vision | `--dolby-vision-rpu … --dolby-vision-profile 8.1 [--dolby-vision-rpu-prm crop=true]` | `--dolby-vision-profile 10.1` (rest unchanged) |
| Metrics | `--ssim --vmaf …` | unchanged (Furnace keeps quality measurement; crucible dropped it) |

**Output elementary stream.** The adapter is extension-agnostic — it writes to
whatever `output_path` the executor passes. Today that is `video.mkv`
(`executor.py:341`). The change lives in the **executor**, not the adapter: the
**encode branch** writes `video.obu` (raw AV1 OBU elementary stream) while the
**passthrough branch keeps `video.mkv`** (a verbatim stream copy whose codec is
the source's, muxed by ffmpeg). mkvmerge then muxes whichever it is given (it
reads AV1 OBU/IVF, including DV, since v79). The adapter's stale
"Outputs raw HEVC bitstream" docstring (`nvencc.py:105`) is corrected to AV1.

Empirical verification is required on the user's NVEncC/mkvmerge (cannot run
here — Windows + RTX): after a DV encode, MediaInfo on the final MKV must show
*"Dolby Vision … Profile 10.1 … BL+RPU … HDR10 compatible"* and codec
`dav1.10.x`.

**Also verify frame rate and total duration** against the source. A raw `.obu`
elementary stream carries no container-level timing — mkvmerge derives the frame
rate from the OBU sequence header's `timing_info` if present, otherwise it falls
back to a default and warns. The current mkvmerge invocation passes no
`--default-duration`, so if the muxed MKV shows the wrong fps or duration (or
mkvmerge warns about missing timing), switch the encode framing from `.obu` to
**`.ivf`** — IVF carries an explicit frame-rate header, so timing is unambiguous,
and it preserves the DV metadata OBUs equally well (mkvmerge reads DV from AV1
IVF & OBU streams alike). This `.obu → .ivf` switch is the one-line
`_video_intermediate_name` change; both DV-loss and timing-loss point to the same
fallback.

**Encoder-settings MKV tag** (`_build_encoder_settings`, `nvencc.py:149-180`).
Replace the hardcoded `hevc_nvenc` → `av1_nvenc`; drop `tier`/`lookahead-level`
from the tag string; update `preset=P5`→`P4`; `dolby-vision=8.1`→`10.1`.

## Quality anchors (`core/quality.py`)

Replace `CQ_ANCHORS` (currently HEVC-tuned 22/24/25/28/31 at `quality.py:12-18`)
with the crucible AV1 set:

```
SD     854×480  → 35
720p  1280×720  → 35
1080p 1920×1080 → 36
1440p 2560×1440 → 38
4K    3840×2160 → 41
```

`interpolate_cq()` math is unchanged (linear interpolation by pixel area, clamp
outside the anchor range). The value still flows verbatim into `--qvbr`.
Update the module's HEVC-specific docstring/comment.

`align_dimensions` (mod-8, `quality.py:40-52`) is **kept** — AV1 only requires
mod-2, so mod-8 remains safe and avoids changing existing output dimensions.
Only its "HEVC CU" comment is corrected.

## Dolby Vision pipeline

- **Extraction (`adapters/dovi_tool.py`) — unchanged.** Source is still HEVC, so
  ffmpeg `-f hevc` + `dovi_tool extract-rpu` stay. The `-m` mode is unchanged:
  P7 dual-layer → single-layer (`-m 2`), already-single-layer → copy (`-m 0`).
  The single-layer RPU produced is exactly what NVEncC needs; NVEncC re-tags it
  as Profile 10.1 at encode time. **No AV1 `inject-rpu` step is added.**
- **`DvMode` enum (`core/models.py`) — kept as-is.** `TO_8_1` (value 2 →
  dovi_tool `-m 2`) accurately names the *extraction* mode: the RPU written to
  disk is an 8.1 single-layer RPU. Profile 10.1 is only how NVEncC tags that same
  RPU once it is wrapped in AV1 — a separate, encoder-side concern. So no rename
  (it is correct at its layer, and renaming would be unrelated churn); a one-line
  comment notes the AV1 output is re-tagged 10.1.
- **Planner (`planner.py:511-513`)** keeps choosing the mode the same way; only
  the downstream NVEncC profile string changes.

## Guards / validation

- **NVEncC version gate (DV-conditional).** The adapter already parses
  `NVEncC --version` (`_get_version`, cached). When a DV encode is requested
  (`rpu_path is not None`) **and** the detected version is known and **< 8.00**,
  raise a clear error ("AV1 Dolby Vision requires NVEncC ≥ 8.00") instead of
  silently producing a broken file. The 8.00 floor is specifically the DV-RPU
  Profile-10.x requirement (profiles added in 8.00b4, AV1 RPU interleaving fixed
  in 8.00b5); a plain non-DV AV1 encode is **not** gated (works on 7.x), and an
  undetectable version (empty string) does **not** block — only a positively-known
  too-old version does.
- **HDR10+ — still rejected**, unchanged. Analyzer (`analyzer.py:156-159`) and
  planner (`planner.py:507-508`) keep raising. (NVEncC *could* do HDR10+ on AV1
  via `--dhdr10-info`, but that is out of scope here.)
- **Passthrough is codec-agnostic** (`planner._classify_passthrough`) — it
  copies the source stream verbatim only when `copy_video` is requested, with no
  HEVC assumption. Unaffected by the codec swap; no change.

## Naming / surface touch-ups

- UI label `Video: HEVC …` → `Video: AV1 …` (`ui/run_tui.py:232`, plus the
  formatter test).
- `_NVDEC_CODECS` (decode-side whitelist, `nvencc.py:86`) already contains
  `av1`; no change.
- The MKV `ENCODER` tag remains `Furnace v{VERSION}` (the `furnace scan` status
  detector keys on that, not on codec) — unaffected.

## Out of scope

- No codec-abstraction layer; no H.264/HEVC fallback mode.
- No MP4 output path (crucible's container; Furnace stays MKV).
- HDR10+ dynamic metadata (still rejected).
- BL+EL (dual-layer) DV output — NVEncC is BL+RPU only, same as today.
- Re-running the QVBR calibration ourselves — crucible's numbers are adopted.

## Testing (TDD, per repo rules)

Failing test before each change; 100% line+branch on touched code; `make check`
green. Affected suites:

- `tests/adapters/test_nvencc_cmd.py` — AV1 codec/profile/preset, dropped
  `--tier`/`--lookahead-level`, DV profile `10.1`, HDR10 flags, encoder-settings
  tag string, version-gate error.
- `tests/core/test_quality.py` — new AV1 anchors / interpolation at each anchor
  and between.
- `tests/adapters/test_dovi_tool.py` — extraction unchanged (regression guard).
- `tests/services/test_planner_dv.py`, `test_planner_passthrough.py` — DV mode
  selection + passthrough unaffected.
- `tests/core/test_models.py` — `DvMode` unchanged (regression guard on the
  value-2 round-trip).
- `tests/test_plan.py` — `video_params` serialization round-trip.
- `tests/ui/test_formatters.py` (+ run-tui tests) — `Video: AV1` label.
- Executor/integration tests referencing the `.hevc` output extension → `.obu`.

## Version

`furnace/__init__.py` `VERSION = "2.0.0"` and `pyproject.toml`
`version = "2.0.0"`, bumped together.
