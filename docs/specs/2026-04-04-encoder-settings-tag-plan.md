# ENCODER_SETTINGS MKV Tag — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write video encoder settings (NVENC params + applied filters) into the output MKV as an `ENCODER_SETTINGS` global tag, readable by MediaInfo.

**Architecture:** Module-level `_build_encoder_settings()` in `ffmpeg.py` builds the string from `VideoParams`. `encode()` return type changes from `int` to `tuple[int, str]` to carry the string up to `Executor`, which passes it to `Tagger`. `MkvpropeditAdapter` writes both `ENCODER` and `ENCODER_SETTINGS` tags in one mkvpropedit call.

**Tech Stack:** Python, mkvpropedit (MKV tag XML)

---

### Task 1: Build encoder settings string

**Files:**
- Modify: `furnace/adapters/ffmpeg.py` (add `_build_encoder_settings` function after `_build_vf_chain`)
- Modify: `tests/test_ffmpeg_encode_cmd.py` (add `TestBuildEncoderSettings` class)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_ffmpeg_encode_cmd.py`:

```python
from furnace.adapters.ffmpeg import _build_encoder_settings


class TestBuildEncoderSettings:
    """ENCODER_SETTINGS string generation from VideoParams."""

    def test_no_filters(self) -> None:
        vp = _make_vp(cq=25)
        result = _build_encoder_settings(vp, use_cuda=True)
        assert result == (
            "hevc_nvenc / main10 / cq=25 / preset=p5 / tune=uhq / rc=vbr"
            " / spatial-aq=1 / temporal-aq=1 / rc-lookahead=32 / multipass=qres"
        )

    def test_deinterlace_cuda(self) -> None:
        vp = _make_vp(deinterlace=True)
        result = _build_encoder_settings(vp, use_cuda=True)
        assert result.endswith("multipass=qres / deinterlace=bwdif_cuda")

    def test_deinterlace_cpu(self) -> None:
        vp = _make_vp(deinterlace=True)
        result = _build_encoder_settings(vp, use_cuda=False)
        assert "deinterlace=bwdif" in result
        assert "bwdif_cuda" not in result

    def test_crop(self) -> None:
        """crop=T:B:L:R — pixels removed from each edge."""
        crop = CropRect(w=1920, h=816, x=0, y=132)
        vp = _make_vp(crop=crop)
        result = _build_encoder_settings(vp, use_cuda=False)
        assert "crop=132:132:0:0" in result

    def test_alignment(self) -> None:
        """align=WxH when dimensions aren't multiples of 8."""
        crop = CropRect(w=1916, h=1076, x=2, y=2)
        vp = _make_vp(crop=crop)
        result = _build_encoder_settings(vp, use_cuda=False)
        assert "crop=2:2:2:2" in result
        assert "align=1912x1072" in result

    def test_no_alignment_when_already_aligned(self) -> None:
        crop = CropRect(w=1920, h=816, x=0, y=132)
        vp = _make_vp(crop=crop)
        result = _build_encoder_settings(vp, use_cuda=False)
        assert "align=" not in result

    def test_all_filters(self) -> None:
        crop = CropRect(w=1916, h=1076, x=2, y=2)
        vp = _make_vp(deinterlace=True, crop=crop)
        result = _build_encoder_settings(vp, use_cuda=False)
        parts = result.split(" / ")
        assert "deinterlace=bwdif" in parts
        assert "crop=2:2:2:2" in parts
        assert "align=1912x1072" in parts

    def test_cq_value_varies(self) -> None:
        vp = _make_vp(cq=31)
        result = _build_encoder_settings(vp, use_cuda=True)
        assert "cq=31" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ffmpeg_encode_cmd.py::TestBuildEncoderSettings -v`
Expected: FAIL — `ImportError: cannot import name '_build_encoder_settings'`

- [ ] **Step 3: Implement `_build_encoder_settings`**

Add to `furnace/adapters/ffmpeg.py`, after the `_build_vf_chain` function (before `class FFmpegAdapter`):

```python
def _build_encoder_settings(vp: VideoParams, *, use_cuda: bool) -> str:
    """Build ENCODER_SETTINGS string for MKV global tag.

    Format: slash-separated, NVENC params always present, filters only when applied.
    Example: hevc_nvenc / main10 / cq=25 / preset=p5 / ... / crop=132:132:0:0
    """
    parts: list[str] = [
        "hevc_nvenc",
        "main10",
        f"cq={vp.cq}",
        "preset=p5",
        "tune=uhq",
        "rc=vbr",
        "spatial-aq=1",
        "temporal-aq=1",
        "rc-lookahead=32",
        "multipass=qres",
    ]

    if vp.deinterlace:
        parts.append(f"deinterlace={'bwdif_cuda' if use_cuda else 'bwdif'}")

    if vp.crop is not None:
        top = vp.crop.y
        bottom = vp.source_height - vp.crop.y - vp.crop.h
        left = vp.crop.x
        right = vp.source_width - vp.crop.x - vp.crop.w
        parts.append(f"crop={top}:{bottom}:{left}:{right}")

    # Check if CU alignment trimmed pixels
    cur_w = vp.crop.w if vp.crop is not None else vp.source_width
    cur_h = vp.crop.h if vp.crop is not None else vp.source_height
    if vp.sar_num != vp.sar_den:
        cur_w = round(cur_w * vp.sar_num / vp.sar_den)
    aligned = align_dimensions(cur_w, cur_h)
    if aligned.w != cur_w or aligned.h != cur_h:
        parts.append(f"align={aligned.w}x{aligned.h}")

    return " / ".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ffmpeg_encode_cmd.py::TestBuildEncoderSettings -v`
Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add furnace/adapters/ffmpeg.py tests/test_ffmpeg_encode_cmd.py
git commit -m "feat: add _build_encoder_settings for ENCODER_SETTINGS MKV tag"
```

---

### Task 2: Return encoder settings from encode()

**Files:**
- Modify: `furnace/core/ports.py:35-48` (Encoder.encode return type)
- Modify: `furnace/adapters/ffmpeg.py:161-286` (FFmpegAdapter.encode)
- Modify: `furnace/services/executor.py:272-280` (unpack encode result)

- [ ] **Step 1: Update Encoder protocol return type**

In `furnace/core/ports.py`, change the `encode` method signature:

```python
    def encode(
        self,
        input_path: Path,
        output_path: Path,
        video_params: VideoParams,
        source_size: int,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> tuple[int, str]:
        """Encode video. Returns (return_code, encoder_settings_string).

        source_size is passed for mid-encoding bloat check (see 12.11).
        on_progress callback receives (progress_pct, status_line).
        """
        ...
```

- [ ] **Step 2: Update FFmpegAdapter.encode() to return tuple**

In `furnace/adapters/ffmpeg.py`, modify the `encode` method:

Change the signature:
```python
    def encode(
        self,
        input_path: Path,
        output_path: Path,
        video_params: VideoParams,
        source_size: int,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> tuple[int, str]:
```

After `cmd = self._build_encode_cmd(...)` (line 172), add:
```python
        # Build encoder settings string
        use_cuda = (
            video_params.source_codec in self._NVDEC_CODECS
            and video_params.crop is None
            and video_params.sar_num == video_params.sar_den
        )
        encoder_settings = _build_encoder_settings(video_params, use_cuda=use_cuda)
```

Change the three return statements:
- `return 1` (bloat abort, line 281) → `return 1, encoder_settings`
- `return process.returncode` (line 286) → `return process.returncode, encoder_settings`

- [ ] **Step 3: Update executor to unpack encode result**

In `furnace/services/executor.py`, line 272-280, change:

```python
        rc = self._encoder.encode(
```
to:
```python
        rc, encoder_settings = self._encoder.encode(
```

Store `encoder_settings` on the local scope — it will be used in Task 3 when we pass it to the tagger. For now, just add a `_ = encoder_settings` line after the `if rc != 0` check to suppress unused-variable warnings:

Actually, no — just use `_encoder_settings` as the variable name until Task 3 wires it up:

```python
        rc, _encoder_settings = self._encoder.encode(
            input_path=main_source,
            output_path=video_output,
            video_params=job.video_params,
            source_size=job.source_size,
            on_progress=_encode_progress,
        )
```

- [ ] **Step 4: Run all tests**

Run: `uv run pytest tests/ -q`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add furnace/core/ports.py furnace/adapters/ffmpeg.py furnace/services/executor.py
git commit -m "feat: return encoder_settings string from encode()"
```

---

### Task 3: Write ENCODER_SETTINGS MKV tag

**Files:**
- Modify: `furnace/core/ports.py:145-149` (Tagger.set_encoder_tag signature)
- Modify: `furnace/adapters/mkvpropedit.py` (XML template + method signature)
- Modify: `furnace/services/executor.py:336-347` (pass encoder_settings to tagger)

- [ ] **Step 1: Update Tagger protocol**

In `furnace/core/ports.py`, change `set_encoder_tag`:

```python
@runtime_checkable
class Tagger(Protocol):
    """Set MKV tags via mkvpropedit."""

    def set_encoder_tag(self, mkv_path: Path, tag_value: str, encoder_settings: str | None = None) -> int:
        """Set global ENCODER tag (and ENCODER_SETTINGS if provided). Returns return code."""
        ...
```

- [ ] **Step 2: Update MkvpropeditAdapter**

In `furnace/adapters/mkvpropedit.py`:

Replace the `_TAGS_XML_TEMPLATE` constant and `set_encoder_tag` method:

```python
def _build_tags_xml(tag_value: str, encoder_settings: str | None = None) -> str:
    """Build MKV global tags XML with ENCODER and optional ENCODER_SETTINGS."""
    lines = [
        "<Tags>",
        "  <Tag>",
        "    <Simple>",
        "      <Name>ENCODER</Name>",
        f"      <String>{tag_value}</String>",
        "    </Simple>",
    ]
    if encoder_settings:
        lines += [
            "    <Simple>",
            "      <Name>ENCODER_SETTINGS</Name>",
            f"      <String>{encoder_settings}</String>",
            "    </Simple>",
        ]
    lines += [
        "  </Tag>",
        "</Tags>",
    ]
    return "\n".join(lines) + "\n"
```

Update the `set_encoder_tag` method signature and body:

```python
    def set_encoder_tag(self, mkv_path: Path, tag_value: str, encoder_settings: str | None = None) -> int:
        """Set global ENCODER and ENCODER_SETTINGS tags.

        Creates a temporary tags.xml, runs:
            mkvpropedit mkv_path --tags global:tags.xml
        then deletes the temp file.
        """
        xml_content = _build_tags_xml(tag_value, encoder_settings)
```

The rest of the method stays the same (temp file write, run mkvpropedit, cleanup).

- [ ] **Step 3: Write test for _build_tags_xml**

Add to `tests/test_ffmpeg_encode_cmd.py` (or a new file — since this tests mkvpropedit, add it at the bottom of the existing test file for simplicity):

```python
from furnace.adapters.mkvpropedit import _build_tags_xml


class TestBuildTagsXml:
    def test_encoder_only(self) -> None:
        xml = _build_tags_xml("Furnace v1.4.0")
        assert "<Name>ENCODER</Name>" in xml
        assert "<String>Furnace v1.4.0</String>" in xml
        assert "ENCODER_SETTINGS" not in xml

    def test_with_encoder_settings(self) -> None:
        xml = _build_tags_xml("Furnace v1.4.0", "hevc_nvenc / main10 / cq=25")
        assert "<Name>ENCODER</Name>" in xml
        assert "<Name>ENCODER_SETTINGS</Name>" in xml
        assert "<String>hevc_nvenc / main10 / cq=25</String>" in xml
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ffmpeg_encode_cmd.py::TestBuildTagsXml -v`
Expected: PASS

- [ ] **Step 5: Wire up in executor**

In `furnace/services/executor.py`, change the variable name back and pass it to tagger.

Line ~272, change `_encoder_settings` to `encoder_settings`:
```python
        rc, encoder_settings = self._encoder.encode(
            input_path=main_source,
            output_path=video_output,
            video_params=job.video_params,
            source_size=job.source_size,
            on_progress=_encode_progress,
        )
```

Lines ~344-345, update the tagger call:
```python
        tag_value = f"Furnace v{FURNACE_VERSION}"
        rc = self._tagger.set_encoder_tag(muxed_path, tag_value, encoder_settings)
```

- [ ] **Step 6: Run all tests**

Run: `uv run pytest tests/ -q`
Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add furnace/core/ports.py furnace/adapters/mkvpropedit.py furnace/services/executor.py tests/test_ffmpeg_encode_cmd.py
git commit -m "feat: write ENCODER_SETTINGS tag to output MKV"
```

---

### Task 4: Quality gates

- [ ] **Step 1: Lint**

Run: `uv run ruff check furnace/`
Expected: clean

- [ ] **Step 2: Type check**

Run: `uv run mypy furnace/ --strict`
Expected: clean

- [ ] **Step 3: Full test suite**

Run: `uv run pytest tests/ -q`
Expected: all pass
