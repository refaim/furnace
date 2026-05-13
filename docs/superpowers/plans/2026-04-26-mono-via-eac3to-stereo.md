# Mono via eac3to stereo intermediate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-rolled ITU multichannel → mono downmix in `FFmpegAdapter` with a two-step chain: eac3to performs `multichannel → stereo` (using its bitstream-aware mix matrix), then ffmpeg averages `(L+R)/2` to mono. Stereo sources flagged as fake-mono go straight through ffmpeg without eac3to.

**Architecture:** Hexagonal split unchanged. `AudioExtractor` port renames `downmix_to_mono_wav(channels, …)` to `stereo_to_mono_wav(…)` (drops `channels`). `FFmpegAdapter` keeps a single `pan=mono|c0=0.5*FL+0.5*FR` filter — the 5.1 / 7.1 ITU branches and `alimiter` are deleted. `Executor._process_audio_track` rewrites the `DownmixMode.MONO` branch into two paths: stereo source → direct ffmpeg; multichannel → extract (or pre-decode) → eac3to `-downStereo` → ffmpeg stereo→mono → qaac. Planner, classifier, models, and serialization stay untouched.

**Tech Stack:** Python 3.13, `unittest.mock` for adapter mocking, pytest with 100 % line + branch coverage enforced via `make check`. TDD strict — failing tests are written before any implementation change. All test/lint/type runs go through the Makefile (`make test`, `make check`), never `uv run pytest` directly. The Makefile only runs the full suite (no per-file selection); intermediate tasks therefore expect a known-red state, and full green is verified at the end.

**Design spec:** `docs/superpowers/specs/2026-04-26-mono-via-eac3to-stereo-design.md`

**Commit policy:** NO intermediate commits. The user commits when explicitly told to at the end (per `CLAUDE.md` and standing preference). No commit step appears inside individual tasks; commits are out of scope for this plan.

---

## File Structure

**Modified files:**

- `furnace/core/ports.py` — rename `downmix_to_mono_wav` → `stereo_to_mono_wav` on `AudioExtractor`, drop `channels` parameter, update docstring.
- `furnace/adapters/ffmpeg.py` — rename method, delete 5.1 / 7.1 branches and `aformat=channel_layouts=…` / `alimiter=limit=0.99`, drop `channels` parameter, update docstring.
- `furnace/services/executor.py` — rewrite the `DownmixMode.MONO` branch in `_process_audio_track` (currently lines 510-540).
- `furnace/__init__.py` — bump `VERSION` to `1.15.0`.
- `pyproject.toml` — bump `version` to `1.15.0`.
- `tests/test_ffmpeg_mono_downmix.py` — drop 5.1 / 7.1 / `alimiter` cases; keep stereo + delay handling + return-code propagation.
- `tests/services/test_executor_downmix.py` — extend `_instr` helper with `delay_ms`; rewrite the `TestDecodeEncodeMonoDownmix` class.
- `tests/core/test_ports.py` — update `_MinimalAudioExtractor` stub method, the `test_audio_extractor_*` signature lock tests, and the lone exercise call in `test_minimal_audio_extractor_method_surface`.

**No new files.**

---

## Task 1: Rewrite all three test files for the new API

**Files:**
- Test: `tests/test_ffmpeg_mono_downmix.py` (full rewrite)
- Test: `tests/services/test_executor_downmix.py` (extend helper + rewrite one class)
- Test: `tests/core/test_ports.py` (update stub + signature-lock tests)

Rationale: TDD up-front. After this task the tests describe the new API completely; production code in Task 2 makes them pass. Running `make test` at the end of this task confirms the tests are red for the *expected* reason — so we know they're testing the right thing.

- [ ] **Step 1: Replace `tests/test_ffmpeg_mono_downmix.py` in full**

Overwrite the entire file with this content:

```python
"""Test the filter chain built by FFmpegAdapter.stereo_to_mono_wav.

``run_tool`` is patched in every test — no real ffmpeg invocation. We only
verify the command line the adapter builds (pan formula, delay handling,
exit-code propagation, log path). Multichannel collapse is no longer this
method's concern; eac3to handles that step in the executor pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from furnace.adapters.ffmpeg import FFmpegAdapter

PAN_STEREO = "pan=mono|c0=0.5*FL+0.5*FR"


@pytest.fixture
def adapter() -> FFmpegAdapter:
    return FFmpegAdapter(
        ffmpeg_path=Path("ffmpeg"),
        ffprobe_path=Path("ffprobe"),
    )


def _af_value(call_args: Any) -> str:
    cmd: list[str] = call_args[0][0]
    return cmd[cmd.index("-af") + 1]


def _invoke(
    adapter: FFmpegAdapter,
    tmp_path: Path,
    *,
    delay_ms: int = 0,
) -> str:
    """Run the adapter with run_tool patched, return the -af value."""
    with patch("furnace.adapters.ffmpeg.run_tool") as run_tool:
        run_tool.return_value = (0, "")
        adapter.stereo_to_mono_wav(
            input_path=tmp_path / "a.mkv",
            stream_index=1,
            output_wav=tmp_path / "out.wav",
            delay_ms=delay_ms,
        )
    return _af_value(run_tool.call_args)


def test_stereo_averages_fronts(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path)
    assert PAN_STEREO in af


def test_no_alimiter(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    """(L+R)/2 cannot exceed unity for normalized PCM — no limiter needed."""
    af = _invoke(adapter, tmp_path)
    assert "alimiter" not in af


def test_no_layout_normalizer(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    """Input is already stereo from eac3to; no aformat needed."""
    af = _invoke(adapter, tmp_path)
    assert "aformat=" not in af


def test_zero_delay_has_no_delay_filter(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path, delay_ms=0)
    assert "adelay" not in af
    assert "atrim" not in af


def test_positive_delay_appends_adelay(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    af = _invoke(adapter, tmp_path, delay_ms=50)
    assert "adelay=50" in af
    assert PAN_STEREO in af


def test_negative_delay_appends_atrim(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    """delay_ms<0 trims |delay_ms|/1000 seconds of lead-in."""
    af = _invoke(adapter, tmp_path, delay_ms=-50)
    assert "atrim=start=0.050" in af
    assert PAN_STEREO in af
    assert "adelay" not in af


def test_returns_run_tool_exit_code(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    """Propagating ffmpeg's exit code lets the executor branch on failure."""
    with patch("furnace.adapters.ffmpeg.run_tool") as run_tool:
        run_tool.return_value = (42, "")
        rc = adapter.stereo_to_mono_wav(
            input_path=tmp_path / "a.mkv",
            stream_index=1,
            output_wav=tmp_path / "out.wav",
            delay_ms=0,
        )
    assert rc == 42


def test_log_path_uses_log_dir_when_set(
    adapter: FFmpegAdapter, tmp_path: Path,
) -> None:
    """When `set_log_dir` is configured, the log_path passes through to
    run_tool. Covers the truthy branch of the `if self._log_dir` guard.
    """
    adapter.set_log_dir(tmp_path)
    with patch("furnace.adapters.ffmpeg.run_tool") as run_tool:
        run_tool.return_value = (0, "")
        adapter.stereo_to_mono_wav(
            input_path=tmp_path / "a.mkv",
            stream_index=7,
            output_wav=tmp_path / "out.wav",
            delay_ms=0,
        )
    assert run_tool.call_args.kwargs["log_path"] == tmp_path / "ffmpeg_mono_s7.log"
```

- [ ] **Step 2: Extend the `_instr` helper in `tests/services/test_executor_downmix.py`**

Locate the `_instr` helper at the top of the file (lines 20-33). Replace it with:

```python
def _instr(
    codec_name: str,
    downmix: DownmixMode | None = None,
    channels: int | None = 8,
    stream_index: int = 1,
    delay_ms: int = 0,
) -> AudioInstruction:
    return make_audio_instruction(
        stream_index=stream_index,
        action=AudioAction.DECODE_ENCODE,
        codec_name=codec_name,
        channels=channels,
        bitrate=4_500_000,
        downmix=downmix,
        delay_ms=delay_ms,
    )
```

- [ ] **Step 3: Replace `TestDecodeEncodeMonoDownmix` in `tests/services/test_executor_downmix.py`**

The existing `TestDecodeEncodeMonoDownmix` class (starts at line 179, runs to end of file) tests the old `downmix_to_mono_wav` direct-mono path. Replace the **entire class** with:

```python
class TestDecodeEncodeMonoDownmix:
    """DECODE_ENCODE + downmix=MONO chains eac3to (multichannel only) and
    ffmpeg's stereo_to_mono_wav, then qaac. Stereo sources skip eac3to.
    """

    # ---- multichannel, eac3to-friendly source ----

    def test_5_1_dts_chains_extract_eac3to_stereo_mono_qaac(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0

        instr = _instr("dts", downmix=DownmixMode.MONO, channels=6, stream_index=1)
        executor._process_audio_track(instr, tmp_path, _job())

        mocks.audio_extractor.extract_track.assert_called_once()
        mocks.audio_extractor.ffmpeg_to_wav.assert_not_called()

        decode_call = mocks.audio_decoder.decode_lossless.call_args
        assert decode_call.kwargs.get("downmix") == DownmixMode.STEREO

        mono_call = mocks.audio_extractor.stereo_to_mono_wav.call_args
        assert mono_call.kwargs["stream_index"] == 0
        assert mono_call.kwargs["delay_ms"] == 0
        assert mono_call.kwargs["output_wav"].suffix == ".wav"

        mocks.aac_encoder.encode_aac.assert_called_once()

    def test_7_1_truehd_chains_extract_eac3to_stereo_mono_qaac(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0

        instr = _instr("truehd", downmix=DownmixMode.MONO, channels=8, stream_index=2)
        executor._process_audio_track(instr, tmp_path, _job())

        mocks.audio_extractor.extract_track.assert_called_once()
        decode_call = mocks.audio_decoder.decode_lossless.call_args
        assert decode_call.kwargs.get("downmix") == DownmixMode.STEREO
        mocks.audio_extractor.stereo_to_mono_wav.assert_called_once()
        mocks.aac_encoder.encode_aac.assert_called_once()

    # ---- multichannel, eac3to-incompatible source ----

    def test_aac_5_1_chains_ffmpeg_to_wav_eac3to_stereo_mono_qaac(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0

        instr = _instr("aac", downmix=DownmixMode.MONO, channels=6, stream_index=3)
        executor._process_audio_track(instr, tmp_path, _job())

        mocks.audio_extractor.ffmpeg_to_wav.assert_called_once()
        mocks.audio_extractor.extract_track.assert_not_called()

        decode_call = mocks.audio_decoder.decode_lossless.call_args
        assert decode_call.kwargs.get("downmix") == DownmixMode.STEREO

        mocks.audio_extractor.stereo_to_mono_wav.assert_called_once()
        mocks.aac_encoder.encode_aac.assert_called_once()

    # ---- delay routing ----

    def test_multichannel_delay_goes_to_eac3to_not_to_mono_step(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        """Delay is applied during the multichannel collapse, not the
        stereo→mono step. The stereo→mono call always sees delay_ms=0.
        """
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0

        instr = _instr(
            "dts", downmix=DownmixMode.MONO, channels=6,
            stream_index=1, delay_ms=125,
        )
        executor._process_audio_track(instr, tmp_path, _job())

        decode_call = mocks.audio_decoder.decode_lossless.call_args
        # decode_lossless positional layout: (input, output, delay_ms, ...)
        assert decode_call.args[2] == 125

        mono_call = mocks.audio_extractor.stereo_to_mono_wav.call_args
        assert mono_call.kwargs["delay_ms"] == 0

    # ---- stereo source (skip eac3to entirely) ----

    def test_stereo_source_skips_eac3to_calls_mono_directly(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        """A 2-channel source flagged as MONO has nothing to collapse —
        ffmpeg averages L and R directly with delay applied here."""
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0

        instr = _instr(
            "ac3", downmix=DownmixMode.MONO, channels=2,
            stream_index=4, delay_ms=-30,
        )
        executor._process_audio_track(instr, tmp_path, _job())

        mocks.audio_extractor.extract_track.assert_not_called()
        mocks.audio_extractor.ffmpeg_to_wav.assert_not_called()
        mocks.audio_decoder.decode_lossless.assert_not_called()

        mono_call = mocks.audio_extractor.stereo_to_mono_wav.call_args
        assert mono_call.kwargs["stream_index"] == 4
        assert mono_call.kwargs["input_path"] == Path(instr.source_file)
        assert mono_call.kwargs["delay_ms"] == -30

        mocks.aac_encoder.encode_aac.assert_called_once()

    # ---- regression: non-MONO path untouched ----

    def test_decode_encode_without_mono_does_not_call_stereo_to_mono(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0

        instr = _instr("truehd", downmix=None, channels=6)
        executor._process_audio_track(instr, tmp_path, _job())

        mocks.audio_extractor.stereo_to_mono_wav.assert_not_called()
        mocks.audio_decoder.decode_lossless.assert_called_once()

    # ---- failure modes ----

    def test_multichannel_raises_when_eac3to_fails(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.audio_decoder.decode_lossless.return_value = 7
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0

        instr = _instr("dts", downmix=DownmixMode.MONO, channels=6, stream_index=1)
        with pytest.raises(RuntimeError, match=r"eac3to -downStereo failed.*rc=7"):
            executor._process_audio_track(instr, tmp_path, _job())

        mocks.audio_extractor.stereo_to_mono_wav.assert_not_called()
        mocks.aac_encoder.encode_aac.assert_not_called()

    def test_multichannel_raises_when_extract_fails(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.extract_track.return_value = 9
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0

        instr = _instr("dts", downmix=DownmixMode.MONO, channels=6, stream_index=1)
        with pytest.raises(RuntimeError, match=r"Audio extract.*MONO.*rc=9"):
            executor._process_audio_track(instr, tmp_path, _job())

        mocks.audio_decoder.decode_lossless.assert_not_called()
        mocks.audio_extractor.stereo_to_mono_wav.assert_not_called()

    def test_multichannel_raises_when_ffmpeg_pre_decode_fails(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.ffmpeg_to_wav.return_value = 11
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0

        instr = _instr("aac", downmix=DownmixMode.MONO, channels=6, stream_index=1)
        with pytest.raises(RuntimeError, match=r"ffmpeg pre-decode.*MONO.*rc=11"):
            executor._process_audio_track(instr, tmp_path, _job())

        mocks.audio_decoder.decode_lossless.assert_not_called()

    def test_multichannel_raises_when_stereo_to_mono_fails(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 5

        instr = _instr("dts", downmix=DownmixMode.MONO, channels=6, stream_index=1)
        with pytest.raises(RuntimeError, match=r"stereo_to_mono_wav failed.*rc=5"):
            executor._process_audio_track(instr, tmp_path, _job())

        mocks.aac_encoder.encode_aac.assert_not_called()

    def test_stereo_raises_when_stereo_to_mono_fails(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 5

        instr = _instr("ac3", downmix=DownmixMode.MONO, channels=2, stream_index=1)
        with pytest.raises(RuntimeError, match=r"stereo_to_mono_wav failed.*rc=5"):
            executor._process_audio_track(instr, tmp_path, _job())

        mocks.aac_encoder.encode_aac.assert_not_called()

    def test_mono_raises_when_encode_aac_fails(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0
        mocks.aac_encoder.encode_aac.return_value = 3

        instr = _instr("ac3", downmix=DownmixMode.MONO, channels=2, stream_index=1)
        with pytest.raises(RuntimeError, match=r"encode_aac failed.*rc=3"):
            executor._process_audio_track(instr, tmp_path, _job())

    def test_mono_raises_when_channels_none(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        instr = _instr("dts", downmix=DownmixMode.MONO, channels=None)
        with pytest.raises(RuntimeError, match="MONO downmix without channel count"):
            executor._process_audio_track(instr, tmp_path, _job())
        mocks.audio_extractor.stereo_to_mono_wav.assert_not_called()
        mocks.audio_extractor.extract_track.assert_not_called()
        mocks.audio_extractor.ffmpeg_to_wav.assert_not_called()
        mocks.audio_decoder.decode_lossless.assert_not_called()
```

Leave the other classes in this file (`TestDecodeEncodeDownmixRouting`, `TestDecodeEncodeDownmixProgressWiring`) untouched.

- [ ] **Step 4: Update `_MinimalAudioExtractor` and signature-lock tests in `tests/core/test_ports.py`**

In `_MinimalAudioExtractor` (around lines 123-157), replace the `downmix_to_mono_wav` method with:

```python
    def stereo_to_mono_wav(
        self,
        input_path: Path,  # noqa: ARG002
        stream_index: int,  # noqa: ARG002
        output_wav: Path,  # noqa: ARG002
        delay_ms: int,  # noqa: ARG002
    ) -> int:
        return 0
```

Then replace the four affected test functions (`test_audio_extractor_has_downmix_to_mono_wav`, `test_audio_extractor_downmix_to_mono_wav_signature`, `test_minimal_audio_extractor_satisfies_runtime_checkable_protocol`, `test_minimal_audio_extractor_method_surface`) with:

```python
def test_audio_extractor_has_stereo_to_mono_wav() -> None:
    assert hasattr(AudioExtractor, "stereo_to_mono_wav")
    assert callable(AudioExtractor.stereo_to_mono_wav)


def test_audio_extractor_stereo_to_mono_wav_signature() -> None:
    sig = inspect.signature(AudioExtractor.stereo_to_mono_wav)
    params = sig.parameters
    # self + four positional args, in fixed order — channels is gone
    assert list(params) == [
        "self",
        "input_path",
        "stream_index",
        "output_wav",
        "delay_ms",
    ]

    hints = typing.get_type_hints(AudioExtractor.stereo_to_mono_wav)
    assert hints["input_path"] is Path
    assert hints["stream_index"] is int
    assert hints["output_wav"] is Path
    assert hints["delay_ms"] is int
    assert hints["return"] is int


def test_minimal_audio_extractor_satisfies_runtime_checkable_protocol(
    tmp_path: Path,
) -> None:
    stub = _MinimalAudioExtractor()
    assert isinstance(stub, AudioExtractor)
    rc = stub.stereo_to_mono_wav(Path("/dev/null"), 1, tmp_path / "out.wav", 0)
    assert rc == 0


def test_minimal_audio_extractor_method_surface(tmp_path: Path) -> None:
    """Exercise every method of the AudioExtractor stub so coverage stays
    at 100% — same rationale as ``test_minimal_prober_method_surface``.
    """
    stub = _MinimalAudioExtractor()
    assert stub.extract_track(Path("/dev/null"), 0, tmp_path / "o.thd") == 0
    assert stub.ffmpeg_to_wav(Path("/dev/null"), 0, tmp_path / "o.wav") == 0
    assert stub.stereo_to_mono_wav(Path("/dev/null"), 0, tmp_path / "o.wav", -50) == 0
```

- [ ] **Step 5: Run `make test` — expected RED**

Run: `make test`
Expected: tests fail. Specifically:
- `tests/test_ffmpeg_mono_downmix.py::*` — `AttributeError: 'FFmpegAdapter' object has no attribute 'stereo_to_mono_wav'`.
- `tests/services/test_executor_downmix.py::TestDecodeEncodeMonoDownmix::*` — `AssertionError: Expected 'stereo_to_mono_wav' to have been called` (executor still calls `downmix_to_mono_wav` on the mock, which auto-creates) **or** the new `RuntimeError(... downmix_to_mono_wav failed: rc=<Mock>)` raised from the old code path.
- `tests/core/test_ports.py::test_audio_extractor_has_stereo_to_mono_wav` and the signature lock — `AttributeError` on the Protocol.
- The runtime-checkable isinstance check — `assert isinstance(stub, AudioExtractor)` may either pass or fail depending on whether the old method (`downmix_to_mono_wav`) is still on the Protocol; once we drop it in Task 2 it will pass naturally.

Confirm the failures are in those three files only and trace back to the rename. Other test files must remain green. If anything else fails, stop and investigate before continuing.

---

## Task 2: Implement the new API across port, adapter, and executor

**Files:**
- Modify: `furnace/core/ports.py:145-161`
- Modify: `furnace/adapters/ffmpeg.py:586-642`
- Modify: `furnace/services/executor.py:510-540`

Rationale: tests from Task 1 lock the contract. This task makes them pass by simultaneously updating the Protocol, the adapter implementation, and the executor's MONO branch. We do all three before the next `make test` because they're tightly coupled: the runtime-checkable Protocol fails if the adapter and stub disagree, and the executor calls the adapter directly.

- [ ] **Step 1: Update `AudioExtractor` port in `furnace/core/ports.py`**

Replace the `downmix_to_mono_wav` declaration on `AudioExtractor` (currently `furnace/core/ports.py:145-161`) with:

```python
    def stereo_to_mono_wav(
        self,
        input_path: Path,
        stream_index: int,
        output_wav: Path,
        delay_ms: int,
    ) -> int:
        """Average a stereo source to a mono WAV via ffmpeg's pan filter.

        Filter: ``pan=mono|c0=0.5*FL+0.5*FR``. ``delay_ms`` applies an
        ``adelay`` (positive) or ``atrim`` (negative). No limiter — for
        normalized PCM ``|0.5L + 0.5R| <= max(|L|, |R|) <= 1.0``.

        Multichannel collapse is the caller's responsibility (typically
        eac3to ``-downStereo``). Returns ffmpeg exit code.
        """
        ...
```

- [ ] **Step 2: Replace the adapter method in `furnace/adapters/ffmpeg.py`**

Locate the existing `downmix_to_mono_wav` method (currently `furnace/adapters/ffmpeg.py:586-642`) and replace the **entire method** with:

```python
    def stereo_to_mono_wav(
        self,
        input_path: Path,
        stream_index: int,
        output_wav: Path,
        delay_ms: int,
    ) -> int:
        """Average a stereo source to a mono WAV via ffmpeg's pan filter.

        Filter chain: ``pan=mono|c0=0.5*FL+0.5*FR`` plus an optional
        ``adelay`` (positive ``delay_ms``) or ``atrim`` (negative
        ``delay_ms``). No limiter is needed: for normalized PCM
        ``|0.5L + 0.5R| <= max(|L|, |R|) <= 1.0``.

        Multichannel collapse is the caller's responsibility — typically
        eac3to ``-downStereo`` runs before this step in the executor's
        DECODE_ENCODE + MONO path.
        """
        filters = ["pan=mono|c0=0.5*FL+0.5*FR"]
        if delay_ms > 0:
            filters.append(f"adelay={delay_ms}")
        elif delay_ms < 0:
            seconds = abs(delay_ms) / 1000.0
            filters.append(f"atrim=start={seconds:.3f}")

        af_value = ",".join(filters)

        cmd = [
            str(self._ffmpeg),
            "-hide_banner", "-loglevel", "warning",
            "-i", str(input_path),
            "-map", f"0:{stream_index}",
            "-af", af_value,
            "-ac", "1",
            "-f", "wav",
            "-rf64", "auto",
            "-y", str(output_wav),
        ]
        log_path = (
            self._log_dir / f"ffmpeg_mono_s{stream_index}.log"
            if self._log_dir
            else None
        )
        rc, _out = run_tool(cmd, on_output=self._on_output, log_path=log_path)
        return rc
```

- [ ] **Step 3: Rewrite the MONO branch in `furnace/services/executor.py`**

In `_process_audio_track`, locate:

```python
        if instr.action == AudioAction.DECODE_ENCODE:
            if instr.downmix == DownmixMode.MONO:
                # Planner guarantees channels is set ...
```

(currently `furnace/services/executor.py:509-540`). Replace the **entire** `if instr.downmix == DownmixMode.MONO:` block (everything from `if instr.downmix == DownmixMode.MONO:` through the matching `return m4a_path` inclusive) with:

```python
            if instr.downmix == DownmixMode.MONO:
                if instr.channels is None:
                    raise RuntimeError(
                        f"MONO downmix without channel count for stream {track_idx}",
                    )

                # Stereo sources have nothing to collapse — go straight to
                # ffmpeg's (L+R)/2 with delay applied here.
                if instr.channels == 2:
                    mono_wav = temp_dir / f"audio_{track_idx}_mono.wav"
                    rc = self._audio_extractor.stereo_to_mono_wav(
                        input_path=source_path,
                        stream_index=track_idx,
                        output_wav=mono_wav,
                        delay_ms=instr.delay_ms,
                    )
                    if rc != 0:
                        raise RuntimeError(
                            f"stereo_to_mono_wav failed: rc={rc}",
                        )

                    m4a_path = temp_dir / f"audio_{track_idx}.m4a"
                    _, on_progress = self._make_progress_callback(total_s=None)
                    rc = self._aac_encoder.encode_aac(
                        mono_wav,
                        m4a_path,
                        on_progress=on_progress,
                    )
                    if rc != 0:
                        raise RuntimeError(f"encode_aac failed: rc={rc}")
                    return m4a_path

                # Multichannel: extract (or pre-decode) -> eac3to -downStereo
                # -> ffmpeg stereo->mono -> qaac. Delay is applied by eac3to.
                if _codec_supported_by_eac3to(instr.codec_name):
                    extracted = temp_dir / f"audio_{track_idx}_raw{ext}"
                    _, on_progress = self._make_progress_callback(
                        total_s=job.duration_s or None,
                    )
                    rc = self._audio_extractor.extract_track(
                        source_path,
                        track_idx,
                        extracted,
                        on_progress=on_progress,
                    )
                    if rc != 0:
                        raise RuntimeError(
                            f"Audio extract (MONO multichannel) failed with "
                            f"rc={rc} for stream {track_idx}",
                        )
                else:
                    extracted = temp_dir / f"audio_{track_idx}_pre.wav"
                    _, on_progress = self._make_progress_callback(
                        total_s=job.duration_s or None,
                    )
                    rc = self._audio_extractor.ffmpeg_to_wav(
                        source_path,
                        track_idx,
                        extracted,
                        on_progress=on_progress,
                    )
                    if rc != 0:
                        raise RuntimeError(
                            f"ffmpeg pre-decode (MONO multichannel) failed "
                            f"with rc={rc} for stream {track_idx}",
                        )

                stereo_wav = temp_dir / f"audio_{track_idx}_stereo.wav"
                _, on_progress = self._make_progress_callback(total_s=None)
                rc = self._audio_decoder.decode_lossless(
                    extracted,
                    stereo_wav,
                    instr.delay_ms,
                    on_progress=on_progress,
                    downmix=DownmixMode.STEREO,
                )
                if rc != 0:
                    raise RuntimeError(
                        f"eac3to -downStereo failed: rc={rc}",
                    )

                mono_wav = temp_dir / f"audio_{track_idx}_mono.wav"
                rc = self._audio_extractor.stereo_to_mono_wav(
                    input_path=stereo_wav,
                    stream_index=0,
                    output_wav=mono_wav,
                    delay_ms=0,
                )
                if rc != 0:
                    raise RuntimeError(
                        f"stereo_to_mono_wav failed: rc={rc}",
                    )

                m4a_path = temp_dir / f"audio_{track_idx}.m4a"
                _, on_progress = self._make_progress_callback(total_s=None)
                rc = self._aac_encoder.encode_aac(
                    mono_wav,
                    m4a_path,
                    on_progress=on_progress,
                )
                if rc != 0:
                    raise RuntimeError(f"encode_aac failed: rc={rc}")
                return m4a_path
```

Indentation note: the outer block sits inside `if instr.action == AudioAction.DECODE_ENCODE:` (12-space indent for the `if instr.downmix == DownmixMode.MONO:` line). Match the surrounding indentation exactly.

- [ ] **Step 4: Run `make test` — expected GREEN**

Run: `make test`
Expected: full suite passes, coverage ≥ 100 %.

If any test fails, the most likely causes are:
- Indentation drift in the executor block — re-check that the new block is nested inside `AudioAction.DECODE_ENCODE`.
- A leftover reference to `downmix_to_mono_wav` somewhere — `grep -rn "downmix_to_mono_wav" furnace/ tests/` should return zero hits.
- Coverage gap on a new branch — see the Task 4 coverage notes.

Fix and re-run until green.

---

## Task 3: Bump version to 1.15.0

**Files:**
- Modify: `furnace/__init__.py`
- Modify: `pyproject.toml`

Rationale: per `CLAUDE.md`, every change to user-facing behaviour bumps SemVer. The MONO downmix produces different bytes (different matrix on multichannel collapse, no limiter) and a different temp-file shape. That's MINOR.

- [ ] **Step 1: Bump `furnace/__init__.py`**

Change the `VERSION` line from `VERSION = "1.14.0"` to:

```python
VERSION = "1.15.0"
```

- [ ] **Step 2: Bump `pyproject.toml`**

Change the `version = "1.14.0"` line under `[project]` to:

```toml
version = "1.15.0"
```

---

## Task 4: Final `make check`

**Rationale:** `make check` is the single source of truth (`CLAUDE.md`). Lint, typecheck, and the full test suite with 100 % line + branch coverage on `furnace/` and `tests/` must all pass before the work is considered done.

- [ ] **Step 1: Run `make check`**

Run: `make check`
Expected: ruff clean, mypy strict clean, pytest with 100 % line + branch coverage.

Coverage checklist for the new code paths (every one should be hit by the tests added in Task 1):

- `if instr.channels is None` true branch — `test_mono_raises_when_channels_none`.
- `if instr.channels == 2` true branch — `test_stereo_source_skips_eac3to_calls_mono_directly`, `test_stereo_raises_when_stereo_to_mono_fails`.
- `if instr.channels == 2` false branch (multichannel) — every multichannel test.
- `_codec_supported_by_eac3to` true branch — `test_5_1_dts_chains_…`, `test_7_1_truehd_chains_…`.
- `_codec_supported_by_eac3to` false branch — `test_aac_5_1_chains_ffmpeg_to_wav_…`.
- `extract_track` rc != 0 branch — `test_multichannel_raises_when_extract_fails`.
- `ffmpeg_to_wav` rc != 0 branch — `test_multichannel_raises_when_ffmpeg_pre_decode_fails`.
- `decode_lossless` rc != 0 branch — `test_multichannel_raises_when_eac3to_fails`.
- Multichannel `stereo_to_mono_wav` rc != 0 — `test_multichannel_raises_when_stereo_to_mono_fails`.
- Stereo-direct `stereo_to_mono_wav` rc != 0 — `test_stereo_raises_when_stereo_to_mono_fails`.
- Final `encode_aac` rc != 0 — `test_mono_raises_when_encode_aac_fails`.
- `delay_ms == 0`, `> 0`, `< 0` branches in `stereo_to_mono_wav` — three dedicated tests in `test_ffmpeg_mono_downmix.py`.
- `self._log_dir` truthy / falsy branches — `test_log_path_uses_log_dir_when_set` and the default fixture (no log_dir).

If `make check` reports an uncovered line or branch outside this list, add a focused test that hits exactly that path. Re-run `make check` until clean.

- [ ] **Step 2: Confirm clean output**

Re-read the `make check` summary. Every line: PASS. Every percentage: 100. The work is done — stop here. Do not commit; the user will commit explicitly.

---

## Self-review notes

Cross-check against the spec (`docs/superpowers/specs/2026-04-26-mono-via-eac3to-stereo-design.md`):

- **Adapter rename + simplification:** Task 1 step 1 + Task 2 step 2.
- **Stereo source skips eac3to:** Task 2 step 3 (`if instr.channels == 2:` branch) + Task 1 step 3 test `test_stereo_source_skips_eac3to_calls_mono_directly`.
- **Multichannel via eac3to-friendly extract:** Task 2 step 3 (`_codec_supported_by_eac3to(...)` true branch) + tests `test_5_1_dts_chains_…`, `test_7_1_truehd_chains_…`.
- **Multichannel via ffmpeg pre-decode:** Task 2 step 3 (`else` branch) + test `test_aac_5_1_chains_ffmpeg_to_wav_…`.
- **Delay routed to eac3to (not to mono step) for multichannel:** test `test_multichannel_delay_goes_to_eac3to_not_to_mono_step`.
- **Delay routed to ffmpeg for stereo direct:** test `test_stereo_source_skips_eac3to_calls_mono_directly` (asserts `delay_ms == -30`).
- **Failure modes (extract, ffmpeg_to_wav, eac3to, stereo_to_mono_wav, encode_aac, channels=None):** dedicated tests in `TestDecodeEncodeMonoDownmix`.
- **No fallback on eac3to failure:** verified by `test_multichannel_raises_when_eac3to_fails` — `stereo_to_mono_wav` is asserted not-called.
- **No `alimiter`:** Task 1 step 1 test `test_no_alimiter`.
- **Port surface matches adapter:** Task 1 step 4 + Task 2 step 1.
- **Version bump:** Task 3.
- **100 % coverage gate:** Task 4.

Untouched on purpose: `tests/services/test_planner_downmix.py`, `tests/core/test_audio_profile.py`, `tests/test_plan.py`, `tests/test_cli.py`, `tests/adapters/test_eac3to_downmix.py`. Planner emits the same `AudioInstruction` shape; classifier verdicts are unchanged; serialization is unchanged; eac3to `-downStereo` is already covered.
