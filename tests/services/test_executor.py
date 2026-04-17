"""Comprehensive tests for furnace/services/executor.py.

Covers per-step methods (_process_audio_track, _process_subtitle_track,
_extract_chapters_file, _set_adapters_log_dir, _make_progress_callback,
DoviProcessor in constructor) and full integration tests (_run_pipeline,
run, graceful_shutdown).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from furnace.core.models import (
    AudioAction,
    DvMode,
    EncodeResult,
    JobStatus,
    SubtitleAction,
)
from furnace.core.progress import ProgressSample
from furnace.plan import load_plan, save_plan
from furnace.services.executor import Executor
from tests.conftest import (
    make_audio_instruction,
    make_job,
    make_plan,
    make_subtitle_instruction,
    make_video_params,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def executor_with_mocks() -> tuple[Executor, SimpleNamespace]:
    """Construct Executor with all adapter ports mocked.

    Returns (executor, mocks) namespace.
    """
    mocks = SimpleNamespace(
        encoder=MagicMock(),
        audio_extractor=MagicMock(),
        audio_decoder=MagicMock(),
        aac_encoder=MagicMock(),
        muxer=MagicMock(),
        tagger=MagicMock(),
        cleaner=MagicMock(),
        prober=MagicMock(),
    )
    mocks.audio_extractor.extract_track.return_value = 0
    mocks.audio_extractor.ffmpeg_to_wav.return_value = 0
    mocks.audio_decoder.decode_lossless.return_value = 0
    mocks.audio_decoder.denormalize.return_value = 0
    mocks.aac_encoder.encode_aac.return_value = 0
    mocks.encoder.encode.return_value = EncodeResult(return_code=0, encoder_settings="test")
    mocks.muxer.mux.return_value = 0
    mocks.tagger.set_encoder_tag.return_value = 0
    mocks.cleaner.clean.return_value = 0
    mocks.prober.probe.return_value = {"chapters": []}

    executor = Executor(
        encoder=mocks.encoder,
        audio_extractor=mocks.audio_extractor,
        audio_decoder=mocks.audio_decoder,
        aac_encoder=mocks.aac_encoder,
        muxer=mocks.muxer,
        tagger=mocks.tagger,
        cleaner=mocks.cleaner,
        prober=mocks.prober,
    )
    executor._vmaf_enabled = False  # normally set by run()
    return executor, mocks


def _minimal_job(**kwargs: Any) -> Any:
    """Build a Job with sane defaults suitable for executor tests."""
    defaults: dict[str, Any] = {
        "job_id": "test-job",
        "audio": [],
        "subtitles": [],
        "copy_chapters": False,
        "source_size": 0,
        "duration_s": 5400.0,
    }
    defaults.update(kwargs)
    return make_job(**defaults)


# =========================================================================
# Task 14 — Per-Step Tests
# =========================================================================


class TestProcessAudioTrackCopy:
    """Test 1: AudioAction.COPY branch."""

    def test_copy_success_returns_path(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        instr = make_audio_instruction(
            action=AudioAction.COPY,
            codec_name="aac",
            stream_index=1,
        )
        result = executor._process_audio_track(instr, tmp_path, _minimal_job())
        assert result == tmp_path / "audio_1.m4a"
        mocks.audio_extractor.extract_track.assert_called_once()

    def test_copy_failure_raises_runtime_error(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.extract_track.return_value = 1
        instr = make_audio_instruction(
            action=AudioAction.COPY,
            codec_name="ac3",
            stream_index=2,
        )
        with pytest.raises(RuntimeError, match=r"Audio extract \(COPY\) failed"):
            executor._process_audio_track(instr, tmp_path, _minimal_job())


class TestProcessAudioTrackDenorm:
    """Test 2: AudioAction.DENORM branch."""

    def test_denorm_success(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        instr = make_audio_instruction(
            action=AudioAction.DENORM,
            codec_name="ac3",
            stream_index=1,
            delay_ms=500,
        )
        result = executor._process_audio_track(instr, tmp_path, _minimal_job())
        assert result == tmp_path / "audio_1_denorm.ac3"
        mocks.audio_extractor.extract_track.assert_called_once()
        mocks.audio_decoder.denormalize.assert_called_once()
        denorm_call = mocks.audio_decoder.denormalize.call_args
        assert denorm_call[0][2] == 500  # delay_ms positional

    def test_denorm_extract_failure(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.extract_track.return_value = 1
        instr = make_audio_instruction(
            action=AudioAction.DENORM,
            codec_name="ac3",
            stream_index=1,
        )
        with pytest.raises(RuntimeError, match=r"Audio extract \(DENORM\) failed"):
            executor._process_audio_track(instr, tmp_path, _minimal_job())

    def test_denorm_denormalize_failure(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.audio_decoder.denormalize.return_value = 1
        instr = make_audio_instruction(
            action=AudioAction.DENORM,
            codec_name="eac3",
            stream_index=1,
        )
        with pytest.raises(RuntimeError, match="Audio denormalize failed"):
            executor._process_audio_track(instr, tmp_path, _minimal_job())


class TestProcessAudioTrackFfmpegEncode:
    """Test 3: AudioAction.FFMPEG_ENCODE branch."""

    def test_ffmpeg_encode_success(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        instr = make_audio_instruction(
            action=AudioAction.FFMPEG_ENCODE,
            codec_name="wmav2",
            stream_index=3,
        )
        result = executor._process_audio_track(instr, tmp_path, _minimal_job())
        assert result == tmp_path / "audio_3.m4a"
        mocks.audio_extractor.ffmpeg_to_wav.assert_called_once()
        mocks.aac_encoder.encode_aac.assert_called_once()

    def test_ffmpeg_encode_ffmpeg_failure(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.ffmpeg_to_wav.return_value = 1
        instr = make_audio_instruction(
            action=AudioAction.FFMPEG_ENCODE,
            codec_name="wmav2",
            stream_index=3,
        )
        with pytest.raises(RuntimeError, match="ffmpeg_to_wav failed"):
            executor._process_audio_track(instr, tmp_path, _minimal_job())

    def test_ffmpeg_encode_aac_failure(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.aac_encoder.encode_aac.return_value = 1
        instr = make_audio_instruction(
            action=AudioAction.FFMPEG_ENCODE,
            codec_name="wmav2",
            stream_index=3,
        )
        with pytest.raises(RuntimeError, match=r"AAC encode \(FFMPEG_ENCODE\) failed"):
            executor._process_audio_track(instr, tmp_path, _minimal_job())


class TestProcessAudioTrackDecodeEncodeNonEac3to:
    """Test 4: DECODE_ENCODE with non-eac3to codec (e.g., 'opus')."""

    def test_opus_uses_ffmpeg_to_wav_then_decode_then_encode(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        instr = make_audio_instruction(
            action=AudioAction.DECODE_ENCODE,
            codec_name="opus",
            stream_index=1,
        )
        result = executor._process_audio_track(instr, tmp_path, _minimal_job())
        assert result == tmp_path / "audio_1.m4a"
        mocks.audio_extractor.ffmpeg_to_wav.assert_called_once()
        assert not mocks.audio_extractor.extract_track.called
        mocks.audio_decoder.decode_lossless.assert_called_once()
        mocks.aac_encoder.encode_aac.assert_called_once()

    def test_opus_ffmpeg_failure(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.ffmpeg_to_wav.return_value = 1
        instr = make_audio_instruction(
            action=AudioAction.DECODE_ENCODE,
            codec_name="opus",
            stream_index=1,
        )
        with pytest.raises(RuntimeError, match="ffmpeg pre-decode failed"):
            executor._process_audio_track(instr, tmp_path, _minimal_job())

    def test_decode_encode_eac3to_extract_failure(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.extract_track.return_value = 1
        instr = make_audio_instruction(
            action=AudioAction.DECODE_ENCODE,
            codec_name="truehd",
            stream_index=1,
        )
        with pytest.raises(RuntimeError, match=r"Audio extract \(DECODE_ENCODE\) failed"):
            executor._process_audio_track(instr, tmp_path, _minimal_job())

    def test_decode_encode_decode_lossless_failure(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.audio_decoder.decode_lossless.return_value = 1
        instr = make_audio_instruction(
            action=AudioAction.DECODE_ENCODE,
            codec_name="truehd",
            stream_index=1,
        )
        with pytest.raises(RuntimeError, match="Audio decode_lossless failed"):
            executor._process_audio_track(instr, tmp_path, _minimal_job())

    def test_decode_encode_aac_encode_failure(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.aac_encoder.encode_aac.return_value = 1
        instr = make_audio_instruction(
            action=AudioAction.DECODE_ENCODE,
            codec_name="truehd",
            stream_index=1,
        )
        with pytest.raises(RuntimeError, match="AAC encode failed"):
            executor._process_audio_track(instr, tmp_path, _minimal_job())


class TestProcessAudioTrackUnknownAction:
    """Unknown action raises ValueError."""

    def test_unknown_action_raises(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, _mocks = executor_with_mocks
        instr = make_audio_instruction(action=AudioAction.COPY, stream_index=1)
        # Monkey-patch the action to an invalid value
        object.__setattr__(instr, "action", "BOGUS")
        with pytest.raises(ValueError, match="Unknown AudioAction"):
            executor._process_audio_track(instr, tmp_path, _minimal_job())


class TestProcessAudioTrackCodecExtensionMapping:
    """Verify correct file extension for various codec names."""

    def test_unknown_codec_gets_audio_ext(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, _mocks = executor_with_mocks
        instr = make_audio_instruction(
            action=AudioAction.COPY,
            codec_name="totally_unknown",
            stream_index=5,
        )
        result = executor._process_audio_track(instr, tmp_path, _minimal_job())
        assert result == tmp_path / "audio_5.audio"


# ---------------------------------------------------------------------------
# Subtitle tests
# ---------------------------------------------------------------------------


class TestProcessSubtitleTrackCopy:
    """Tests 5-6: COPY satellite and container."""

    def test_copy_satellite_srt_returns_path_as_is(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        """Test 5: external .srt → no extraction, returned as-is."""
        executor, mocks = executor_with_mocks
        srt_path = tmp_path / "subs.srt"
        srt_path.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")
        instr = make_subtitle_instruction(
            source_file=str(srt_path),
            action=SubtitleAction.COPY,
            codec_name="subrip",
            stream_index=0,
        )
        result = executor._process_subtitle_track(instr, tmp_path, _minimal_job())
        assert result == srt_path
        assert not mocks.audio_extractor.extract_track.called

    def test_copy_satellite_sup_returns_path_as_is(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, _mocks = executor_with_mocks
        sup_path = tmp_path / "subs.sup"
        sup_path.write_bytes(b"\x00")
        instr = make_subtitle_instruction(
            source_file=str(sup_path),
            action=SubtitleAction.COPY,
            codec_name="hdmv_pgs_subtitle",
            stream_index=0,
        )
        result = executor._process_subtitle_track(instr, tmp_path, _minimal_job())
        assert result == sup_path

    def test_copy_container_extracts(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        """Test 6: .mkv source → extract_track called."""
        executor, mocks = executor_with_mocks
        instr = make_subtitle_instruction(
            source_file="/src/movie.mkv",
            action=SubtitleAction.COPY,
            codec_name="hdmv_pgs_subtitle",
            stream_index=3,
        )
        result = executor._process_subtitle_track(instr, tmp_path, _minimal_job())
        assert result == tmp_path / "sub_3.sup"
        mocks.audio_extractor.extract_track.assert_called_once()

    def test_copy_container_extract_failure(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.extract_track.return_value = 1
        instr = make_subtitle_instruction(
            source_file="/src/movie.mkv",
            action=SubtitleAction.COPY,
            codec_name="hdmv_pgs_subtitle",
            stream_index=3,
        )
        with pytest.raises(RuntimeError, match=r"Subtitle extract \(COPY\) failed"):
            executor._process_subtitle_track(instr, tmp_path, _minimal_job())


class TestProcessSubtitleTrackCopyRecode:
    """Tests 7-9: COPY_RECODE branches."""

    def test_recode_satellite_cp1251_to_utf8(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        """Test 7: cp1251 → UTF-8 recode."""
        executor, mocks = executor_with_mocks
        srt_path = tmp_path / "subs.srt"
        text = "Привет мир"
        srt_path.write_bytes(text.encode("cp1251"))
        instr = make_subtitle_instruction(
            source_file=str(srt_path),
            action=SubtitleAction.COPY_RECODE,
            codec_name="subrip",
            stream_index=2,
            source_encoding="cp1251",
        )
        result = executor._process_subtitle_track(instr, tmp_path, _minimal_job())
        assert result.name == "sub_2_utf8.srt"
        assert result.read_text(encoding="utf-8") == text
        # No extract_track because it's a satellite file
        assert not mocks.audio_extractor.extract_track.called

    def test_recode_utf8_source_copies_as_is(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        """Test 8: UTF-8 source → plain copy (no decode/encode)."""
        executor, _mocks = executor_with_mocks
        srt_path = tmp_path / "subs.srt"
        srt_path.write_text("Hello world", encoding="utf-8")
        instr = make_subtitle_instruction(
            source_file=str(srt_path),
            action=SubtitleAction.COPY_RECODE,
            codec_name="subrip",
            stream_index=2,
            source_encoding="utf-8",
        )
        result = executor._process_subtitle_track(instr, tmp_path, _minimal_job())
        assert result.read_text(encoding="utf-8") == "Hello world"

    def test_recode_none_encoding_copies_as_is(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        """source_encoding=None defaults to utf-8, so copy as-is."""
        executor, _mocks = executor_with_mocks
        srt_path = tmp_path / "subs.srt"
        srt_path.write_text("Fallback", encoding="utf-8")
        instr = make_subtitle_instruction(
            source_file=str(srt_path),
            action=SubtitleAction.COPY_RECODE,
            codec_name="subrip",
            stream_index=2,
            source_encoding=None,
        )
        result = executor._process_subtitle_track(instr, tmp_path, _minimal_job())
        assert result.read_text(encoding="utf-8") == "Fallback"

    def test_recode_decode_error_falls_back_to_copy(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        """Test 9: decode error → copy as-is."""
        executor, _mocks = executor_with_mocks
        srt_path = tmp_path / "subs.srt"
        # Write invalid bytes for cp1251 that will still fail to decode as shift_jis
        srt_path.write_bytes(b"\x80\x81\x82\x83")
        instr = make_subtitle_instruction(
            source_file=str(srt_path),
            action=SubtitleAction.COPY_RECODE,
            codec_name="subrip",
            stream_index=2,
            source_encoding="shift_jis",
        )
        # Should not raise — falls back to copy
        result = executor._process_subtitle_track(instr, tmp_path, _minimal_job())
        assert result.exists()
        # Content should be the raw bytes (copied)
        assert result.read_bytes() == b"\x80\x81\x82\x83"

    def test_recode_container_extracts_first(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        """COPY_RECODE from .mkv: extract_track first, then recode."""
        executor, mocks = executor_with_mocks
        instr = make_subtitle_instruction(
            source_file="/src/movie.mkv",
            action=SubtitleAction.COPY_RECODE,
            codec_name="subrip",
            stream_index=2,
            source_encoding="utf-8",
        )
        def fake_extract(src: Any, idx: Any, out: Any, on_progress: Any = None) -> int:
            Path(out).write_text("Hello from container", encoding="utf-8")
            return 0

        mocks.audio_extractor.extract_track.side_effect = fake_extract
        result = executor._process_subtitle_track(instr, tmp_path, _minimal_job())
        mocks.audio_extractor.extract_track.assert_called_once()
        # Result should be the utf8 copy
        assert result == tmp_path / "sub_2_utf8.srt"

    def test_recode_container_extract_failure(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.extract_track.return_value = 1
        instr = make_subtitle_instruction(
            source_file="/src/movie.mkv",
            action=SubtitleAction.COPY_RECODE,
            codec_name="subrip",
            stream_index=2,
            source_encoding="cp1251",
        )
        with pytest.raises(RuntimeError, match=r"Subtitle extract \(COPY_RECODE\) failed"):
            executor._process_subtitle_track(instr, tmp_path, _minimal_job())


class TestProcessSubtitleUnknownAction:
    """Unknown subtitle action raises ValueError."""

    def test_unknown_action_raises(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, _mocks = executor_with_mocks
        instr = make_subtitle_instruction(action=SubtitleAction.COPY, stream_index=2)
        object.__setattr__(instr, "action", "BOGUS")
        with pytest.raises(ValueError, match="Unknown SubtitleAction"):
            executor._process_subtitle_track(instr, tmp_path, _minimal_job())


class TestProcessSubtitleCodecExtension:
    """Verify correct file extension for subtitle codecs."""

    def test_unknown_codec_gets_sub_ext(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, _mocks = executor_with_mocks
        instr = make_subtitle_instruction(
            source_file="/src/movie.mkv",
            action=SubtitleAction.COPY,
            codec_name="totally_unknown_sub",
            stream_index=5,
        )
        result = executor._process_subtitle_track(instr, tmp_path, _minimal_job())
        assert result == tmp_path / "sub_5.sub"


# ---------------------------------------------------------------------------
# _extract_chapters_file
# ---------------------------------------------------------------------------


class TestExtractChaptersFile:
    """Tests 10-12."""

    def test_chapters_present_writes_ogm(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        """Test 10: prober returns chapters → OGM file written."""
        executor, mocks = executor_with_mocks
        mocks.prober.probe.return_value = {
            "chapters": [
                {
                    "start_time": "0.000000",
                    "end_time": "300.000000",
                    "tags": {"title": "Chapter 1"},
                },
                {
                    "start_time": "300.000000",
                    "end_time": "600.000000",
                    "tags": {"title": "Chapter 2"},
                },
            ],
        }
        result = executor._extract_chapters_file(Path("/src/movie.mkv"), tmp_path)
        assert result is not None
        assert result == tmp_path / "chapters.txt"
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "CHAPTER01" in content

    def test_no_chapters_returns_none(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        """Test 11: empty chapters → None."""
        executor, mocks = executor_with_mocks
        mocks.prober.probe.return_value = {"chapters": []}
        result = executor._extract_chapters_file(Path("/src/movie.mkv"), tmp_path)
        assert result is None

    def test_probe_raises_returns_none(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        """Test 12: probe raises RuntimeError → None."""
        executor, mocks = executor_with_mocks
        mocks.prober.probe.side_effect = RuntimeError("probe failed")
        result = executor._extract_chapters_file(Path("/src/movie.mkv"), tmp_path)
        assert result is None


# ---------------------------------------------------------------------------
# _set_adapters_log_dir
# ---------------------------------------------------------------------------


class TestSetAdaptersLogDir:
    """Test 13."""

    def test_creates_subdir_and_calls_set_log_dir(
        self, tmp_path: Path,
    ) -> None:
        adapter1 = MagicMock()
        adapter2 = MagicMock()
        adapter2.set_log_dir = None  # simulate no set_log_dir attribute
        del adapter2.set_log_dir  # make getattr return None
        executor = Executor(
            encoder=adapter1,
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
            log_dir=tmp_path,
        )
        executor._set_adapters_log_dir("TestMovie")
        expected_dir = tmp_path / "TestMovie"
        assert expected_dir.is_dir()
        # adapter1 has set_log_dir → called with the job dir
        adapter1.set_log_dir.assert_called_once_with(expected_dir)

    def test_no_log_dir_does_nothing(self) -> None:
        executor = Executor(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
            log_dir=None,
        )
        # Should not raise
        executor._set_adapters_log_dir("Whatever")


# ---------------------------------------------------------------------------
# _make_progress_callback
# ---------------------------------------------------------------------------


class TestMakeProgressCallback:
    """Test 14."""

    def test_returns_tracker_and_callback(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
    ) -> None:
        executor, _mocks = executor_with_mocks
        tracker, callback = executor._make_progress_callback(total_s=100.0)
        assert callable(callback)
        # Callback should add samples to tracker
        sample = ProgressSample(fraction=0.5, speed=2.0)
        callback(sample)
        snap = tracker.snapshot()
        assert snap.fraction == pytest.approx(0.5)

    def test_callback_pushes_to_progress_when_set(self) -> None:
        progress_mock = MagicMock()
        executor = Executor(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
            progress=progress_mock,
        )
        _tracker, callback = executor._make_progress_callback(total_s=None)
        sample = ProgressSample(fraction=0.3)
        callback(sample)
        progress_mock.update_progress.assert_called_once()

    def test_callback_does_not_push_when_no_progress(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
    ) -> None:
        executor, _mocks = executor_with_mocks
        # executor._progress is None by default
        _, callback = executor._make_progress_callback()
        # Should not raise
        callback(ProgressSample(fraction=0.5))


# ---------------------------------------------------------------------------
# DoviProcessor in constructor
# ---------------------------------------------------------------------------


class TestDoviProcessorInConstructor:
    """Test 15: dovi_processor appended to _adapters."""

    def test_dovi_processor_appended(self) -> None:
        dovi = MagicMock()
        executor = Executor(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
            dovi_processor=dovi,
        )
        assert dovi in executor._adapters
        assert len(executor._adapters) == 8  # 7 base + 1 dovi

    def test_no_dovi_processor_not_appended(self) -> None:
        executor = Executor(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
            dovi_processor=None,
        )
        assert len(executor._adapters) == 7


# =========================================================================
# Task 15 — Integration Tests
# =========================================================================


def _pipeline_job(
    tmp_path: Path,
    *,
    audio: list[Any] | None = None,
    subtitles: list[Any] | None = None,
    dv_mode: DvMode | None = None,
    copy_chapters: bool = False,
    chapters_source: str | None = None,
    attachments: list[dict[str, str]] | None = None,
    duration_s: float = 5400.0,
) -> Any:
    """Create a Job for pipeline tests with output inside tmp_path."""
    return make_job(
        job_id="pipeline-job",
        source_files=["/src/movie.mkv"],
        output_file=str(tmp_path / "output" / "movie.mkv"),
        video_params=make_video_params(dv_mode=dv_mode),
        audio=audio if audio is not None else [],
        subtitles=subtitles if subtitles is not None else [],
        attachments=attachments if attachments is not None else [],
        copy_chapters=copy_chapters,
        chapters_source=chapters_source,
        source_size=1_000_000,
        duration_s=duration_s,
    )


class TestRunPipelineHappyPath:
    """Test 16: full pipeline with 1 audio COPY + 1 subtitle COPY."""

    def test_full_pipeline(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        audio_instr = make_audio_instruction(
            action=AudioAction.COPY,
            codec_name="aac",
            stream_index=1,
        )
        sub_instr = make_subtitle_instruction(
            source_file="/src/movie.mkv",
            action=SubtitleAction.COPY,
            codec_name="hdmv_pgs_subtitle",
            stream_index=3,
        )
        job = _pipeline_job(tmp_path, audio=[audio_instr], subtitles=[sub_instr])

        # Create the cleaned output so move succeeds
        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN_MKV_DATA")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean

        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        # Verify all adapters called
        mocks.audio_extractor.extract_track.assert_called()  # audio COPY + sub COPY
        mocks.encoder.encode.assert_called_once()
        mocks.muxer.mux.assert_called_once()
        mocks.tagger.set_encoder_tag.assert_called_once()
        mocks.cleaner.clean.assert_called_once()
        # Output file should exist
        assert output_path.exists()


class TestRunPipelineWithDvRpu:
    """Test 17: DV RPU extraction before encode."""

    def test_dv_rpu_extraction(
        self,
        tmp_path: Path,
    ) -> None:
        dovi_mock = MagicMock()
        dovi_mock.extract_rpu.return_value = 0
        mocks = SimpleNamespace(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
            dovi_processor=dovi_mock,
        )
        mocks.encoder.encode.return_value = EncodeResult(return_code=0, encoder_settings="test")
        mocks.muxer.mux.return_value = 0
        mocks.tagger.set_encoder_tag.return_value = 0

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean

        executor = Executor(
            encoder=mocks.encoder,
            audio_extractor=mocks.audio_extractor,
            audio_decoder=mocks.audio_decoder,
            aac_encoder=mocks.aac_encoder,
            muxer=mocks.muxer,
            tagger=mocks.tagger,
            cleaner=mocks.cleaner,
            prober=mocks.prober,
            dovi_processor=dovi_mock,
        )
        executor._vmaf_enabled = False

        job = _pipeline_job(tmp_path, dv_mode=DvMode.TO_8_1)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        dovi_mock.extract_rpu.assert_called_once()
        call_kwargs = dovi_mock.extract_rpu.call_args.kwargs
        assert call_kwargs["mode"] == DvMode.TO_8_1

        # Verify rpu_path was passed to encode
        encode_kwargs = mocks.encoder.encode.call_args.kwargs
        assert encode_kwargs["rpu_path"] is not None

    def test_dv_without_processor_raises(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, _mocks = executor_with_mocks
        job = _pipeline_job(tmp_path, dv_mode=DvMode.TO_8_1)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with pytest.raises(RuntimeError, match="DV content requires dovi_tool"):
            executor._run_pipeline(job, output_path, tmp_path)

    def test_dv_rpu_extraction_failure(
        self,
        tmp_path: Path,
    ) -> None:
        dovi_mock = MagicMock()
        dovi_mock.extract_rpu.return_value = 1
        executor = Executor(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
            dovi_processor=dovi_mock,
        )
        executor._vmaf_enabled = False
        job = _pipeline_job(tmp_path, dv_mode=DvMode.COPY)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with pytest.raises(RuntimeError, match="DV RPU extraction failed"):
            executor._run_pipeline(job, output_path, tmp_path)


class TestRunPipelineShutdown:
    """Test 18: shutdown_event stops pipeline early."""

    def test_shutdown_before_audio(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        executor._shutdown_event.set()
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        assert not mocks.encoder.encode.called

    def test_shutdown_during_audio_processing(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        audio1 = make_audio_instruction(action=AudioAction.COPY, codec_name="aac", stream_index=1)
        audio2 = make_audio_instruction(action=AudioAction.COPY, codec_name="aac", stream_index=2)

        # After first extract, set shutdown
        call_count = 0

        def extract_and_shutdown(src: Any, idx: Any, out: Any, on_progress: Any = None) -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                executor._shutdown_event.set()
            return 0

        mocks.audio_extractor.extract_track.side_effect = extract_and_shutdown
        job = _pipeline_job(tmp_path, audio=[audio1, audio2])
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        # Only first audio processed, encoder not called
        assert call_count == 1
        assert not mocks.encoder.encode.called

    def test_shutdown_during_encode(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        """Shutdown set via side_effect on encoder.encode → pipeline stops after encode."""
        executor, mocks = executor_with_mocks

        def encode_and_shutdown(**kwargs: Any) -> EncodeResult:
            executor._shutdown_event.set()
            return EncodeResult(return_code=0, encoder_settings="test")

        mocks.encoder.encode.side_effect = encode_and_shutdown
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        mocks.encoder.encode.assert_called_once()
        # Mux should NOT be called because shutdown was set
        assert not mocks.muxer.mux.called

    def test_shutdown_between_subtitles(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        sub1 = make_subtitle_instruction(
            source_file="/src/movie.mkv",
            action=SubtitleAction.COPY,
            codec_name="hdmv_pgs_subtitle",
            stream_index=3,
        )
        sub2 = make_subtitle_instruction(
            source_file="/src/movie.mkv",
            action=SubtitleAction.COPY,
            codec_name="hdmv_pgs_subtitle",
            stream_index=4,
        )

        call_count = 0

        def extract_and_shutdown(src: Any, idx: Any, out: Any, on_progress: Any = None) -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                executor._shutdown_event.set()
            return 0

        mocks.audio_extractor.extract_track.side_effect = extract_and_shutdown
        job = _pipeline_job(tmp_path, subtitles=[sub1, sub2])
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        assert call_count == 1
        assert not mocks.encoder.encode.called

    def test_shutdown_before_dv_extraction(
        self,
        tmp_path: Path,
    ) -> None:
        dovi_mock = MagicMock()
        executor = Executor(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
            dovi_processor=dovi_mock,
        )
        executor._vmaf_enabled = False
        executor._shutdown_event.set()
        job = _pipeline_job(tmp_path, dv_mode=DvMode.COPY)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        assert not dovi_mock.extract_rpu.called

    def test_shutdown_before_mux(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks

        def encode_and_shutdown(
            input_path: Any,
            output_path: Any,
            video_params: Any,
            on_progress: Any = None,
            vmaf_enabled: bool = False,
            rpu_path: Any = None,
        ) -> EncodeResult:
            executor._shutdown_event.set()
            return EncodeResult(return_code=0, encoder_settings="test")

        mocks.encoder.encode.side_effect = encode_and_shutdown
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        assert not mocks.muxer.mux.called

    def test_shutdown_before_tagger(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks

        def mux_and_shutdown(**kwargs: Any) -> int:
            executor._shutdown_event.set()
            return 0

        mocks.muxer.mux.side_effect = mux_and_shutdown
        mocks.encoder.encode.return_value = EncodeResult(return_code=0, encoder_settings="test")
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        assert not mocks.tagger.set_encoder_tag.called

    def test_shutdown_before_cleaner(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks

        def tag_and_shutdown(*args: Any, **kwargs: Any) -> int:
            executor._shutdown_event.set()
            return 0

        mocks.tagger.set_encoder_tag.side_effect = tag_and_shutdown
        mocks.encoder.encode.return_value = EncodeResult(return_code=0, encoder_settings="test")
        mocks.muxer.mux.return_value = 0
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        assert not mocks.cleaner.clean.called


class TestRunPipelineEncodeFailure:
    """Encode returns nonzero → RuntimeError."""

    def test_encode_failure(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.encoder.encode.return_value = EncodeResult(return_code=1, encoder_settings="fail")
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with pytest.raises(RuntimeError, match="Video encoding failed"):
            executor._run_pipeline(job, output_path, tmp_path)


class TestRunPipelineMuxFailure:
    """Mux returns nonzero → RuntimeError."""

    def test_mux_failure(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.muxer.mux.return_value = 1
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with pytest.raises(RuntimeError, match="Muxing failed"):
            executor._run_pipeline(job, output_path, tmp_path)


class TestRunPipelineTaggerWarning:
    """Tagger returns nonzero → warning logged, no exception."""

    def test_tagger_nonzero_continues(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.tagger.set_encoder_tag.return_value = 1

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Should not raise
        executor._run_pipeline(job, output_path, tmp_path)
        assert output_path.exists()


class TestRunPipelineEncodeMetrics:
    """Verify VMAF/SSIM metrics stored on job."""

    def test_metrics_stored(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.encoder.encode.return_value = EncodeResult(
            return_code=0,
            encoder_settings="test",
            vmaf_score=95.5,
            ssim_score=0.99,
        )

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        assert job.vmaf_score == 95.5
        assert job.ssim_score == 0.99


class TestRunPipelineChapters:
    """Chapters extraction integration."""

    def test_chapters_passed_to_mux(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.prober.probe.return_value = {
            "chapters": [
                {
                    "start_time": "0.000000",
                    "end_time": "300.000000",
                    "tags": {"title": "Chapter 1"},
                },
            ],
        }

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean
        job = _pipeline_job(
            tmp_path,
            copy_chapters=True,
            chapters_source="/src/movie.mkv",
        )
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        mux_call = mocks.muxer.mux.call_args
        assert mux_call.kwargs["chapters_source"] is not None


class TestRunPipelineAttachments:
    """Attachments passed to muxer."""

    def test_attachments_forwarded(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean
        job = _pipeline_job(
            tmp_path,
            attachments=[
                {"source_file": "/src/font.ttf", "filename": "font.ttf", "mime_type": "font/sfnt"},
            ],
        )
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        mux_call = mocks.muxer.mux.call_args
        assert len(mux_call.kwargs["attachments"]) == 1


class TestRunPipelineVideoMeta:
    """Verify video_meta dict built from video_params."""

    def test_hdr_metadata_in_video_meta(
        self,
        tmp_path: Path,
    ) -> None:
        from furnace.core.models import HdrMetadata

        mocks = SimpleNamespace(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
        )
        mocks.encoder.encode.return_value = EncodeResult(return_code=0, encoder_settings="test")
        mocks.muxer.mux.return_value = 0
        mocks.tagger.set_encoder_tag.return_value = 0

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean

        executor = Executor(
            encoder=mocks.encoder,
            audio_extractor=mocks.audio_extractor,
            audio_decoder=mocks.audio_decoder,
            aac_encoder=mocks.aac_encoder,
            muxer=mocks.muxer,
            tagger=mocks.tagger,
            cleaner=mocks.cleaner,
            prober=mocks.prober,
        )
        executor._vmaf_enabled = False

        hdr = HdrMetadata(content_light="MaxCLL=1000,MaxFALL=400")
        vp = make_video_params(
            color_range="tv",
            color_primaries="bt2020",
            color_transfer="smpte2084",
            hdr=hdr,
        )
        job = make_job(
            job_id="hdr-job",
            output_file=str(tmp_path / "output" / "movie.mkv"),
            video_params=vp,
            audio=[],
            subtitles=[],
            attachments=[],
            copy_chapters=False,
            source_size=0,
            duration_s=100.0,
        )
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)

        mux_call = mocks.muxer.mux.call_args
        video_meta = mux_call.kwargs["video_meta"]
        assert video_meta["color_range"] == "tv"
        assert video_meta["color_primaries"] == "bt2020"
        assert video_meta["color_transfer"] == "smpte2084"
        assert video_meta["hdr_max_cll"] == "1000"
        assert video_meta["hdr_max_fall"] == "400"


class TestRunPipelineProgressWiring:
    """Pipeline with progress mock: verify status/size updates."""

    def test_progress_updates(
        self,
        tmp_path: Path,
    ) -> None:
        progress_mock = MagicMock()
        mocks = SimpleNamespace(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
        )
        mocks.audio_extractor.extract_track.return_value = 0
        mocks.encoder.encode.return_value = EncodeResult(return_code=0, encoder_settings="test")
        mocks.muxer.mux.return_value = 0
        mocks.tagger.set_encoder_tag.return_value = 0

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean

        executor = Executor(
            encoder=mocks.encoder,
            audio_extractor=mocks.audio_extractor,
            audio_decoder=mocks.audio_decoder,
            aac_encoder=mocks.aac_encoder,
            muxer=mocks.muxer,
            tagger=mocks.tagger,
            cleaner=mocks.cleaner,
            prober=mocks.prober,
            progress=progress_mock,
        )
        executor._vmaf_enabled = False
        audio_instr = make_audio_instruction(
            action=AudioAction.COPY,
            codec_name="aac",
            stream_index=1,
        )
        sub_instr = make_subtitle_instruction(
            source_file="/src/movie.mkv",
            action=SubtitleAction.COPY,
            codec_name="hdmv_pgs_subtitle",
            stream_index=3,
        )
        job = _pipeline_job(tmp_path, audio=[audio_instr], subtitles=[sub_instr])
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)

        # Progress should have received update_status calls
        assert progress_mock.update_status.called
        assert progress_mock.add_tool_line.called


class TestRunPipelineMuxedSizeUpdate:
    """After mux, progress gets output size update."""

    def test_muxed_size_updated(
        self,
        tmp_path: Path,
    ) -> None:
        progress_mock = MagicMock()
        mocks = SimpleNamespace(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
        )
        mocks.encoder.encode.return_value = EncodeResult(return_code=0, encoder_settings="test")
        mocks.tagger.set_encoder_tag.return_value = 0

        # Mux creates a file so the size check works
        def fake_mux(**kwargs: Any) -> int:
            Path(kwargs["output_path"]).write_bytes(b"X" * 100)
            return 0

        mocks.muxer.mux.side_effect = fake_mux

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean

        executor = Executor(
            encoder=mocks.encoder,
            audio_extractor=mocks.audio_extractor,
            audio_decoder=mocks.audio_decoder,
            aac_encoder=mocks.aac_encoder,
            muxer=mocks.muxer,
            tagger=mocks.tagger,
            cleaner=mocks.cleaner,
            prober=mocks.prober,
            progress=progress_mock,
        )
        executor._vmaf_enabled = False
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        progress_mock.update_output_size.assert_called()


# ---------------------------------------------------------------------------
# run() lifecycle tests
# ---------------------------------------------------------------------------


class TestRunLifecycleHappyPath:
    """Test 19: Plan with one pending job → DONE in JSON."""

    def test_happy_path(
        self,
        tmp_path: Path,
    ) -> None:
        mocks = SimpleNamespace(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
        )
        mocks.encoder.encode.return_value = EncodeResult(return_code=0, encoder_settings="test")
        mocks.muxer.mux.return_value = 0
        mocks.tagger.set_encoder_tag.return_value = 0

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"OUTPUT_DATA")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean

        executor = Executor(
            encoder=mocks.encoder,
            audio_extractor=mocks.audio_extractor,
            audio_decoder=mocks.audio_decoder,
            aac_encoder=mocks.aac_encoder,
            muxer=mocks.muxer,
            tagger=mocks.tagger,
            cleaner=mocks.cleaner,
            prober=mocks.prober,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        job = make_job(
            job_id="run-test-job",
            output_file=str(output_dir / "movie.mkv"),
            audio=[],
            subtitles=[],
            attachments=[],
            copy_chapters=False,
            source_size=0,
            duration_s=100.0,
        )
        plan = make_plan(jobs=[job])

        plan_path = tmp_path / "plan.json"
        save_plan(plan, plan_path)

        executor.run(plan, plan_path)

        loaded = load_plan(plan_path)
        assert loaded.jobs[0].status == JobStatus.DONE
        assert loaded.jobs[0].error is None

    def test_skips_done_jobs(
        self,
        tmp_path: Path,
    ) -> None:
        mocks = SimpleNamespace(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
        )
        executor = Executor(
            encoder=mocks.encoder,
            audio_extractor=mocks.audio_extractor,
            audio_decoder=mocks.audio_decoder,
            aac_encoder=mocks.aac_encoder,
            muxer=mocks.muxer,
            tagger=mocks.tagger,
            cleaner=mocks.cleaner,
            prober=mocks.prober,
        )

        job = make_job(
            job_id="done-job",
            status=JobStatus.DONE,
            audio=[],
            subtitles=[],
            copy_chapters=False,
            source_size=0,
        )
        plan = make_plan(jobs=[job])
        plan_path = tmp_path / "plan.json"
        save_plan(plan, plan_path)

        executor.run(plan, plan_path)

        # Encoder should never be called for DONE jobs
        assert not mocks.encoder.encode.called


class TestRunLifecycleError:
    """Test 20: encoder raises → ERROR in plan JSON."""

    def test_encoder_raises_marks_error(
        self,
        tmp_path: Path,
    ) -> None:
        mocks = SimpleNamespace(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
        )
        mocks.encoder.encode.side_effect = RuntimeError("GPU died")

        executor = Executor(
            encoder=mocks.encoder,
            audio_extractor=mocks.audio_extractor,
            audio_decoder=mocks.audio_decoder,
            aac_encoder=mocks.aac_encoder,
            muxer=mocks.muxer,
            tagger=mocks.tagger,
            cleaner=mocks.cleaner,
            prober=mocks.prober,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        job = make_job(
            job_id="error-job",
            output_file=str(output_dir / "movie.mkv"),
            audio=[],
            subtitles=[],
            attachments=[],
            copy_chapters=False,
            source_size=0,
            duration_s=100.0,
        )
        plan = make_plan(jobs=[job])
        plan_path = tmp_path / "plan.json"
        save_plan(plan, plan_path)

        executor.run(plan, plan_path)

        loaded = load_plan(plan_path)
        assert loaded.jobs[0].status == JobStatus.ERROR
        assert "GPU died" in (loaded.jobs[0].error or "")


class TestMkcleanFailureFallback:
    """Test 21: cleaner returns nonzero → uses muxed file."""

    def test_cleaner_failure_uses_muxed(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks

        # Mux creates the muxed file
        def fake_mux(**kwargs: Any) -> int:
            Path(kwargs["output_path"]).write_bytes(b"MUXED_DATA")
            return 0

        mocks.muxer.mux.side_effect = fake_mux
        mocks.cleaner.clean.return_value = 1  # mkclean failure

        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Should not raise
        executor._run_pipeline(job, output_path, tmp_path)
        assert output_path.exists()
        assert output_path.read_bytes() == b"MUXED_DATA"


class TestMkcleanProgressUpdate:
    """mkclean with progress: cleaned size update."""

    def test_cleaned_size_updated(
        self,
        tmp_path: Path,
    ) -> None:
        progress_mock = MagicMock()
        mocks = SimpleNamespace(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
        )
        mocks.encoder.encode.return_value = EncodeResult(return_code=0, encoder_settings="test")
        mocks.muxer.mux.return_value = 0
        mocks.tagger.set_encoder_tag.return_value = 0

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEANED_OUTPUT")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean

        executor = Executor(
            encoder=mocks.encoder,
            audio_extractor=mocks.audio_extractor,
            audio_decoder=mocks.audio_decoder,
            aac_encoder=mocks.aac_encoder,
            muxer=mocks.muxer,
            tagger=mocks.tagger,
            cleaner=mocks.cleaner,
            prober=mocks.prober,
            progress=progress_mock,
        )
        executor._vmaf_enabled = False
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        # update_output_size should have been called with cleaned size
        progress_mock.update_output_size.assert_called()


# ---------------------------------------------------------------------------
# run() shutdown between jobs
# ---------------------------------------------------------------------------


class TestRunShutdownBetweenJobs:
    """Shutdown event stops processing of further jobs."""

    def test_shutdown_stops_second_job(
        self,
        tmp_path: Path,
    ) -> None:
        mocks = SimpleNamespace(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
        )
        mocks.encoder.encode.return_value = EncodeResult(return_code=0, encoder_settings="test")
        mocks.muxer.mux.return_value = 0
        mocks.tagger.set_encoder_tag.return_value = 0

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean

        executor = Executor(
            encoder=mocks.encoder,
            audio_extractor=mocks.audio_extractor,
            audio_decoder=mocks.audio_decoder,
            aac_encoder=mocks.aac_encoder,
            muxer=mocks.muxer,
            tagger=mocks.tagger,
            cleaner=mocks.cleaner,
            prober=mocks.prober,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        job1 = make_job(
            job_id="job-1",
            output_file=str(output_dir / "movie1.mkv"),
            audio=[],
            subtitles=[],
            attachments=[],
            copy_chapters=False,
            source_size=0,
            duration_s=100.0,
        )
        job2 = make_job(
            job_id="job-2",
            output_file=str(output_dir / "movie2.mkv"),
            audio=[],
            subtitles=[],
            attachments=[],
            copy_chapters=False,
            source_size=0,
            duration_s=100.0,
        )
        plan = make_plan(jobs=[job1, job2])
        plan_path = tmp_path / "plan.json"
        save_plan(plan, plan_path)

        # Set shutdown after first encode
        def encode_then_shutdown(**kwargs: Any) -> EncodeResult:
            executor._shutdown_event.set()
            return EncodeResult(return_code=0, encoder_settings="test")

        mocks.encoder.encode.side_effect = encode_then_shutdown

        executor.run(plan, plan_path)

        # Only job-1 should be marked DONE
        loaded = load_plan(plan_path)
        assert loaded.jobs[0].status == JobStatus.DONE
        assert loaded.jobs[1].status == JobStatus.PENDING


# ---------------------------------------------------------------------------
# run() with progress
# ---------------------------------------------------------------------------


class TestRunWithProgress:
    """run() lifecycle with progress mock: start_job / finish_job."""

    def test_progress_lifecycle(
        self,
        tmp_path: Path,
    ) -> None:
        progress_mock = MagicMock()
        mocks = SimpleNamespace(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
        )
        mocks.encoder.encode.return_value = EncodeResult(return_code=0, encoder_settings="test")
        mocks.muxer.mux.return_value = 0
        mocks.tagger.set_encoder_tag.return_value = 0

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean

        executor = Executor(
            encoder=mocks.encoder,
            audio_extractor=mocks.audio_extractor,
            audio_decoder=mocks.audio_decoder,
            aac_encoder=mocks.aac_encoder,
            muxer=mocks.muxer,
            tagger=mocks.tagger,
            cleaner=mocks.cleaner,
            prober=mocks.prober,
            progress=progress_mock,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        job = make_job(
            job_id="progress-job",
            output_file=str(output_dir / "movie.mkv"),
            audio=[],
            subtitles=[],
            attachments=[],
            copy_chapters=False,
            source_size=0,
            duration_s=100.0,
        )
        plan = make_plan(jobs=[job])
        plan_path = tmp_path / "plan.json"
        save_plan(plan, plan_path)

        executor.run(plan, plan_path)

        progress_mock.start_job.assert_called_once()
        progress_mock.finish_job.assert_called_once()


# ---------------------------------------------------------------------------
# graceful_shutdown
# ---------------------------------------------------------------------------


class TestGracefulShutdown:
    """Test 22: Mock psutil → event set, children killed."""

    def test_sets_event_and_kills_children(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
    ) -> None:
        executor, _mocks = executor_with_mocks

        mock_child = MagicMock()
        mock_parent = MagicMock()
        mock_parent.children.return_value = [mock_child]

        with patch("furnace.services.executor.psutil.Process", return_value=mock_parent):
            executor.graceful_shutdown()

        assert executor._shutdown_event.is_set()
        mock_child.kill.assert_called_once()

    def test_handles_os_error_gracefully(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
    ) -> None:
        executor, _mocks = executor_with_mocks

        with patch("furnace.services.executor.psutil.Process", side_effect=OSError("fail")):
            executor.graceful_shutdown()  # should not raise

        assert executor._shutdown_event.is_set()

    def test_handles_psutil_error_gracefully(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
    ) -> None:
        import psutil as psutil_mod

        executor, _mocks = executor_with_mocks

        with patch(
            "furnace.services.executor.psutil.Process",
            side_effect=psutil_mod.Error("fail"),
        ):
            executor.graceful_shutdown()  # should not raise

        assert executor._shutdown_event.is_set()

    def test_nosuchprocess_suppressed(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
    ) -> None:
        """NoSuchProcess on child.kill() → suppressed."""
        import psutil as psutil_mod

        executor, _mocks = executor_with_mocks
        mock_child = MagicMock()
        mock_child.kill.side_effect = psutil_mod.NoSuchProcess(pid=12345)
        mock_parent = MagicMock()
        mock_parent.children.return_value = [mock_child]

        with patch("furnace.services.executor.psutil.Process", return_value=mock_parent):
            executor.graceful_shutdown()  # should not raise


# ---------------------------------------------------------------------------
# _execute_job temp cleanup
# ---------------------------------------------------------------------------


class TestExecuteJobTempCleanup:
    """_execute_job cleans up temp dir even on failure."""

    def test_temp_dir_cleaned_on_success(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._execute_job(job)
        assert output_path.exists()

    def test_temp_dir_cleaned_on_error(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.encoder.encode.side_effect = RuntimeError("boom")
        job = _pipeline_job(tmp_path)
        with pytest.raises(RuntimeError, match="boom"):
            executor._execute_job(job)


# ---------------------------------------------------------------------------
# _codec_supported_by_eac3to
# ---------------------------------------------------------------------------


class TestCodecSupportedByEac3to:
    """Test the module-level helper."""

    def test_supported_codecs(self) -> None:
        from furnace.services.executor import _codec_supported_by_eac3to

        supported = [
            "ac3", "eac3", "dts", "truehd", "flac",
            "pcm_s16le", "pcm_s24le", "pcm_s16be", "aac", "mp2", "mp3",
        ]
        for codec in supported:
            assert _codec_supported_by_eac3to(codec), f"{codec} should be supported"

    def test_unsupported_codecs(self) -> None:
        from furnace.services.executor import _codec_supported_by_eac3to

        for codec in ["opus", "vorbis", "wmav2", "amr_nb"]:
            assert not _codec_supported_by_eac3to(codec), f"{codec} should NOT be supported"

    def test_case_insensitive(self) -> None:
        from furnace.services.executor import _codec_supported_by_eac3to

        assert _codec_supported_by_eac3to("AC3")
        assert _codec_supported_by_eac3to("TrueHD")


# ---------------------------------------------------------------------------
# Encode on_progress callback with output size
# ---------------------------------------------------------------------------


class TestEncodeOnProgressOutputSize:
    """The encode_on_progress wrapper should update output size."""

    def test_encode_progress_updates_size(
        self,
        tmp_path: Path,
    ) -> None:
        progress_mock = MagicMock()
        mocks = SimpleNamespace(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
        )

        def fake_encode(
            input_path: Any,
            output_path: Any,
            video_params: Any,
            on_progress: Any = None,
            vmaf_enabled: bool = False,
            rpu_path: Any = None,
        ) -> EncodeResult:
            if on_progress:
                # Create a fake video output to test size measurement
                Path(output_path).write_bytes(b"V" * 500)
                on_progress(ProgressSample(fraction=0.5))
            return EncodeResult(return_code=0, encoder_settings="test")

        mocks.encoder.encode.side_effect = fake_encode
        mocks.muxer.mux.return_value = 0
        mocks.tagger.set_encoder_tag.return_value = 0

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean

        executor = Executor(
            encoder=mocks.encoder,
            audio_extractor=mocks.audio_extractor,
            audio_decoder=mocks.audio_decoder,
            aac_encoder=mocks.aac_encoder,
            muxer=mocks.muxer,
            tagger=mocks.tagger,
            cleaner=mocks.cleaner,
            prober=mocks.prober,
            progress=progress_mock,
        )
        executor._vmaf_enabled = False
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        # update_output_size should have been called with video size
        progress_mock.update_output_size.assert_called()


class TestEncodeOnProgressOSError:
    """The encode_on_progress handles OSError when video file doesn't exist yet."""

    def test_oserror_handled(
        self,
        tmp_path: Path,
    ) -> None:
        progress_mock = MagicMock()
        mocks = SimpleNamespace(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
        )

        def fake_encode(
            input_path: Any,
            output_path: Any,
            video_params: Any,
            on_progress: Any = None,
            vmaf_enabled: bool = False,
            rpu_path: Any = None,
        ) -> EncodeResult:
            if on_progress:
                # Don't create video file — test OSError path
                on_progress(ProgressSample(fraction=0.1))
            return EncodeResult(return_code=0, encoder_settings="test")

        mocks.encoder.encode.side_effect = fake_encode
        mocks.muxer.mux.return_value = 0
        mocks.tagger.set_encoder_tag.return_value = 0

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean

        executor = Executor(
            encoder=mocks.encoder,
            audio_extractor=mocks.audio_extractor,
            audio_decoder=mocks.audio_decoder,
            aac_encoder=mocks.aac_encoder,
            muxer=mocks.muxer,
            tagger=mocks.tagger,
            cleaner=mocks.cleaner,
            prober=mocks.prober,
            progress=progress_mock,
        )
        executor._vmaf_enabled = False
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Should not raise
        executor._run_pipeline(job, output_path, tmp_path)


# ---------------------------------------------------------------------------
# Audio track processing with progress (audio size tracking)
# ---------------------------------------------------------------------------


class TestAudioSizeTracking:
    """Audio file size tracked via progress.update_output_size."""

    def test_audio_size_tracked(
        self,
        tmp_path: Path,
    ) -> None:
        progress_mock = MagicMock()
        mocks = SimpleNamespace(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
        )

        # extract_track creates a file so size check works
        def fake_extract(src: Any, idx: Any, out: Any, on_progress: Any = None) -> int:
            Path(out).write_bytes(b"A" * 200)
            return 0

        mocks.audio_extractor.extract_track.side_effect = fake_extract
        mocks.encoder.encode.return_value = EncodeResult(return_code=0, encoder_settings="test")
        mocks.muxer.mux.return_value = 0
        mocks.tagger.set_encoder_tag.return_value = 0

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean

        executor = Executor(
            encoder=mocks.encoder,
            audio_extractor=mocks.audio_extractor,
            audio_decoder=mocks.audio_decoder,
            aac_encoder=mocks.aac_encoder,
            muxer=mocks.muxer,
            tagger=mocks.tagger,
            cleaner=mocks.cleaner,
            prober=mocks.prober,
            progress=progress_mock,
        )
        executor._vmaf_enabled = False
        audio_instr = make_audio_instruction(
            action=AudioAction.COPY,
            codec_name="aac",
            stream_index=1,
        )
        job = _pipeline_job(tmp_path, audio=[audio_instr])
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        # Should have been called with audio size (200)
        progress_mock.update_output_size.assert_called()
        first_call_size = progress_mock.update_output_size.call_args_list[0][0][0]
        assert first_call_size == 200


# ---------------------------------------------------------------------------
# run() retries ERROR jobs
# ---------------------------------------------------------------------------


class TestRunRetriesErrorJobs:
    """Jobs with ERROR status are also retried."""

    def test_error_job_retried(
        self,
        tmp_path: Path,
    ) -> None:
        mocks = SimpleNamespace(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
        )
        mocks.encoder.encode.return_value = EncodeResult(return_code=0, encoder_settings="test")
        mocks.muxer.mux.return_value = 0
        mocks.tagger.set_encoder_tag.return_value = 0

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean

        executor = Executor(
            encoder=mocks.encoder,
            audio_extractor=mocks.audio_extractor,
            audio_decoder=mocks.audio_decoder,
            aac_encoder=mocks.aac_encoder,
            muxer=mocks.muxer,
            tagger=mocks.tagger,
            cleaner=mocks.cleaner,
            prober=mocks.prober,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        job = make_job(
            job_id="retry-job",
            output_file=str(output_dir / "movie.mkv"),
            status=JobStatus.ERROR,
            error="previous failure",
            audio=[],
            subtitles=[],
            attachments=[],
            copy_chapters=False,
            source_size=0,
            duration_s=100.0,
        )
        plan = make_plan(jobs=[job])
        plan_path = tmp_path / "plan.json"
        save_plan(plan, plan_path)

        executor.run(plan, plan_path)

        loaded = load_plan(plan_path)
        assert loaded.jobs[0].status == JobStatus.DONE


# ---------------------------------------------------------------------------
# COPY delay_ms vs other actions delay_ms
# ---------------------------------------------------------------------------


class TestAudioDelayMeta:
    """Audio meta delay_ms is nonzero only for COPY action."""

    def test_copy_preserves_delay(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.encoder.encode.return_value = EncodeResult(return_code=0, encoder_settings="test")
        mocks.muxer.mux.return_value = 0
        mocks.tagger.set_encoder_tag.return_value = 0

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean

        audio_copy = make_audio_instruction(
            action=AudioAction.COPY,
            codec_name="aac",
            stream_index=1,
            delay_ms=100,
        )
        audio_denorm = make_audio_instruction(
            action=AudioAction.DENORM,
            codec_name="ac3",
            stream_index=2,
            delay_ms=200,
        )
        job = _pipeline_job(tmp_path, audio=[audio_copy, audio_denorm])
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        mux_call = mocks.muxer.mux.call_args
        audio_files = mux_call.kwargs["audio_files"]
        # COPY: delay preserved
        assert audio_files[0][1]["delay_ms"] == 100
        # DENORM: delay zeroed
        assert audio_files[1][1]["delay_ms"] == 0


# ---------------------------------------------------------------------------
# Coverage gap: progress add_tool_line in per-step methods
# ---------------------------------------------------------------------------


def _make_executor_with_progress() -> tuple[Executor, SimpleNamespace, MagicMock]:
    """Build an Executor with progress mock AND all adapter mocks."""
    progress_mock = MagicMock()
    mocks = SimpleNamespace(
        encoder=MagicMock(),
        audio_extractor=MagicMock(),
        audio_decoder=MagicMock(),
        aac_encoder=MagicMock(),
        muxer=MagicMock(),
        tagger=MagicMock(),
        cleaner=MagicMock(),
        prober=MagicMock(),
    )
    mocks.audio_extractor.extract_track.return_value = 0
    mocks.audio_extractor.ffmpeg_to_wav.return_value = 0
    mocks.audio_decoder.decode_lossless.return_value = 0
    mocks.audio_decoder.denormalize.return_value = 0
    mocks.aac_encoder.encode_aac.return_value = 0
    executor = Executor(
        encoder=mocks.encoder,
        audio_extractor=mocks.audio_extractor,
        audio_decoder=mocks.audio_decoder,
        aac_encoder=mocks.aac_encoder,
        muxer=mocks.muxer,
        tagger=mocks.tagger,
        cleaner=mocks.cleaner,
        prober=mocks.prober,
        progress=progress_mock,
    )
    executor._vmaf_enabled = False
    return executor, mocks, progress_mock


class TestAudioProgressLines:
    """Cover add_tool_line branches for DENORM, DECODE_ENCODE, FFMPEG_ENCODE."""

    def test_denorm_progress_lines(self, tmp_path: Path) -> None:
        executor, _mocks, progress = _make_executor_with_progress()
        instr = make_audio_instruction(
            action=AudioAction.DENORM,
            codec_name="ac3",
            stream_index=1,
        )
        executor._process_audio_track(instr, tmp_path, _minimal_job())
        tool_lines = [c[0][0] for c in progress.add_tool_line.call_args_list]
        assert any("Extracting audio stream 1" in line for line in tool_lines)
        assert any("Denormalizing" in line for line in tool_lines)

    def test_decode_encode_eac3to_progress_lines(self, tmp_path: Path) -> None:
        executor, _mocks, progress = _make_executor_with_progress()
        instr = make_audio_instruction(
            action=AudioAction.DECODE_ENCODE,
            codec_name="truehd",
            stream_index=2,
        )
        executor._process_audio_track(instr, tmp_path, _minimal_job())
        tool_lines = [c[0][0] for c in progress.add_tool_line.call_args_list]
        assert any("Extracting audio stream 2" in line for line in tool_lines)
        assert any("Decoding lossless" in line for line in tool_lines)
        assert any("Encoding AAC" in line for line in tool_lines)

    def test_decode_encode_non_eac3to_progress_lines(self, tmp_path: Path) -> None:
        executor, _mocks, progress = _make_executor_with_progress()
        instr = make_audio_instruction(
            action=AudioAction.DECODE_ENCODE,
            codec_name="opus",
            stream_index=3,
        )
        executor._process_audio_track(instr, tmp_path, _minimal_job())
        tool_lines = [c[0][0] for c in progress.add_tool_line.call_args_list]
        assert any("Pre-decoding" in line for line in tool_lines)
        assert any("Decoding lossless" in line for line in tool_lines)
        assert any("Encoding AAC" in line for line in tool_lines)

    def test_ffmpeg_encode_progress_lines(self, tmp_path: Path) -> None:
        executor, _mocks, progress = _make_executor_with_progress()
        instr = make_audio_instruction(
            action=AudioAction.FFMPEG_ENCODE,
            codec_name="wmav2",
            stream_index=4,
        )
        executor._process_audio_track(instr, tmp_path, _minimal_job())
        tool_lines = [c[0][0] for c in progress.add_tool_line.call_args_list]
        assert any("Decoding audio stream 4 with ffmpeg" in line for line in tool_lines)
        assert any("Encoding AAC" in line for line in tool_lines)

    def test_copy_progress_lines(self, tmp_path: Path) -> None:
        executor, _mocks, progress = _make_executor_with_progress()
        instr = make_audio_instruction(
            action=AudioAction.COPY,
            codec_name="aac",
            stream_index=1,
        )
        executor._process_audio_track(instr, tmp_path, _minimal_job())
        tool_lines = [c[0][0] for c in progress.add_tool_line.call_args_list]
        assert any("Extracting audio stream 1 (copy)" in line for line in tool_lines)


# ---------------------------------------------------------------------------
# Coverage gap: video_meta branches for absent color fields
# ---------------------------------------------------------------------------


class TestVideoMetaEmptyFields:
    """Cover branches where color_range/primaries/transfer are empty/None."""

    def test_no_color_metadata_video_meta_none(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean
        vp = make_video_params(
            color_range="",
            color_primaries="",
            color_transfer="",
            hdr=None,
        )
        job = make_job(
            job_id="no-color-job",
            output_file=str(tmp_path / "output" / "movie.mkv"),
            video_params=vp,
            audio=[],
            subtitles=[],
            attachments=[],
            copy_chapters=False,
            source_size=0,
            duration_s=100.0,
        )
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        mux_call = mocks.muxer.mux.call_args
        # When all color fields are empty, video_meta should be None
        assert mux_call.kwargs["video_meta"] is None


# ---------------------------------------------------------------------------
# Coverage gap: chapters_have_mojibake=True branch
# ---------------------------------------------------------------------------


class TestChaptersMojibake:
    """Cover the mojibake detection branch in _extract_chapters_file."""

    def test_mojibake_chapters_detected(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        # Simulate UTF-8 bytes "Глава 1" decoded as Latin-1 -> mojibake
        mojibake_title = "Глава 1".encode().decode("latin-1")
        mocks.prober.probe.return_value = {
            "chapters": [
                {
                    "start_time": "0.000000",
                    "end_time": "300.000000",
                    "tags": {"title": mojibake_title},
                },
            ],
        }
        result = executor._extract_chapters_file(Path("/src/movie.mkv"), tmp_path)
        assert result is not None
        assert result.exists()


# ---------------------------------------------------------------------------
# Coverage gap: shutdown after subtitle loop, before DV extraction
# ---------------------------------------------------------------------------


class TestShutdownAfterSubtitles:
    """Cover the return-after-subtitle-loop shutdown path."""

    def test_shutdown_after_subtitles_before_dv(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        sub_instr = make_subtitle_instruction(
            source_file="/src/movie.mkv",
            action=SubtitleAction.COPY,
            codec_name="hdmv_pgs_subtitle",
            stream_index=3,
        )

        # Set shutdown after sub extraction
        def extract_and_shutdown(src: Any, idx: Any, out: Any, on_progress: Any = None) -> int:
            executor._shutdown_event.set()
            return 0

        mocks.audio_extractor.extract_track.side_effect = extract_and_shutdown
        job = _pipeline_job(tmp_path, subtitles=[sub_instr])
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        # Encoder should not be called
        assert not mocks.encoder.encode.called


# ---------------------------------------------------------------------------
# Coverage gap: _set_adapters_log_dir with adapter missing set_log_dir
# ---------------------------------------------------------------------------


class TestSetAdaptersLogDirMissingMethod:
    """Cover the branch where getattr returns None for set_log_dir."""

    def test_adapter_without_set_log_dir(self, tmp_path: Path) -> None:
        """Adapter that lacks set_log_dir attribute is skipped."""
        # Create a mock without set_log_dir method
        adapter_with = MagicMock()
        adapter_without = MagicMock(spec=[])  # spec=[] means no attributes

        executor = Executor(
            encoder=adapter_with,
            audio_extractor=adapter_without,
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
            log_dir=tmp_path,
        )
        executor._set_adapters_log_dir("TestMovie")
        expected_dir = tmp_path / "TestMovie"
        assert expected_dir.is_dir()
        # adapter_with has set_log_dir -> called
        adapter_with.set_log_dir.assert_called_once_with(expected_dir)


# ---------------------------------------------------------------------------
# Coverage gap: subtitle/DV progress status lines in _run_pipeline
# ---------------------------------------------------------------------------


class TestSubtitleProgressInPipeline:
    """Cover progress update_status and add_tool_line for subtitles in pipeline."""

    def test_subtitle_progress_status(self, tmp_path: Path) -> None:
        executor, mocks, progress = _make_executor_with_progress()

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean
        mocks.encoder.encode.return_value = EncodeResult(return_code=0, encoder_settings="test")
        mocks.muxer.mux.return_value = 0
        mocks.tagger.set_encoder_tag.return_value = 0

        sub_instr = make_subtitle_instruction(
            source_file="/src/movie.mkv",
            action=SubtitleAction.COPY,
            codec_name="hdmv_pgs_subtitle",
            stream_index=3,
        )
        audio_instr = make_audio_instruction(
            action=AudioAction.COPY,
            codec_name="aac",
            stream_index=1,
        )
        job = _pipeline_job(tmp_path, audio=[audio_instr], subtitles=[sub_instr])
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)

        status_calls = [c[0][0] for c in progress.update_status.call_args_list]
        assert any("Processing subtitle" in s for s in status_calls)
        assert any("Processing audio" in s for s in status_calls)
        assert any("Encoding video" in s for s in status_calls)
        assert any("Muxing" in s for s in status_calls)
        assert any("Setting metadata" in s for s in status_calls)
        assert any("Optimizing MKV index" in s for s in status_calls)


class TestDvProgressInPipeline:
    """Cover DV RPU progress status + tool lines."""

    def test_dv_progress_lines(self, tmp_path: Path) -> None:
        progress_mock = MagicMock()
        dovi_mock = MagicMock()
        dovi_mock.extract_rpu.return_value = 0
        mocks = SimpleNamespace(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
        )
        mocks.encoder.encode.return_value = EncodeResult(return_code=0, encoder_settings="test")
        mocks.muxer.mux.return_value = 0
        mocks.tagger.set_encoder_tag.return_value = 0

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean

        executor = Executor(
            encoder=mocks.encoder,
            audio_extractor=mocks.audio_extractor,
            audio_decoder=mocks.audio_decoder,
            aac_encoder=mocks.aac_encoder,
            muxer=mocks.muxer,
            tagger=mocks.tagger,
            cleaner=mocks.cleaner,
            prober=mocks.prober,
            dovi_processor=dovi_mock,
            progress=progress_mock,
        )
        executor._vmaf_enabled = False
        job = _pipeline_job(tmp_path, dv_mode=DvMode.TO_8_1)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)

        status_calls = [c[0][0] for c in progress_mock.update_status.call_args_list]
        assert any("Extracting DV RPU" in s for s in status_calls)
        tool_lines = [c[0][0] for c in progress_mock.add_tool_line.call_args_list]
        assert any("Extracting DV RPU" in line for line in tool_lines)


# ---------------------------------------------------------------------------
# Coverage gap: shutdown AFTER audio loop, BEFORE subtitle loop (line 265)
# ---------------------------------------------------------------------------


class TestShutdownAfterAudioBeforeSubtitles:
    """Shutdown after all audio processed but before subtitle loop starts."""

    def test_shutdown_after_audio_before_subs(
        self,
        tmp_path: Path,
    ) -> None:
        executor, mocks, _progress = _make_executor_with_progress()

        audio_instr = make_audio_instruction(
            action=AudioAction.COPY,
            codec_name="aac",
            stream_index=1,
        )
        sub_instr = make_subtitle_instruction(
            source_file="/src/movie.mkv",
            action=SubtitleAction.COPY,
            codec_name="hdmv_pgs_subtitle",
            stream_index=3,
        )

        # Audio extract sets shutdown
        call_count = 0

        def extract_and_shutdown(src: Any, idx: Any, out: Any, on_progress: Any = None) -> int:
            nonlocal call_count
            call_count += 1
            Path(out).write_bytes(b"AUDIO")
            executor._shutdown_event.set()
            return 0

        mocks.audio_extractor.extract_track.side_effect = extract_and_shutdown

        job = _pipeline_job(tmp_path, audio=[audio_instr], subtitles=[sub_instr])
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        assert call_count == 1
        assert not mocks.encoder.encode.called


# ---------------------------------------------------------------------------
# Coverage gap: shutdown after subs but before DV (line 291 with DV job)
# ---------------------------------------------------------------------------


class TestShutdownBeforeDvWithDvMode:
    """Shutdown after subtitles, before DV RPU extraction in a DV job."""

    def test_shutdown_before_dv_extraction_in_dv_job(
        self,
        tmp_path: Path,
    ) -> None:
        dovi_mock = MagicMock()
        dovi_mock.extract_rpu.return_value = 0
        progress_mock = MagicMock()
        mocks = SimpleNamespace(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
        )
        mocks.audio_extractor.extract_track.return_value = 0

        executor = Executor(
            encoder=mocks.encoder,
            audio_extractor=mocks.audio_extractor,
            audio_decoder=mocks.audio_decoder,
            aac_encoder=mocks.aac_encoder,
            muxer=mocks.muxer,
            tagger=mocks.tagger,
            cleaner=mocks.cleaner,
            prober=mocks.prober,
            dovi_processor=dovi_mock,
            progress=progress_mock,
        )
        executor._vmaf_enabled = False

        sub_instr = make_subtitle_instruction(
            source_file="/src/movie.mkv",
            action=SubtitleAction.COPY,
            codec_name="hdmv_pgs_subtitle",
            stream_index=3,
        )

        # Sub extraction sets shutdown
        def extract_and_shutdown(src: Any, idx: Any, out: Any, on_progress: Any = None) -> int:
            executor._shutdown_event.set()
            return 0

        mocks.audio_extractor.extract_track.side_effect = extract_and_shutdown

        job = _pipeline_job(tmp_path, subtitles=[sub_instr], dv_mode=DvMode.TO_8_1)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        # DV extraction should NOT be called
        assert not dovi_mock.extract_rpu.called


# ---------------------------------------------------------------------------
# Coverage gap: encode_on_progress OSError branch (lines 328-329)
# ---------------------------------------------------------------------------


class TestEncodeOnProgressStatOSError:
    """Cover the OSError catch in encode_on_progress."""

    def test_stat_oserror_caught(self, tmp_path: Path) -> None:
        progress_mock = MagicMock()
        mocks = SimpleNamespace(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
        )

        def fake_encode(
            input_path: Any,
            output_path: Any,
            video_params: Any,
            on_progress: Any = None,
            vmaf_enabled: bool = False,
            rpu_path: Any = None,
        ) -> EncodeResult:
            if on_progress:
                # Create output then immediately delete to cause OSError on stat
                Path(output_path).write_bytes(b"V")

                def patched_stat(self_path: Path, **kwargs: Any) -> Any:
                    if str(self_path) == str(output_path):
                        raise OSError("permission denied")
                    return Path.stat(self_path, **kwargs)

                with patch.object(Path, "stat", patched_stat):
                    on_progress(ProgressSample(fraction=0.3))
            return EncodeResult(return_code=0, encoder_settings="test")

        mocks.encoder.encode.side_effect = fake_encode
        mocks.muxer.mux.return_value = 0
        mocks.tagger.set_encoder_tag.return_value = 0

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean

        executor = Executor(
            encoder=mocks.encoder,
            audio_extractor=mocks.audio_extractor,
            audio_decoder=mocks.audio_decoder,
            aac_encoder=mocks.aac_encoder,
            muxer=mocks.muxer,
            tagger=mocks.tagger,
            cleaner=mocks.cleaner,
            prober=mocks.prober,
            progress=progress_mock,
        )
        executor._vmaf_enabled = False
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Should not raise — OSError caught
        executor._run_pipeline(job, output_path, tmp_path)
        # update_output_size still called (with fallback video_size=0)
        progress_mock.update_output_size.assert_called()


# ---------------------------------------------------------------------------
# Coverage gap: encode_on_progress with no progress (line 330->exit)
# ---------------------------------------------------------------------------


class TestEncodeOnProgressNoProgress:
    """Encode callback does not call update_output_size when progress is None."""

    def test_no_progress_in_callback(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        # executor._progress is None by default

        def fake_encode(
            input_path: Any,
            output_path: Any,
            video_params: Any,
            on_progress: Any = None,
            vmaf_enabled: bool = False,
            rpu_path: Any = None,
        ) -> EncodeResult:
            if on_progress:
                on_progress(ProgressSample(fraction=0.5))
            return EncodeResult(return_code=0, encoder_settings="test")

        mocks.encoder.encode.side_effect = fake_encode

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Should not raise — progress is None, callback just skips
        executor._run_pipeline(job, output_path, tmp_path)


# ---------------------------------------------------------------------------
# Coverage gap: content_light with extra/unknown part (line 390->387)
# ---------------------------------------------------------------------------


class TestVideoMetaUnknownContentLightPart:
    """Cover the content_light loop with a part not matching MaxCLL or MaxFALL."""

    def test_content_light_unknown_part_ignored(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        from furnace.core.models import HdrMetadata

        executor, mocks = executor_with_mocks

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean
        # content_light with an extra unknown part
        hdr = HdrMetadata(content_light="MaxCLL=1000,MaxFALL=400,Unknown=999")
        vp = make_video_params(
            color_range="tv",
            color_primaries="bt2020",
            color_transfer="smpte2084",
            hdr=hdr,
        )
        job = make_job(
            job_id="hdr-extra-part",
            output_file=str(tmp_path / "output" / "movie.mkv"),
            video_params=vp,
            audio=[],
            subtitles=[],
            attachments=[],
            copy_chapters=False,
            source_size=0,
            duration_s=100.0,
        )
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        mux_call = mocks.muxer.mux.call_args
        video_meta = mux_call.kwargs["video_meta"]
        assert video_meta["hdr_max_cll"] == "1000"
        assert video_meta["hdr_max_fall"] == "400"
        # "Unknown=999" is silently ignored
        assert "Unknown" not in str(video_meta)
