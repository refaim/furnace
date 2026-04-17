"""Tests for the DECODE_ENCODE branch with downmix in the Executor."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from furnace.core.models import (
    AudioAction,
    AudioInstruction,
    DownmixMode,
    Job,
)
from furnace.services.executor import Executor
from tests.conftest import make_audio_instruction, make_job


def _instr(
    codec_name: str,
    downmix: DownmixMode | None = None,
    channels: int | None = 8,
    stream_index: int = 1,
) -> AudioInstruction:
    return make_audio_instruction(
        stream_index=stream_index,
        action=AudioAction.DECODE_ENCODE,
        codec_name=codec_name,
        channels=channels,
        bitrate=4_500_000,
        downmix=downmix,
    )


def _job(duration_s: float = 5400.0) -> Job:
    """Minimal Job instance sufficient for _process_audio_track."""
    return make_job(
        job_id="test-job",
        audio=[],
        subtitles=[],
        copy_chapters=False,
        source_size=0,
        duration_s=duration_s,
    )


@pytest.fixture
def executor_with_mocks() -> tuple[Executor, SimpleNamespace]:
    """Construct an Executor with all adapter ports mocked.
    Returns (executor, mocks) where mocks holds the adapter MagicMocks."""
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
    )
    return executor, mocks


class TestDecodeEncodeDownmixRouting:
    def test_truehd_downmix_uses_extract_track(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        """TrueHD is eac3to-supported -> extract_track, not ffmpeg_to_wav."""
        executor, mocks = executor_with_mocks
        instr = _instr("truehd", downmix=DownmixMode.STEREO)
        executor._process_audio_track(instr, tmp_path, _job())

        assert mocks.audio_extractor.extract_track.called
        assert not mocks.audio_extractor.ffmpeg_to_wav.called
        decode_call = mocks.audio_decoder.decode_lossless.call_args
        assert decode_call.kwargs.get("downmix") == DownmixMode.STEREO

    def test_opus_downmix_uses_ffmpeg_to_wav(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        """Opus is NOT eac3to-supported -> ffmpeg_to_wav, then eac3to downmix."""
        executor, mocks = executor_with_mocks
        instr = _instr("opus", downmix=DownmixMode.STEREO)
        executor._process_audio_track(instr, tmp_path, _job())

        assert mocks.audio_extractor.ffmpeg_to_wav.called
        assert not mocks.audio_extractor.extract_track.called
        decode_call = mocks.audio_decoder.decode_lossless.call_args
        assert decode_call.kwargs.get("downmix") == DownmixMode.STEREO

    def test_vorbis_downmix_uses_ffmpeg_to_wav(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        instr = _instr("vorbis", downmix=DownmixMode.DOWN6)
        executor._process_audio_track(instr, tmp_path, _job())

        assert mocks.audio_extractor.ffmpeg_to_wav.called
        assert not mocks.audio_extractor.extract_track.called

    def test_no_downmix_on_truehd_passes_none(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        """Regression guard: existing DECODE_ENCODE flow passes downmix=None."""
        executor, mocks = executor_with_mocks
        instr = _instr("truehd", downmix=None)
        executor._process_audio_track(instr, tmp_path, _job())

        decode_call = mocks.audio_decoder.decode_lossless.call_args
        assert decode_call.kwargs.get("downmix") is None

    def test_dts_downmix_uses_extract_track(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        instr = _instr("dts", downmix=DownmixMode.STEREO)
        executor._process_audio_track(instr, tmp_path, _job())

        assert mocks.audio_extractor.extract_track.called
        assert not mocks.audio_extractor.ffmpeg_to_wav.called


class TestDecodeEncodeDownmixProgressWiring:
    """Each tool step in the DECODE_ENCODE branch must receive its own
    on_progress callback — this is the contract with the unified progress
    tracking refactor from commit 0d6e0c2."""

    def test_eac3to_supported_path_wires_three_progress_callbacks(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        """extract_track + decode_lossless + encode_aac each get a callback."""
        executor, mocks = executor_with_mocks
        instr = _instr("truehd", downmix=DownmixMode.STEREO)
        executor._process_audio_track(instr, tmp_path, _job())

        extract_call = mocks.audio_extractor.extract_track.call_args
        assert callable(extract_call.kwargs.get("on_progress"))

        decode_call = mocks.audio_decoder.decode_lossless.call_args
        assert callable(decode_call.kwargs.get("on_progress"))

        encode_call = mocks.aac_encoder.encode_aac.call_args
        assert callable(encode_call.kwargs.get("on_progress"))

    def test_non_eac3to_path_wires_three_progress_callbacks(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        """ffmpeg_to_wav + decode_lossless + encode_aac each get a callback."""
        executor, mocks = executor_with_mocks
        instr = _instr("opus", downmix=DownmixMode.STEREO)
        executor._process_audio_track(instr, tmp_path, _job())

        ffmpeg_call = mocks.audio_extractor.ffmpeg_to_wav.call_args
        assert callable(ffmpeg_call.kwargs.get("on_progress"))

        decode_call = mocks.audio_decoder.decode_lossless.call_args
        assert callable(decode_call.kwargs.get("on_progress"))

        encode_call = mocks.aac_encoder.encode_aac.call_args
        assert callable(encode_call.kwargs.get("on_progress"))
