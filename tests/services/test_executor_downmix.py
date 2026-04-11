"""Tests for the DECODE_ENCODE branch with downmix in the Executor."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from furnace.core.models import (
    AudioAction,
    AudioInstruction,
    DownmixMode,
    Job,
    JobStatus,
    VideoParams,
)


def _instr(
    codec_name: str,
    downmix: DownmixMode | None = None,
    channels: int | None = 8,
    stream_index: int = 1,
) -> AudioInstruction:
    return AudioInstruction(
        source_file="/src/movie.mkv",
        stream_index=stream_index,
        language="eng",
        action=AudioAction.DECODE_ENCODE,
        delay_ms=0,
        is_default=True,
        codec_name=codec_name,
        channels=channels,
        bitrate=4_500_000,
        downmix=downmix,
    )


def _job(duration_s: float = 5400.0) -> Job:
    """Minimal Job instance sufficient for _process_audio_track."""
    return Job(
        id="test-job",
        source_files=["/src/movie.mkv"],
        output_file="/out/movie.mkv",
        video_params=VideoParams(
            cq=25,
            crop=None,
            deinterlace=False,
            color_matrix="bt709",
            color_range="tv",
            color_transfer="bt709",
            color_primaries="bt709",
            hdr=None,
            gop=120,
            fps_num=24,
            fps_den=1,
            source_width=1920,
            source_height=1080,
        ),
        audio=[],
        subtitles=[],
        attachments=[],
        copy_chapters=False,
        chapters_source=None,
        status=JobStatus.PENDING,
        source_size=0,
        duration_s=duration_s,
    )


@pytest.fixture
def executor_with_mocks():
    """Construct an Executor with all adapter ports mocked.
    Returns (executor, mocks) where mocks holds the adapter MagicMocks."""
    from furnace.services.executor import Executor

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
    def test_truehd_downmix_uses_extract_track(self, executor_with_mocks, tmp_path):
        """TrueHD is eac3to-supported -> extract_track, not ffmpeg_to_wav."""
        executor, mocks = executor_with_mocks
        instr = _instr("truehd", downmix=DownmixMode.STEREO)
        executor._process_audio_track(instr, tmp_path, _job())

        assert mocks.audio_extractor.extract_track.called
        assert not mocks.audio_extractor.ffmpeg_to_wav.called
        decode_call = mocks.audio_decoder.decode_lossless.call_args
        assert decode_call.kwargs.get("downmix") == DownmixMode.STEREO

    def test_opus_downmix_uses_ffmpeg_to_wav(self, executor_with_mocks, tmp_path):
        """Opus is NOT eac3to-supported -> ffmpeg_to_wav, then eac3to downmix."""
        executor, mocks = executor_with_mocks
        instr = _instr("opus", downmix=DownmixMode.STEREO)
        executor._process_audio_track(instr, tmp_path, _job())

        assert mocks.audio_extractor.ffmpeg_to_wav.called
        assert not mocks.audio_extractor.extract_track.called
        decode_call = mocks.audio_decoder.decode_lossless.call_args
        assert decode_call.kwargs.get("downmix") == DownmixMode.STEREO

    def test_vorbis_downmix_uses_ffmpeg_to_wav(self, executor_with_mocks, tmp_path):
        executor, mocks = executor_with_mocks
        instr = _instr("vorbis", downmix=DownmixMode.DOWN6)
        executor._process_audio_track(instr, tmp_path, _job())

        assert mocks.audio_extractor.ffmpeg_to_wav.called
        assert not mocks.audio_extractor.extract_track.called

    def test_no_downmix_on_truehd_passes_none(self, executor_with_mocks, tmp_path):
        """Regression guard: existing DECODE_ENCODE flow passes downmix=None."""
        executor, mocks = executor_with_mocks
        instr = _instr("truehd", downmix=None)
        executor._process_audio_track(instr, tmp_path, _job())

        decode_call = mocks.audio_decoder.decode_lossless.call_args
        assert decode_call.kwargs.get("downmix") is None

    def test_dts_downmix_uses_extract_track(self, executor_with_mocks, tmp_path):
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
        self, executor_with_mocks, tmp_path,
    ):
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
        self, executor_with_mocks, tmp_path,
    ):
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
