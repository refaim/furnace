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


class TestDecodeEncodeMonoDownmix:
    """DECODE_ENCODE + downmix=MONO uses a three-path flow:

    - stereo source (channels == 2) -> stereo_to_mono_wav -> encode_aac
    - multichannel + eac3to-supported codec -> extract_track ->
      decode_lossless(downmix=STEREO) -> stereo_to_mono_wav -> encode_aac
    - multichannel + non-eac3to codec -> ffmpeg_to_wav ->
      decode_lossless(downmix=STEREO) -> stereo_to_mono_wav -> encode_aac

    Delay is applied at the eac3to step (multichannel) or at the
    stereo_to_mono_wav step (stereo direct) — never at both.
    """

    def test_5_1_dts_chains_extract_eac3to_stereo_mono_qaac(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        """5.1 DTS: extract_track -> decode_lossless(STEREO) -> stereo_to_mono_wav -> encode_aac."""
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0

        instr = _instr("dts", downmix=DownmixMode.MONO, channels=6, stream_index=1)
        executor._process_audio_track(instr, tmp_path, _job())

        mocks.audio_extractor.extract_track.assert_called_once()
        mocks.audio_extractor.ffmpeg_to_wav.assert_not_called()

        decode_call = mocks.audio_decoder.decode_lossless.call_args
        assert decode_call.kwargs.get("downmix") == DownmixMode.STEREO

        mocks.audio_extractor.stereo_to_mono_wav.assert_called_once()
        s2m_kwargs = mocks.audio_extractor.stereo_to_mono_wav.call_args.kwargs
        assert s2m_kwargs["stream_index"] == 0
        assert s2m_kwargs["delay_ms"] == 0
        assert s2m_kwargs["output_wav"].suffix == ".wav"

        mocks.aac_encoder.encode_aac.assert_called_once()

    def test_7_1_truehd_chains_extract_eac3to_stereo_mono_qaac(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        """7.1 TrueHD: extract_track -> decode_lossless(STEREO) -> stereo_to_mono_wav -> encode_aac."""
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0

        instr = _instr("truehd", downmix=DownmixMode.MONO, channels=8, stream_index=1)
        executor._process_audio_track(instr, tmp_path, _job())

        mocks.audio_extractor.extract_track.assert_called_once()
        mocks.audio_extractor.ffmpeg_to_wav.assert_not_called()
        decode_call = mocks.audio_decoder.decode_lossless.call_args
        assert decode_call.kwargs.get("downmix") == DownmixMode.STEREO
        mocks.audio_extractor.stereo_to_mono_wav.assert_called_once()
        mocks.aac_encoder.encode_aac.assert_called_once()

    def test_aac_5_1_chains_ffmpeg_to_wav_eac3to_stereo_mono_qaac(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        """AAC 5.1 (not eac3to-supported as a source here per spec):
        ffmpeg_to_wav -> decode_lossless(STEREO) -> stereo_to_mono_wav -> encode_aac."""
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0

        instr = _instr("opus", downmix=DownmixMode.MONO, channels=6, stream_index=1)
        executor._process_audio_track(instr, tmp_path, _job())

        mocks.audio_extractor.ffmpeg_to_wav.assert_called_once()
        mocks.audio_extractor.extract_track.assert_not_called()
        decode_call = mocks.audio_decoder.decode_lossless.call_args
        assert decode_call.kwargs.get("downmix") == DownmixMode.STEREO
        mocks.audio_extractor.stereo_to_mono_wav.assert_called_once()
        mocks.aac_encoder.encode_aac.assert_called_once()

    def test_multichannel_delay_goes_to_eac3to_not_to_mono_step(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        """For multichannel sources, the delay is applied at decode_lossless
        (positional index 2), and stereo_to_mono_wav receives delay_ms=0."""
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0

        instr = _instr("dts", downmix=DownmixMode.MONO, channels=6, delay_ms=125)
        executor._process_audio_track(instr, tmp_path, _job())

        decode_call = mocks.audio_decoder.decode_lossless.call_args
        assert decode_call.args[2] == 125

        s2m_kwargs = mocks.audio_extractor.stereo_to_mono_wav.call_args.kwargs
        assert s2m_kwargs["delay_ms"] == 0

    def test_stereo_source_skips_eac3to_calls_mono_directly(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        """channels == 2: stereo_to_mono_wav direct from source, delay applied here."""
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0

        instr = _instr(
            "ac3", downmix=DownmixMode.MONO, channels=2, stream_index=3, delay_ms=-30,
        )
        executor._process_audio_track(instr, tmp_path, _job())

        mocks.audio_extractor.extract_track.assert_not_called()
        mocks.audio_extractor.ffmpeg_to_wav.assert_not_called()
        mocks.audio_decoder.decode_lossless.assert_not_called()

        mocks.audio_extractor.stereo_to_mono_wav.assert_called_once()
        s2m_kwargs = mocks.audio_extractor.stereo_to_mono_wav.call_args.kwargs
        assert s2m_kwargs["stream_index"] == instr.stream_index
        assert s2m_kwargs["input_path"] == Path(instr.source_file)
        assert s2m_kwargs["delay_ms"] == -30

        mocks.aac_encoder.encode_aac.assert_called_once()

    def test_decode_encode_without_mono_does_not_call_stereo_to_mono(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        """Regression: DECODE_ENCODE without downmix=MONO never touches stereo_to_mono_wav."""
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0

        instr = _instr("truehd", downmix=None, channels=6)
        executor._process_audio_track(instr, tmp_path, _job())

        mocks.audio_extractor.stereo_to_mono_wav.assert_not_called()
        mocks.audio_decoder.decode_lossless.assert_called_once()

    def test_multichannel_raises_when_eac3to_fails(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        """rc != 0 from decode_lossless raises; downstream steps not called."""
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0
        mocks.audio_decoder.decode_lossless.return_value = 7

        instr = _instr("dts", downmix=DownmixMode.MONO, channels=6)
        with pytest.raises(RuntimeError, match=r"eac3to -downStereo failed.*rc=7"):
            executor._process_audio_track(instr, tmp_path, _job())

        mocks.audio_extractor.stereo_to_mono_wav.assert_not_called()
        mocks.aac_encoder.encode_aac.assert_not_called()

    def test_multichannel_raises_when_extract_fails(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        """rc != 0 from extract_track raises; decode and mono steps not called."""
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.extract_track.return_value = 9
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0

        instr = _instr("dts", downmix=DownmixMode.MONO, channels=6)
        with pytest.raises(RuntimeError, match=r"Audio extract.*MONO.*rc=9"):
            executor._process_audio_track(instr, tmp_path, _job())

        mocks.audio_decoder.decode_lossless.assert_not_called()
        mocks.audio_extractor.stereo_to_mono_wav.assert_not_called()

    def test_multichannel_raises_when_ffmpeg_pre_decode_fails(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        """rc != 0 from ffmpeg_to_wav raises; decode_lossless not called."""
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.ffmpeg_to_wav.return_value = 11
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0

        instr = _instr("opus", downmix=DownmixMode.MONO, channels=6)
        with pytest.raises(RuntimeError, match=r"ffmpeg pre-decode.*MONO.*rc=11"):
            executor._process_audio_track(instr, tmp_path, _job())

        mocks.audio_decoder.decode_lossless.assert_not_called()

    def test_multichannel_raises_when_stereo_to_mono_fails(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        """rc != 0 from stereo_to_mono_wav (multichannel path) raises; encode_aac not called."""
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 5

        instr = _instr("dts", downmix=DownmixMode.MONO, channels=6)
        with pytest.raises(RuntimeError, match=r"stereo_to_mono_wav failed.*rc=5"):
            executor._process_audio_track(instr, tmp_path, _job())

        mocks.aac_encoder.encode_aac.assert_not_called()

    def test_stereo_raises_when_stereo_to_mono_fails(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        """rc != 0 from stereo_to_mono_wav (stereo path) raises; encode_aac not called."""
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 5

        instr = _instr("ac3", downmix=DownmixMode.MONO, channels=2)
        with pytest.raises(RuntimeError, match=r"stereo_to_mono_wav failed.*rc=5"):
            executor._process_audio_track(instr, tmp_path, _job())

        mocks.aac_encoder.encode_aac.assert_not_called()

    def test_mono_raises_when_encode_aac_fails(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        """After a successful mono WAV, rc != 0 from encode_aac raises."""
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0
        mocks.aac_encoder.encode_aac.return_value = 3

        instr = _instr("ac3", downmix=DownmixMode.MONO, channels=2)
        with pytest.raises(RuntimeError, match=r"encode_aac failed.*rc=3"):
            executor._process_audio_track(instr, tmp_path, _job())

    def test_multichannel_raises_when_encode_aac_fails(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        """Multichannel path: rc != 0 from encode_aac raises after the eac3to+ffmpeg
        chain completed; the failure happens at the final encode step, not earlier.
        """
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0
        mocks.aac_encoder.encode_aac.return_value = 4

        instr = _instr("dts", downmix=DownmixMode.MONO, channels=6)
        with pytest.raises(RuntimeError, match=r"encode_aac failed.*rc=4"):
            executor._process_audio_track(instr, tmp_path, _job())

        mocks.audio_extractor.extract_track.assert_called_once()
        mocks.audio_decoder.decode_lossless.assert_called_once()
        mocks.audio_extractor.stereo_to_mono_wav.assert_called_once()
        mocks.aac_encoder.encode_aac.assert_called_once()

    def test_stereo_direct_passes_on_progress_to_stereo_to_mono(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        """Stereo source wires on_progress to stereo_to_mono_wav so the
        per-step bar advances during the ffmpeg pan step."""
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0

        instr = _instr("ac3", downmix=DownmixMode.MONO, channels=2)
        executor._process_audio_track(instr, tmp_path, _job())

        mono_call = mocks.audio_extractor.stereo_to_mono_wav.call_args
        assert mono_call.kwargs.get("on_progress") is not None
        assert callable(mono_call.kwargs["on_progress"])

    def test_multichannel_passes_on_progress_to_stereo_to_mono(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        """Multichannel post-eac3to step also wires on_progress."""
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0

        instr = _instr("dts", downmix=DownmixMode.MONO, channels=6)
        executor._process_audio_track(instr, tmp_path, _job())

        mono_call = mocks.audio_extractor.stereo_to_mono_wav.call_args
        assert mono_call.kwargs.get("on_progress") is not None
        assert callable(mono_call.kwargs["on_progress"])

    def test_mono_raises_when_channels_none(
        self, executor_with_mocks: tuple[Executor, SimpleNamespace], tmp_path: Path,
    ) -> None:
        """Defensive guard: MONO without channel count raises before any subprocess work."""
        executor, mocks = executor_with_mocks
        instr = _instr("dts", downmix=DownmixMode.MONO, channels=None)
        with pytest.raises(RuntimeError, match="MONO downmix without channel count"):
            executor._process_audio_track(instr, tmp_path, _job())
        mocks.audio_extractor.stereo_to_mono_wav.assert_not_called()
        mocks.audio_extractor.extract_track.assert_not_called()
        mocks.audio_extractor.ffmpeg_to_wav.assert_not_called()
        mocks.audio_decoder.decode_lossless.assert_not_called()
