from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from furnace.core.models import (
    AudioAction,
    DownmixMode,
    DvMode,
    EncodeResult,
    JobStatus,
    SubtitleAction,
)
from furnace.core.progress import ProgressSample
from furnace.core.target_quality import KnobSearchResult
from furnace.plan import load_plan, save_plan
from furnace.services.executor import Executor, _video_intermediate_name
from tests.conftest import (
    make_audio_instruction,
    make_job,
    make_plan,
    make_subtitle_instruction,
    make_video_params,
)


@pytest.fixture
def executor_with_mocks() -> tuple[Executor, SimpleNamespace]:
    mocks = SimpleNamespace(
        encoder=MagicMock(),
        audio_extractor=MagicMock(),
        audio_decoder=MagicMock(),
        aac_encoder=MagicMock(),
        muxer=MagicMock(),
        tagger=MagicMock(),
        cleaner=MagicMock(),
        prober=MagicMock(),
        video_copier=MagicMock(),
    )
    mocks.audio_extractor.extract_track.return_value = 0
    mocks.audio_extractor.ffmpeg_to_wav.return_value = 0
    mocks.audio_extractor.decode_full_wav.return_value = 0
    mocks.audio_extractor.stereo_to_mono_wav.return_value = 0
    mocks.audio_decoder.decode_lossless.return_value = 0
    mocks.audio_decoder.denormalize.return_value = 0
    mocks.aac_encoder.encode_aac.return_value = 0
    mocks.encoder.encode.return_value = EncodeResult(return_code=0, encoder_settings="test")
    mocks.muxer.mux.return_value = 0
    mocks.tagger.set_encoder_tag.return_value = 0
    mocks.cleaner.clean.return_value = 0
    mocks.prober.probe.return_value = {"chapters": []}
    mocks.video_copier.copy_video.return_value = 0

    executor = Executor(
        encoder=mocks.encoder,
        audio_extractor=mocks.audio_extractor,
        audio_decoder=mocks.audio_decoder,
        aac_encoder=mocks.aac_encoder,
        muxer=mocks.muxer,
        tagger=mocks.tagger,
        cleaner=mocks.cleaner,
        prober=mocks.prober,
        video_copier=mocks.video_copier,
    )
    return executor, mocks


def _minimal_job(**kwargs: Any) -> Any:
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


def _stream0_probe(seconds: float | None) -> dict[str, Any]:
    if seconds is None:
        return {"chapters": []}
    return {"streams": [{"index": 0, "duration": str(seconds)}]}


def _source_probe(seconds: float | None, stream_index: int = 1) -> dict[str, Any]:
    if seconds is None:
        return {"chapters": []}
    return {"streams": [{"index": stream_index, "duration": str(seconds)}]}


def _repair_probe_router(
    *,
    declared: float | None,
    produced: float | None,
    decoded: float | None = None,
    repaired: float | None = None,
    source_file: str = "/src/movie.mkv",
    stream_index: int = 1,
) -> Any:
    def probe(path: Any) -> dict[str, Any]:
        p = str(path)
        if p == source_file:
            return _source_probe(declared, stream_index)
        if "_healed" in p:
            return _stream0_probe(decoded)
        if "_repaired.m4a" in p:
            return _stream0_probe(repaired)
        return _stream0_probe(produced)

    return probe


class TestVerifyAndRepairAudio:
    def test_full_length_returns_produced_unchanged(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        produced = tmp_path / "audio_1.m4a"
        mocks.prober.probe.side_effect = _repair_probe_router(declared=7345.0, produced=7345.0)
        instr = make_audio_instruction(action=AudioAction.DECODE_ENCODE, stream_index=1, codec_name="ac3")
        result, repaired = executor._verify_and_repair_audio(instr, produced, tmp_path, _minimal_job())
        assert result == produced
        assert repaired is False
        mocks.audio_extractor.decode_full_wav.assert_not_called()

    def test_declared_unknown_skips_and_returns_produced(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        produced = tmp_path / "audio_1.m4a"
        mocks.prober.probe.side_effect = _repair_probe_router(declared=None, produced=1794.0)
        instr = make_audio_instruction(action=AudioAction.DECODE_ENCODE, stream_index=1, codec_name="ac3")
        result, repaired = executor._verify_and_repair_audio(instr, produced, tmp_path, _minimal_job())
        assert result == produced
        assert repaired is False
        mocks.audio_extractor.decode_full_wav.assert_not_called()

    def test_produced_unknown_skips_and_returns_produced(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        produced = tmp_path / "audio_1.m4a"
        mocks.prober.probe.side_effect = _repair_probe_router(declared=7345.0, produced=None)
        instr = make_audio_instruction(action=AudioAction.DECODE_ENCODE, stream_index=1, codec_name="ac3")
        result, repaired = executor._verify_and_repair_audio(instr, produced, tmp_path, _minimal_job())
        assert result == produced
        assert repaired is False

    def test_genuine_short_accepts_produced(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        produced = tmp_path / "audio_1.m4a"
        mocks.prober.probe.side_effect = _repair_probe_router(declared=7345.0, produced=1794.0, decoded=1800.0)
        instr = make_audio_instruction(action=AudioAction.DECODE_ENCODE, stream_index=1, codec_name="ac3")
        result, repaired = executor._verify_and_repair_audio(instr, produced, tmp_path, _minimal_job())
        assert result == produced
        assert repaired is False
        mocks.audio_extractor.decode_full_wav.assert_called_once()
        mocks.aac_encoder.encode_aac.assert_not_called()

    def test_bug_repairs_and_returns_repaired(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        produced = tmp_path / "audio_1.m4a"
        mocks.prober.probe.side_effect = _repair_probe_router(
            declared=7345.0, produced=1794.0, decoded=7340.0, repaired=7340.0
        )
        instr = make_audio_instruction(action=AudioAction.DECODE_ENCODE, stream_index=1, codec_name="ac3")
        result, repaired = executor._verify_and_repair_audio(instr, produced, tmp_path, _minimal_job())
        assert result == tmp_path / "audio_1_repaired.m4a"
        assert repaired is True
        mocks.audio_decoder.decode_lossless.assert_called_once()
        assert mocks.audio_decoder.decode_lossless.call_args.kwargs["downmix"] is None
        mocks.aac_encoder.encode_aac.assert_called_once()

    def test_oracle_decode_failure_raises(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        produced = tmp_path / "audio_1.m4a"
        mocks.prober.probe.side_effect = _repair_probe_router(declared=7345.0, produced=1794.0)
        mocks.audio_extractor.decode_full_wav.return_value = 1
        instr = make_audio_instruction(action=AudioAction.DECODE_ENCODE, stream_index=1, codec_name="ac3")
        with pytest.raises(RuntimeError, match="oracle decode failed"):
            executor._verify_and_repair_audio(instr, produced, tmp_path, _minimal_job())

    def test_decoded_length_unknown_raises(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        produced = tmp_path / "audio_1.m4a"
        mocks.prober.probe.side_effect = _repair_probe_router(declared=7345.0, produced=1794.0, decoded=None)
        instr = make_audio_instruction(action=AudioAction.DECODE_ENCODE, stream_index=1, codec_name="ac3")
        with pytest.raises(RuntimeError, match="Cannot measure decoded"):
            executor._verify_and_repair_audio(instr, produced, tmp_path, _minimal_job())

    def test_repaired_length_unknown_raises(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        produced = tmp_path / "audio_1.m4a"
        mocks.prober.probe.side_effect = _repair_probe_router(
            declared=7345.0, produced=1794.0, decoded=7340.0, repaired=None
        )
        instr = make_audio_instruction(action=AudioAction.DECODE_ENCODE, stream_index=1, codec_name="ac3")
        with pytest.raises(RuntimeError, match="Cannot measure repaired"):
            executor._verify_and_repair_audio(instr, produced, tmp_path, _minimal_job())

    def test_repair_still_truncated_raises(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        produced = tmp_path / "audio_1.m4a"
        mocks.prober.probe.side_effect = _repair_probe_router(
            declared=7345.0, produced=1794.0, decoded=7340.0, repaired=1800.0
        )
        instr = make_audio_instruction(action=AudioAction.DECODE_ENCODE, stream_index=1, codec_name="ac3")
        with pytest.raises(RuntimeError, match="repair failed"):
            executor._verify_and_repair_audio(instr, produced, tmp_path, _minimal_job())

    def test_ac3_source_disables_drc_on_oracle(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        produced = tmp_path / "audio_1.m4a"
        mocks.prober.probe.side_effect = _repair_probe_router(declared=7345.0, produced=1794.0, decoded=1800.0)
        instr = make_audio_instruction(action=AudioAction.DECODE_ENCODE, stream_index=1, codec_name="ac3")
        executor._verify_and_repair_audio(instr, produced, tmp_path, _minimal_job())
        assert mocks.audio_extractor.decode_full_wav.call_args.kwargs["disable_drc"] is True

    def test_lpcm_source_keeps_drc_on_oracle(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        produced = tmp_path / "audio_1.m4a"
        mocks.prober.probe.side_effect = _repair_probe_router(declared=7345.0, produced=1794.0, decoded=1800.0)
        instr = make_audio_instruction(action=AudioAction.DECODE_ENCODE, stream_index=1, codec_name="pcm_s24le")
        executor._verify_and_repair_audio(instr, produced, tmp_path, _minimal_job())
        assert mocks.audio_extractor.decode_full_wav.call_args.kwargs["disable_drc"] is False


class TestRepairAudioFromWav:
    def test_no_downmix_decodes_and_encodes(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        instr = make_audio_instruction(action=AudioAction.DECODE_ENCODE, stream_index=1, downmix=None)
        result = executor._repair_audio_from_wav(instr, tmp_path / "audio_1_healed.wav", tmp_path)
        assert result == tmp_path / "audio_1_repaired.m4a"
        assert mocks.audio_decoder.decode_lossless.call_args.kwargs["downmix"] is None
        mocks.aac_encoder.encode_aac.assert_called_once()

    def test_stereo_downmix_passes_downmix_arg(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        instr = make_audio_instruction(
            action=AudioAction.DECODE_ENCODE, stream_index=1, channels=6, downmix=DownmixMode.STEREO
        )
        executor._repair_audio_from_wav(instr, tmp_path / "audio_1_healed.wav", tmp_path)
        assert mocks.audio_decoder.decode_lossless.call_args.kwargs["downmix"] == DownmixMode.STEREO

    def test_decode_lossless_failure_raises(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.audio_decoder.decode_lossless.return_value = 1
        instr = make_audio_instruction(action=AudioAction.DECODE_ENCODE, stream_index=1, downmix=None)
        with pytest.raises(RuntimeError, match="repair decode failed"):
            executor._repair_audio_from_wav(instr, tmp_path / "audio_1_healed.wav", tmp_path)

    def test_encode_failure_raises(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.aac_encoder.encode_aac.return_value = 1
        instr = make_audio_instruction(action=AudioAction.DECODE_ENCODE, stream_index=1, downmix=None)
        with pytest.raises(RuntimeError, match="AAC encode failed"):
            executor._repair_audio_from_wav(instr, tmp_path / "audio_1_healed.wav", tmp_path)

    def test_mono_downmix_routes_to_mono(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        instr = make_audio_instruction(
            action=AudioAction.DECODE_ENCODE, stream_index=1, channels=2, downmix=DownmixMode.MONO
        )
        executor._repair_audio_from_wav(instr, tmp_path / "audio_1_healed.wav", tmp_path)
        mocks.audio_extractor.stereo_to_mono_wav.assert_called_once()
        mocks.audio_decoder.decode_lossless.assert_not_called()


class TestRepairMonoWav:
    def test_mono_without_channels_raises(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, _mocks = executor_with_mocks
        instr = make_audio_instruction(
            action=AudioAction.DECODE_ENCODE, stream_index=1, channels=None, downmix=DownmixMode.MONO
        )
        with pytest.raises(RuntimeError, match="without channel count"):
            executor._repair_mono_wav(instr, tmp_path / "audio_1_healed.wav", tmp_path)

    def test_mono_stereo_source_uses_healed_directly(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        healed = tmp_path / "audio_1_healed.wav"
        instr = make_audio_instruction(
            action=AudioAction.DECODE_ENCODE, stream_index=1, channels=2, delay_ms=100, downmix=DownmixMode.MONO
        )
        result = executor._repair_mono_wav(instr, healed, tmp_path)
        assert result == tmp_path / "audio_1_repaired_mono.wav"
        mocks.audio_decoder.decode_lossless.assert_not_called()
        call = mocks.audio_extractor.stereo_to_mono_wav.call_args.kwargs
        assert call["input_path"] == healed
        assert call["delay_ms"] == 100

    def test_mono_multichannel_downmixes_first(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        instr = make_audio_instruction(
            action=AudioAction.DECODE_ENCODE, stream_index=1, channels=6, delay_ms=100, downmix=DownmixMode.MONO
        )
        executor._repair_mono_wav(instr, tmp_path / "audio_1_healed.wav", tmp_path)
        assert mocks.audio_decoder.decode_lossless.call_args.kwargs["downmix"] == DownmixMode.STEREO
        assert mocks.audio_extractor.stereo_to_mono_wav.call_args.kwargs["delay_ms"] == 0

    def test_mono_stereo_downmix_failure_raises(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.audio_decoder.decode_lossless.return_value = 1
        instr = make_audio_instruction(
            action=AudioAction.DECODE_ENCODE, stream_index=1, channels=6, downmix=DownmixMode.MONO
        )
        with pytest.raises(RuntimeError, match="stereo downmix failed"):
            executor._repair_mono_wav(instr, tmp_path / "audio_1_healed.wav", tmp_path)

    def test_mono_average_failure_raises(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 1
        instr = make_audio_instruction(
            action=AudioAction.DECODE_ENCODE, stream_index=1, channels=2, downmix=DownmixMode.MONO
        )
        with pytest.raises(RuntimeError, match="mono average failed"):
            executor._repair_mono_wav(instr, tmp_path / "audio_1_healed.wav", tmp_path)


class TestProcessAudioTrackCopy:
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
        assert denorm_call[0][2] == 500

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
    def test_unknown_action_raises(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, _mocks = executor_with_mocks
        instr = make_audio_instruction(action=AudioAction.COPY, stream_index=1)
        object.__setattr__(instr, "action", "BOGUS")
        with pytest.raises(ValueError, match="Unknown AudioAction"):
            executor._process_audio_track(instr, tmp_path, _minimal_job())


class TestProcessAudioTrackCodecExtensionMapping:
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


class TestProcessSubtitleTrackCopy:
    def test_copy_satellite_srt_returns_path_as_is(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
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
    def test_recode_satellite_cp1251_to_utf8(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
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
        assert not mocks.audio_extractor.extract_track.called

    def test_recode_utf8_source_copies_as_is(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
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
        executor, _mocks = executor_with_mocks
        srt_path = tmp_path / "subs.srt"
        srt_path.write_bytes(b"\x80\x81\x82\x83")
        instr = make_subtitle_instruction(
            source_file=str(srt_path),
            action=SubtitleAction.COPY_RECODE,
            codec_name="subrip",
            stream_index=2,
            source_encoding="shift_jis",
        )
        result = executor._process_subtitle_track(instr, tmp_path, _minimal_job())
        assert result.exists()
        assert result.read_bytes() == b"\x80\x81\x82\x83"

    def test_recode_container_extracts_first(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
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


class TestExtractChaptersFile:
    def test_chapters_present_writes_ogm(
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
        executor, mocks = executor_with_mocks
        mocks.prober.probe.return_value = {"chapters": []}
        result = executor._extract_chapters_file(Path("/src/movie.mkv"), tmp_path)
        assert result is None

    def test_probe_raises_returns_none(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.prober.probe.side_effect = RuntimeError("probe failed")
        result = executor._extract_chapters_file(Path("/src/movie.mkv"), tmp_path)
        assert result is None


class TestSetAdaptersLogDir:
    def test_creates_subdir_and_calls_set_log_dir(
        self,
        tmp_path: Path,
    ) -> None:
        adapter1 = MagicMock()
        adapter2 = MagicMock()
        adapter2.set_log_dir = None
        del adapter2.set_log_dir
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
        executor._set_adapters_log_dir("Whatever")


class TestMakeProgressCallback:
    def test_returns_tracker_and_callback(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
    ) -> None:
        executor, _mocks = executor_with_mocks
        tracker, callback = executor._make_progress_callback(total_s=100.0)
        assert callable(callback)
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
        _, callback = executor._make_progress_callback()
        callback(ProgressSample(fraction=0.5))


class TestDoviProcessorInConstructor:
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
        assert len(executor._adapters) == 8

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

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN_MKV_DATA")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean

        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        mocks.audio_extractor.extract_track.assert_called()
        mocks.encoder.encode.assert_called_once()
        mocks.muxer.mux.assert_called_once()
        mocks.tagger.set_encoder_tag.assert_called_once()
        mocks.cleaner.clean.assert_called_once()
        assert output_path.exists()


def _fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
    Path(output_path).write_bytes(b"CLEAN")
    return 0


class TestRunPipelineAudioTruncation:
    def test_unrepairable_truncation_aborts_before_mux(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        audio_instr = make_audio_instruction(action=AudioAction.DENORM, codec_name="ac3", stream_index=1)
        job = _pipeline_job(tmp_path, audio=[audio_instr])
        mocks.prober.probe.side_effect = _repair_probe_router(
            declared=7345.0, produced=1794.0, decoded=7340.0, repaired=1800.0
        )
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with pytest.raises(RuntimeError, match="repair failed"):
            executor._run_pipeline(job, output_path, tmp_path)

        mocks.muxer.mux.assert_not_called()

    def test_repaired_audio_is_muxed(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.cleaner.clean.side_effect = _fake_clean
        audio_instr = make_audio_instruction(action=AudioAction.DENORM, codec_name="ac3", stream_index=1)
        job = _pipeline_job(tmp_path, audio=[audio_instr])
        mocks.prober.probe.side_effect = _repair_probe_router(
            declared=7345.0, produced=1794.0, decoded=7340.0, repaired=7340.0
        )
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        muxed_audio = mocks.muxer.mux.call_args.kwargs["audio_files"]
        assert muxed_audio[0][0] == tmp_path / "audio_1_repaired.m4a"

    def test_genuine_short_audio_is_muxed(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.cleaner.clean.side_effect = _fake_clean
        audio_instr = make_audio_instruction(action=AudioAction.DENORM, codec_name="ac3", stream_index=1)
        job = _pipeline_job(tmp_path, audio=[audio_instr])
        mocks.prober.probe.side_effect = _repair_probe_router(declared=7345.0, produced=1794.0, decoded=1800.0)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        muxed_audio = mocks.muxer.mux.call_args.kwargs["audio_files"]
        assert muxed_audio[0][0] == tmp_path / "audio_1_denorm.ac3"

    def test_repaired_copy_track_zeroes_mux_delay(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.cleaner.clean.side_effect = _fake_clean
        audio_instr = make_audio_instruction(
            action=AudioAction.COPY, codec_name="aac", stream_index=1, delay_ms=200
        )
        job = _pipeline_job(tmp_path, audio=[audio_instr])
        mocks.prober.probe.side_effect = _repair_probe_router(
            declared=7345.0, produced=1794.0, decoded=7340.0, repaired=7340.0
        )
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        muxed_audio = mocks.muxer.mux.call_args.kwargs["audio_files"]
        assert muxed_audio[0][0] == tmp_path / "audio_1_repaired.m4a"
        assert muxed_audio[0][1]["delay_ms"] == 0

    def test_genuine_short_copy_track_keeps_mux_delay(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.cleaner.clean.side_effect = _fake_clean
        audio_instr = make_audio_instruction(
            action=AudioAction.COPY, codec_name="aac", stream_index=1, delay_ms=200
        )
        job = _pipeline_job(tmp_path, audio=[audio_instr])
        mocks.prober.probe.side_effect = _repair_probe_router(declared=7345.0, produced=1794.0, decoded=1800.0)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        muxed_audio = mocks.muxer.mux.call_args.kwargs["audio_files"]
        assert muxed_audio[0][0] == tmp_path / "audio_1.m4a"
        assert muxed_audio[0][1]["delay_ms"] == 200


class TestRunPipelineWithDvRpu:
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

        job = _pipeline_job(tmp_path, dv_mode=DvMode.TO_8_1)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        dovi_mock.extract_rpu.assert_called_once()
        call_kwargs = dovi_mock.extract_rpu.call_args.kwargs
        assert call_kwargs["mode"] == DvMode.TO_8_1

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
        job = _pipeline_job(tmp_path, dv_mode=DvMode.COPY)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with pytest.raises(RuntimeError, match="DV RPU extraction failed"):
            executor._run_pipeline(job, output_path, tmp_path)


class TestRunPipelineShutdown:
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

        call_count = 0

        def extract_and_shutdown(src: Any, idx: Any, out: Any, on_progress: Any = None) -> int:
            nonlocal call_count
            call_count += 1
            executor._shutdown_event.set()
            return 0

        mocks.audio_extractor.extract_track.side_effect = extract_and_shutdown
        job = _pipeline_job(tmp_path, audio=[audio1, audio2])
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        assert call_count == 1
        assert not mocks.encoder.encode.called

    def test_shutdown_during_encode(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
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
            rpu_path: Any = None,
            cq_override: Any = None,
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
        executor._run_pipeline(job, output_path, tmp_path)
        assert output_path.exists()


def _tq_result(*, knob: int = 27, score: float = 85.0, hit: bool = True) -> KnobSearchResult:
    return KnobSearchResult(knob=knob, score=score, hit=hit, probes=((knob, score),))


def _fake_clean_writing(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
    Path(output_path).write_bytes(b"CLEAN")
    return 0


class TestRunPipelineTargetQuality:
    def test_search_sets_cq_override_and_drops_final_metrics(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        svc = MagicMock()
        svc.search.return_value = _tq_result(knob=27, hit=True)
        executor._target_quality = svc
        mocks.cleaner.clean.side_effect = _fake_clean_writing
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        svc.search.assert_called_once()
        enc_kwargs = mocks.encoder.encode.call_args.kwargs
        assert enc_kwargs["cq_override"] == 27
        assert job.chosen_cq == 27

    def test_search_miss_still_encodes_at_closest(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        svc = MagicMock()
        svc.search.return_value = _tq_result(knob=44, score=70.0, hit=False)
        executor._target_quality = svc
        mocks.cleaner.clean.side_effect = _fake_clean_writing
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        enc_kwargs = mocks.encoder.encode.call_args.kwargs
        assert enc_kwargs["cq_override"] == 44
        assert job.chosen_cq == 44

    def test_grain_searches_when_service_supports_it(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        svc = MagicMock()
        svc.can_search.return_value = True
        svc.search.return_value = _tq_result(knob=20, hit=True)
        executor._target_quality = svc
        mocks.cleaner.clean.side_effect = _fake_clean_writing
        job = _pipeline_job(tmp_path)
        job.video_params = make_video_params(grain=True, source_width=720, source_height=576)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        svc.search.assert_called_once()
        assert mocks.encoder.encode.call_args.kwargs["cq_override"] == 20
        assert job.chosen_cq == 20

    def test_grain_hd_uses_fixed_qvbr_without_searching(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        svc = MagicMock()
        svc.can_search.return_value = True
        executor._target_quality = svc
        mocks.cleaner.clean.side_effect = _fake_clean_writing
        job = _pipeline_job(tmp_path)
        job.video_params = make_video_params(grain=True, source_width=1920, source_height=1080)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        svc.search.assert_not_called()
        assert mocks.encoder.encode.call_args.kwargs["cq_override"] == 32
        assert job.chosen_cq == 32

    def test_grain_skipped_when_service_cannot_search(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        svc = MagicMock()
        svc.can_search.return_value = False
        executor._target_quality = svc
        mocks.cleaner.clean.side_effect = _fake_clean_writing
        job = _pipeline_job(tmp_path)
        job.video_params = make_video_params(grain=True, source_width=720, source_height=576)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        svc.search.assert_not_called()
        assert mocks.encoder.encode.call_args.kwargs["cq_override"] is None
        assert job.chosen_cq is None

    def test_search_skipped_without_duration(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        svc = MagicMock()
        executor._target_quality = svc
        mocks.cleaner.clean.side_effect = _fake_clean_writing
        job = _pipeline_job(tmp_path, duration_s=0.0)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        svc.search.assert_not_called()
        assert mocks.encoder.encode.call_args.kwargs["cq_override"] is None

    def test_search_skipped_without_service(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.cleaner.clean.side_effect = _fake_clean_writing
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        enc_kwargs = mocks.encoder.encode.call_args.kwargs
        assert enc_kwargs["cq_override"] is None

    def test_reuses_cached_chosen_cq(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        svc = MagicMock()
        executor._target_quality = svc
        mocks.cleaner.clean.side_effect = _fake_clean_writing
        job = _pipeline_job(tmp_path)
        job.chosen_cq = 33
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        svc.search.assert_not_called()
        assert mocks.encoder.encode.call_args.kwargs["cq_override"] == 33
        assert job.chosen_cq == 33

    def test_search_skipped_duration_progress_warns(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        progress = MagicMock()
        executor._progress = progress
        svc = MagicMock()
        executor._target_quality = svc
        mocks.cleaner.clean.side_effect = _fake_clean_writing
        job = _pipeline_job(tmp_path, duration_s=0.0)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        svc.search.assert_not_called()
        tool_lines = [c.args[0] for c in progress.add_tool_line.call_args_list]
        assert any("unknown duration" in line.lower() for line in tool_lines)

    def test_chosen_cq_persisted_on_error_path(self, tmp_path: Path) -> None:
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
        mocks.muxer.mux.return_value = 1
        svc = MagicMock()
        svc.search.return_value = _tq_result(knob=31, hit=True)

        executor = Executor(
            encoder=mocks.encoder,
            audio_extractor=mocks.audio_extractor,
            audio_decoder=mocks.audio_decoder,
            aac_encoder=mocks.aac_encoder,
            muxer=mocks.muxer,
            tagger=mocks.tagger,
            cleaner=mocks.cleaner,
            prober=mocks.prober,
            target_quality=svc,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        job = make_job(
            job_id="tq-err-job",
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
        assert loaded.jobs[0].chosen_cq == 31

    def test_search_hit_progress_wiring(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        progress = MagicMock()
        executor._progress = progress
        svc = MagicMock()
        svc.search.return_value = _tq_result(knob=30, hit=True)
        executor._target_quality = svc
        mocks.cleaner.clean.side_effect = _fake_clean_writing
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        tool_lines = [c.args[0] for c in progress.add_tool_line.call_args_list]
        assert any("target-quality qvbr 30" in line.lower() for line in tool_lines)

    def test_search_miss_progress_warns(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        progress = MagicMock()
        executor._progress = progress
        svc = MagicMock()
        svc.search.return_value = _tq_result(knob=44, score=70.0, hit=False)
        executor._target_quality = svc
        mocks.cleaner.clean.side_effect = _fake_clean_writing
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        tool_lines = [c.args[0] for c in progress.add_tool_line.call_args_list]
        assert any("not hit" in line.lower() for line in tool_lines)

    def test_search_mutes_raw_output_and_wires_narration(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, _mocks = executor_with_mocks
        progress = MagicMock()
        executor._progress = progress
        svc = MagicMock()
        svc.can_search.return_value = True
        order: list[str] = []
        progress.mute_tool_output.side_effect = lambda: order.append("mute")
        progress.unmute_tool_output.side_effect = lambda: order.append("unmute")

        def _search(*_a: Any, **_k: Any) -> KnobSearchResult:
            order.append("search")
            return _tq_result(knob=27, hit=True)

        svc.search.side_effect = _search
        executor._target_quality = svc
        job = _pipeline_job(tmp_path)

        result = executor._maybe_search_target_quality(job, Path("/src/movie.mkv"), tmp_path)

        assert result == 27
        assert order == ["mute", "search", "unmute"]
        assert callable(svc.search.call_args.kwargs["on_event"])

    def test_search_unmutes_even_when_search_raises(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, _mocks = executor_with_mocks
        progress = MagicMock()
        executor._progress = progress
        svc = MagicMock()
        svc.can_search.return_value = True
        svc.search.side_effect = RuntimeError("probe blew up")
        executor._target_quality = svc
        job = _pipeline_job(tmp_path)

        with pytest.raises(RuntimeError, match="probe blew up"):
            executor._maybe_search_target_quality(job, Path("/src/movie.mkv"), tmp_path)
        progress.unmute_tool_output.assert_called_once()

    def test_narrate_routes_to_furnace_channel(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
    ) -> None:
        executor, _mocks = executor_with_mocks
        progress = MagicMock()
        executor._progress = progress
        executor._narrate("CRF 24 -> SSIMULACRA2 67.0")
        progress.add_tool_line.assert_called_once_with("[furnace] CRF 24 -> SSIMULACRA2 67.0")

    def test_narrate_without_progress_is_noop(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
    ) -> None:
        executor, _mocks = executor_with_mocks
        executor._progress = None
        executor._narrate("ignored")

    def test_chosen_cq_persisted_by_run(self, tmp_path: Path) -> None:
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
        mocks.cleaner.clean.side_effect = _fake_clean_writing
        svc = MagicMock()
        svc.search.return_value = _tq_result(knob=29, hit=True)

        executor = Executor(
            encoder=mocks.encoder,
            audio_extractor=mocks.audio_extractor,
            audio_decoder=mocks.audio_decoder,
            aac_encoder=mocks.aac_encoder,
            muxer=mocks.muxer,
            tagger=mocks.tagger,
            cleaner=mocks.cleaner,
            prober=mocks.prober,
            target_quality=svc,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        job = make_job(
            job_id="tq-run-job",
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
        assert loaded.jobs[0].chosen_cq == 29


class TestRunPipelineChapters:
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

        hdr = HdrMetadata(
            content_light="MaxCLL=1000,MaxFALL=400",
            mastering_display=(
                "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,0)"
            ),
        )
        vp = make_video_params(
            color_range="tv",
            color_primaries="bt2020",
            color_transfer="smpte2084",
            color_matrix="bt2020nc",
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
        assert video_meta["color_matrix"] == "bt2020nc"
        assert video_meta["hdr_max_cll"] == "1000"
        assert video_meta["hdr_max_fall"] == "400"
        assert video_meta["hdr_mastering_display"] == (
            "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,0)"
        )

    def test_fps_in_video_meta_for_reencode(self, tmp_path: Path) -> None:
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

        vp = make_video_params(fps_num=24000, fps_den=1001)
        job = make_job(
            job_id="fps-job",
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

        video_meta = mocks.muxer.mux.call_args.kwargs["video_meta"]
        assert video_meta["fps_num"] == 24000
        assert video_meta["fps_den"] == 1001

    def test_no_fps_in_video_meta_for_passthrough(self, tmp_path: Path) -> None:
        mocks = SimpleNamespace(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
            video_copier=MagicMock(),
        )
        mocks.video_copier.copy_video.return_value = 0
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
            video_copier=mocks.video_copier,
        )

        vp = make_video_params(passthrough=True)
        job = make_job(
            job_id="pt-job",
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

        video_meta = mocks.muxer.mux.call_args.kwargs["video_meta"]
        assert video_meta is not None
        assert "fps_num" not in video_meta
        assert "fps_den" not in video_meta


class TestRunPipelineProgressWiring:
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

        assert progress_mock.update_status.called
        assert progress_mock.add_tool_line.called


class TestRunPipelineMuxedSizeUpdate:
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
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        progress_mock.update_output_size.assert_called()


class TestRunLifecycleHappyPath:
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

        assert not mocks.encoder.encode.called


class TestRunLifecycleError:
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
    def test_cleaner_failure_uses_muxed(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks

        def fake_mux(**kwargs: Any) -> int:
            Path(kwargs["output_path"]).write_bytes(b"MUXED_DATA")
            return 0

        mocks.muxer.mux.side_effect = fake_mux
        mocks.cleaner.clean.return_value = 1

        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        assert output_path.exists()
        assert output_path.read_bytes() == b"MUXED_DATA"


class TestMkcleanProgressUpdate:
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
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        progress_mock.update_output_size.assert_called()


class TestRunShutdownBetweenJobs:
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

        def encode_then_shutdown(**kwargs: Any) -> EncodeResult:
            executor._shutdown_event.set()
            return EncodeResult(return_code=0, encoder_settings="test")

        mocks.encoder.encode.side_effect = encode_then_shutdown

        executor.run(plan, plan_path)

        loaded = load_plan(plan_path)
        assert loaded.jobs[0].status == JobStatus.DONE
        assert loaded.jobs[1].status == JobStatus.PENDING


class TestRunWithProgress:
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


class TestGracefulShutdown:
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
            executor.graceful_shutdown()

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
            executor.graceful_shutdown()

        assert executor._shutdown_event.is_set()

    def test_nosuchprocess_suppressed(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
    ) -> None:
        import psutil as psutil_mod

        executor, _mocks = executor_with_mocks
        mock_child = MagicMock()
        mock_child.kill.side_effect = psutil_mod.NoSuchProcess(pid=12345)
        mock_parent = MagicMock()
        mock_parent.children.return_value = [mock_child]

        with patch("furnace.services.executor.psutil.Process", return_value=mock_parent):
            executor.graceful_shutdown()


class TestExecuteJobTempCleanup:
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


class TestCodecSupportedByEac3to:
    def test_supported_codecs(self) -> None:
        from furnace.services.executor import _codec_supported_by_eac3to

        supported = [
            "ac3",
            "eac3",
            "dts",
            "truehd",
            "flac",
            "pcm_s16le",
            "pcm_s24le",
            "pcm_s16be",
            "mp2",
            "mp3",
        ]
        for codec in supported:
            assert _codec_supported_by_eac3to(codec), f"{codec} should be supported"

    def test_unsupported_codecs(self) -> None:
        from furnace.services.executor import _codec_supported_by_eac3to

        for codec in ["opus", "vorbis", "wmav2", "amr_nb"]:
            assert not _codec_supported_by_eac3to(codec), f"{codec} should NOT be supported"

    def test_aac_is_not_supported(self) -> None:
        from furnace.services.executor import _codec_supported_by_eac3to

        assert not _codec_supported_by_eac3to("aac")
        assert not _codec_supported_by_eac3to("AAC")

    def test_case_insensitive(self) -> None:
        from furnace.services.executor import _codec_supported_by_eac3to

        assert _codec_supported_by_eac3to("AC3")
        assert _codec_supported_by_eac3to("TrueHD")


class TestEncodeOnProgressOutputSize:
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
            rpu_path: Any = None,
            cq_override: Any = None,
        ) -> EncodeResult:
            assert on_progress is not None
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
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        progress_mock.update_output_size.assert_called()


class TestEncodeOnProgressOSError:
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
            rpu_path: Any = None,
            cq_override: Any = None,
        ) -> EncodeResult:
            assert on_progress is not None
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
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)


class TestAudioSizeTracking:
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
        audio_instr = make_audio_instruction(
            action=AudioAction.COPY,
            codec_name="aac",
            stream_index=1,
        )
        job = _pipeline_job(tmp_path, audio=[audio_instr])
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        progress_mock.update_output_size.assert_called()
        first_call_size = progress_mock.update_output_size.call_args_list[0][0][0]
        assert first_call_size == 200


class TestRunRetriesErrorJobs:
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


class TestAudioDelayMeta:
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
        assert audio_files[0][1]["delay_ms"] == 100
        assert audio_files[1][1]["delay_ms"] == 0


def _make_executor_with_progress() -> tuple[Executor, SimpleNamespace, MagicMock]:
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
    mocks.audio_extractor.decode_full_wav.return_value = 0
    mocks.audio_extractor.stereo_to_mono_wav.return_value = 0
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
    return executor, mocks, progress_mock


class TestAudioProgressLines:
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

    def test_decode_encode_mono_stereo_source_progress_lines(
        self,
        tmp_path: Path,
    ) -> None:
        executor, mocks, progress = _make_executor_with_progress()
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0
        instr = make_audio_instruction(
            action=AudioAction.DECODE_ENCODE,
            codec_name="aac",
            channels=2,
            downmix=DownmixMode.MONO,
            stream_index=5,
        )
        executor._process_audio_track(instr, tmp_path, _minimal_job())
        tool_lines = [c[0][0] for c in progress.add_tool_line.call_args_list]
        assert any("Averaging audio stream 5 to mono" in line for line in tool_lines)
        assert any("Encoding AAC for stream 5" in line for line in tool_lines)

    def test_decode_encode_mono_stereo_drc_codec_progress_lines(
        self,
        tmp_path: Path,
    ) -> None:
        executor, mocks, progress = _make_executor_with_progress()
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0
        instr = make_audio_instruction(
            action=AudioAction.DECODE_ENCODE,
            codec_name="ac3",
            channels=2,
            downmix=DownmixMode.MONO,
            stream_index=8,
        )
        executor._process_audio_track(instr, tmp_path, _minimal_job())
        tool_lines = [c[0][0] for c in progress.add_tool_line.call_args_list]
        assert any("Extracting audio stream 8 for MONO downmix" in line for line in tool_lines)
        assert any("Decoding audio stream 8 with eac3to" in line for line in tool_lines)
        assert any("Averaging audio stream 8 to mono" in line for line in tool_lines)
        assert any("Encoding AAC for stream 8" in line for line in tool_lines)

    def test_decode_encode_mono_multichannel_eac3to_progress_lines(
        self,
        tmp_path: Path,
    ) -> None:
        executor, mocks, progress = _make_executor_with_progress()
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0
        instr = make_audio_instruction(
            action=AudioAction.DECODE_ENCODE,
            codec_name="dts",
            channels=6,
            downmix=DownmixMode.MONO,
            stream_index=6,
        )
        executor._process_audio_track(instr, tmp_path, _minimal_job())
        tool_lines = [c[0][0] for c in progress.add_tool_line.call_args_list]
        assert any("Extracting audio stream 6 for MONO downmix" in line for line in tool_lines)
        assert any("Downmixing audio stream 6 to stereo with eac3to" in line for line in tool_lines)
        assert any("Averaging audio stream 6 to mono" in line for line in tool_lines)
        assert any("Encoding AAC for stream 6" in line for line in tool_lines)

    def test_decode_encode_mono_multichannel_non_eac3to_progress_lines(
        self,
        tmp_path: Path,
    ) -> None:
        executor, mocks, progress = _make_executor_with_progress()
        mocks.audio_extractor.stereo_to_mono_wav.return_value = 0
        instr = make_audio_instruction(
            action=AudioAction.DECODE_ENCODE,
            codec_name="opus",
            channels=6,
            downmix=DownmixMode.MONO,
            stream_index=7,
        )
        executor._process_audio_track(instr, tmp_path, _minimal_job())
        tool_lines = [c[0][0] for c in progress.add_tool_line.call_args_list]
        assert any("Pre-decoding audio stream 7" in line and "MONO downmix" in line for line in tool_lines)
        assert any("Downmixing audio stream 7 to stereo with eac3to" in line for line in tool_lines)
        assert any("Averaging audio stream 7 to mono" in line for line in tool_lines)
        assert any("Encoding AAC for stream 7" in line for line in tool_lines)


class TestVideoMetaEmptyFields:
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
            passthrough=True,
            color_range="",
            color_primaries="",
            color_transfer="",
            color_matrix="",
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
        assert mux_call.kwargs["video_meta"] is None


class TestChaptersMojibake:
    def test_mojibake_chapters_detected(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
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


class TestShutdownAfterSubtitles:
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

        def extract_and_shutdown(src: Any, idx: Any, out: Any, on_progress: Any = None) -> int:
            executor._shutdown_event.set()
            return 0

        mocks.audio_extractor.extract_track.side_effect = extract_and_shutdown
        job = _pipeline_job(tmp_path, subtitles=[sub_instr])
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        assert not mocks.encoder.encode.called


class TestSetAdaptersLogDirMissingMethod:
    def test_adapter_without_set_log_dir(self, tmp_path: Path) -> None:
        adapter_with = MagicMock()
        adapter_without = MagicMock(spec=[])

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
        adapter_with.set_log_dir.assert_called_once_with(expected_dir)


class TestSubtitleProgressInPipeline:
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
        job = _pipeline_job(tmp_path, dv_mode=DvMode.TO_8_1)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)

        status_calls = [c[0][0] for c in progress_mock.update_status.call_args_list]
        assert any("Extracting DV RPU" in s for s in status_calls)
        tool_lines = [c[0][0] for c in progress_mock.add_tool_line.call_args_list]
        assert any("Extracting DV RPU" in line for line in tool_lines)


class TestShutdownAfterAudioBeforeSubtitles:
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


class TestShutdownBeforeDvWithDvMode:
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

        sub_instr = make_subtitle_instruction(
            source_file="/src/movie.mkv",
            action=SubtitleAction.COPY,
            codec_name="hdmv_pgs_subtitle",
            stream_index=3,
        )

        def extract_and_shutdown(src: Any, idx: Any, out: Any, on_progress: Any = None) -> int:
            executor._shutdown_event.set()
            return 0

        mocks.audio_extractor.extract_track.side_effect = extract_and_shutdown

        job = _pipeline_job(tmp_path, subtitles=[sub_instr], dv_mode=DvMode.TO_8_1)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        assert not dovi_mock.extract_rpu.called


class TestEncodeOnProgressStatOSError:
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
            rpu_path: Any = None,
            cq_override: Any = None,
        ) -> EncodeResult:
            assert on_progress is not None
            Path(output_path).write_bytes(b"V")

            def patched_stat(self_path: Path, **kwargs: Any) -> Any:
                assert str(self_path) == str(output_path)
                raise OSError("permission denied")

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
        job = _pipeline_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)
        progress_mock.update_output_size.assert_called()


class TestEncodeOnProgressNoProgress:
    def test_no_progress_in_callback(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks

        def fake_encode(
            input_path: Any,
            output_path: Any,
            video_params: Any,
            on_progress: Any = None,
            rpu_path: Any = None,
            cq_override: Any = None,
        ) -> EncodeResult:
            assert on_progress is not None
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
        executor._run_pipeline(job, output_path, tmp_path)


class TestVideoMetaUnknownContentLightPart:
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
        assert "Unknown" not in str(video_meta)


def _passthrough_job(tmp_path: Path, *, dv_mode: DvMode | None = None) -> Any:
    return make_job(
        job_id="passthrough-job",
        source_files=["/src/movie.mkv"],
        output_file=str(tmp_path / "output" / "movie.mkv"),
        video_params=make_video_params(passthrough=True, dv_mode=dv_mode),
        audio=[],
        subtitles=[],
        attachments=[],
        copy_chapters=False,
        source_size=1_000_000,
        duration_s=5400.0,
    )


class TestVideoCopierInConstructor:
    def test_video_copier_appended(self) -> None:
        copier = MagicMock()
        executor = Executor(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
            video_copier=copier,
        )
        assert copier in executor._adapters

    def test_no_video_copier_not_appended(self) -> None:
        executor = Executor(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
            video_copier=None,
        )
        assert executor._video_copier is None


class TestRunPipelinePassthrough:
    def test_passthrough_calls_copy_video_not_encode(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean
        job = _passthrough_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)

        mocks.video_copier.copy_video.assert_called_once()
        assert not mocks.encoder.encode.called
        mocks.muxer.mux.assert_called_once()
        mocks.tagger.set_encoder_tag.assert_called_once()
        mocks.cleaner.clean.assert_called_once()
        assert output_path.exists()

    def test_passthrough_tag_settings_string(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean
        job = _passthrough_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)

        tag_call = mocks.tagger.set_encoder_tag.call_args
        assert tag_call[0][2] == "video stream copied (passthrough)"

    def test_passthrough_skips_dv_rpu_extraction(
        self,
        tmp_path: Path,
    ) -> None:
        dovi_mock = MagicMock()
        copier_mock = MagicMock()
        copier_mock.copy_video.return_value = 0
        muxer_mock = MagicMock()
        muxer_mock.mux.return_value = 0
        tagger_mock = MagicMock()
        tagger_mock.set_encoder_tag.return_value = 0
        cleaner_mock = MagicMock()

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        cleaner_mock.clean.side_effect = fake_clean
        executor = Executor(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=muxer_mock,
            tagger=tagger_mock,
            cleaner=cleaner_mock,
            prober=MagicMock(),
            dovi_processor=dovi_mock,
            video_copier=copier_mock,
        )
        job = _passthrough_job(tmp_path, dv_mode=DvMode.COPY)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)

        assert not dovi_mock.extract_rpu.called
        copier_mock.copy_video.assert_called_once()

    def test_passthrough_forwards_hdr_video_meta_to_mux(
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
        vp = make_video_params(
            passthrough=True,
            color_range="tv",
            color_primaries="bt2020",
            color_transfer="smpte2084",
            hdr=HdrMetadata(content_light="MaxCLL=1000,MaxFALL=400"),
        )
        job = make_job(
            job_id="passthrough-hdr",
            source_files=["/src/movie.mkv"],
            output_file=str(tmp_path / "output" / "movie.mkv"),
            video_params=vp,
            audio=[],
            subtitles=[],
            attachments=[],
            copy_chapters=False,
            source_size=1_000_000,
            duration_s=5400.0,
        )
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)

        video_meta = mocks.muxer.mux.call_args.kwargs["video_meta"]
        assert video_meta["color_range"] == "tv"
        assert video_meta["color_primaries"] == "bt2020"
        assert video_meta["color_transfer"] == "smpte2084"
        assert video_meta["hdr_max_cll"] == "1000"
        assert video_meta["hdr_max_fall"] == "400"

    def test_passthrough_without_copier_raises(
        self,
        tmp_path: Path,
    ) -> None:
        executor = Executor(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=MagicMock(),
            tagger=MagicMock(),
            cleaner=MagicMock(),
            prober=MagicMock(),
            video_copier=None,
        )
        job = _passthrough_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with pytest.raises(RuntimeError, match=r"passthrough.*video_copier"):
            executor._run_pipeline(job, output_path, tmp_path)

    def test_passthrough_copy_failure_raises(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks
        mocks.video_copier.copy_video.return_value = 1
        job = _passthrough_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with pytest.raises(RuntimeError, match=r"passthrough copy failed with return code 1"):
            executor._run_pipeline(job, output_path, tmp_path)

    def test_passthrough_progress_updates_output_size(
        self,
        tmp_path: Path,
    ) -> None:
        progress_mock = MagicMock()
        copier_mock = MagicMock()

        def fake_copy(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"VIDEO_DATA")
            assert on_progress is not None
            on_progress(ProgressSample(fraction=0.5))
            return 0

        copier_mock.copy_video.side_effect = fake_copy
        muxer_mock = MagicMock()
        muxer_mock.mux.return_value = 0
        tagger_mock = MagicMock()
        tagger_mock.set_encoder_tag.return_value = 0
        cleaner_mock = MagicMock()

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        cleaner_mock.clean.side_effect = fake_clean
        executor = Executor(
            encoder=MagicMock(),
            audio_extractor=MagicMock(),
            audio_decoder=MagicMock(),
            aac_encoder=MagicMock(),
            muxer=muxer_mock,
            tagger=tagger_mock,
            cleaner=cleaner_mock,
            prober=MagicMock(),
            video_copier=copier_mock,
            progress=progress_mock,
        )
        job = _passthrough_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)

        progress_mock.update_output_size.assert_called()
        progress_mock.update_status.assert_called()

    def test_passthrough_progress_callback_no_progress(
        self,
        executor_with_mocks: tuple[Executor, SimpleNamespace],
        tmp_path: Path,
    ) -> None:
        executor, mocks = executor_with_mocks

        def fake_copy(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            assert on_progress is not None
            on_progress(ProgressSample(fraction=0.5))
            return 0

        mocks.video_copier.copy_video.side_effect = fake_copy

        def fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
            Path(output_path).write_bytes(b"CLEAN")
            return 0

        mocks.cleaner.clean.side_effect = fake_clean
        job = _passthrough_job(tmp_path)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        executor._run_pipeline(job, output_path, tmp_path)


class TestVideoIntermediateName:
    def test_encode_branch_uses_obu(self) -> None:
        assert _video_intermediate_name(passthrough=False) == "video.obu"

    def test_passthrough_branch_uses_mkv(self) -> None:
        assert _video_intermediate_name(passthrough=True) == "video.mkv"


def _grain_executor(
    *,
    grain_encoder: Any | None,
) -> tuple[Executor, SimpleNamespace]:
    mocks = SimpleNamespace(
        encoder=MagicMock(),
        grain_encoder=grain_encoder,
        audio_extractor=MagicMock(),
        audio_decoder=MagicMock(),
        aac_encoder=MagicMock(),
        muxer=MagicMock(),
        tagger=MagicMock(),
        cleaner=MagicMock(),
        prober=MagicMock(),
        video_copier=MagicMock(),
    )
    mocks.encoder.encode.return_value = EncodeResult(return_code=0, encoder_settings="main")
    if grain_encoder is not None:
        grain_encoder.encode.return_value = EncodeResult(return_code=0, encoder_settings="grain")
    mocks.muxer.mux.return_value = 0
    mocks.tagger.set_encoder_tag.return_value = 0
    mocks.video_copier.copy_video.return_value = 0

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
        video_copier=mocks.video_copier,
        grain_encoder=grain_encoder,
    )
    return executor, mocks


def _grain_job(
    tmp_path: Path,
    *,
    grain: bool,
    passthrough: bool = False,
    source_width: int = 1920,
    source_height: int = 1080,
) -> Any:
    return make_job(
        job_id="grain-job",
        source_files=["/src/movie.mkv"],
        output_file=str(tmp_path / "output" / "movie.mkv"),
        video_params=make_video_params(
            grain=grain,
            passthrough=passthrough,
            source_width=source_width,
            source_height=source_height,
        ),
        audio=[],
        subtitles=[],
        attachments=[],
        copy_chapters=False,
        source_size=0,
        duration_s=100.0,
    )


class TestGrainEncoderRouting:
    def test_grain_sd_routes_to_grain_encoder(self, tmp_path: Path) -> None:
        grain_enc = MagicMock()
        executor, mocks = _grain_executor(grain_encoder=grain_enc)
        job = _grain_job(tmp_path, grain=True, source_width=720, source_height=576)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        grain_enc.encode.assert_called_once()
        assert not mocks.encoder.encode.called

    def test_grain_hd_routes_to_main_encoder_at_fixed_qvbr(self, tmp_path: Path) -> None:
        grain_enc = MagicMock()
        executor, mocks = _grain_executor(grain_encoder=grain_enc)
        job = _grain_job(tmp_path, grain=True, source_width=1920, source_height=1080)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        mocks.encoder.encode.assert_called_once()
        assert not grain_enc.encode.called
        assert mocks.encoder.encode.call_args.kwargs["cq_override"] == 32
        assert job.chosen_cq == 32

    def test_non_grain_job_routes_to_main_encoder(self, tmp_path: Path) -> None:
        grain_enc = MagicMock()
        executor, mocks = _grain_executor(grain_encoder=grain_enc)
        job = _grain_job(tmp_path, grain=False)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        mocks.encoder.encode.assert_called_once()
        assert not grain_enc.encode.called

    def test_grain_sd_falls_back_to_main_when_no_grain_encoder(self, tmp_path: Path) -> None:
        executor, mocks = _grain_executor(grain_encoder=None)
        job = _grain_job(tmp_path, grain=True, source_width=720, source_height=576)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        mocks.encoder.encode.assert_called_once()

    def test_passthrough_job_uses_no_encoder(self, tmp_path: Path) -> None:
        grain_enc = MagicMock()
        executor, mocks = _grain_executor(grain_encoder=grain_enc)
        job = _grain_job(tmp_path, grain=False, passthrough=True)
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        executor._run_pipeline(job, output_path, tmp_path)

        assert not mocks.encoder.encode.called
        assert not grain_enc.encode.called
        mocks.video_copier.copy_video.assert_called_once()

    def test_grain_encoder_in_adapters_when_set(self) -> None:
        grain_enc = MagicMock()
        executor, _mocks = _grain_executor(grain_encoder=grain_enc)
        assert grain_enc in executor._adapters

    def test_grain_encoder_absent_when_none(self) -> None:
        executor, mocks = _grain_executor(grain_encoder=None)
        assert mocks.encoder in executor._adapters
        assert None not in executor._adapters
