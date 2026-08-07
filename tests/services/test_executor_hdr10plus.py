from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from furnace.core.models import (
    AudioAction,
    DvMode,
    EncodeResult,
    HdrMetadata,
    SubtitleAction,
    VideoParams,
)
from furnace.services.executor import Executor, HdrMetadataLostError
from tests.conftest import (
    make_audio_instruction,
    make_job,
    make_subtitle_instruction,
    make_video_params,
)

_MASTERING_DISPLAY = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)"

_HDR10_PLUS_SIDE_DATA = {"side_data_type": "HDR Dynamic Metadata SMPTE2094-40 (HDR10+)"}
_DOLBY_VISION_SIDE_DATA = {"side_data_type": "Dolby Vision Metadata"}
_MASTERING_DISPLAY_SIDE_DATA = {"side_data_type": "Mastering display metadata"}
_CONTENT_LIGHT_SIDE_DATA = {"side_data_type": "Content light level metadata"}

_DURATION_S = 100.0
_FPS = 24
_SOURCE_FRAMES = 2400

_LONG_DURATION_S = 500.0
_LONG_SOURCE_FRAMES = 12000


def _hdr10plus_payload(frames: int) -> str:
    return json.dumps({"SceneInfo": [{"SceneFrameIndex": i} for i in range(frames)]})


def _light_payload(frames: int) -> str:
    return json.dumps({"SceneInfo": [{}] * frames})


def _hdr10plus_params(**kwargs: Any) -> VideoParams:
    hdr = kwargs.pop("hdr", HdrMetadata(is_hdr10_plus=True))
    return make_video_params(hdr=hdr, fps_num=_FPS, fps_den=1, **kwargs)


def _fake_clean(input_path: Any, output_path: Any, on_progress: Any = None) -> int:
    Path(output_path).write_bytes(b"CLEAN")
    return 0


def _make_hdr10plus_processor(
    *,
    rc: int = 0,
    frames: int = _SOURCE_FRAMES,
    payload: str | None = None,
    write: bool = True,
) -> MagicMock:
    processor = MagicMock()

    def extract(input_path: Any, output_json: Any) -> int:
        if write:
            text = payload if payload is not None else _hdr10plus_payload(frames)
            Path(output_json).write_text(text, encoding="utf-8")
        return rc

    processor.extract.side_effect = extract
    return processor


def _make_executor(
    *,
    hdr10plus_processor: Any = None,
    dovi_processor: Any = None,
    side_data: list[dict[str, Any]] | None = None,
    grain_encoder: Any = None,
    progress: Any = None,
) -> tuple[Executor, SimpleNamespace]:
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
        hdr10plus_processor=hdr10plus_processor,
        dovi_processor=dovi_processor,
        grain_encoder=grain_encoder,
    )
    mocks.encoder.encode.return_value = EncodeResult(return_code=0, encoder_settings="test")
    if grain_encoder is not None:
        grain_encoder.encode.return_value = EncodeResult(return_code=0, encoder_settings="svt")
    mocks.muxer.mux.return_value = 0
    mocks.tagger.set_encoder_tag.return_value = 0
    mocks.cleaner.clean.side_effect = _fake_clean
    mocks.prober.probe.return_value = {"chapters": []}
    mocks.prober.probe_hdr_side_data_strict.return_value = (
        side_data if side_data is not None else [_HDR10_PLUS_SIDE_DATA]
    )
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
        dovi_processor=dovi_processor,
        hdr10plus_processor=hdr10plus_processor,
        video_copier=mocks.video_copier,
        grain_encoder=grain_encoder,
        progress=progress,
    )
    return executor, mocks


def _job(
    tmp_path: Path,
    video_params: VideoParams,
    *,
    duration_s: float = _DURATION_S,
    subtitles: list[Any] | None = None,
) -> Any:
    return make_job(
        job_id="hdr10plus-job",
        source_files=["/src/movie.mkv"],
        output_file=str(tmp_path / "output" / "movie.mkv"),
        video_params=video_params,
        audio=[],
        subtitles=subtitles if subtitles is not None else [],
        attachments=[],
        copy_chapters=False,
        source_size=1_000_000,
        duration_s=duration_s,
    )


def _run(executor: Executor, job: Any, tmp_path: Path) -> None:
    output_path = Path(job.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    executor._run_pipeline(job, output_path, tmp_path)


class TestHdr10PlusProcessorInConstructor:
    def test_hdr10plus_processor_appended(self) -> None:
        processor = MagicMock()
        executor, _mocks = _make_executor(hdr10plus_processor=processor)
        assert processor in executor._adapters

    def test_no_hdr10plus_processor_not_appended(self) -> None:
        executor, mocks = _make_executor()
        assert mocks.video_copier in executor._adapters
        assert len(executor._adapters) == 8


class TestHdr10PlusExtraction:
    def test_extracts_metadata_into_temp_dir(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor()
        executor, _mocks = _make_executor(hdr10plus_processor=processor)
        _run(executor, _job(tmp_path, _hdr10plus_params()), tmp_path)

        processor.extract.assert_called_once()
        call_kwargs = processor.extract.call_args.kwargs
        assert call_kwargs["input_path"] == Path("/src/movie.mkv")
        assert call_kwargs["output_json"] == tmp_path / "hdr10plus.json"

    def test_encoder_receives_the_json(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor()
        executor, mocks = _make_executor(hdr10plus_processor=processor)
        _run(executor, _job(tmp_path, _hdr10plus_params()), tmp_path)

        encode_kwargs = mocks.encoder.encode.call_args.kwargs
        assert encode_kwargs["dhdr10_json"] == tmp_path / "hdr10plus.json"

    def test_non_hdr10_plus_job_passes_no_json(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor()
        executor, mocks = _make_executor(hdr10plus_processor=processor, side_data=[])
        _run(executor, _job(tmp_path, make_video_params()), tmp_path)

        processor.extract.assert_not_called()
        assert mocks.encoder.encode.call_args.kwargs["dhdr10_json"] is None

    def test_missing_tool_raises(self, tmp_path: Path) -> None:
        executor, _mocks = _make_executor()
        with pytest.raises(RuntimeError, match="HDR10\\+ content requires hdr10plus_tool"):
            _run(executor, _job(tmp_path, _hdr10plus_params()), tmp_path)

    def test_missing_tool_raises_before_any_audio_work(self, tmp_path: Path) -> None:
        executor, mocks = _make_executor()
        job = _job(tmp_path, _hdr10plus_params())
        job.audio = [make_audio_instruction(action=AudioAction.COPY, codec_name="aac", stream_index=1)]
        with pytest.raises(RuntimeError, match="HDR10\\+ content requires hdr10plus_tool"):
            _run(executor, job, tmp_path)

        mocks.audio_extractor.extract_track.assert_not_called()

    def test_svt_routing_error_wins_over_the_missing_tool(self, tmp_path: Path) -> None:
        grain_encoder = MagicMock()
        executor, _mocks = _make_executor(grain_encoder=grain_encoder)
        params = make_video_params(
            hdr=HdrMetadata(is_hdr10_plus=True),
            grain=True,
            source_width=720,
            source_height=576,
        )
        with pytest.raises(RuntimeError, match="carries no HDR metadata"):
            _run(executor, _job(tmp_path, params), tmp_path)

    def test_passthrough_does_not_require_the_tool(self, tmp_path: Path) -> None:
        executor, mocks = _make_executor()
        _run(executor, _job(tmp_path, _hdr10plus_params(passthrough=True)), tmp_path)

        mocks.video_copier.copy_video.assert_called_once()

    def test_extraction_failure_raises(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor(rc=3, write=False)
        executor, _mocks = _make_executor(hdr10plus_processor=processor)
        with pytest.raises(RuntimeError, match="HDR10\\+ metadata extraction failed"):
            _run(executor, _job(tmp_path, _hdr10plus_params()), tmp_path)

    def test_passthrough_skips_extraction(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor()
        executor, mocks = _make_executor(hdr10plus_processor=processor)
        _run(executor, _job(tmp_path, _hdr10plus_params(passthrough=True)), tmp_path)

        processor.extract.assert_not_called()
        mocks.video_copier.copy_video.assert_called_once()

    def test_shutdown_before_extraction(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor()
        executor, mocks = _make_executor(hdr10plus_processor=processor)
        executor._shutdown_event.set()
        _run(executor, _job(tmp_path, _hdr10plus_params()), tmp_path)

        processor.extract.assert_not_called()
        mocks.encoder.encode.assert_not_called()

    def test_shutdown_during_subtitles_stops_before_extraction(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor()
        executor, mocks = _make_executor(hdr10plus_processor=processor)

        def extract_and_shutdown(src: Any, idx: Any, out: Any, on_progress: Any = None) -> int:
            executor._shutdown_event.set()
            return 0

        mocks.audio_extractor.extract_track.side_effect = extract_and_shutdown
        job = _job(
            tmp_path,
            _hdr10plus_params(),
            subtitles=[make_subtitle_instruction(action=SubtitleAction.COPY, stream_index=3)],
        )
        _run(executor, job, tmp_path)

        processor.extract.assert_not_called()
        mocks.encoder.encode.assert_not_called()

    def test_extraction_is_narrated(self, tmp_path: Path) -> None:
        progress = MagicMock()
        processor = _make_hdr10plus_processor()
        executor, _mocks = _make_executor(hdr10plus_processor=processor, progress=progress)
        _run(executor, _job(tmp_path, _hdr10plus_params()), tmp_path)

        status_calls = [call[0][0] for call in progress.update_status.call_args_list]
        assert any("Extracting HDR10+ metadata" in status for status in status_calls)
        tool_lines = [call[0][0] for call in progress.add_tool_line.call_args_list]
        assert any("Extracting HDR10+ metadata" in line for line in tool_lines)


class TestHdr10PlusJsonVerification:
    def test_missing_scene_info_raises(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor(payload=json.dumps({"JSONInfo": {}}))
        executor, _mocks = _make_executor(hdr10plus_processor=processor)
        with pytest.raises(RuntimeError, match="no per-frame HDR10\\+ metadata"):
            _run(executor, _job(tmp_path, _hdr10plus_params()), tmp_path)

    def test_empty_scene_info_raises(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor(frames=0)
        executor, _mocks = _make_executor(hdr10plus_processor=processor)
        with pytest.raises(RuntimeError, match="no per-frame HDR10\\+ metadata"):
            _run(executor, _job(tmp_path, _hdr10plus_params()), tmp_path)

    def test_unreadable_json_raises(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor(payload="{not json")
        executor, _mocks = _make_executor(hdr10plus_processor=processor)
        with pytest.raises(RuntimeError, match="unreadable"):
            _run(executor, _job(tmp_path, _hdr10plus_params()), tmp_path)

    def test_absent_json_raises(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor(write=False)
        executor, _mocks = _make_executor(hdr10plus_processor=processor)
        with pytest.raises(RuntimeError, match="unreadable"):
            _run(executor, _job(tmp_path, _hdr10plus_params()), tmp_path)

    def test_non_object_json_raises(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor(payload=json.dumps(["SceneInfo"]))
        executor, _mocks = _make_executor(hdr10plus_processor=processor)
        with pytest.raises(RuntimeError, match="not a JSON object"):
            _run(executor, _job(tmp_path, _hdr10plus_params()), tmp_path)

    def test_scalar_scene_info_raises(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor(payload=json.dumps({"SceneInfo": 42}))
        executor, _mocks = _make_executor(hdr10plus_processor=processor)
        with pytest.raises(RuntimeError, match="SceneInfo is not a list"):
            _run(executor, _job(tmp_path, _hdr10plus_params()), tmp_path)

    def test_truncated_metadata_raises(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor(frames=1200)
        executor, _mocks = _make_executor(hdr10plus_processor=processor)
        with pytest.raises(RuntimeError, match="covers 1200 frames"):
            _run(executor, _job(tmp_path, _hdr10plus_params()), tmp_path)

    def test_frame_floor_slack_accepted(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor(frames=_SOURCE_FRAMES - 5)
        executor, mocks = _make_executor(hdr10plus_processor=processor)
        _run(executor, _job(tmp_path, _hdr10plus_params()), tmp_path)

        mocks.encoder.encode.assert_called_once()

    def test_one_frame_past_the_floor_raises(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor(frames=_SOURCE_FRAMES - 6)
        executor, _mocks = _make_executor(hdr10plus_processor=processor)
        with pytest.raises(RuntimeError, match="covers 2394 frames"):
            _run(executor, _job(tmp_path, _hdr10plus_params()), tmp_path)

    def test_ratio_slack_accepted(self, tmp_path: Path) -> None:
        expected = _LONG_SOURCE_FRAMES
        processor = _make_hdr10plus_processor(payload=_light_payload(expected - 12))
        executor, mocks = _make_executor(hdr10plus_processor=processor)
        _run(executor, _job(tmp_path, _hdr10plus_params(), duration_s=_LONG_DURATION_S), tmp_path)

        mocks.encoder.encode.assert_called_once()

    def test_one_frame_past_the_ratio_raises(self, tmp_path: Path) -> None:
        expected = _LONG_SOURCE_FRAMES
        processor = _make_hdr10plus_processor(payload=_light_payload(expected - 13))
        executor, _mocks = _make_executor(hdr10plus_processor=processor)
        with pytest.raises(RuntimeError, match="covers 11987 frames"):
            _run(executor, _job(tmp_path, _hdr10plus_params(), duration_s=_LONG_DURATION_S), tmp_path)

    def test_slight_surplus_accepted(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor(frames=_SOURCE_FRAMES + 5)
        executor, mocks = _make_executor(hdr10plus_processor=processor)
        _run(executor, _job(tmp_path, _hdr10plus_params()), tmp_path)

        mocks.encoder.encode.assert_called_once()

    def test_surplus_past_the_tolerance_raises(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor(frames=_SOURCE_FRAMES + 6)
        executor, _mocks = _make_executor(hdr10plus_processor=processor)
        with pytest.raises(RuntimeError, match="covers 2406 frames"):
            _run(executor, _job(tmp_path, _hdr10plus_params()), tmp_path)

    def test_large_surplus_raises(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor(payload=_light_payload(_SOURCE_FRAMES * 2))
        executor, _mocks = _make_executor(hdr10plus_processor=processor)
        with pytest.raises(RuntimeError, match="covers 4800 frames"):
            _run(executor, _job(tmp_path, _hdr10plus_params()), tmp_path)

    def test_unknown_duration_raises(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor()
        executor, _mocks = _make_executor(hdr10plus_processor=processor)
        with pytest.raises(RuntimeError, match="cannot be verified"):
            _run(executor, _job(tmp_path, _hdr10plus_params(), duration_s=0.0), tmp_path)

    def test_unknown_frame_rate_raises(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor()
        executor, _mocks = _make_executor(hdr10plus_processor=processor)
        params = _hdr10plus_params()
        params.fps_den = 0
        with pytest.raises(RuntimeError, match="cannot be verified"):
            _run(executor, _job(tmp_path, params), tmp_path)


class TestEncodedHdrMetadataVerification:
    def test_hdr10_plus_present_passes(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor()
        executor, mocks = _make_executor(
            hdr10plus_processor=processor,
            side_data=[_HDR10_PLUS_SIDE_DATA],
        )
        _run(executor, _job(tmp_path, _hdr10plus_params()), tmp_path)

        assert mocks.prober.probe_hdr_side_data_strict.call_args_list[0].args[0] == tmp_path / "video.obu"

    def test_hdr10_plus_absent_raises(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor()
        executor, mocks = _make_executor(hdr10plus_processor=processor, side_data=[])
        with pytest.raises(RuntimeError, match="HDR10\\+"):
            _run(executor, _job(tmp_path, _hdr10plus_params()), tmp_path)

        mocks.muxer.mux.assert_not_called()

    def test_dolby_vision_present_passes(self, tmp_path: Path) -> None:
        dovi = MagicMock()
        dovi.extract_rpu.return_value = 0
        executor, mocks = _make_executor(
            dovi_processor=dovi,
            side_data=[_DOLBY_VISION_SIDE_DATA],
        )
        _run(executor, _job(tmp_path, make_video_params(dv_mode=DvMode.COPY)), tmp_path)

        mocks.muxer.mux.assert_called_once()

    def test_dolby_vision_absent_raises(self, tmp_path: Path) -> None:
        dovi = MagicMock()
        dovi.extract_rpu.return_value = 0
        executor, _mocks = _make_executor(dovi_processor=dovi, side_data=[])
        with pytest.raises(RuntimeError, match="Dolby Vision"):
            _run(executor, _job(tmp_path, make_video_params(dv_mode=DvMode.COPY)), tmp_path)

    def test_mastering_display_present_passes(self, tmp_path: Path) -> None:
        hdr = HdrMetadata(mastering_display=_MASTERING_DISPLAY)
        executor, mocks = _make_executor(side_data=[_MASTERING_DISPLAY_SIDE_DATA])
        _run(executor, _job(tmp_path, make_video_params(hdr=hdr)), tmp_path)

        mocks.muxer.mux.assert_called_once()

    def test_mastering_display_absent_raises(self, tmp_path: Path) -> None:
        hdr = HdrMetadata(mastering_display=_MASTERING_DISPLAY)
        executor, _mocks = _make_executor(side_data=[_CONTENT_LIGHT_SIDE_DATA])
        with pytest.raises(RuntimeError, match="mastering display"):
            _run(executor, _job(tmp_path, make_video_params(hdr=hdr)), tmp_path)

    def test_content_light_present_passes(self, tmp_path: Path) -> None:
        hdr = HdrMetadata(content_light="MaxCLL=1000,MaxFALL=147")
        executor, mocks = _make_executor(side_data=[_CONTENT_LIGHT_SIDE_DATA])
        _run(executor, _job(tmp_path, make_video_params(hdr=hdr)), tmp_path)

        mocks.muxer.mux.assert_called_once()

    def test_content_light_absent_raises(self, tmp_path: Path) -> None:
        hdr = HdrMetadata(content_light="MaxCLL=1000,MaxFALL=147")
        executor, _mocks = _make_executor(side_data=[_MASTERING_DISPLAY_SIDE_DATA])
        with pytest.raises(RuntimeError, match="content light level"):
            _run(executor, _job(tmp_path, make_video_params(hdr=hdr)), tmp_path)

    def test_zero_content_light_is_required(self, tmp_path: Path) -> None:
        hdr = HdrMetadata(content_light="MaxCLL=0,MaxFALL=0")
        executor, mocks = _make_executor(side_data=[_CONTENT_LIGHT_SIDE_DATA])
        _run(executor, _job(tmp_path, make_video_params(hdr=hdr)), tmp_path)

        mocks.muxer.mux.assert_called_once()

    def test_unparseable_content_light_is_not_required(self, tmp_path: Path) -> None:
        hdr = HdrMetadata(
            mastering_display=_MASTERING_DISPLAY,
            content_light="MaxCLL=,MaxFALL=147",
        )
        executor, mocks = _make_executor(side_data=[_MASTERING_DISPLAY_SIDE_DATA])
        _run(executor, _job(tmp_path, make_video_params(hdr=hdr)), tmp_path)

        mocks.muxer.mux.assert_called_once()

    def test_probe_failure_is_not_reported_as_lost_metadata(self, tmp_path: Path) -> None:
        hdr = HdrMetadata(mastering_display=_MASTERING_DISPLAY)
        executor, mocks = _make_executor()
        mocks.prober.probe_hdr_side_data_strict.side_effect = RuntimeError(
            "HDR side-data probe failed (rc=1) for video.obu"
        )
        with pytest.raises(RuntimeError, match="probe failed"):
            _run(executor, _job(tmp_path, make_video_params(hdr=hdr)), tmp_path)

        mocks.muxer.mux.assert_not_called()

    def test_message_names_every_missing_kind(self, tmp_path: Path) -> None:
        hdr = HdrMetadata(
            mastering_display=_MASTERING_DISPLAY,
            content_light="MaxCLL=1000,MaxFALL=147",
            is_hdr10_plus=True,
        )
        dovi = MagicMock()
        dovi.extract_rpu.return_value = 0
        processor = _make_hdr10plus_processor()
        executor, _mocks = _make_executor(
            hdr10plus_processor=processor,
            dovi_processor=dovi,
            side_data=[],
        )
        params = _hdr10plus_params(hdr=hdr, dv_mode=DvMode.COPY)
        with pytest.raises(RuntimeError) as excinfo:
            _run(executor, _job(tmp_path, params), tmp_path)

        message = str(excinfo.value)
        assert "mastering display" in message
        assert "content light level" in message
        assert "Dolby Vision" in message
        assert "HDR10+" in message

    def test_all_kinds_present_passes(self, tmp_path: Path) -> None:
        hdr = HdrMetadata(
            mastering_display=_MASTERING_DISPLAY,
            content_light="MaxCLL=1000,MaxFALL=147",
            is_hdr10_plus=True,
        )
        dovi = MagicMock()
        dovi.extract_rpu.return_value = 0
        processor = _make_hdr10plus_processor()
        executor, mocks = _make_executor(
            hdr10plus_processor=processor,
            dovi_processor=dovi,
            side_data=[
                _MASTERING_DISPLAY_SIDE_DATA,
                _CONTENT_LIGHT_SIDE_DATA,
                _HDR10_PLUS_SIDE_DATA,
                _DOLBY_VISION_SIDE_DATA,
            ],
        )
        params = _hdr10plus_params(hdr=hdr, dv_mode=DvMode.COPY)
        _run(executor, _job(tmp_path, params), tmp_path)

        mocks.muxer.mux.assert_called_once()

    def test_sdr_job_is_not_probed(self, tmp_path: Path) -> None:
        executor, mocks = _make_executor()
        _run(executor, _job(tmp_path, make_video_params()), tmp_path)

        mocks.prober.probe_hdr_side_data_strict.assert_not_called()

    def test_passthrough_is_not_probed(self, tmp_path: Path) -> None:
        hdr = HdrMetadata(mastering_display=_MASTERING_DISPLAY)
        executor, mocks = _make_executor()
        _run(executor, _job(tmp_path, make_video_params(hdr=hdr, passthrough=True)), tmp_path)

        mocks.prober.probe_hdr_side_data_strict.assert_not_called()

    def test_sdr_svt_grain_path_is_not_probed(self, tmp_path: Path) -> None:
        grain_encoder = MagicMock()
        executor, mocks = _make_executor(grain_encoder=grain_encoder)
        params = make_video_params(grain=True, source_width=720, source_height=576)
        _run(executor, _job(tmp_path, params), tmp_path)

        grain_encoder.encode.assert_called_once()
        mocks.prober.probe_hdr_side_data_strict.assert_not_called()


class TestHdrOnGrainEncoderRefused:
    def test_hdr_transfer_on_the_grain_encoder_raises(self, tmp_path: Path) -> None:
        hdr = HdrMetadata(mastering_display=_MASTERING_DISPLAY)
        grain_encoder = MagicMock()
        executor, mocks = _make_executor(grain_encoder=grain_encoder)
        params = make_video_params(
            hdr=hdr,
            color_transfer="smpte2084",
            grain=True,
            source_width=720,
            source_height=576,
        )
        with pytest.raises(RuntimeError, match="carries no HDR metadata"):
            _run(executor, _job(tmp_path, params), tmp_path)

        grain_encoder.encode.assert_not_called()
        mocks.muxer.mux.assert_not_called()

    def test_dolby_vision_on_the_grain_encoder_raises(self, tmp_path: Path) -> None:
        dovi = MagicMock()
        dovi.extract_rpu.return_value = 0
        grain_encoder = MagicMock()
        executor, _mocks = _make_executor(dovi_processor=dovi, grain_encoder=grain_encoder)
        params = make_video_params(
            dv_mode=DvMode.COPY,
            grain=True,
            source_width=720,
            source_height=576,
        )
        with pytest.raises(RuntimeError, match="carries no HDR metadata"):
            _run(executor, _job(tmp_path, params), tmp_path)

        grain_encoder.encode.assert_not_called()

    def test_hdr10_plus_on_the_grain_encoder_raises(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor()
        grain_encoder = MagicMock()
        executor, _mocks = _make_executor(hdr10plus_processor=processor, grain_encoder=grain_encoder)
        params = make_video_params(
            hdr=HdrMetadata(is_hdr10_plus=True),
            grain=True,
            source_width=720,
            source_height=576,
        )
        with pytest.raises(RuntimeError, match="carries no HDR metadata"):
            _run(executor, _job(tmp_path, params), tmp_path)

        grain_encoder.encode.assert_not_called()
        processor.extract.assert_not_called()

    def test_sdr_with_stray_content_light_still_encodes_on_svt(self, tmp_path: Path) -> None:
        hdr = HdrMetadata(content_light="MaxCLL=1000,MaxFALL=400")
        grain_encoder = MagicMock()
        executor, mocks = _make_executor(grain_encoder=grain_encoder)
        params = make_video_params(
            hdr=hdr,
            color_transfer="bt709",
            grain=True,
            source_width=720,
            source_height=576,
        )
        _run(executor, _job(tmp_path, params), tmp_path)

        grain_encoder.encode.assert_called_once()
        mocks.muxer.mux.assert_called_once()
        mocks.prober.probe_hdr_side_data_strict.assert_not_called()

    def test_grain_encoder_guard_fires_before_audio_work(self, tmp_path: Path) -> None:
        grain_encoder = MagicMock()
        executor, mocks = _make_executor(grain_encoder=grain_encoder)
        params = make_video_params(
            dv_mode=DvMode.COPY,
            grain=True,
            source_width=720,
            source_height=576,
        )
        job = _job(tmp_path, params)
        job.audio = [make_audio_instruction(action=AudioAction.COPY, codec_name="aac", stream_index=1)]
        with pytest.raises(RuntimeError, match="carries no HDR metadata"):
            _run(executor, job, tmp_path)

        mocks.audio_extractor.extract_track.assert_not_called()


class TestDeliveredFileVerification:
    def test_delivered_file_is_verified_after_the_move(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor()
        executor, mocks = _make_executor(hdr10plus_processor=processor)
        job = _job(tmp_path, _hdr10plus_params())
        _run(executor, job, tmp_path)

        probed = [call.args[0] for call in mocks.prober.probe_hdr_side_data_strict.call_args_list]
        assert probed == [tmp_path / "video.obu", Path(job.output_file)]

    def test_metadata_lost_in_the_container_raises(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor()
        executor, mocks = _make_executor(hdr10plus_processor=processor)
        mocks.prober.probe_hdr_side_data_strict.side_effect = [[_HDR10_PLUS_SIDE_DATA], []]
        job = _job(tmp_path, _hdr10plus_params())
        with pytest.raises(RuntimeError, match="HDR10\\+"):
            _run(executor, job, tmp_path)

        mocks.muxer.mux.assert_called_once()

    def test_unverifiable_delivered_file_is_removed(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor()
        executor, mocks = _make_executor(hdr10plus_processor=processor)
        mocks.prober.probe_hdr_side_data_strict.side_effect = [[_HDR10_PLUS_SIDE_DATA], []]
        job = _job(tmp_path, _hdr10plus_params())
        with pytest.raises(HdrMetadataLostError):
            _run(executor, job, tmp_path)

        assert not Path(job.output_file).exists()

    def test_delivered_probe_failure_moves_the_file_aside(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor()
        executor, mocks = _make_executor(hdr10plus_processor=processor)
        mocks.prober.probe_hdr_side_data_strict.side_effect = [
            [_HDR10_PLUS_SIDE_DATA],
            RuntimeError("HDR side-data probe failed (rc=1) for movie.mkv"),
        ]
        job = _job(tmp_path, _hdr10plus_params())
        with pytest.raises(RuntimeError, match="probe failed") as excinfo:
            _run(executor, job, tmp_path)

        output_path = Path(job.output_file)
        quarantined = output_path.with_name(output_path.name + ".unverified")
        assert not output_path.exists()
        assert quarantined.exists()
        assert str(quarantined) in str(excinfo.value)

    def test_unmovable_output_keeps_the_probe_error(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor()
        executor, mocks = _make_executor(hdr10plus_processor=processor)
        mocks.prober.probe_hdr_side_data_strict.side_effect = [
            [_HDR10_PLUS_SIDE_DATA],
            RuntimeError("HDR side-data probe failed (rc=1) for movie.mkv"),
        ]
        job = _job(tmp_path, _hdr10plus_params())

        def refuse_rename(self: Path, target: Any) -> Path:
            raise PermissionError("file in use")

        with patch.object(Path, "rename", refuse_rename):
            with pytest.raises(RuntimeError, match="probe failed"):
                _run(executor, job, tmp_path)

        output_path = Path(job.output_file)
        assert output_path.exists()
        assert not output_path.with_name(output_path.name + ".unverified").exists()

    def test_undeletable_output_keeps_the_metadata_error(self, tmp_path: Path) -> None:
        processor = _make_hdr10plus_processor()
        executor, mocks = _make_executor(hdr10plus_processor=processor)
        mocks.prober.probe_hdr_side_data_strict.side_effect = [[_HDR10_PLUS_SIDE_DATA], []]
        job = _job(tmp_path, _hdr10plus_params())

        def refuse_unlink(self: Path, missing_ok: bool = False) -> None:
            raise PermissionError("file in use")

        with patch.object(Path, "unlink", refuse_unlink):
            with pytest.raises(HdrMetadataLostError, match="lost HDR metadata"):
                _run(executor, job, tmp_path)

        assert Path(job.output_file).exists()

    def test_passthrough_delivery_is_not_probed(self, tmp_path: Path) -> None:
        executor, mocks = _make_executor()
        _run(executor, _job(tmp_path, _hdr10plus_params(passthrough=True)), tmp_path)

        mocks.prober.probe_hdr_side_data_strict.assert_not_called()
