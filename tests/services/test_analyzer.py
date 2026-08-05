from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from furnace.core.audio_profile import AudioMetrics, Verdict
from furnace.core.models import (
    AnalysisOutcome,
    AnalyzeStatus,
    AudioCodecId,
    HdrMetadata,
    ScanResult,
    SubtitleCodecId,
    TrackType,
)
from furnace.services.analyzer import _TEXT_SUBTITLE_CODECS, Analyzer


def make_prober(
    probe_data: dict[str, Any] | None = None,
    encoder_tag: str | None = None,
    hdr_side_data: list[dict[str, Any]] | None = None,
) -> MagicMock:
    prober = MagicMock()
    prober.get_encoder_tag.return_value = encoder_tag
    prober.probe.return_value = probe_data or {}
    prober.run_idet.return_value = 0.0
    prober.probe_hdr_side_data.return_value = hdr_side_data or []
    prober.sample_repeat_pict.return_value = []
    prober.sample_grain.return_value = [0.8, 0.9, 0.7, 0.8, 0.8]
    prober.sample_field_pairing.return_value = (0, 0)
    return prober


def make_scan_result(tmp_path: Path, filename: str = "movie.mkv") -> ScanResult:
    main_file = tmp_path / filename
    main_file.write_bytes(b"\x00" * 1024)
    output_path = tmp_path / "out" / filename
    return ScanResult(
        main_file=main_file,
        satellite_files=[],
        output_path=output_path,
    )


def _h264_probe_data() -> dict[str, Any]:
    return {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "24000/1001",
                "duration": "5400.0",
                "field_order": "progressive",
                "pix_fmt": "yuv420p",
                "color_space": "bt709",
                "color_primaries": "bt709",
                "color_transfer": "bt709",
                "color_range": "tv",
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "profile": "LC",
                "channels": 2,
                "channel_layout": "stereo",
                "sample_rate": "48000",
                "tags": {
                    "language": "eng",
                    "title": "English Stereo",
                },
                "disposition": {"default": 1, "forced": 0},
            },
            {
                "index": 2,
                "codec_type": "audio",
                "codec_name": "ac3",
                "channels": 6,
                "channel_layout": "5.1(side)",
                "sample_rate": "48000",
                "bit_rate": "640000",
                "tags": {
                    "language": "rus",
                    "title": "",
                },
                "disposition": {"default": 0, "forced": 0},
            },
            {
                "index": 3,
                "codec_type": "subtitle",
                "codec_name": "hdmv_pgs_subtitle",
                "tags": {
                    "language": "rus",
                    "title": "",
                    "NUMBER_OF_FRAMES": "120",
                },
                "disposition": {"default": 0, "forced": 0},
            },
        ],
        "format": {
            "duration": "5400.0",
        },
        "chapters": [{"id": 0, "start_time": "0.0", "end_time": "600.0", "tags": {"title": "Chapter 1"}}],
    }


def _dv_probe_data() -> dict[str, Any]:
    return {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "dvhe",
                "width": 3840,
                "height": 2160,
                "avg_frame_rate": "24/1",
                "duration": "5400.0",
                "field_order": "progressive",
                "pix_fmt": "yuv420p10le",
                "color_space": "bt2020nc",
                "color_primaries": "bt2020",
                "color_transfer": "smpte2084",
                "color_range": "tv",
                "side_data_list": [
                    {"side_data_type": "DOVI configuration record"},
                ],
            },
        ],
        "format": {"duration": "5400.0"},
        "chapters": [],
    }


class TestAnalyzerParsesTracks:
    def test_analyzer_parses_tracks(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_h264_probe_data())

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            analyzer = Analyzer(prober=prober)
            outcome = analyzer.analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        assert outcome.detail == "h264 1920x1080 23fps SDR, 2 audio (eng,rus), 1 subs"
        movie = outcome.movie
        assert movie is not None
        assert movie.video.codec_name == "h264"
        assert movie.video.width == 1920
        assert movie.video.height == 1080
        assert len(movie.audio_tracks) == 2
        eng_track = next(t for t in movie.audio_tracks if t.language == "eng")
        rus_track = next(t for t in movie.audio_tracks if t.language == "rus")
        assert eng_track.codec_id == AudioCodecId.AAC_LC
        assert eng_track.track_type == TrackType.AUDIO
        assert rus_track.codec_id == AudioCodecId.AC3
        assert rus_track.channels == 6
        assert len(movie.subtitle_tracks) == 1
        sub = movie.subtitle_tracks[0]
        assert sub.codec_id == SubtitleCodecId.PGS
        assert sub.language == "rus"
        assert movie.has_chapters is True

    def test_analyzer_parses_audio_language_and_disposition(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_h264_probe_data())

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            analyzer = Analyzer(prober=prober)
            outcome = analyzer.analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        eng_track = next(t for t in movie.audio_tracks if t.language == "eng")
        assert eng_track.is_default is True
        rus_track = next(t for t in movie.audio_tracks if t.language == "rus")
        assert rus_track.is_default is False


class TestAnalyzerDVProceeds:
    def test_analyzer_dv_returns_movie(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_dv_probe_data())

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            with patch("furnace.services.analyzer.detect_hdr") as mock_detect_hdr:
                mock_detect_hdr.return_value = HdrMetadata(is_dolby_vision=True)
                with patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None):
                    analyzer = Analyzer(prober=prober)
                    outcome = analyzer.analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert movie.video.codec_name == "dvhe"


class TestAnalyzerHdrSideDataMerge:
    def test_stream_dovi_and_frame_mdcv_cll_both_detected(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        probe_data = _dv_probe_data()
        frame_side_data: list[dict[str, Any]] = [
            {
                "side_data_type": "Mastering display metadata",
                "red_x": "35400/50000",
                "red_y": "14600/50000",
                "green_x": "8500/50000",
                "green_y": "39850/50000",
                "blue_x": "6550/50000",
                "blue_y": "2300/50000",
                "white_point_x": "15635/50000",
                "white_point_y": "16450/50000",
                "min_luminance": "50/10000",
                "max_luminance": "40000000/10000",
            },
            {
                "side_data_type": "Content light level metadata",
                "max_content": 4342,
                "max_average": 2342,
            },
            {"side_data_type": "Dolby Vision RPU Data"},
        ]
        probe_data["streams"][0]["side_data_list"] = [
            {
                "side_data_type": "DOVI configuration record",
                "dv_profile": 7,
                "dv_bl_signal_compatibility_id": 0,
            }
        ]
        prober = make_prober(probe_data=probe_data, hdr_side_data=frame_side_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            with patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None):
                analyzer = Analyzer(prober=prober)
                outcome = analyzer.analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        hdr = movie.video.hdr
        assert hdr.is_dolby_vision is True
        assert hdr.dv_profile == 7
        assert hdr.mastering_display is not None
        assert "L(40000000,50)" in hdr.mastering_display
        assert hdr.content_light == "MaxCLL=4342,MaxFALL=2342"
        prober.probe_hdr_side_data.assert_called_once()

    def test_sdr_skips_frame_side_data_probe(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_h264_probe_data())

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            analyzer = Analyzer(prober=prober)
            analyzer.analyze(scan_result)

        prober.probe_hdr_side_data.assert_not_called()


class TestAnalyzerHDR10PlusError:
    def test_analyzer_hdr10plus_returns_failed(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_dv_probe_data())

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            with patch("furnace.services.analyzer.detect_hdr") as mock_detect_hdr:
                mock_detect_hdr.return_value = HdrMetadata(is_hdr10_plus=True)
                analyzer = Analyzer(prober=prober)
                outcome = analyzer.analyze(scan_result)

        assert outcome == AnalysisOutcome(None, AnalyzeStatus.FAILED, "HDR10+ not supported")


class TestAnalyzerHdrSummary:
    @staticmethod
    def _probe_with_transfer(transfer: str) -> dict[str, Any]:
        data = _h264_probe_data()
        data["streams"][0]["color_transfer"] = transfer
        return data

    def test_smpte2084_transfer_summarised_as_hdr10(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=self._probe_with_transfer("smpte2084"))

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            analyzer = Analyzer(prober=prober)
            outcome = analyzer.analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        assert outcome.detail == "h264 1920x1080 23fps HDR10, 2 audio (eng,rus), 1 subs"

    def test_hlg_transfer_summarised_as_hlg(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=self._probe_with_transfer("arib-std-b67"))

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            analyzer = Analyzer(prober=prober)
            outcome = analyzer.analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        assert outcome.detail == "h264 1920x1080 23fps HLG, 2 audio (eng,rus), 1 subs"

    def test_dolby_vision_summary_locks_profile_and_bl(self, tmp_path: Path) -> None:
        data = _dv_probe_data()
        data["streams"][0]["side_data_list"] = [
            {
                "side_data_type": "DOVI configuration record",
                "dv_profile": 8,
                "dv_bl_signal_compatibility_id": 1,
            },
        ]
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            analyzer = Analyzer(prober=prober)
            outcome = analyzer.analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        assert outcome.detail == "dvhe 3840x2160 24fps DV P8 (BL=HDR10), 0 audio, 0 subs"

    def test_interlaced_marker_in_summary(self, tmp_path: Path) -> None:
        data = _h264_probe_data()
        data["streams"][0]["field_order"] = "tt"
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            analyzer = Analyzer(prober=prober)
            outcome = analyzer.analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        assert outcome.detail == "h264 1920x1080 23fps SDR (interlaced), 2 audio (eng,rus), 1 subs"


class TestAnalyzerProgress:
    def test_on_progress_reports_stage_fractions(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_h264_probe_data())
        fractions: list[float] = []

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            analyzer = Analyzer(prober=prober)
            outcome = analyzer.analyze(scan_result, on_progress=fractions.append)

        assert outcome.status is AnalyzeStatus.DONE
        assert fractions == pytest.approx([1 / 3, 2 / 3, 1.0, 1.0])


class TestAnalyzerDelay:
    def test_analyzer_delay_from_start_pts(self, tmp_path: Path) -> None:
        probe_data = {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "24/1",
                    "duration": "100.0",
                    "field_order": "progressive",
                    "pix_fmt": "yuv420p",
                },
                {
                    "index": 1,
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "profile": "LC",
                    "channels": 2,
                    "sample_rate": "48000",
                    "start_pts": 500,
                    "tags": {"language": "eng"},
                    "disposition": {"default": 1, "forced": 0},
                },
            ],
            "format": {"duration": "100.0"},
            "chapters": [],
        }
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=probe_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            with patch("furnace.services.analyzer.detect_hdr", return_value=HdrMetadata()):
                with patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None):
                    analyzer = Analyzer(prober=prober)
                    outcome = analyzer.analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert movie.audio_tracks[0].delay_ms == 500

    def test_analyzer_delay_fallback_start_time(self, tmp_path: Path) -> None:
        probe_data = {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "24/1",
                    "duration": "100.0",
                    "field_order": "progressive",
                    "pix_fmt": "yuv420p",
                },
                {
                    "index": 1,
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "profile": "LC",
                    "channels": 2,
                    "sample_rate": "48000",
                    "start_time": "0.5",
                    "tags": {"language": "eng"},
                    "disposition": {"default": 1, "forced": 0},
                },
            ],
            "format": {"duration": "100.0"},
            "chapters": [],
        }
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=probe_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            with patch("furnace.services.analyzer.detect_hdr", return_value=HdrMetadata()):
                with patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None):
                    analyzer = Analyzer(prober=prober)
                    outcome = analyzer.analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert movie.audio_tracks[0].delay_ms == 500

    def test_analyzer_delay_default(self, tmp_path: Path) -> None:
        probe_data = {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "24/1",
                    "duration": "100.0",
                    "field_order": "progressive",
                    "pix_fmt": "yuv420p",
                },
                {
                    "index": 1,
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "profile": "LC",
                    "channels": 2,
                    "sample_rate": "48000",
                    "tags": {"language": "eng"},
                    "disposition": {"default": 1, "forced": 0},
                },
            ],
            "format": {"duration": "100.0"},
            "chapters": [],
        }
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=probe_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            with patch("furnace.services.analyzer.detect_hdr", return_value=HdrMetadata()):
                with patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None):
                    analyzer = Analyzer(prober=prober)
                    outcome = analyzer.analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert movie.audio_tracks[0].delay_ms == 0


class TestAnalyzerDelayDirect:
    def test_detect_delay_from_start_pts(self) -> None:
        prober = MagicMock()
        analyzer = Analyzer(prober=prober)
        result = analyzer._detect_audio_delay({"start_pts": 500})
        assert result == 500

    def test_detect_delay_fallback_start_time(self) -> None:
        prober = MagicMock()
        analyzer = Analyzer(prober=prober)
        result = analyzer._detect_audio_delay({"start_time": "0.5"})
        assert result == 500

    def test_detect_delay_default(self) -> None:
        prober = MagicMock()
        analyzer = Analyzer(prober=prober)
        result = analyzer._detect_audio_delay({})
        assert result == 0

    def test_detect_delay_start_pts_takes_priority(self) -> None:
        prober = MagicMock()
        analyzer = Analyzer(prober=prober)
        result = analyzer._detect_audio_delay({"start_pts": 100, "start_time": "5.0"})
        assert result == 100


class TestAnalyzeEarlyReturns:
    def test_skip_when_should_skip_file_returns_true(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_h264_probe_data(), encoder_tag="Furnace 1.0")

        analyzer = Analyzer(prober=prober)
        outcome = analyzer.analyze(scan_result)

        assert outcome.movie is None
        assert outcome.status is AnalyzeStatus.SKIPPED
        assert outcome.detail == "file already encoded by Furnace (tag: Furnace 1.0)"

    def test_skip_when_output_exists(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        scan_result.output_path.parent.mkdir(parents=True, exist_ok=True)
        scan_result.output_path.write_bytes(b"x")
        prober = make_prober(probe_data=_h264_probe_data())

        analyzer = Analyzer(prober=prober)
        outcome = analyzer.analyze(scan_result)

        assert outcome.movie is None
        assert outcome.status is AnalyzeStatus.SKIPPED
        assert outcome.detail == f"output file already exists: {scan_result.output_path}"

    def test_force_processes_furnace_tagged_file(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_h264_probe_data(), encoder_tag="Furnace 1.17.0")

        analyzer = Analyzer(prober=prober, force=True)
        outcome = analyzer.analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        result = outcome.movie
        assert result is not None

    def test_force_processes_when_output_exists(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        scan_result.output_path.parent.mkdir(parents=True, exist_ok=True)
        scan_result.output_path.write_bytes(b"x")
        prober = make_prober(probe_data=_h264_probe_data())

        analyzer = Analyzer(prober=prober, force=True)
        outcome = analyzer.analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        result = outcome.movie
        assert result is not None

    def test_probe_raises_oserror(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober()
        prober.probe.side_effect = OSError("disk failure")

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            analyzer = Analyzer(prober=prober)
            outcome = analyzer.analyze(scan_result)

        assert outcome.movie is None
        assert outcome.status is AnalyzeStatus.FAILED
        assert outcome.detail == "probe failed"

    def test_probe_raises_runtime_error(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober()
        prober.probe.side_effect = RuntimeError("ffprobe crash")

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.movie is None
        assert outcome.status is AnalyzeStatus.FAILED
        assert outcome.detail == "probe failed"

    def test_probe_raises_value_error(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober()
        prober.probe.side_effect = ValueError("bad data")

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.movie is None
        assert outcome.status is AnalyzeStatus.FAILED
        assert outcome.detail == "probe failed"

    def test_no_video_stream(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(
            probe_data={
                "streams": [
                    {
                        "index": 0,
                        "codec_type": "audio",
                        "codec_name": "aac",
                        "profile": "LC",
                        "channels": 2,
                        "sample_rate": "48000",
                        "tags": {"language": "eng"},
                        "disposition": {"default": 1, "forced": 0},
                    },
                ],
                "format": {"duration": "100.0"},
                "chapters": [],
            }
        )

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.movie is None
        assert outcome.status is AnalyzeStatus.SKIPPED
        assert outcome.detail == "no video stream"

    def test_parse_video_info_raises_key_error(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(
            probe_data={
                "streams": [{"index": 0, "codec_type": "video", "codec_name": "h264"}],
                "format": {},
                "chapters": [],
            }
        )

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            analyzer = Analyzer(prober=prober)
            with patch.object(analyzer, "_parse_video_info", side_effect=KeyError("missing")):
                outcome = analyzer.analyze(scan_result)

        assert outcome.movie is None
        assert outcome.status is AnalyzeStatus.FAILED
        assert outcome.detail == "parse failed"

    def test_check_unsupported_codecs_returns_warning(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_h264_probe_data())

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch(
                "furnace.services.analyzer.check_unsupported_codecs",
                return_value="unsupported codecs detected: audio stream #1",
            ),
        ):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.movie is None
        assert outcome.status is AnalyzeStatus.SKIPPED
        assert outcome.detail == "unsupported codecs detected: audio stream #1"


class TestParseVideoInfoFallbacks:
    def _make_base_stream(self) -> dict[str, Any]:
        return {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": "24000/1001",
            "duration": "100.0",
            "field_order": "progressive",
            "pix_fmt": "yuv420p",
        }

    def test_fps_non_fraction_string(self, tmp_path: Path) -> None:
        stream = self._make_base_stream()
        stream["avg_frame_rate"] = "25"
        prober = MagicMock()
        prober.probe_hdr_side_data.return_value = []

        analyzer = Analyzer(prober=prober)
        vi = analyzer._parse_video_info(stream, {}, tmp_path / "movie.mkv")

        assert vi.fps_num == 25
        assert vi.fps_den == 1

    def test_fps_float_non_fraction_string(self, tmp_path: Path) -> None:
        stream = self._make_base_stream()
        stream["avg_frame_rate"] = "23.976"
        prober = MagicMock()
        prober.probe_hdr_side_data.return_value = []

        analyzer = Analyzer(prober=prober)
        vi = analyzer._parse_video_info(stream, {}, tmp_path / "movie.mkv")

        assert vi.fps_num == 23
        assert vi.fps_den == 1

    def test_fps_avg_zero_over_zero_falls_back_to_r_frame_rate(
        self,
        tmp_path: Path,
    ) -> None:
        stream = self._make_base_stream()
        stream["avg_frame_rate"] = "0/0"
        stream["r_frame_rate"] = "24/1"
        prober = MagicMock()
        prober.probe_hdr_side_data.return_value = []

        analyzer = Analyzer(prober=prober)
        vi = analyzer._parse_video_info(stream, {}, tmp_path / "movie.mkv")

        assert vi.fps_num == 24
        assert vi.fps_den == 1

    def test_fps_both_zero_over_zero_defaults_to_25(self, tmp_path: Path) -> None:
        stream = self._make_base_stream()
        stream["avg_frame_rate"] = "0/0"
        stream["r_frame_rate"] = "0/0"
        prober = MagicMock()
        prober.probe_hdr_side_data.return_value = []

        analyzer = Analyzer(prober=prober)
        vi = analyzer._parse_video_info(stream, {}, tmp_path / "movie.mkv")

        assert vi.fps_num == 25
        assert vi.fps_den == 1

    def test_duration_zero_in_stream_fallback_to_format(self, tmp_path: Path) -> None:
        stream = self._make_base_stream()
        stream["duration"] = "0"
        format_data = {"duration": "7200.5"}
        prober = MagicMock()
        prober.probe_hdr_side_data.return_value = []

        analyzer = Analyzer(prober=prober)
        vi = analyzer._parse_video_info(stream, format_data, tmp_path / "movie.mkv")

        assert vi.duration_s == pytest.approx(7200.5)

    def test_no_duration_in_stream_fallback_to_format(self, tmp_path: Path) -> None:
        stream = self._make_base_stream()
        del stream["duration"]
        format_data = {"duration": "3600.0"}
        prober = MagicMock()
        prober.probe_hdr_side_data.return_value = []

        analyzer = Analyzer(prober=prober)
        vi = analyzer._parse_video_info(stream, format_data, tmp_path / "movie.mkv")

        assert vi.duration_s == pytest.approx(3600.0)

    def test_bitrate_zero_in_stream_fallback_to_format(self, tmp_path: Path) -> None:
        stream = self._make_base_stream()
        stream["bit_rate"] = "0"
        format_data = {"bit_rate": "5000000"}
        prober = MagicMock()
        prober.probe_hdr_side_data.return_value = []

        analyzer = Analyzer(prober=prober)
        vi = analyzer._parse_video_info(stream, format_data, tmp_path / "movie.mkv")

        assert vi.bitrate == 5000000

    def test_bitrate_not_in_stream_fallback_to_format(self, tmp_path: Path) -> None:
        stream = self._make_base_stream()
        format_data = {"bit_rate": "8000000"}
        prober = MagicMock()
        prober.probe_hdr_side_data.return_value = []

        analyzer = Analyzer(prober=prober)
        vi = analyzer._parse_video_info(stream, format_data, tmp_path / "movie.mkv")

        assert vi.bitrate == 8000000

    def test_sar_parse_failure_defaults_to_1_1(self, tmp_path: Path) -> None:
        stream = self._make_base_stream()
        stream["sample_aspect_ratio"] = "bad:data"
        prober = MagicMock()
        prober.probe_hdr_side_data.return_value = []

        analyzer = Analyzer(prober=prober)
        vi = analyzer._parse_video_info(stream, {}, tmp_path / "movie.mkv")

        assert vi.sar_num == 1
        assert vi.sar_den == 1

    def test_sample_rate_not_parseable(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        probe_data = _h264_probe_data()
        probe_data["streams"][1]["sample_rate"] = "invalid"
        prober = make_prober(probe_data=probe_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        eng_track = next(t for t in movie.audio_tracks if t.language == "eng")
        assert eng_track.sample_rate is None


class TestIdetPath:
    def _interlaced_probe_data(self) -> dict[str, Any]:
        return {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "mpeg2video",
                    "width": 720,
                    "height": 576,
                    "avg_frame_rate": "25/1",
                    "r_frame_rate": "25/1",
                    "duration": "3600.0",
                    "field_order": "tt",
                    "pix_fmt": "yuv420p",
                },
            ],
            "format": {"duration": "3600.0"},
            "chapters": [],
        }

    def test_idet_triggered_and_deinterlace_set(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=self._interlaced_probe_data())
        prober.run_idet.return_value = 0.9

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None),
        ):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert movie.video.interlaced is True
        prober.run_idet.assert_called_once()

    def test_idet_low_ratio_stays_progressive(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=self._interlaced_probe_data())
        prober.run_idet.return_value = 0.01

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None),
        ):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert movie.video.interlaced is False

    def test_idet_exception_logged_and_continues(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=self._interlaced_probe_data())
        prober.run_idet.side_effect = RuntimeError("idet crash")

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None),
        ):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert movie.video.interlaced is False

    def test_r_frame_rate_non_fraction(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        probe_data = self._interlaced_probe_data()
        probe_data["streams"][0]["r_frame_rate"] = "25"
        prober = make_prober(probe_data=probe_data)
        prober.run_idet.return_value = 0.9

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None),
        ):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert movie.video.interlaced is True

    def _interlaced_hd_probe_data(self) -> dict[str, Any]:
        return {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "25/1",
                    "r_frame_rate": "25/1",
                    "duration": "3600.0",
                    "field_order": "tt",
                    "pix_fmt": "yuv420p",
                },
            ],
            "format": {"duration": "3600.0"},
            "chapters": [],
        }

    def test_hd_interlace_deinterlaces_without_idet(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=self._interlaced_hd_probe_data())

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None),
        ):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert movie.video.interlaced is True
        prober.run_idet.assert_not_called()


def _ntsc_dvd_probe_data(field_order: str = "tt") -> dict[str, Any]:
    return {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "mpeg2video",
                "width": 720,
                "height": 480,
                "avg_frame_rate": "30000/1001",
                "r_frame_rate": "30000/1001",
                "duration": "4889.0",
                "field_order": field_order,
                "pix_fmt": "yuv420p",
            },
        ],
        "format": {"duration": "4889.0"},
        "chapters": [],
    }


class TestSoftTelecinePath:
    def test_soft_telecine_overrides_fps_to_film_rate(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_ntsc_dvd_probe_data())
        prober.sample_repeat_pict.return_value = [0, 1] * 250

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert (movie.video.fps_num, movie.video.fps_den) == (24000, 1001)
        prober.sample_repeat_pict.assert_called_once_with(
            scan_result.main_file,
            4889.0,
        )
        assert outcome.detail == "mpeg2video 720x480 23fps SDR, 0 audio, 0 subs"

    def test_no_rff_flags_keeps_display_rate(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_ntsc_dvd_probe_data())
        prober.sample_repeat_pict.return_value = [0] * 500

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert (movie.video.fps_num, movie.video.fps_den) == (30000, 1001)

    def test_non_mpeg2_never_probed(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        probe_data = _ntsc_dvd_probe_data()
        probe_data["streams"][0]["codec_name"] = "h264"
        prober = make_prober(probe_data=probe_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        prober.sample_repeat_pict.assert_not_called()

    def test_pal_dvd_never_probed(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        probe_data = _ntsc_dvd_probe_data()
        probe_data["streams"][0]["height"] = 576
        probe_data["streams"][0]["avg_frame_rate"] = "25/1"
        probe_data["streams"][0]["r_frame_rate"] = "25/1"
        prober = make_prober(probe_data=probe_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        prober.sample_repeat_pict.assert_not_called()

    def test_probe_failure_keeps_display_rate(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_ntsc_dvd_probe_data())
        prober.sample_repeat_pict.side_effect = RuntimeError("ffprobe crash")

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert (movie.video.fps_num, movie.video.fps_den) == (30000, 1001)

    def test_pulldown_probe_counts_as_progress_stage(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_ntsc_dvd_probe_data())
        prober.sample_repeat_pict.return_value = [0, 1] * 250
        fractions: list[float] = []

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result, on_progress=fractions.append)

        assert outcome.status is AnalyzeStatus.DONE
        assert fractions == pytest.approx([1 / 3, 2 / 3, 1.0, 1.0])

    def test_deinterlace_and_soft_telecine_can_coexist(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_ntsc_dvd_probe_data())
        prober.run_idet.return_value = 0.9
        prober.sample_repeat_pict.return_value = [0, 1] * 250

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert movie.video.interlaced is True
        assert (movie.video.fps_num, movie.video.fps_den) == (24000, 1001)


class TestGrainPath:
    def test_grainy_source_flagged(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_ntsc_dvd_probe_data())
        prober.sample_grain.return_value = [0.8, 0.9, 0.7, 0.8, 0.8]

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert movie.video.grainy is True
        prober.sample_grain.assert_called_once_with(
            scan_result.main_file,
            4889.0,
            hdr_transfer=None,
        )

    def test_clean_source_not_flagged(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_ntsc_dvd_probe_data())
        prober.sample_grain.return_value = [0.2, 0.2, 0.2, 0.2, 0.2]

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert movie.video.grainy is False

    def test_hd_sdr_source_is_probed(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_h264_probe_data())
        prober.sample_grain.return_value = [0.8, 0.9, 0.7, 0.8, 0.8]

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert movie.video.grainy is True
        prober.sample_grain.assert_called_once()

    @pytest.mark.parametrize("transfer", ["smpte2084", "arib-std-b67"])
    def test_hdr_source_probed_through_a_tonemap(self, tmp_path: Path, transfer: str) -> None:
        scan_result = make_scan_result(tmp_path)
        data = _h264_probe_data()
        data["streams"][0]["color_transfer"] = transfer
        prober = make_prober(probe_data=data)
        prober.sample_grain.return_value = [0.8, 0.9, 0.7, 0.8, 0.8]

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert movie.video.grainy is True
        prober.sample_grain.assert_called_once_with(
            scan_result.main_file,
            movie.video.duration_s,
            hdr_transfer=transfer,
        )

    def test_passthrough_skips_the_probe_entirely(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_h264_probe_data())

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result, copy_video=True)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert movie.video.grainy is False
        prober.sample_grain.assert_not_called()

    @pytest.mark.parametrize("broken", [RuntimeError("ffmpeg crash"), None])
    def test_passthrough_survives_a_broken_probe(self, tmp_path: Path, broken: Exception | None) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_h264_probe_data())
        if broken is not None:
            prober.sample_grain.side_effect = broken
        else:
            prober.sample_grain.return_value = []

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result, copy_video=True)

        assert outcome.status is AnalyzeStatus.DONE
        assert outcome.movie is not None

    def test_copy_video_on_interlaced_is_not_passthrough_so_it_still_probes(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_ntsc_dvd_probe_data())
        prober.run_idet.return_value = 0.9

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result, copy_video=True)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert movie.video.interlaced is True
        prober.sample_grain.assert_called_once()

    @pytest.mark.parametrize("override", [True, False])
    def test_manual_override_skips_the_probe(self, tmp_path: Path, override: bool) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_ntsc_dvd_probe_data())
        prober.sample_grain.side_effect = RuntimeError("ffmpeg crash")

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result, grain_override=override)

        assert outcome.status is AnalyzeStatus.DONE
        assert outcome.movie is not None
        prober.sample_grain.assert_not_called()

    def test_a_skipped_probe_keeps_the_progress_shape(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_ntsc_dvd_probe_data())
        fractions: list[float] = []

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(
                scan_result,
                on_progress=fractions.append,
                grain_override=True,
            )

        assert outcome.status is AnalyzeStatus.DONE
        assert fractions == pytest.approx([1 / 3, 2 / 3, 1.0, 1.0])

    @pytest.mark.parametrize("exc", [RuntimeError("ffmpeg crash"), OSError("gone"), ValueError("junk")])
    def test_probe_failure_fails_the_file(self, tmp_path: Path, exc: Exception) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_ntsc_dvd_probe_data())
        prober.sample_grain.side_effect = exc
        fractions: list[float] = []

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result, on_progress=fractions.append)

        assert outcome.status is AnalyzeStatus.FAILED
        assert outcome.movie is None
        assert outcome.detail == "grain probe failed"
        assert fractions == pytest.approx([1 / 3, 2 / 3])

    def test_probe_measuring_no_window_fails_the_file(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_ntsc_dvd_probe_data())
        prober.sample_grain.return_value = []

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.FAILED
        assert outcome.movie is None
        assert outcome.detail == "grain probe measured nothing"

    def test_a_failed_probe_stops_before_the_audio_profiling(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        healthy = make_prober(probe_data=_h264_probe_data())

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            good = Analyzer(prober=healthy).analyze(scan_result)

        assert good.status is AnalyzeStatus.DONE
        assert healthy.profile_audio_track.call_count == 2

        broken = make_prober(probe_data=_h264_probe_data())
        broken.sample_grain.side_effect = RuntimeError("ffmpeg crash")

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=broken).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.FAILED
        broken.profile_audio_track.assert_not_called()

    def test_grain_probe_counts_as_progress_stage(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_ntsc_dvd_probe_data())
        prober.sample_grain.return_value = [0.8, 0.9, 0.7, 0.8, 0.8]
        fractions: list[float] = []

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result, on_progress=fractions.append)

        assert outcome.status is AnalyzeStatus.DONE
        assert fractions == pytest.approx([1 / 3, 2 / 3, 1.0, 1.0])


class TestExternalSubtitle:
    def test_srt_satellite_language_from_filename(self, tmp_path: Path) -> None:
        srt_path = tmp_path / "movie.eng.srt"
        srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")

        scan_result = make_scan_result(tmp_path)
        scan_result = ScanResult(
            main_file=scan_result.main_file,
            satellite_files=[srt_path],
            output_path=scan_result.output_path,
        )
        prober = make_prober(probe_data=_h264_probe_data())

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer._from_path") as mock_from_path,
        ):
            mock_result = MagicMock()
            mock_best = MagicMock()
            mock_best.encoding = "utf-8"
            mock_result.best.return_value = mock_best
            mock_from_path.return_value = mock_result

            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        sat_subs = [t for t in movie.subtitle_tracks if t.source_file == srt_path]
        assert len(sat_subs) == 1
        sub = sat_subs[0]
        assert sub.language == "eng"
        assert sub.codec_id == SubtitleCodecId.SRT
        assert sub.is_forced is False
        assert sub.encoding == "utf-8"

    def test_forced_keyword_in_filename(self, tmp_path: Path) -> None:
        srt_path = tmp_path / "movie.rus.forced.srt"
        srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nTest\n", encoding="utf-8")

        scan_result = ScanResult(
            main_file=make_scan_result(tmp_path).main_file,
            satellite_files=[srt_path],
            output_path=make_scan_result(tmp_path).output_path,
        )
        prober = make_prober(probe_data=_h264_probe_data())

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer._from_path") as mock_from_path,
        ):
            mock_result = MagicMock()
            mock_best = MagicMock()
            mock_best.encoding = "utf-8"
            mock_result.best.return_value = mock_best
            mock_from_path.return_value = mock_result

            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        sat_subs = [t for t in movie.subtitle_tracks if t.source_file == srt_path]
        assert len(sat_subs) == 1
        assert sat_subs[0].is_forced is True
        assert sat_subs[0].language == "rus"

    def test_sup_satellite_no_encoding_detection(self, tmp_path: Path) -> None:
        sup_path = tmp_path / "movie.jpn.sup"
        sup_path.write_bytes(b"\x00" * 16)

        scan_result = ScanResult(
            main_file=make_scan_result(tmp_path).main_file,
            satellite_files=[sup_path],
            output_path=make_scan_result(tmp_path).output_path,
        )
        prober = make_prober(probe_data=_h264_probe_data())

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        sat_subs = [t for t in movie.subtitle_tracks if t.source_file == sup_path]
        assert len(sat_subs) == 1
        sub = sat_subs[0]
        assert sub.codec_id == SubtitleCodecId.PGS
        assert sub.language == "jpn"
        assert sub.encoding is None

    def test_ass_satellite_encoding_detection(self, tmp_path: Path) -> None:
        ass_path = tmp_path / "movie.rus.ass"
        ass_path.write_text("[Script Info]\n", encoding="utf-8")

        scan_result = ScanResult(
            main_file=make_scan_result(tmp_path).main_file,
            satellite_files=[ass_path],
            output_path=make_scan_result(tmp_path).output_path,
        )
        prober = make_prober(probe_data=_h264_probe_data())

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer._from_path") as mock_from_path,
        ):
            mock_result = MagicMock()
            mock_best = MagicMock()
            mock_best.encoding = "windows-1251"
            mock_result.best.return_value = mock_best
            mock_from_path.return_value = mock_result

            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        sat_subs = [t for t in movie.subtitle_tracks if t.source_file == ass_path]
        assert len(sat_subs) == 1
        assert sat_subs[0].encoding == "windows-1251"


class TestExternalAudio:
    def _audio_probe_data(self) -> dict[str, Any]:
        return {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "audio",
                    "codec_name": "flac",
                    "channels": 6,
                    "channel_layout": "5.1",
                    "sample_rate": "48000",
                    "bit_rate": "2000000",
                    "tags": {"language": "eng", "title": "English 5.1"},
                    "disposition": {"default": 0, "forced": 0},
                },
            ],
            "format": {},
            "chapters": [],
        }

    def test_external_audio_satellite_parsed(self, tmp_path: Path) -> None:
        flac_path = tmp_path / "movie.eng.flac"
        flac_path.write_bytes(b"\x00" * 256)

        main_scan = make_scan_result(tmp_path)
        scan_result = ScanResult(
            main_file=main_scan.main_file,
            satellite_files=[flac_path],
            output_path=main_scan.output_path,
        )
        prober = make_prober(probe_data=_h264_probe_data())
        sat_probe = self._audio_probe_data()

        def probe_side_effect(path: Path) -> dict[str, Any]:
            if path == flac_path:
                return sat_probe
            return _h264_probe_data()

        prober.probe.side_effect = probe_side_effect

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        sat_audio = [t for t in movie.audio_tracks if t.source_file == flac_path]
        assert len(sat_audio) == 1
        track = sat_audio[0]
        assert track.codec_name == "flac"
        assert track.channels == 6
        assert track.index == 2

    def test_external_audio_probe_fails(self, tmp_path: Path) -> None:
        ac3_path = tmp_path / "movie.eng.ac3"
        ac3_path.write_bytes(b"\x00" * 256)

        main_scan = make_scan_result(tmp_path)
        scan_result = ScanResult(
            main_file=main_scan.main_file,
            satellite_files=[ac3_path],
            output_path=main_scan.output_path,
        )
        prober = make_prober(probe_data=_h264_probe_data())

        call_count = 0

        def probe_side_effect(path: Path) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if path == ac3_path:
                raise OSError("cannot read satellite")
            return _h264_probe_data()

        prober.probe.side_effect = probe_side_effect

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert len(movie.audio_tracks) == 2

    def test_external_audio_no_audio_streams(self, tmp_path: Path) -> None:
        wav_path = tmp_path / "movie.eng.wav"
        wav_path.write_bytes(b"\x00" * 256)

        main_scan = make_scan_result(tmp_path)
        scan_result = ScanResult(
            main_file=main_scan.main_file,
            satellite_files=[wav_path],
            output_path=main_scan.output_path,
        )
        prober = make_prober(probe_data=_h264_probe_data())

        def probe_side_effect(path: Path) -> dict[str, Any]:
            if path == wav_path:
                return {"streams": [], "format": {}, "chapters": []}
            return _h264_probe_data()

        prober.probe.side_effect = probe_side_effect

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert len(movie.audio_tracks) == 2


class TestAttachments:
    def test_attachments_parsed(self, tmp_path: Path) -> None:
        probe_data = _h264_probe_data()
        probe_data["streams"].append(
            {
                "index": 4,
                "codec_type": "attachment",
                "tags": {
                    "filename": "Arial.ttf",
                    "mimetype": "application/x-truetype-font",
                },
            }
        )
        probe_data["streams"].append(
            {
                "index": 5,
                "codec_type": "attachment",
                "tags": {
                    "filename": "OpenSans.otf",
                    "mime_type": "font/otf",
                },
            }
        )
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=probe_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert len(movie.attachments) == 2
        assert movie.attachments[0].filename == "Arial.ttf"
        assert movie.attachments[0].mime_type == "application/x-truetype-font"
        assert movie.attachments[0].stream_index == 4
        assert movie.attachments[1].filename == "OpenSans.otf"
        assert movie.attachments[1].mime_type == "font/otf"
        assert movie.attachments[1].stream_index == 5

    def test_attachment_no_filename_skipped(self, tmp_path: Path) -> None:
        probe_data = _h264_probe_data()
        probe_data["streams"].append(
            {
                "index": 4,
                "codec_type": "attachment",
                "tags": {"mimetype": "application/octet-stream"},
            }
        )
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=probe_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert len(movie.attachments) == 0


class TestTextEncodingDetection:
    def test_encoding_detected_successfully(self, tmp_path: Path) -> None:
        prober = MagicMock()
        analyzer = Analyzer(prober=prober)
        srt = tmp_path / "test.srt"

        with patch("furnace.services.analyzer._from_path") as mock_from_path:
            mock_result = MagicMock()
            mock_best = MagicMock()
            mock_best.encoding = "windows-1251"
            mock_result.best.return_value = mock_best
            mock_from_path.return_value = mock_result

            result = analyzer._detect_text_encoding(srt)

        assert result == "windows-1251"

    def test_encoding_detection_returns_none_when_best_is_none(self, tmp_path: Path) -> None:
        prober = MagicMock()
        analyzer = Analyzer(prober=prober)
        srt = tmp_path / "test.srt"

        with patch("furnace.services.analyzer._from_path") as mock_from_path:
            mock_result = MagicMock()
            mock_result.best.return_value = None
            mock_from_path.return_value = mock_result

            result = analyzer._detect_text_encoding(srt)

        assert result is None

    def test_encoding_detection_os_error(self, tmp_path: Path) -> None:
        prober = MagicMock()
        analyzer = Analyzer(prober=prober)
        srt = tmp_path / "test.srt"

        with patch("furnace.services.analyzer._from_path", side_effect=OSError("file gone")):
            result = analyzer._detect_text_encoding(srt)

        assert result is None

    def test_encoding_detection_value_error(self, tmp_path: Path) -> None:
        prober = MagicMock()
        analyzer = Analyzer(prober=prober)
        srt = tmp_path / "test.srt"

        with patch("furnace.services.analyzer._from_path", side_effect=ValueError("bad data")):
            result = analyzer._detect_text_encoding(srt)

        assert result is None


class TestNumFramesParsing:
    def test_number_of_frames_tag_parsed(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        probe_data = _h264_probe_data()
        prober = make_prober(probe_data=probe_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        sub = movie.subtitle_tracks[0]
        assert sub.num_frames == 120

    def test_number_of_frames_eng_tag(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        probe_data = _h264_probe_data()
        sub_stream = probe_data["streams"][3]
        del sub_stream["tags"]["NUMBER_OF_FRAMES"]
        sub_stream["tags"]["NUMBER_OF_FRAMES-eng"] = "250"
        prober = make_prober(probe_data=probe_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        sub = movie.subtitle_tracks[0]
        assert sub.num_frames == 250

    def test_number_of_frames_invalid_value(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        probe_data = _h264_probe_data()
        probe_data["streams"][3]["tags"]["NUMBER_OF_FRAMES"] = "not_a_number"
        prober = make_prober(probe_data=probe_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        sub = movie.subtitle_tracks[0]
        assert sub.num_frames is None


class TestAudioBitrateParsing:
    def test_bitrate_from_tags_bps(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        probe_data = _h264_probe_data()
        audio_stream = probe_data["streams"][2]
        audio_stream.pop("bit_rate", None)
        audio_stream["tags"]["BPS"] = "640000"
        prober = make_prober(probe_data=probe_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        rus_track = next(t for t in movie.audio_tracks if t.language == "rus")
        assert rus_track.bitrate == 640000


class TestTextSubtitleCodecsSet:
    def test_text_subtitle_codecs_contains_srt_and_ass(self) -> None:
        assert SubtitleCodecId.SRT in _TEXT_SUBTITLE_CODECS
        assert SubtitleCodecId.ASS in _TEXT_SUBTITLE_CODECS
        assert SubtitleCodecId.PGS not in _TEXT_SUBTITLE_CODECS


class TestSatelliteUnknownExtension:
    def test_unknown_extension_satellite_skipped(self, tmp_path: Path) -> None:
        nfo_path = tmp_path / "movie.nfo"
        nfo_path.write_text("info", encoding="utf-8")

        main_scan = make_scan_result(tmp_path)
        scan_result = ScanResult(
            main_file=main_scan.main_file,
            satellite_files=[nfo_path],
            output_path=main_scan.output_path,
        )
        prober = make_prober(probe_data=_h264_probe_data())

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert len(movie.audio_tracks) == 2
        assert len(movie.subtitle_tracks) == 1


class TestSarNoColon:
    def test_sar_no_colon_defaults(self, tmp_path: Path) -> None:
        stream = {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": "24/1",
            "duration": "100.0",
            "field_order": "progressive",
            "pix_fmt": "yuv420p",
            "sample_aspect_ratio": "1",
        }
        prober = MagicMock()
        prober.probe_hdr_side_data.return_value = []

        vi = Analyzer(prober=prober)._parse_video_info(stream, {}, tmp_path / "m.mkv")

        assert vi.sar_num == 1
        assert vi.sar_den == 1

    def test_sar_empty_string_defaults(self, tmp_path: Path) -> None:
        stream = {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": "24/1",
            "duration": "100.0",
            "field_order": "progressive",
            "pix_fmt": "yuv420p",
            "sample_aspect_ratio": "",
        }
        prober = MagicMock()
        prober.probe_hdr_side_data.return_value = []

        vi = Analyzer(prober=prober)._parse_video_info(stream, {}, tmp_path / "m.mkv")

        assert vi.sar_num == 1
        assert vi.sar_den == 1


class TestAudioNoBitrate:
    def test_no_bitrate_anywhere(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        probe_data = _h264_probe_data()
        audio_stream = probe_data["streams"][1]
        audio_stream.pop("bit_rate", None)
        audio_stream["tags"].pop("BPS", None)
        audio_stream["tags"].pop("BPS-eng", None)
        prober = make_prober(probe_data=probe_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        eng_track = next(t for t in movie.audio_tracks if t.language == "eng")
        assert eng_track.bitrate is None


class TestSubtitleNoFramesTags:
    def test_no_frames_tags(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        probe_data = _h264_probe_data()
        del probe_data["streams"][3]["tags"]["NUMBER_OF_FRAMES"]
        prober = make_prober(probe_data=probe_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert movie.subtitle_tracks[0].num_frames is None


class TestExternalSubtitleNoLanguageCode:
    def test_no_language_code_in_filename(self, tmp_path: Path) -> None:
        srt_path = tmp_path / "movie.srt"
        srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")

        main_scan = make_scan_result(tmp_path)
        scan_result = ScanResult(
            main_file=main_scan.main_file,
            satellite_files=[srt_path],
            output_path=main_scan.output_path,
        )
        prober = make_prober(probe_data=_h264_probe_data())

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer._from_path") as mock_from_path,
        ):
            mock_result = MagicMock()
            mock_best = MagicMock()
            mock_best.encoding = "utf-8"
            mock_result.best.return_value = mock_best
            mock_from_path.return_value = mock_result

            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        sat_subs = [t for t in movie.subtitle_tracks if t.source_file == srt_path]
        assert len(sat_subs) == 1
        assert sat_subs[0].language == "und"

    def test_non_alpha_part_in_filename(self, tmp_path: Path) -> None:
        srt_path = tmp_path / "movie.720p.srt"
        srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")

        main_scan = make_scan_result(tmp_path)
        scan_result = ScanResult(
            main_file=main_scan.main_file,
            satellite_files=[srt_path],
            output_path=main_scan.output_path,
        )
        prober = make_prober(probe_data=_h264_probe_data())

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer._from_path") as mock_from_path,
        ):
            mock_result = MagicMock()
            mock_best = MagicMock()
            mock_best.encoding = "utf-8"
            mock_result.best.return_value = mock_best
            mock_from_path.return_value = mock_result

            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        sat_subs = [t for t in movie.subtitle_tracks if t.source_file == srt_path]
        assert len(sat_subs) == 1
        assert sat_subs[0].language == "und"


class TestExternalAudioEmptyTracks:
    def test_parse_audio_tracks_returns_empty(self, tmp_path: Path) -> None:
        ac3_path = tmp_path / "movie.eng.ac3"
        ac3_path.write_bytes(b"\x00" * 256)

        prober = MagicMock()
        analyzer = Analyzer(prober=prober)

        prober.probe.return_value = {
            "streams": [{"index": 0, "codec_type": "audio", "codec_name": "ac3"}],
            "format": {},
        }
        with patch.object(analyzer, "_parse_audio_tracks", return_value=[]):
            result = analyzer._parse_external_audio(ac3_path, 5)

        assert result is None


class TestAudioNoSampleRate:
    def test_no_sample_rate_key(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        probe_data: dict[str, Any] = {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "24/1",
                    "duration": "100.0",
                    "field_order": "progressive",
                    "pix_fmt": "yuv420p",
                },
                {
                    "index": 1,
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "profile": "LC",
                    "channels": 2,
                    "tags": {"language": "eng"},
                    "disposition": {"default": 1, "forced": 0},
                },
            ],
            "format": {"duration": "100.0"},
            "chapters": [],
        }
        prober = make_prober(probe_data=probe_data)

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer.detect_hdr", return_value=HdrMetadata()),
            patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None),
        ):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert movie.audio_tracks[0].sample_rate is None


class TestExternalSubtitleReturnsNone:
    def test_external_subtitle_returns_none_skipped(self, tmp_path: Path) -> None:
        srt_path = tmp_path / "movie.eng.srt"
        srt_path.write_text("sub", encoding="utf-8")

        main_scan = make_scan_result(tmp_path)
        scan_result = ScanResult(
            main_file=main_scan.main_file,
            satellite_files=[srt_path],
            output_path=main_scan.output_path,
        )
        prober = make_prober(probe_data=_h264_probe_data())

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            analyzer = Analyzer(prober=prober)
            with patch.object(analyzer, "_parse_external_subtitle", return_value=None):
                outcome = analyzer.analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert len(movie.subtitle_tracks) == 1


def _real_5_1_metrics() -> AudioMetrics:
    return AudioMetrics(
        channels=6,
        rms_l=-22.0,
        rms_r=-22.0,
        rms_c=-18.0,
        rms_lfe=-20.0,
        rms_ls=-25.0,
        rms_rs=-25.0,
        rms_lb=None,
        rms_rb=None,
        corr_lr=0.3,
        corr_ls_l=0.1,
        corr_rs_r=0.1,
        corr_ls_rs=0.4,
        corr_lb_ls=None,
        corr_rb_rs=None,
    )


def _dead_lfe_2_1_metrics() -> AudioMetrics:
    return AudioMetrics(
        channels=3,
        rms_l=-29.6,
        rms_r=-30.4,
        rms_c=None,
        rms_lfe=-120.0,
        rms_ls=None,
        rms_rs=None,
        rms_lb=None,
        rms_rb=None,
        corr_lr=0.765,
        corr_ls_l=None,
        corr_rs_r=None,
        corr_ls_rs=None,
        corr_lb_ls=None,
        corr_rb_rs=None,
    )


def _silent_surrounds_5_0_metrics() -> AudioMetrics:
    return AudioMetrics(
        channels=5,
        rms_l=-25.0,
        rms_r=-25.5,
        rms_c=-10.0,
        rms_lfe=None,
        rms_ls=-70.0,
        rms_rs=-70.0,
        rms_lb=None,
        rms_rb=None,
        corr_lr=0.3,
        corr_ls_l=0.1,
        corr_rs_r=0.1,
        corr_ls_rs=0.2,
        corr_lb_ls=None,
        corr_rb_rs=None,
    )


def _derived_center_3_0_metrics() -> AudioMetrics:
    return AudioMetrics(
        channels=3,
        rms_l=-25.0,
        rms_r=-25.5,
        rms_c=-24.0,
        rms_lfe=None,
        rms_ls=None,
        rms_rs=None,
        rms_lb=None,
        rms_rb=None,
        corr_lr=0.4,
        corr_ls_l=None,
        corr_rs_r=None,
        corr_ls_rs=None,
        corr_lb_ls=None,
        corr_rb_rs=None,
        corr_c_lr=0.99,
    )


class TestAudioProfiling:
    def test_profiles_6ch_track_and_attaches_verdict(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        probe_data: dict[str, Any] = {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "24/1",
                    "duration": "100.0",
                    "field_order": "progressive",
                    "pix_fmt": "yuv420p",
                },
                {
                    "index": 1,
                    "codec_type": "audio",
                    "codec_name": "ac3",
                    "channels": 6,
                    "tags": {"language": "eng"},
                    "disposition": {"default": 1, "forced": 0},
                },
            ],
            "format": {"duration": "100.0"},
            "chapters": [],
        }
        prober = make_prober(probe_data=probe_data)
        prober.profile_audio_track.return_value = _real_5_1_metrics()

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer.detect_hdr", return_value=HdrMetadata()),
            patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None),
        ):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert movie.audio_tracks[0].audio_profile is not None
        assert movie.audio_tracks[0].audio_profile.verdict == Verdict.REAL
        prober.profile_audio_track.assert_called_once_with(
            path=scan_result.main_file,
            stream_index=1,
            channels=6,
            duration_s=100.0,
            channel_layout=None,
        )

    def _three_channel_probe_data(self, layout: str | None) -> dict[str, Any]:
        audio: dict[str, Any] = {
            "index": 1,
            "codec_type": "audio",
            "codec_name": "ac3",
            "channels": 3,
            "tags": {"language": "rus"},
            "disposition": {"default": 1, "forced": 0},
        }
        if layout is not None:
            audio["channel_layout"] = layout
        return {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 720,
                    "height": 576,
                    "avg_frame_rate": "25/1",
                    "duration": "100.0",
                    "field_order": "progressive",
                    "pix_fmt": "yuv420p",
                },
                audio,
            ],
            "format": {"duration": "100.0"},
            "chapters": [],
        }

    def test_forwards_the_2_1_layout_and_attaches_the_verdict(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=self._three_channel_probe_data("2.1"))
        prober.profile_audio_track.return_value = _dead_lfe_2_1_metrics()

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer.detect_hdr", return_value=HdrMetadata()),
            patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None),
        ):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        movie = outcome.movie
        assert movie is not None
        profile = movie.audio_tracks[0].audio_profile
        assert profile is not None
        assert profile.verdict == Verdict.FAKE
        prober.profile_audio_track.assert_called_once_with(
            path=scan_result.main_file,
            stream_index=1,
            channels=3,
            duration_s=100.0,
            channel_layout="2.1",
        )

    def test_forwards_the_3_0_layout(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=self._three_channel_probe_data("3.0"))
        prober.profile_audio_track.return_value = _derived_center_3_0_metrics()

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer.detect_hdr", return_value=HdrMetadata()),
            patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None),
        ):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert prober.profile_audio_track.call_args.kwargs["channel_layout"] == "3.0"
        movie = outcome.movie
        assert movie is not None
        profile = movie.audio_tracks[0].audio_profile
        assert profile is not None
        assert profile.verdict == Verdict.FAKE
        assert any("mix of the fronts" in r for r in profile.reasons)

    @pytest.mark.parametrize("layout", ["3.0(back)", None])
    def test_skips_unhandled_three_channel_layouts(self, tmp_path: Path, layout: str | None) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=self._three_channel_probe_data(layout))

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer.detect_hdr", return_value=HdrMetadata()),
            patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None),
        ):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        movie = outcome.movie
        assert movie is not None
        assert movie.audio_tracks[0].audio_profile is None
        prober.profile_audio_track.assert_not_called()

    def _five_channel_probe_data(self, layout: str | None) -> dict[str, Any]:
        audio: dict[str, Any] = {
            "index": 1,
            "codec_type": "audio",
            "codec_name": "ac3",
            "channels": 5,
            "tags": {"language": "rus"},
            "disposition": {"default": 1, "forced": 0},
        }
        if layout is not None:
            audio["channel_layout"] = layout
        return {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "mpeg2video",
                    "width": 720,
                    "height": 576,
                    "avg_frame_rate": "25/1",
                    "duration": "100.0",
                    "field_order": "progressive",
                    "pix_fmt": "yuv420p",
                },
                audio,
            ],
            "format": {"duration": "100.0"},
            "chapters": [],
        }

    @pytest.mark.parametrize("layout", ["5.0", "5.0(side)"])
    def test_forwards_the_5_0_layout_and_attaches_the_verdict(self, tmp_path: Path, layout: str) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=self._five_channel_probe_data(layout))
        prober.profile_audio_track.return_value = _silent_surrounds_5_0_metrics()

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer.detect_hdr", return_value=HdrMetadata()),
            patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None),
        ):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        movie = outcome.movie
        assert movie is not None
        profile = movie.audio_tracks[0].audio_profile
        assert profile is not None
        assert profile.verdict == Verdict.FAKE
        prober.profile_audio_track.assert_called_once_with(
            path=scan_result.main_file,
            stream_index=1,
            channels=5,
            duration_s=100.0,
            channel_layout=layout,
        )

    @pytest.mark.parametrize("layout", ["4.1", None])
    def test_skips_unhandled_five_channel_layouts(self, tmp_path: Path, layout: str | None) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=self._five_channel_probe_data(layout))

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer.detect_hdr", return_value=HdrMetadata()),
            patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None),
        ):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        movie = outcome.movie
        assert movie is not None
        assert movie.audio_tracks[0].audio_profile is None
        prober.profile_audio_track.assert_not_called()

    def test_an_unprofileable_five_channel_layout_is_logged_as_a_warning(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=self._five_channel_probe_data("4.1"))

        with (
            caplog.at_level(logging.WARNING, logger="furnace.services.analyzer"),
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer.detect_hdr", return_value=HdrMetadata()),
            patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None),
        ):
            Analyzer(prober=prober).analyze(scan_result)

        assert any("Not profiling track" in r.message for r in caplog.records)

    def test_a_mono_track_is_skipped_without_a_warning(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        scan_result = make_scan_result(tmp_path)
        probe_data = self._five_channel_probe_data("4.1")
        probe_data["streams"][1]["channels"] = 1
        del probe_data["streams"][1]["channel_layout"]
        prober = make_prober(probe_data=probe_data)

        with (
            caplog.at_level(logging.INFO, logger="furnace.services.analyzer"),
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer.detect_hdr", return_value=HdrMetadata()),
            patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None),
        ):
            Analyzer(prober=prober).analyze(scan_result)

        skipped = [r for r in caplog.records if "Not profiling track" in r.message]
        assert skipped, "the skip must still be logged"
        assert all(r.levelno == logging.INFO for r in skipped)

    def test_skips_1ch_track(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        probe_data: dict[str, Any] = {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "24/1",
                    "duration": "100.0",
                    "field_order": "progressive",
                    "pix_fmt": "yuv420p",
                },
                {
                    "index": 1,
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "channels": 1,
                    "tags": {"language": "eng"},
                    "disposition": {"default": 1, "forced": 0},
                },
            ],
            "format": {"duration": "100.0"},
            "chapters": [],
        }
        prober = make_prober(probe_data=probe_data)

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer.detect_hdr", return_value=HdrMetadata()),
            patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None),
        ):
            outcome = Analyzer(prober=prober).analyze(scan_result)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert movie.audio_tracks[0].audio_profile is None
        prober.profile_audio_track.assert_not_called()


class TestFieldSeparatedPath:
    def _field_rate_probe_data(self) -> dict[str, Any]:
        return {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "50/1",
                    "r_frame_rate": "50/1",
                    "duration": "5372.48",
                    "field_order": "tt",
                    "pix_fmt": "yuv420p",
                },
            ],
            "format": {"duration": "5372.48"},
            "chapters": [],
        }

    def _analyze(self, tmp_path: Path, prober: MagicMock) -> AnalysisOutcome:
        scan_result = make_scan_result(tmp_path)
        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None),
        ):
            return Analyzer(prober=prober).analyze(scan_result)

    def test_field_separated_source_carries_the_coded_frame_rate(self, tmp_path: Path) -> None:
        prober = make_prober(probe_data=self._field_rate_probe_data())
        prober.sample_field_pairing.return_value = (1500, 3000)

        outcome = self._analyze(tmp_path, prober)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert (movie.video.fps_num, movie.video.fps_den) == (25, 1)
        assert movie.video.interlaced is True
        prober.sample_field_pairing.assert_called_once_with(tmp_path / "movie.mkv")

    def test_frame_coded_source_keeps_the_container_rate(self, tmp_path: Path) -> None:
        prober = make_prober(probe_data=self._field_rate_probe_data())
        prober.sample_field_pairing.return_value = (1500, 1500)

        outcome = self._analyze(tmp_path, prober)

        movie = outcome.movie
        assert movie is not None
        assert (movie.video.fps_num, movie.video.fps_den) == (50, 1)

    def test_probe_skipped_for_ordinary_interlaced_source(self, tmp_path: Path) -> None:
        probe_data = self._field_rate_probe_data()
        probe_data["streams"][0]["avg_frame_rate"] = "25/1"
        probe_data["streams"][0]["r_frame_rate"] = "50/1"
        prober = make_prober(probe_data=probe_data)

        outcome = self._analyze(tmp_path, prober)

        movie = outcome.movie
        assert movie is not None
        assert (movie.video.fps_num, movie.video.fps_den) == (25, 1)
        prober.sample_field_pairing.assert_not_called()

    def test_probe_skipped_for_progressive_source(self, tmp_path: Path) -> None:
        probe_data = self._field_rate_probe_data()
        probe_data["streams"][0]["field_order"] = "progressive"
        prober = make_prober(probe_data=probe_data)

        outcome = self._analyze(tmp_path, prober)

        movie = outcome.movie
        assert movie is not None
        assert (movie.video.fps_num, movie.video.fps_den) == (50, 1)
        prober.sample_field_pairing.assert_not_called()

    def test_probe_exception_logged_and_keeps_container_rate(self, tmp_path: Path) -> None:
        prober = make_prober(probe_data=self._field_rate_probe_data())
        prober.sample_field_pairing.side_effect = RuntimeError("ffprobe crash")

        outcome = self._analyze(tmp_path, prober)

        assert outcome.status is AnalyzeStatus.DONE
        movie = outcome.movie
        assert movie is not None
        assert (movie.video.fps_num, movie.video.fps_den) == (50, 1)

    def test_probe_reports_progress(self, tmp_path: Path) -> None:
        prober = make_prober(probe_data=self._field_rate_probe_data())
        prober.sample_field_pairing.return_value = (1500, 3000)
        scan_result = make_scan_result(tmp_path)
        seen: list[float] = []

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None),
        ):
            Analyzer(prober=prober).analyze(scan_result, on_progress=seen.append)

        assert seen == [0.5, 1.0, 1.0]
