from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from furnace.core.models import (
    AudioCodecId,
    HdrMetadata,
    ScanResult,
    SubtitleCodecId,
    TrackType,
)
from furnace.services.analyzer import _TEXT_SUBTITLE_CODECS, Analyzer

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

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
    return prober


def make_scan_result(tmp_path: Path, filename: str = "movie.mkv") -> ScanResult:
    main_file = tmp_path / filename
    main_file.write_bytes(b"\x00" * 1024)  # give it a real size for stat()
    output_path = tmp_path / "out" / filename
    return ScanResult(
        main_file=main_file,
        satellite_files=[],
        output_path=output_path,
    )


def _h264_probe_data() -> dict[str, Any]:
    """Realistic ffprobe-like dict for a standard H.264 SDR MKV."""
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
    """ffprobe-like dict for a Dolby Vision file."""
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAnalyzerParsesTracks:
    def test_analyzer_parses_tracks(self, tmp_path: Path) -> None:
        """Mock Prober returns ffprobe-like data -> correct Track objects."""
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_h264_probe_data())

        # Patch should_skip_file to always allow processing
        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            analyzer = Analyzer(prober=prober)
            movie = analyzer.analyze(scan_result)

        assert movie is not None
        # Video info
        assert movie.video.codec_name == "h264"
        assert movie.video.width == 1920
        assert movie.video.height == 1080
        # Audio tracks: eng AAC + rus AC3
        assert len(movie.audio_tracks) == 2
        eng_track = next(t for t in movie.audio_tracks if t.language == "eng")
        rus_track = next(t for t in movie.audio_tracks if t.language == "rus")
        assert eng_track.codec_id == AudioCodecId.AAC_LC
        assert eng_track.track_type == TrackType.AUDIO
        assert rus_track.codec_id == AudioCodecId.AC3
        assert rus_track.channels == 6
        # Subtitle track
        assert len(movie.subtitle_tracks) == 1
        sub = movie.subtitle_tracks[0]
        assert sub.codec_id == SubtitleCodecId.PGS
        assert sub.language == "rus"
        # Chapters
        assert movie.has_chapters is True

    def test_analyzer_parses_audio_language_and_disposition(self, tmp_path: Path) -> None:
        """Audio track language and default disposition are parsed correctly."""
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_h264_probe_data())

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            analyzer = Analyzer(prober=prober)
            movie = analyzer.analyze(scan_result)

        assert movie is not None
        eng_track = next(t for t in movie.audio_tracks if t.language == "eng")
        assert eng_track.is_default is True
        rus_track = next(t for t in movie.audio_tracks if t.language == "rus")
        assert rus_track.is_default is False


class TestAnalyzerDVProceeds:
    def test_analyzer_dv_returns_movie(self, tmp_path: Path) -> None:
        """DV file (dvhe codec) -> analyze returns Movie (no longer skipped)."""
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_dv_probe_data())

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            with patch("furnace.services.analyzer.detect_hdr") as mock_detect_hdr:
                mock_detect_hdr.return_value = HdrMetadata(is_dolby_vision=True)
                with patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None):
                    analyzer = Analyzer(prober=prober)
                    movie = analyzer.analyze(scan_result)

        assert movie is not None
        assert movie.video.codec_name == "dvhe"


class TestAnalyzerHdrSideDataMerge:
    """Stream-level DOVI config + frame-level MDCV/CLL must merge into hdr metadata.

    Real-world UHD Blu-Ray DV P7 remuxes carry DOVI configuration record at
    packet (stream) level, but MDCV and Content Light at frame level.
    The analyzer must probe both and merge.
    """

    def test_stream_dovi_and_frame_mdcv_cll_both_detected(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        probe_data = _dv_probe_data()  # stream has DOVI configuration record, PQ transfer
        # Real ffprobe-like frame-level side data
        frame_side_data: list[dict[str, Any]] = [
            {
                "side_data_type": "Mastering display metadata",
                "red_x": "35400/50000", "red_y": "14600/50000",
                "green_x": "8500/50000", "green_y": "39850/50000",
                "blue_x": "6550/50000", "blue_y": "2300/50000",
                "white_point_x": "15635/50000", "white_point_y": "16450/50000",
                "min_luminance": "50/10000", "max_luminance": "40000000/10000",
            },
            {
                "side_data_type": "Content light level metadata",
                "max_content": 4342,
                "max_average": 2342,
            },
            {"side_data_type": "Dolby Vision RPU Data"},
        ]
        # Inject dv_profile on the stream-level DOVI config so the analyzer
        # can compute dv_mode correctly.
        probe_data["streams"][0]["side_data_list"] = [{
            "side_data_type": "DOVI configuration record",
            "dv_profile": 7,
            "dv_bl_signal_compatibility_id": 0,
        }]
        prober = make_prober(probe_data=probe_data, hdr_side_data=frame_side_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            with patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None):
                analyzer = Analyzer(prober=prober)
                movie = analyzer.analyze(scan_result)

        assert movie is not None
        hdr = movie.video.hdr
        assert hdr.is_dolby_vision is True
        assert hdr.dv_profile == 7
        assert hdr.mastering_display is not None
        assert "L(40000000,50)" in hdr.mastering_display
        assert hdr.content_light == "MaxCLL=4342,MaxFALL=2342"
        # Frame-level probe must have been called for PQ content
        prober.probe_hdr_side_data.assert_called_once()

    def test_sdr_skips_frame_side_data_probe(self, tmp_path: Path) -> None:
        """SDR content (bt709 transfer) -> frame-level probe not called."""
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_h264_probe_data())

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            analyzer = Analyzer(prober=prober)
            analyzer.analyze(scan_result)

        prober.probe_hdr_side_data.assert_not_called()


class TestAnalyzerHDR10PlusError:
    def test_analyzer_hdr10plus_raises(self, tmp_path: Path) -> None:
        """HDR10+ content -> analyze raises ValueError."""
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_dv_probe_data())

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            with patch("furnace.services.analyzer.detect_hdr") as mock_detect_hdr:
                mock_detect_hdr.return_value = HdrMetadata(is_hdr10_plus=True)
                analyzer = Analyzer(prober=prober)
                with pytest.raises(ValueError, match="HDR10\\+ not supported"):
                    analyzer.analyze(scan_result)


class TestAnalyzerDelay:
    def test_analyzer_delay_from_start_pts(self, tmp_path: Path) -> None:
        """start_pts=500 -> delay_ms=500 (used directly as integer ms)."""
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
                    movie = analyzer.analyze(scan_result)

        assert movie is not None
        assert movie.audio_tracks[0].delay_ms == 500

    def test_analyzer_delay_fallback_start_time(self, tmp_path: Path) -> None:
        """No start_pts, start_time=0.5 -> delay_ms=500."""
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
                    movie = analyzer.analyze(scan_result)

        assert movie is not None
        assert movie.audio_tracks[0].delay_ms == 500

    def test_analyzer_delay_default(self, tmp_path: Path) -> None:
        """No start_pts, no start_time -> delay_ms=0."""
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
                    movie = analyzer.analyze(scan_result)

        assert movie is not None
        assert movie.audio_tracks[0].delay_ms == 0


class TestAnalyzerDelayDirect:
    """Unit-test _detect_audio_delay directly (no full analyze pipeline needed)."""

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
        """When both start_pts and start_time present, start_pts wins."""
        prober = MagicMock()
        analyzer = Analyzer(prober=prober)
        result = analyzer._detect_audio_delay({"start_pts": 100, "start_time": "5.0"})
        assert result == 100


# ---------------------------------------------------------------------------
# Early returns in analyze()
# ---------------------------------------------------------------------------


class TestAnalyzeEarlyReturns:
    """Cover early-return branches in analyze()."""

    def test_skip_when_should_skip_file_returns_true(self, tmp_path: Path) -> None:
        """should_skip_file returns (True, reason) -> analyze returns None."""
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_h264_probe_data(), encoder_tag="Furnace 1.0")

        # should_skip_file recognises "Furnace" prefix in encoder tag
        analyzer = Analyzer(prober=prober)
        result = analyzer.analyze(scan_result)

        assert result is None

    def test_skip_when_output_exists(self, tmp_path: Path) -> None:
        """Output file already exists -> should_skip_file -> None."""
        scan_result = make_scan_result(tmp_path)
        # Create the output so should_skip_file sees it
        scan_result.output_path.parent.mkdir(parents=True, exist_ok=True)
        scan_result.output_path.write_bytes(b"x")
        prober = make_prober(probe_data=_h264_probe_data())

        analyzer = Analyzer(prober=prober)
        result = analyzer.analyze(scan_result)

        assert result is None

    def test_probe_raises_oserror(self, tmp_path: Path) -> None:
        """prober.probe raises OSError -> analyze returns None."""
        scan_result = make_scan_result(tmp_path)
        prober = make_prober()
        prober.probe.side_effect = OSError("disk failure")

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            analyzer = Analyzer(prober=prober)
            result = analyzer.analyze(scan_result)

        assert result is None

    def test_probe_raises_runtime_error(self, tmp_path: Path) -> None:
        """prober.probe raises RuntimeError -> analyze returns None."""
        scan_result = make_scan_result(tmp_path)
        prober = make_prober()
        prober.probe.side_effect = RuntimeError("ffprobe crash")

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            result = Analyzer(prober=prober).analyze(scan_result)

        assert result is None

    def test_probe_raises_value_error(self, tmp_path: Path) -> None:
        """prober.probe raises ValueError -> analyze returns None."""
        scan_result = make_scan_result(tmp_path)
        prober = make_prober()
        prober.probe.side_effect = ValueError("bad data")

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            result = Analyzer(prober=prober).analyze(scan_result)

        assert result is None

    def test_no_video_stream(self, tmp_path: Path) -> None:
        """Probe data with no video stream -> returns None."""
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data={
            "streams": [
                {"index": 0, "codec_type": "audio", "codec_name": "aac", "profile": "LC",
                 "channels": 2, "sample_rate": "48000",
                 "tags": {"language": "eng"}, "disposition": {"default": 1, "forced": 0}},
            ],
            "format": {"duration": "100.0"},
            "chapters": [],
        })

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            result = Analyzer(prober=prober).analyze(scan_result)

        assert result is None

    def test_parse_video_info_raises_key_error(self, tmp_path: Path) -> None:
        """_parse_video_info raises KeyError -> analyze returns None."""
        scan_result = make_scan_result(tmp_path)
        # Provide a video stream but patch _parse_video_info to raise
        prober = make_prober(probe_data={
            "streams": [{"index": 0, "codec_type": "video", "codec_name": "h264"}],
            "format": {},
            "chapters": [],
        })

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            analyzer = Analyzer(prober=prober)
            with patch.object(analyzer, "_parse_video_info", side_effect=KeyError("missing")):
                result = analyzer.analyze(scan_result)

        assert result is None

    def test_check_unsupported_codecs_returns_warning(self, tmp_path: Path) -> None:
        """check_unsupported_codecs returns a string -> analyze returns None."""
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=_h264_probe_data())

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch(
                "furnace.services.analyzer.check_unsupported_codecs",
                return_value="unsupported codecs detected: audio stream #1",
            ),
        ):
            result = Analyzer(prober=prober).analyze(scan_result)

        assert result is None


# ---------------------------------------------------------------------------
# Parsing fallbacks in _parse_video_info()
# ---------------------------------------------------------------------------


class TestParseVideoInfoFallbacks:
    """Cover fallback branches in _parse_video_info."""

    def _make_base_stream(self) -> dict[str, Any]:
        """Minimal video stream dict."""
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
        """FPS as plain number '25' (not fraction) -> parsed correctly."""
        stream = self._make_base_stream()
        stream["avg_frame_rate"] = "25"
        prober = MagicMock()
        prober.probe_hdr_side_data.return_value = []

        analyzer = Analyzer(prober=prober)
        vi = analyzer._parse_video_info(stream, {}, tmp_path / "movie.mkv")

        assert vi.fps_num == 25
        assert vi.fps_den == 1

    def test_fps_float_non_fraction_string(self, tmp_path: Path) -> None:
        """FPS as '23.976' -> parsed via int(float(...))."""
        stream = self._make_base_stream()
        stream["avg_frame_rate"] = "23.976"
        prober = MagicMock()
        prober.probe_hdr_side_data.return_value = []

        analyzer = Analyzer(prober=prober)
        vi = analyzer._parse_video_info(stream, {}, tmp_path / "movie.mkv")

        assert vi.fps_num == 23
        assert vi.fps_den == 1

    def test_duration_zero_in_stream_fallback_to_format(self, tmp_path: Path) -> None:
        """duration=0 in stream -> fallback to format duration."""
        stream = self._make_base_stream()
        stream["duration"] = "0"
        format_data = {"duration": "7200.5"}
        prober = MagicMock()
        prober.probe_hdr_side_data.return_value = []

        analyzer = Analyzer(prober=prober)
        vi = analyzer._parse_video_info(stream, format_data, tmp_path / "movie.mkv")

        assert vi.duration_s == pytest.approx(7200.5)

    def test_no_duration_in_stream_fallback_to_format(self, tmp_path: Path) -> None:
        """No duration key at all in stream -> fallback to format."""
        stream = self._make_base_stream()
        del stream["duration"]
        format_data = {"duration": "3600.0"}
        prober = MagicMock()
        prober.probe_hdr_side_data.return_value = []

        analyzer = Analyzer(prober=prober)
        vi = analyzer._parse_video_info(stream, format_data, tmp_path / "movie.mkv")

        assert vi.duration_s == pytest.approx(3600.0)

    def test_bitrate_zero_in_stream_fallback_to_format(self, tmp_path: Path) -> None:
        """bit_rate=0 in stream -> fallback to format bit_rate."""
        stream = self._make_base_stream()
        stream["bit_rate"] = "0"
        format_data = {"bit_rate": "5000000"}
        prober = MagicMock()
        prober.probe_hdr_side_data.return_value = []

        analyzer = Analyzer(prober=prober)
        vi = analyzer._parse_video_info(stream, format_data, tmp_path / "movie.mkv")

        assert vi.bitrate == 5000000

    def test_bitrate_not_in_stream_fallback_to_format(self, tmp_path: Path) -> None:
        """No bit_rate in stream at all -> fallback to format."""
        stream = self._make_base_stream()
        format_data = {"bit_rate": "8000000"}
        prober = MagicMock()
        prober.probe_hdr_side_data.return_value = []

        analyzer = Analyzer(prober=prober)
        vi = analyzer._parse_video_info(stream, format_data, tmp_path / "movie.mkv")

        assert vi.bitrate == 8000000

    def test_sar_parse_failure_defaults_to_1_1(self, tmp_path: Path) -> None:
        """Invalid SAR string -> defaults to 1:1."""
        stream = self._make_base_stream()
        stream["sample_aspect_ratio"] = "bad:data"
        prober = MagicMock()
        prober.probe_hdr_side_data.return_value = []

        analyzer = Analyzer(prober=prober)
        vi = analyzer._parse_video_info(stream, {}, tmp_path / "movie.mkv")

        assert vi.sar_num == 1
        assert vi.sar_den == 1

    def test_sample_rate_not_parseable(self, tmp_path: Path) -> None:
        """sample_rate with unparseable value -> None."""
        scan_result = make_scan_result(tmp_path)
        probe_data = _h264_probe_data()
        # Set a bad sample_rate on the first audio stream
        probe_data["streams"][1]["sample_rate"] = "invalid"
        prober = make_prober(probe_data=probe_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        eng_track = next(t for t in movie.audio_tracks if t.language == "eng")
        assert eng_track.sample_rate is None


# ---------------------------------------------------------------------------
# idet path
# ---------------------------------------------------------------------------


class TestIdetPath:
    """Cover interlace detection via idet."""

    def _interlaced_probe_data(self) -> dict[str, Any]:
        """Probe data with field_order=tt and low FPS (needs idet)."""
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
        """field_order=tt, fps<48 -> needs_idet -> run_idet called -> interlaced set."""
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=self._interlaced_probe_data())
        # idet returns high ratio -> should deinterlace
        prober.run_idet.return_value = 0.9

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None),
        ):
            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        assert movie.video.interlaced is True
        prober.run_idet.assert_called_once()

    def test_idet_low_ratio_stays_progressive(self, tmp_path: Path) -> None:
        """idet returns low ratio -> not interlaced."""
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=self._interlaced_probe_data())
        prober.run_idet.return_value = 0.01

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None),
        ):
            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        assert movie.video.interlaced is False

    def test_idet_exception_logged_and_continues(self, tmp_path: Path) -> None:
        """run_idet raises -> warning logged, continues with progressive."""
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=self._interlaced_probe_data())
        prober.run_idet.side_effect = RuntimeError("idet crash")

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None),
        ):
            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        # idet failed with ratio=0.0, field_order=tt, fps=25 -> should_deinterlace(tt,25,0.0)=False
        assert movie.video.interlaced is False

    def test_r_frame_rate_non_fraction(self, tmp_path: Path) -> None:
        """r_frame_rate as plain number (not fraction) -> parsed correctly for idet logic."""
        scan_result = make_scan_result(tmp_path)
        probe_data = self._interlaced_probe_data()
        probe_data["streams"][0]["r_frame_rate"] = "25"
        prober = make_prober(probe_data=probe_data)
        prober.run_idet.return_value = 0.9

        with (
            patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")),
            patch("furnace.services.analyzer.check_unsupported_codecs", return_value=None),
        ):
            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        assert movie.video.interlaced is True


# ---------------------------------------------------------------------------
# Satellite file processing
# ---------------------------------------------------------------------------


class TestExternalSubtitle:
    """Cover _parse_external_subtitle."""

    def test_srt_satellite_language_from_filename(self, tmp_path: Path) -> None:
        """SRT satellite with language code in filename -> parsed correctly."""
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

            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        # Should have original PGS sub + external SRT
        sat_subs = [t for t in movie.subtitle_tracks if t.source_file == srt_path]
        assert len(sat_subs) == 1
        sub = sat_subs[0]
        assert sub.language == "eng"
        assert sub.codec_id == SubtitleCodecId.SRT
        assert sub.is_forced is False
        assert sub.encoding == "utf-8"

    def test_forced_keyword_in_filename(self, tmp_path: Path) -> None:
        """'forced' in filename -> is_forced=True."""
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

            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        sat_subs = [t for t in movie.subtitle_tracks if t.source_file == srt_path]
        assert len(sat_subs) == 1
        assert sat_subs[0].is_forced is True
        assert sat_subs[0].language == "rus"

    def test_sup_satellite_no_encoding_detection(self, tmp_path: Path) -> None:
        """SUP (PGS) satellite -> no encoding detection (binary sub)."""
        sup_path = tmp_path / "movie.jpn.sup"
        sup_path.write_bytes(b"\x00" * 16)

        scan_result = ScanResult(
            main_file=make_scan_result(tmp_path).main_file,
            satellite_files=[sup_path],
            output_path=make_scan_result(tmp_path).output_path,
        )
        prober = make_prober(probe_data=_h264_probe_data())

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        sat_subs = [t for t in movie.subtitle_tracks if t.source_file == sup_path]
        assert len(sat_subs) == 1
        sub = sat_subs[0]
        assert sub.codec_id == SubtitleCodecId.PGS
        assert sub.language == "jpn"
        assert sub.encoding is None

    def test_ass_satellite_encoding_detection(self, tmp_path: Path) -> None:
        """ASS satellite -> encoding detection called."""
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

            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        sat_subs = [t for t in movie.subtitle_tracks if t.source_file == ass_path]
        assert len(sat_subs) == 1
        assert sat_subs[0].encoding == "windows-1251"


class TestExternalAudio:
    """Cover _parse_external_audio."""

    def _audio_probe_data(self) -> dict[str, Any]:
        """Probe data for an external audio file."""
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
        """External .flac audio satellite -> probed and added to audio_tracks."""
        flac_path = tmp_path / "movie.eng.flac"
        flac_path.write_bytes(b"\x00" * 256)

        main_scan = make_scan_result(tmp_path)
        scan_result = ScanResult(
            main_file=main_scan.main_file,
            satellite_files=[flac_path],
            output_path=main_scan.output_path,
        )
        prober = make_prober(probe_data=_h264_probe_data())
        # Make prober.probe return different data depending on the path
        sat_probe = self._audio_probe_data()

        def probe_side_effect(path: Path) -> dict[str, Any]:
            if path == flac_path:
                return sat_probe
            return _h264_probe_data()

        prober.probe.side_effect = probe_side_effect

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        sat_audio = [t for t in movie.audio_tracks if t.source_file == flac_path]
        assert len(sat_audio) == 1
        track = sat_audio[0]
        assert track.codec_name == "flac"
        assert track.channels == 6
        # Index should be base_index = len(audio_tracks from main)
        assert track.index == 2  # main has 2 audio tracks, so base_index=2

    def test_external_audio_probe_fails(self, tmp_path: Path) -> None:
        """External audio probe raises -> satellite skipped, no crash."""
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
            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        # Only the 2 audio tracks from main file
        assert len(movie.audio_tracks) == 2

    def test_external_audio_no_audio_streams(self, tmp_path: Path) -> None:
        """External audio file with no audio streams -> returns None."""
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
            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        assert len(movie.audio_tracks) == 2  # only main tracks


# ---------------------------------------------------------------------------
# Attachments
# ---------------------------------------------------------------------------


class TestAttachments:
    """Cover _parse_attachments."""

    def test_attachments_parsed(self, tmp_path: Path) -> None:
        """Probe data with attachment streams -> Attachment objects."""
        probe_data = _h264_probe_data()
        probe_data["streams"].append({
            "index": 4,
            "codec_type": "attachment",
            "tags": {
                "filename": "Arial.ttf",
                "mimetype": "application/x-truetype-font",
            },
        })
        probe_data["streams"].append({
            "index": 5,
            "codec_type": "attachment",
            "tags": {
                "filename": "OpenSans.otf",
                "mime_type": "font/otf",
            },
        })
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=probe_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        assert len(movie.attachments) == 2
        assert movie.attachments[0].filename == "Arial.ttf"
        assert movie.attachments[0].mime_type == "application/x-truetype-font"
        assert movie.attachments[1].filename == "OpenSans.otf"
        assert movie.attachments[1].mime_type == "font/otf"

    def test_attachment_no_filename_skipped(self, tmp_path: Path) -> None:
        """Attachment stream with no filename -> not added."""
        probe_data = _h264_probe_data()
        probe_data["streams"].append({
            "index": 4,
            "codec_type": "attachment",
            "tags": {"mimetype": "application/octet-stream"},
        })
        scan_result = make_scan_result(tmp_path)
        prober = make_prober(probe_data=probe_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        assert len(movie.attachments) == 0


# ---------------------------------------------------------------------------
# Encoding detection
# ---------------------------------------------------------------------------


class TestTextEncodingDetection:
    """Cover _detect_text_encoding."""

    def test_encoding_detected_successfully(self, tmp_path: Path) -> None:
        """charset_normalizer returns a best result -> encoding string."""
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
        """charset_normalizer best() returns None -> None."""
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
        """charset_normalizer raises OSError -> None."""
        prober = MagicMock()
        analyzer = Analyzer(prober=prober)
        srt = tmp_path / "test.srt"

        with patch("furnace.services.analyzer._from_path", side_effect=OSError("file gone")):
            result = analyzer._detect_text_encoding(srt)

        assert result is None

    def test_encoding_detection_value_error(self, tmp_path: Path) -> None:
        """charset_normalizer raises ValueError -> None."""
        prober = MagicMock()
        analyzer = Analyzer(prober=prober)
        srt = tmp_path / "test.srt"

        with patch("furnace.services.analyzer._from_path", side_effect=ValueError("bad data")):
            result = analyzer._detect_text_encoding(srt)

        assert result is None


# ---------------------------------------------------------------------------
# num_frames tag parsing
# ---------------------------------------------------------------------------


class TestNumFramesParsing:
    """Cover NUMBER_OF_FRAMES tag parsing in subtitle tracks."""

    def test_number_of_frames_tag_parsed(self, tmp_path: Path) -> None:
        """Subtitle stream with NUMBER_OF_FRAMES tag -> num_frames set."""
        scan_result = make_scan_result(tmp_path)
        probe_data = _h264_probe_data()
        # The existing probe data already has NUMBER_OF_FRAMES: "120" on subtitle
        prober = make_prober(probe_data=probe_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        sub = movie.subtitle_tracks[0]
        assert sub.num_frames == 120

    def test_number_of_frames_eng_tag(self, tmp_path: Path) -> None:
        """NUMBER_OF_FRAMES-eng tag -> parsed as num_frames."""
        scan_result = make_scan_result(tmp_path)
        probe_data = _h264_probe_data()
        # Replace NUMBER_OF_FRAMES with NUMBER_OF_FRAMES-eng
        sub_stream = probe_data["streams"][3]
        del sub_stream["tags"]["NUMBER_OF_FRAMES"]
        sub_stream["tags"]["NUMBER_OF_FRAMES-eng"] = "250"
        prober = make_prober(probe_data=probe_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        sub = movie.subtitle_tracks[0]
        assert sub.num_frames == 250

    def test_number_of_frames_invalid_value(self, tmp_path: Path) -> None:
        """Invalid NUMBER_OF_FRAMES value -> num_frames stays None."""
        scan_result = make_scan_result(tmp_path)
        probe_data = _h264_probe_data()
        probe_data["streams"][3]["tags"]["NUMBER_OF_FRAMES"] = "not_a_number"
        prober = make_prober(probe_data=probe_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        sub = movie.subtitle_tracks[0]
        assert sub.num_frames is None


# ---------------------------------------------------------------------------
# Audio bitrate parsing
# ---------------------------------------------------------------------------


class TestAudioBitrateParsing:
    """Cover bitrate fallback logic in _parse_audio_tracks."""

    def test_bitrate_from_tags_bps(self, tmp_path: Path) -> None:
        """Bitrate from tags.BPS when stream bit_rate is missing."""
        scan_result = make_scan_result(tmp_path)
        probe_data = _h264_probe_data()
        # Remove bit_rate from stream, add BPS tag
        audio_stream = probe_data["streams"][2]  # rus AC3
        audio_stream.pop("bit_rate", None)
        audio_stream["tags"]["BPS"] = "640000"
        prober = make_prober(probe_data=probe_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        rus_track = next(t for t in movie.audio_tracks if t.language == "rus")
        assert rus_track.bitrate == 640000


# ---------------------------------------------------------------------------
# _TEXT_SUBTITLE_CODECS import
# ---------------------------------------------------------------------------


class TestTextSubtitleCodecsSet:
    """Verify the internal set is correctly defined."""

    def test_text_subtitle_codecs_contains_srt_and_ass(self) -> None:
        assert SubtitleCodecId.SRT in _TEXT_SUBTITLE_CODECS
        assert SubtitleCodecId.ASS in _TEXT_SUBTITLE_CODECS
        assert SubtitleCodecId.PGS not in _TEXT_SUBTITLE_CODECS


# ---------------------------------------------------------------------------
# Branch coverage: satellite file loop — unknown extensions ignored
# ---------------------------------------------------------------------------


class TestSatelliteUnknownExtension:
    """Satellite files with unrecognised extensions are silently skipped."""

    def test_unknown_extension_satellite_skipped(self, tmp_path: Path) -> None:
        """A .nfo satellite is neither subtitle nor audio -> silently ignored."""
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
            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        # Track counts unchanged from main file
        assert len(movie.audio_tracks) == 2
        assert len(movie.subtitle_tracks) == 1


# ---------------------------------------------------------------------------
# Branch coverage: SAR without colon -> skip parsing
# ---------------------------------------------------------------------------


class TestSarNoColon:
    """SAR string without colon -> defaults kept at 1:1."""

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
            "sample_aspect_ratio": "1",  # no colon
        }
        prober = MagicMock()
        prober.probe_hdr_side_data.return_value = []

        vi = Analyzer(prober=prober)._parse_video_info(stream, {}, tmp_path / "m.mkv")

        assert vi.sar_num == 1
        assert vi.sar_den == 1

    def test_sar_empty_string_defaults(self, tmp_path: Path) -> None:
        """Empty SAR string -> defaults to 1:1."""
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


# ---------------------------------------------------------------------------
# Branch coverage: audio bitrate — none at all
# ---------------------------------------------------------------------------


class TestAudioNoBitrate:
    """Audio stream with no bit_rate and no BPS tags -> bitrate is None."""

    def test_no_bitrate_anywhere(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        probe_data = _h264_probe_data()
        # Remove all bitrate sources from the eng AAC track
        audio_stream = probe_data["streams"][1]
        audio_stream.pop("bit_rate", None)
        audio_stream["tags"].pop("BPS", None)
        audio_stream["tags"].pop("BPS-eng", None)
        prober = make_prober(probe_data=probe_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        eng_track = next(t for t in movie.audio_tracks if t.language == "eng")
        assert eng_track.bitrate is None


# ---------------------------------------------------------------------------
# Branch coverage: subtitle with no NUMBER_OF_FRAMES tags
# ---------------------------------------------------------------------------


class TestSubtitleNoFramesTags:
    """Subtitle stream with neither NUMBER_OF_FRAMES nor NUMBER_OF_FRAMES-eng."""

    def test_no_frames_tags(self, tmp_path: Path) -> None:
        scan_result = make_scan_result(tmp_path)
        probe_data = _h264_probe_data()
        del probe_data["streams"][3]["tags"]["NUMBER_OF_FRAMES"]
        prober = make_prober(probe_data=probe_data)

        with patch("furnace.services.analyzer.should_skip_file", return_value=(False, "")):
            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        assert movie.subtitle_tracks[0].num_frames is None


# ---------------------------------------------------------------------------
# Branch coverage: external subtitle language code not found
# ---------------------------------------------------------------------------


class TestExternalSubtitleNoLanguageCode:
    """External subtitle with no valid 3-letter language code in filename."""

    def test_no_language_code_in_filename(self, tmp_path: Path) -> None:
        """movie.srt (no language part) -> language='und'."""
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

            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        sat_subs = [t for t in movie.subtitle_tracks if t.source_file == srt_path]
        assert len(sat_subs) == 1
        assert sat_subs[0].language == "und"

    def test_non_alpha_part_in_filename(self, tmp_path: Path) -> None:
        """movie.720p.srt -> no valid lang code -> und."""
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

            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        sat_subs = [t for t in movie.subtitle_tracks if t.source_file == srt_path]
        assert len(sat_subs) == 1
        assert sat_subs[0].language == "und"


# ---------------------------------------------------------------------------
# Branch coverage: _parse_audio_tracks returns empty list (line 403)
# ---------------------------------------------------------------------------


class TestExternalAudioEmptyTracks:
    """Cover the edge case where _parse_audio_tracks returns [] despite audio streams."""

    def test_parse_audio_tracks_returns_empty(self, tmp_path: Path) -> None:
        """_parse_audio_tracks returning [] from satellite -> satellite skipped."""
        ac3_path = tmp_path / "movie.eng.ac3"
        ac3_path.write_bytes(b"\x00" * 256)

        prober = MagicMock()
        analyzer = Analyzer(prober=prober)

        # Probe returns audio stream but we patch _parse_audio_tracks to return []
        prober.probe.return_value = {
            "streams": [{"index": 0, "codec_type": "audio", "codec_name": "ac3"}],
            "format": {},
        }
        with patch.object(analyzer, "_parse_audio_tracks", return_value=[]):
            result = analyzer._parse_external_audio(ac3_path, 5)

        assert result is None


# ---------------------------------------------------------------------------
# Branch coverage: audio stream with no sample_rate key
# ---------------------------------------------------------------------------


class TestAudioNoSampleRate:
    """Audio stream with no sample_rate key at all -> sample_rate is None."""

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
                    # no sample_rate key at all
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
            movie = Analyzer(prober=prober).analyze(scan_result)

        assert movie is not None
        assert movie.audio_tracks[0].sample_rate is None


# ---------------------------------------------------------------------------
# Branch coverage: external subtitle returns None (mock)
# ---------------------------------------------------------------------------


class TestExternalSubtitleReturnsNone:
    """Cover the 95->90 branch where _parse_external_subtitle returns None."""

    def test_external_subtitle_returns_none_skipped(self, tmp_path: Path) -> None:
        """When _parse_external_subtitle returns None, the satellite is skipped."""
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
                movie = analyzer.analyze(scan_result)

        assert movie is not None
        # Only the embedded PGS subtitle, no satellite
        assert len(movie.subtitle_tracks) == 1
