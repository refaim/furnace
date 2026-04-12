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
from furnace.services.analyzer import Analyzer

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
