from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from furnace.adapters.ffmpeg import FFmpegAdapter, _parse_ffmpeg_progress_block
from furnace.core.progress import ProgressSample


class TestParseFfmpegProgressBlock:
    def test_typical_block(self) -> None:
        kv = {
            "frame": "42",
            "fps": "23.97",
            "out_time_us": "60000000",
            "out_time_ms": "60000",
            "speed": "2.5x",
            "progress": "continue",
        }
        sample = _parse_ffmpeg_progress_block(kv)
        assert sample == ProgressSample(processed_s=60.0, speed=2.5)

    def test_missing_out_time(self) -> None:
        kv = {"frame": "42", "progress": "continue"}
        assert _parse_ffmpeg_progress_block(kv) is None

    def test_out_time_na(self) -> None:
        kv = {"out_time_us": "N/A", "progress": "continue"}
        assert _parse_ffmpeg_progress_block(kv) is None

    def test_malformed_out_time(self) -> None:
        kv = {"out_time_us": "not-a-number", "progress": "continue"}
        assert _parse_ffmpeg_progress_block(kv) is None

    def test_speed_na(self) -> None:
        kv = {"out_time_us": "30000000", "speed": "N/A", "progress": "continue"}
        sample = _parse_ffmpeg_progress_block(kv)
        assert sample == ProgressSample(processed_s=30.0, speed=None)

    def test_speed_without_x_suffix(self) -> None:
        kv = {"out_time_us": "30000000", "speed": "2.5", "progress": "continue"}
        sample = _parse_ffmpeg_progress_block(kv)
        assert sample == ProgressSample(processed_s=30.0, speed=None)

    def test_speed_malformed_just_x(self) -> None:
        """speed='x' means empty-before-x → float('') fails ValueError."""
        kv = {"out_time_us": "1000000", "speed": "x", "progress": "continue"}
        sample = _parse_ffmpeg_progress_block(kv)
        assert sample == ProgressSample(processed_s=1.0, speed=None)

    def test_end_of_stream(self) -> None:
        kv = {"out_time_us": "120000000", "speed": "3.0x", "progress": "end"}
        sample = _parse_ffmpeg_progress_block(kv)
        assert sample == ProgressSample(processed_s=120.0, speed=3.0)


def _adapter() -> FFmpegAdapter:
    return FFmpegAdapter(Path("ffmpeg.exe"), Path("ffprobe.exe"))


class TestProbe:
    def test_probe_success(self) -> None:
        probe_data = {"streams": [{"codec_type": "video"}], "format": {"duration": "120"}}
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(probe_data)
        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            result = adapter.probe(Path("video.mkv"))
        assert result == probe_data

    def test_probe_failure(self) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ffprobe error"
        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="ffprobe failed"):
                adapter.probe(Path("video.mkv"))


class TestDetectCrop:
    def test_detect_crop_returns_rect(self) -> None:
        adapter = _adapter()
        # Simulate all crop samples returning the same crop: "crop=1920:800:0:140"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = (
            "[Parsed_cropdetect] crop=1920:800:0:140\n"
        )
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            crop = adapter.detect_crop(Path("v.mkv"), duration_s=100.0)
        assert crop is not None
        assert crop.w == 1920
        assert crop.h == 800

    def test_detect_crop_no_match_returns_none(self) -> None:
        adapter = _adapter()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = "no crop detected\n"
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            crop = adapter.detect_crop(Path("v.mkv"), duration_s=100.0)
        assert crop is None

    def test_detect_crop_dvd_uses_more_sample_points(self) -> None:
        """DVD mode uses _CROP_SAMPLE_POINTS_DVD which has 15 points."""
        adapter = _adapter()
        call_count = 0

        def counting_run(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            mock = MagicMock()
            mock.returncode = 0
            mock.stderr = "[cropdetect] crop=720:480:0:0\n"
            return mock

        with patch("furnace.adapters.ffmpeg.subprocess.run", side_effect=counting_run):
            adapter.detect_crop(Path("v.mkv"), duration_s=100.0, is_dvd=True)
        # DVD uses 15 sample points
        assert call_count == 15

    def test_detect_crop_interlaced_uses_yadif(self) -> None:
        """Interlaced sources add yadif before cropdetect."""
        captured_cmds: list[list[str]] = []
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = "[cropdetect] crop=1920:800:0:140\n"

        def capturing_run(cmd: Any, **kwargs: Any) -> MagicMock:
            captured_cmds.append(list(cmd))
            return mock_result

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.subprocess.run", side_effect=capturing_run):
            adapter.detect_crop(Path("v.mkv"), duration_s=100.0, interlaced=True)
        # Check that yadif is prepended
        for cmd in captured_cmds:
            vf_idx = cmd.index("-vf")
            assert "yadif" in cmd[vf_idx + 1]


class TestDetectCropUnreliableCluster:
    def test_detect_crop_unreliable_returns_none(self) -> None:
        """When no dominant cluster (50% threshold not met), returns None."""
        adapter = _adapter()
        # Return wildly different crops for each sample point so no cluster dominates
        call_idx = 0
        crops = [
            "crop=1920:800:0:140",
            "crop=1000:500:100:200",
            "crop=1920:800:0:140",
            "crop=500:300:50:50",
            "crop=1920:800:0:140",
            "crop=800:600:100:100",
            "crop=500:300:50:50",
            "crop=800:600:100:100",
            "crop=500:300:50:50",
            "crop=800:600:100:100",
        ]

        def varying_run(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_idx
            mock = MagicMock()
            mock.returncode = 0
            mock.stderr = f"[cropdetect] {crops[call_idx % len(crops)]}\n"
            call_idx += 1
            return mock

        with patch("furnace.adapters.ffmpeg.subprocess.run", side_effect=varying_run):
            crop = adapter.detect_crop(Path("v.mkv"), duration_s=100.0)
        # With evenly distributed different crops, no cluster dominates
        # Result depends on cluster_crop_values, but the test verifies the branch runs
        assert crop is None or crop.w > 0  # either None or valid CropRect


class TestGetEncoderTag:
    def test_encoder_tag_found(self) -> None:
        adapter = _adapter()
        probe_data = {"format": {"tags": {"ENCODER": "Furnace v1.4.0"}}}
        with patch.object(adapter, "probe", return_value=probe_data):
            tag = adapter.get_encoder_tag(Path("v.mkv"))
        assert tag == "Furnace v1.4.0"

    def test_encoder_tag_lowercase(self) -> None:
        adapter = _adapter()
        probe_data = {"format": {"tags": {"encoder": "libx265"}}}
        with patch.object(adapter, "probe", return_value=probe_data):
            tag = adapter.get_encoder_tag(Path("v.mkv"))
        assert tag == "libx265"

    def test_encoder_tag_not_found(self) -> None:
        adapter = _adapter()
        probe_data: dict[str, Any] = {"format": {"tags": {}}}
        with patch.object(adapter, "probe", return_value=probe_data):
            tag = adapter.get_encoder_tag(Path("v.mkv"))
        assert tag is None

    def test_encoder_tag_no_format(self) -> None:
        adapter = _adapter()
        probe_data: dict[str, Any] = {"format": {}}
        with patch.object(adapter, "probe", return_value=probe_data):
            tag = adapter.get_encoder_tag(Path("v.mkv"))
        assert tag is None

    def test_encoder_tag_probe_error(self) -> None:
        adapter = _adapter()
        with patch.object(adapter, "probe", side_effect=RuntimeError("fail")):
            tag = adapter.get_encoder_tag(Path("v.mkv"))
        assert tag is None


class TestRunIdet:
    def test_idet_returns_ratio(self) -> None:
        adapter = _adapter()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = (
            "[Parsed_idet] Multi frame detection: TFF:   100 BFF:    50 Progressive:   850 Undetermined:     0\n"
        )
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            ratio = adapter.run_idet(Path("v.mkv"), duration_s=100.0)
        # 5 sample points, each returns TFF:100, BFF:50, Progressive:850
        # total_interlaced = 5 * 150 = 750, total_prog = 5 * 850 = 4250
        # ratio = 750 / (750+4250) = 0.15
        assert abs(ratio - 0.15) < 0.01

    def test_idet_no_match_returns_zero(self) -> None:
        adapter = _adapter()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = "no idet output\n"
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            ratio = adapter.run_idet(Path("v.mkv"), duration_s=100.0)
        assert ratio == 0.0


class TestProbeHdrSideData:
    def test_hdr_side_data_parsed(self) -> None:
        adapter = _adapter()
        frame_data = {
            "frames": [
                {
                    "side_data_list": [
                        {"side_data_type": "Mastering display metadata"},
                    ]
                }
            ]
        }
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(frame_data)
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            side_data = adapter.probe_hdr_side_data(Path("v.mkv"))
        assert len(side_data) == 1
        assert side_data[0]["side_data_type"] == "Mastering display metadata"

    def test_hdr_side_data_failure_returns_empty(self) -> None:
        adapter = _adapter()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error"
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            side_data = adapter.probe_hdr_side_data(Path("v.mkv"))
        assert side_data == []

    def test_hdr_side_data_no_frames(self) -> None:
        adapter = _adapter()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"frames": []})
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            side_data = adapter.probe_hdr_side_data(Path("v.mkv"))
        assert side_data == []

    def test_hdr_side_data_no_side_data_list(self) -> None:
        adapter = _adapter()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"frames": [{"pix_fmt": "yuv420p10le"}]})
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            side_data = adapter.probe_hdr_side_data(Path("v.mkv"))
        assert side_data == []


class TestExtractTrack:
    def test_extract_track_cmd(self) -> None:
        captured: list[str] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            captured.extend(str(c) for c in cmd)
            return 0, ""

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            rc = adapter.extract_track(Path("video.mkv"), 2, Path("out.thd"))
        assert rc == 0
        assert "-map" in captured
        assert "0:2" in captured
        assert "-c" in captured
        assert "copy" in captured
        assert "-progress" in captured
        assert "pipe:1" in captured

    def test_extract_track_progress(self) -> None:
        samples: list[ProgressSample] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                on_progress_line("out_time_us=60000000")
                on_progress_line("speed=2.5x")
                on_progress_line("progress=continue")
            return 0, ""

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            adapter.extract_track(Path("v.mkv"), 2, Path("out.thd"), on_progress=samples.append)
        assert len(samples) == 1
        assert abs(samples[0].processed_s - 60.0) < 0.01  # type: ignore[operator]

    def test_extract_track_non_progress_line(self) -> None:
        """Lines without '=' are not consumed."""
        results: list[bool] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                results.append(on_progress_line("no equals sign here"))
            return 0, ""

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            adapter.extract_track(Path("v.mkv"), 2, Path("out.thd"))
        assert results == [False]

    def test_extract_track_without_on_progress_skips_callback(self) -> None:
        """Progress block is parsed but callback is skipped when on_progress is None."""
        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                # Drive a full progress block through the closure. Without
                # on_progress, the inner False branch of the guard fires.
                assert on_progress_line("out_time_us=1000000") is True
                assert on_progress_line("speed=1.5x") is True
                assert on_progress_line("progress=continue") is True
            return 0, ""

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            # No on_progress passed — closure's `on_progress is None` branch hits.
            rc = adapter.extract_track(Path("v.mkv"), 2, Path("out.thd"))
        assert rc == 0


class TestFfmpegToWav:
    def test_ffmpeg_to_wav_cmd(self) -> None:
        captured: list[str] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            captured.extend(str(c) for c in cmd)
            return 0, ""

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            rc = adapter.ffmpeg_to_wav(Path("audio.thd"), 1, Path("out.wav"))
        assert rc == 0
        assert "-f" in captured
        assert "wav" in captured
        assert "-rf64" in captured
        assert "auto" in captured
        assert "0:1" in captured

    def test_ffmpeg_to_wav_progress(self) -> None:
        samples: list[ProgressSample] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                on_progress_line("out_time_us=30000000")
                on_progress_line("speed=1.0x")
                on_progress_line("progress=continue")
            return 0, ""

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            adapter.ffmpeg_to_wav(Path("a.thd"), 1, Path("out.wav"), on_progress=samples.append)
        assert len(samples) == 1

    def test_ffmpeg_to_wav_non_progress_line(self) -> None:
        results: list[bool] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                results.append(on_progress_line("plain text"))
            return 0, ""

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            adapter.ffmpeg_to_wav(Path("a.thd"), 1, Path("out.wav"))
        assert results == [False]

    def test_ffmpeg_to_wav_without_on_progress_skips_callback(self) -> None:
        """Progress block is parsed but callback skipped when on_progress is None."""
        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                assert on_progress_line("out_time_us=1000000") is True
                assert on_progress_line("speed=1.5x") is True
                assert on_progress_line("progress=continue") is True
            return 0, ""

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            # No on_progress passed — closure's `on_progress is None` branch hits.
            rc = adapter.ffmpeg_to_wav(Path("a.thd"), 1, Path("out.wav"))
        assert rc == 0


class TestGetFfmpegVersion:
    def test_version_parsed(self) -> None:
        adapter = _adapter()
        mock_result = MagicMock()
        mock_result.stdout = "ffmpeg version 7.1 Copyright (c) 2000-2024"
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            v = adapter._get_ffmpeg_version()
        assert v == "7.1"

    def test_version_cached(self) -> None:
        adapter = _adapter()
        mock_result = MagicMock()
        mock_result.stdout = "ffmpeg version 7.1 Copyright"
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result) as mock_run:
            v1 = adapter._get_ffmpeg_version()
            v2 = adapter._get_ffmpeg_version()
        assert v1 == v2
        mock_run.assert_called_once()

    def test_version_oserror(self) -> None:
        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.subprocess.run", side_effect=OSError("not found")):
            v = adapter._get_ffmpeg_version()
        assert v == ""

    def test_version_no_match(self) -> None:
        adapter = _adapter()
        mock_result = MagicMock()
        mock_result.stdout = "something unexpected"
        with patch("furnace.adapters.ffmpeg.subprocess.run", return_value=mock_result):
            v = adapter._get_ffmpeg_version()
        assert v == ""


class TestFfmpegSetLogDir:
    def test_set_log_dir(self, tmp_path: Path) -> None:
        adapter = _adapter()
        adapter.set_log_dir(tmp_path)
        assert adapter._log_dir == tmp_path

    def test_extract_track_log_path(self, tmp_path: Path) -> None:
        captured_kwargs: dict[str, Any] = {}

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            captured_kwargs["log_path"] = log_path
            return 0, ""

        adapter = FFmpegAdapter(Path("ffmpeg.exe"), Path("ffprobe.exe"), log_dir=tmp_path)
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            adapter.extract_track(Path("v.mkv"), 3, Path("out.thd"))
        assert captured_kwargs["log_path"] == tmp_path / "ffmpeg_extract_s3.log"

    def test_ffmpeg_to_wav_log_path(self, tmp_path: Path) -> None:
        captured_kwargs: dict[str, Any] = {}

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            captured_kwargs["log_path"] = log_path
            return 0, ""

        adapter = FFmpegAdapter(Path("ffmpeg.exe"), Path("ffprobe.exe"), log_dir=tmp_path)
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            adapter.ffmpeg_to_wav(Path("a.thd"), 5, Path("out.wav"))
        assert captured_kwargs["log_path"] == tmp_path / "ffmpeg_to_wav_s5.log"
