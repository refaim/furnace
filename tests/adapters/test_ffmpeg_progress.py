from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from furnace.adapters.ffmpeg import FFmpegAdapter, _parse_ffmpeg_progress_block
from furnace.core.models import CropRect
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
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = "[Parsed_cropdetect] crop=1920:800:0:140\n"
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

    def test_detect_crop_dvd_uses_larger_batches(self) -> None:
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
        assert call_count == 30

    def test_detect_crop_interlaced_uses_yadif(self) -> None:
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
        for cmd in captured_cmds:
            vf_idx = cmd.index("-vf")
            assert "yadif" in cmd[vf_idx + 1]


class TestDetectCropAggregation:
    def test_detect_crop_converges_on_dominant_cluster(self) -> None:
        adapter = _adapter()
        call_count = 0
        pattern = [
            "crop=1600:800:160:140",
            "crop=1600:800:160:140",
            "crop=1600:800:160:140",
            "crop=1600:800:160:140",
            "crop=1600:800:160:140",
            "crop=1600:760:160:160",
            "crop=1600:720:160:180",
            "crop=1600:680:160:200",
            "crop=1600:640:160:220",
            "crop=1600:600:160:240",
        ]

        def cycling_run(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            mock = MagicMock()
            mock.returncode = 0
            mock.stderr = f"[cropdetect] {pattern[call_count % len(pattern)]}\n"
            call_count += 1
            return mock

        with patch("furnace.adapters.ffmpeg.subprocess.run", side_effect=cycling_run):
            crop = adapter.detect_crop(Path("v.mkv"), duration_s=100.0)
        assert crop == CropRect(w=1600, h=800, x=160, y=140)
        assert call_count == 20

    def test_detect_crop_runs_to_cap_when_never_converging(self) -> None:
        adapter = _adapter()
        batch_dominant = [1000, 1100, 1200, 1300]
        batch_fillers = [
            [10, 20, 30, 40, 50],
            [60, 70, 80, 90],
            [110, 120, 130],
            [140, 150],
        ]
        heights: list[int] = []
        for b, dom in enumerate(batch_dominant):
            heights += [dom] * (10 - len(batch_fillers[b])) + batch_fillers[b]

        call_count = 0

        def shifting_run(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            mock = MagicMock()
            mock.returncode = 0
            mock.stderr = f"[cropdetect] crop=100:{heights[call_count]}:0:0\n"
            call_count += 1
            return mock

        with patch("furnace.adapters.ffmpeg.subprocess.run", side_effect=shifting_run):
            crop = adapter.detect_crop(Path("v.mkv"), duration_s=100.0)
        assert call_count == 40
        assert crop == CropRect(w=100, h=1300, x=0, y=0)


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
        results: list[bool] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            results.append(on_progress_line("no equals sign here"))
            return 0, ""

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            adapter.extract_track(Path("v.mkv"), 2, Path("out.thd"))
        assert results == [False]

    def test_extract_track_without_on_progress_skips_callback(self) -> None:
        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            assert on_progress_line("out_time_us=1000000") is True
            assert on_progress_line("speed=1.5x") is True
            assert on_progress_line("progress=continue") is True
            return 0, ""

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            rc = adapter.extract_track(Path("v.mkv"), 2, Path("out.thd"))
        assert rc == 0


class TestFfmpegToWav:
    @staticmethod
    def _run(input_name: str = "audio.thd") -> tuple[int, list[str]]:
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
            rc = adapter.ffmpeg_to_wav(Path(input_name), 1, Path("out.wav"))
        return rc, captured

    def test_ffmpeg_to_wav_cmd(self) -> None:
        rc, captured = self._run()
        assert rc == 0
        assert "-f" in captured
        assert "wav" in captured
        assert "-rf64" in captured
        assert "auto" in captured
        assert "0:1" in captured

    def test_ffmpeg_to_wav_asks_for_24_bit(self) -> None:
        _rc, captured = self._run(input_name="audio.m4a")
        assert ("-c:a", "pcm_s24le") in pairwise(captured)

    def test_ffmpeg_to_wav_does_not_downmix(self) -> None:
        _rc, captured = self._run(input_name="audio.m4a")
        assert "-ac" not in captured

    def test_ffmpeg_to_wav_progress(self) -> None:
        samples: list[ProgressSample] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
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
            results.append(on_progress_line("plain text"))
            return 0, ""

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            adapter.ffmpeg_to_wav(Path("a.thd"), 1, Path("out.wav"))
        assert results == [False]

    def test_ffmpeg_to_wav_without_on_progress_skips_callback(self) -> None:
        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            assert on_progress_line("out_time_us=1000000") is True
            assert on_progress_line("speed=1.5x") is True
            assert on_progress_line("progress=continue") is True
            return 0, ""

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            rc = adapter.ffmpeg_to_wav(Path("a.thd"), 1, Path("out.wav"))
        assert rc == 0


class TestDecodeFullWav:
    @staticmethod
    def _run(*, disable_drc: bool = False, rc_value: int = 0) -> tuple[int, list[str]]:
        captured: list[str] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            captured.extend(str(c) for c in cmd)
            return rc_value, ""

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            rc = adapter.decode_full_wav(Path("audio.ac3"), 1, Path("out.wav"), disable_drc=disable_drc)
        return rc, captured

    def test_decode_full_wav_cmd_is_tolerant_wav(self) -> None:
        rc, captured = self._run()
        assert rc == 0
        assert ("-err_detect", "ignore_err") in pairwise(captured)
        assert ("-c:a", "pcm_s24le") in pairwise(captured)
        assert "-f" in captured
        assert "wav" in captured
        assert "-rf64" in captured
        assert "0:1" in captured

    def test_decode_full_wav_disable_drc_adds_flag_before_input(self) -> None:
        _rc, captured = self._run(disable_drc=True)
        assert ("-drc_scale", "0") in pairwise(captured)
        assert captured.index("-drc_scale") < captured.index("-i")

    def test_decode_full_wav_default_omits_drc_flag(self) -> None:
        _rc, captured = self._run()
        assert "-drc_scale" not in captured

    def test_decode_full_wav_passes_return_code(self) -> None:
        rc, _captured = self._run(rc_value=3)
        assert rc == 3

    def test_decode_full_wav_progress(self) -> None:
        samples: list[ProgressSample] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            on_progress_line("out_time_us=30000000")
            on_progress_line("speed=1.0x")
            on_progress_line("progress=continue")
            return 0, ""

        adapter = _adapter()
        with patch("furnace.adapters.ffmpeg.run_tool", side_effect=fake_run_tool):
            adapter.decode_full_wav(Path("a.ac3"), 1, Path("out.wav"), on_progress=samples.append)
        assert len(samples) == 1


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
