from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from furnace.adapters.mkvmerge import MkvmergeAdapter
from furnace.core.progress import ProgressSample

_UHD_1000_NITS = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,0)"


def _build_cmd(
    video_meta: dict[str, Any] | None = None,
    audio_files: list[tuple[Path, dict[str, Any]]] | None = None,
    subtitle_files: list[tuple[Path, dict[str, Any]]] | None = None,
    attachments: list[tuple[Path, str, str]] | None = None,
    chapters_source: Path | None = None,
) -> list[str]:
    adapter = MkvmergeAdapter(Path("mkvmerge.exe"))
    return adapter._build_mux_cmd(
        video_path=Path("video.obu"),
        audio_files=audio_files or [],
        subtitle_files=subtitle_files or [],
        attachments=attachments or [],
        chapters_source=chapters_source,
        output_path=Path("output.mkv"),
        video_meta=video_meta,
    )


class TestMkvmergeSetLogDir:
    def test_set_log_dir(self, tmp_path: Path) -> None:
        adapter = MkvmergeAdapter(Path("mkvmerge.exe"))
        adapter.set_log_dir(tmp_path)
        assert adapter._log_dir == tmp_path


class TestMkvmergeGlobalFlags:
    def test_title_empty(self) -> None:
        cmd = _build_cmd()
        idx = cmd.index("--title")
        assert cmd[idx + 1] == ""

    def test_normalize_language_ietf(self) -> None:
        cmd = _build_cmd()
        idx = cmd.index("--normalize-language-ietf")
        assert cmd[idx + 1] == "canonical"


class TestMkvmergeColorRange:
    def test_color_range_tv(self) -> None:
        cmd = _build_cmd({"color_range": "tv"})
        idx = cmd.index("--color-range")
        assert cmd[idx + 1] == "0:1"

    def test_color_range_pc(self) -> None:
        cmd = _build_cmd({"color_range": "pc"})
        idx = cmd.index("--color-range")
        assert cmd[idx + 1] == "0:2"

    def test_color_range_unknown_skipped(self) -> None:
        cmd = _build_cmd({"color_range": "unknown"})
        assert "--color-range" not in cmd

    def test_no_video_meta_no_color_range(self) -> None:
        cmd = _build_cmd(None)
        assert "--color-range" not in cmd


class TestMkvmergeColorPrimaries:
    def test_bt709(self) -> None:
        cmd = _build_cmd({"color_primaries": "bt709"})
        idx = cmd.index("--color-primaries")
        assert cmd[idx + 1] == "0:1"

    def test_bt470bg(self) -> None:
        cmd = _build_cmd({"color_primaries": "bt470bg"})
        idx = cmd.index("--color-primaries")
        assert cmd[idx + 1] == "0:5"

    def test_smpte170m(self) -> None:
        cmd = _build_cmd({"color_primaries": "smpte170m"})
        idx = cmd.index("--color-primaries")
        assert cmd[idx + 1] == "0:6"

    def test_bt2020(self) -> None:
        cmd = _build_cmd({"color_primaries": "bt2020"})
        idx = cmd.index("--color-primaries")
        assert cmd[idx + 1] == "0:9"

    def test_unknown_skipped(self) -> None:
        cmd = _build_cmd({"color_primaries": "xyz"})
        assert "--color-primaries" not in cmd


class TestMkvmergeColorTransfer:
    def test_bt709(self) -> None:
        cmd = _build_cmd({"color_transfer": "bt709"})
        idx = cmd.index("--color-transfer-characteristics")
        assert cmd[idx + 1] == "0:1"

    def test_smpte2084_hdr10(self) -> None:
        cmd = _build_cmd({"color_transfer": "smpte2084"})
        idx = cmd.index("--color-transfer-characteristics")
        assert cmd[idx + 1] == "0:16"

    def test_hlg(self) -> None:
        cmd = _build_cmd({"color_transfer": "arib-std-b67"})
        idx = cmd.index("--color-transfer-characteristics")
        assert cmd[idx + 1] == "0:18"

    def test_unknown_skipped(self) -> None:
        cmd = _build_cmd({"color_transfer": "nope"})
        assert "--color-transfer-characteristics" not in cmd


class TestMkvmergeColorMatrix:
    def test_bt709(self) -> None:
        cmd = _build_cmd({"color_matrix": "bt709"})
        idx = cmd.index("--color-matrix-coefficients")
        assert cmd[idx + 1] == "0:1"

    def test_bt470bg(self) -> None:
        cmd = _build_cmd({"color_matrix": "bt470bg"})
        idx = cmd.index("--color-matrix-coefficients")
        assert cmd[idx + 1] == "0:5"

    def test_smpte170m(self) -> None:
        cmd = _build_cmd({"color_matrix": "smpte170m"})
        idx = cmd.index("--color-matrix-coefficients")
        assert cmd[idx + 1] == "0:6"

    def test_bt2020nc(self) -> None:
        cmd = _build_cmd({"color_matrix": "bt2020nc"})
        idx = cmd.index("--color-matrix-coefficients")
        assert cmd[idx + 1] == "0:9"

    def test_unknown_skipped(self) -> None:
        cmd = _build_cmd({"color_matrix": "nope"})
        assert "--color-matrix-coefficients" not in cmd

    def test_no_video_meta_no_matrix(self) -> None:
        cmd = _build_cmd(None)
        assert "--color-matrix-coefficients" not in cmd


class TestMkvmergeHdrMetadata:
    def test_max_content_light(self) -> None:
        cmd = _build_cmd({"hdr_max_cll": "1000"})
        idx = cmd.index("--max-content-light")
        assert cmd[idx + 1] == "0:1000"

    def test_max_frame_light(self) -> None:
        cmd = _build_cmd({"hdr_max_fall": "400"})
        idx = cmd.index("--max-frame-light")
        assert cmd[idx + 1] == "0:400"

    def test_both_hdr_values(self) -> None:
        cmd = _build_cmd({"hdr_max_cll": "1000", "hdr_max_fall": "400"})
        assert "--max-content-light" in cmd
        assert "--max-frame-light" in cmd

    def test_no_hdr_no_flags(self) -> None:
        cmd = _build_cmd({"color_range": "tv"})
        assert "--max-content-light" not in cmd
        assert "--max-frame-light" not in cmd


class TestMkvmergeMasteringDisplay:
    def test_chromaticity_coordinates(self) -> None:
        cmd = _build_cmd({"hdr_mastering_display": _UHD_1000_NITS})
        idx = cmd.index("--chromaticity-coordinates")
        assert cmd[idx + 1] == "0:0.68,0.32,0.265,0.69,0.15,0.06"

    def test_white_color_coordinates(self) -> None:
        cmd = _build_cmd({"hdr_mastering_display": _UHD_1000_NITS})
        idx = cmd.index("--white-color-coordinates")
        assert cmd[idx + 1] == "0:0.3127,0.329"

    def test_luminance(self) -> None:
        cmd = _build_cmd({"hdr_mastering_display": _UHD_1000_NITS})
        assert cmd[cmd.index("--max-luminance") + 1] == "0:1000"
        assert cmd[cmd.index("--min-luminance") + 1] == "0:0"

    def test_fractional_min_luminance_not_rounded(self) -> None:
        cmd = _build_cmd(
            {
                "hdr_mastering_display": (
                    "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(20000000,1)"
                )
            }
        )
        assert cmd[cmd.index("--max-luminance") + 1] == "0:2000"
        assert cmd[cmd.index("--min-luminance") + 1] == "0:0.0001"

    def test_no_mastering_display_no_flags(self) -> None:
        cmd = _build_cmd({"hdr_max_cll": "1000"})
        assert "--chromaticity-coordinates" not in cmd
        assert "--white-color-coordinates" not in cmd
        assert "--max-luminance" not in cmd
        assert "--min-luminance" not in cmd

    def test_malformed_mastering_display_raises(self) -> None:
        with pytest.raises(ValueError, match="mastering display"):
            _build_cmd({"hdr_mastering_display": "L(10000000,0)"})


class TestMkvmergeFullHdrPipeline:
    def test_hdr10_full_metadata(self) -> None:
        cmd = _build_cmd(
            {
                "color_range": "tv",
                "color_primaries": "bt2020",
                "color_transfer": "smpte2084",
                "hdr_max_cll": "1000",
                "hdr_max_fall": "400",
                "hdr_mastering_display": _UHD_1000_NITS,
            }
        )
        assert "--color-range" in cmd
        assert "--color-primaries" in cmd
        assert "--color-transfer-characteristics" in cmd
        assert "--max-content-light" in cmd
        assert "--max-frame-light" in cmd
        assert "--chromaticity-coordinates" in cmd
        assert "--white-color-coordinates" in cmd
        assert "--max-luminance" in cmd
        assert "--min-luminance" in cmd
        assert cmd[cmd.index("--color-range") + 1] == "0:1"
        assert cmd[cmd.index("--color-primaries") + 1] == "0:9"
        assert cmd[cmd.index("--color-transfer-characteristics") + 1] == "0:16"

    def test_sdr_bt709(self) -> None:
        cmd = _build_cmd(
            {
                "color_range": "tv",
                "color_primaries": "bt709",
                "color_transfer": "bt709",
            }
        )
        assert "--color-range" in cmd
        assert "--color-primaries" in cmd
        assert "--color-transfer-characteristics" in cmd
        assert "--max-content-light" not in cmd
        assert "--max-frame-light" not in cmd


class TestMkvmergeAudioArgs:
    def test_language_flag(self) -> None:
        cmd = _build_cmd(audio_files=[(Path("a.flac"), {"language": "rus"})])
        audio_lang_indices = [i for i, x in enumerate(cmd) if x == "--language" and cmd[i + 1].startswith("0:rus")]
        assert len(audio_lang_indices) == 1
        assert cmd[audio_lang_indices[0] + 1] == "0:rus"

    def test_default_flag_yes(self) -> None:
        cmd = _build_cmd(audio_files=[(Path("a.flac"), {"language": "eng", "default": True})])
        yes_indices = [i for i, x in enumerate(cmd) if x == "--default-track-flag" and cmd[i + 1] == "0:yes"]
        assert len(yes_indices) == 1

    def test_default_flag_no(self) -> None:
        cmd = _build_cmd(audio_files=[(Path("a.flac"), {"language": "eng", "default": False})])
        no_indices = [i for i, x in enumerate(cmd) if x == "--default-track-flag" and cmd[i + 1] == "0:no"]
        assert len(no_indices) == 1

    def test_sync_delay(self) -> None:
        cmd = _build_cmd(audio_files=[(Path("a.flac"), {"language": "eng", "delay_ms": -200})])
        idx = cmd.index("--sync")
        assert cmd[idx + 1] == "0:-200"

    def test_no_sync_when_zero_delay(self) -> None:
        cmd = _build_cmd(audio_files=[(Path("a.flac"), {"language": "eng", "delay_ms": 0})])
        assert "--sync" not in cmd

    def test_no_chapters_on_audio(self) -> None:
        cmd = _build_cmd(audio_files=[(Path("a.flac"), {"language": "eng"})])
        count = cmd.count("--no-chapters")
        assert count >= 2


class TestMkvmergeSubtitleArgs:
    def test_subtitle_language(self) -> None:
        cmd = _build_cmd(subtitle_files=[(Path("s.sup"), {"language": "rus"})])
        lang_indices = [i for i, x in enumerate(cmd) if x == "--language" and cmd[i + 1] == "0:rus"]
        assert len(lang_indices) == 1

    def test_subtitle_default_yes(self) -> None:
        cmd = _build_cmd(subtitle_files=[(Path("s.sup"), {"language": "eng", "default": True})])
        yes_indices = [i for i, x in enumerate(cmd) if x == "--default-track-flag" and cmd[i + 1] == "0:yes"]
        assert len(yes_indices) == 1

    def test_subtitle_default_no(self) -> None:
        cmd = _build_cmd(subtitle_files=[(Path("s.sup"), {"language": "eng", "default": False})])
        no_indices = [i for i, x in enumerate(cmd) if x == "--default-track-flag" and cmd[i + 1] == "0:no"]
        assert len(no_indices) == 1

    def test_forced_display_flag(self) -> None:
        cmd = _build_cmd(subtitle_files=[(Path("s.sup"), {"language": "eng", "forced": True})])
        idx = cmd.index("--forced-display-flag")
        assert cmd[idx + 1] == "0:yes"

    def test_forced_not_present_when_false(self) -> None:
        cmd = _build_cmd(subtitle_files=[(Path("s.sup"), {"language": "eng", "forced": False})])
        assert "--forced-display-flag" not in cmd

    def test_sub_charset(self) -> None:
        cmd = _build_cmd(subtitle_files=[(Path("s.srt"), {"language": "eng", "encoding": "UTF-8"})])
        idx = cmd.index("--sub-charset")
        assert cmd[idx + 1] == "0:UTF-8"

    def test_no_charset_when_none(self) -> None:
        cmd = _build_cmd(subtitle_files=[(Path("s.sup"), {"language": "eng"})])
        assert "--sub-charset" not in cmd

    def test_no_chapters_on_subtitle(self) -> None:
        cmd = _build_cmd(subtitle_files=[(Path("s.sup"), {"language": "eng"})])
        count = cmd.count("--no-chapters")
        assert count >= 2


class TestMkvmergeAttachments:
    def test_attachment_flags(self) -> None:
        cmd = _build_cmd(attachments=[(Path("/fonts/Arial.ttf"), "Arial.ttf", "application/x-truetype-font")])
        idx_name = cmd.index("--attachment-name")
        assert cmd[idx_name + 1] == "Arial.ttf"
        idx_mime = cmd.index("--attachment-mime-type")
        assert cmd[idx_mime + 1] == "application/x-truetype-font"
        idx_file = cmd.index("--attach-file")
        assert Path(cmd[idx_file + 1]) == Path("/fonts/Arial.ttf")

    def test_multiple_attachments(self) -> None:
        cmd = _build_cmd(
            attachments=[
                (Path("/fonts/A.ttf"), "A.ttf", "application/x-truetype-font"),
                (Path("/fonts/B.otf"), "B.otf", "font/otf"),
            ]
        )
        assert cmd.count("--attach-file") == 2
        assert cmd.count("--attachment-name") == 2
        assert cmd.count("--attachment-mime-type") == 2


class TestMkvmergeChapters:
    def test_chapters_present(self) -> None:
        chap = Path("/work/chapters.txt")
        cmd = _build_cmd(chapters_source=chap)
        idx = cmd.index("--chapters")
        assert cmd[idx + 1] == str(chap)

    def test_no_chapters_when_none(self) -> None:
        cmd = _build_cmd(chapters_source=None)
        assert "--chapters" not in cmd


class TestMkvmergeTrackOrder:
    def test_track_order_with_audio_and_subs(self) -> None:
        cmd = _build_cmd(
            audio_files=[
                (Path("a1.flac"), {"language": "eng"}),
                (Path("a2.flac"), {"language": "rus"}),
            ],
            subtitle_files=[
                (Path("s1.sup"), {"language": "eng"}),
            ],
        )
        idx = cmd.index("--track-order")
        order = cmd[idx + 1]
        assert order == "0:0,1:0,2:0,3:0"


class TestMkvmergeMuxExecution:
    def _fake_run_tool(
        self,
        rc: int,
    ) -> tuple[list[str], Any]:
        captured_cmd: list[str] = []

        def fake(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            captured_cmd.extend(str(c) for c in cmd)
            return rc, "some output"

        return captured_cmd, fake

    def test_mux_rc_zero_ok(self) -> None:
        captured_cmd, fake = self._fake_run_tool(0)
        adapter = MkvmergeAdapter(Path("mkvmerge.exe"))
        with patch("furnace.adapters.mkvmerge.run_tool", side_effect=fake):
            rc = adapter.mux(
                video_path=Path("video.obu"),
                audio_files=[],
                subtitle_files=[],
                attachments=[],
                chapters_source=None,
                output_path=Path("output.mkv"),
            )
        assert rc == 0
        assert "mkvmerge.exe" in captured_cmd

    def test_mux_rc_one_warning(self) -> None:
        _captured_cmd, fake = self._fake_run_tool(1)
        adapter = MkvmergeAdapter(Path("mkvmerge.exe"))
        with patch("furnace.adapters.mkvmerge.run_tool", side_effect=fake):
            rc = adapter.mux(
                video_path=Path("video.obu"),
                audio_files=[],
                subtitle_files=[],
                attachments=[],
                chapters_source=None,
                output_path=Path("output.mkv"),
            )
        assert rc == 1

    def test_mux_rc_two_error(self) -> None:
        _captured_cmd, fake = self._fake_run_tool(2)
        adapter = MkvmergeAdapter(Path("mkvmerge.exe"))
        with patch("furnace.adapters.mkvmerge.run_tool", side_effect=fake):
            rc = adapter.mux(
                video_path=Path("video.obu"),
                audio_files=[],
                subtitle_files=[],
                attachments=[],
                chapters_source=None,
                output_path=Path("output.mkv"),
            )
        assert rc == 2

    def test_mux_log_path_from_log_dir(self, tmp_path: Path) -> None:
        captured_kwargs: dict[str, Any] = {}

        def fake(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            captured_kwargs["log_path"] = log_path
            return 0, ""

        adapter = MkvmergeAdapter(Path("mkvmerge.exe"), log_dir=tmp_path)
        with patch("furnace.adapters.mkvmerge.run_tool", side_effect=fake):
            adapter.mux(
                video_path=Path("video.obu"),
                audio_files=[],
                subtitle_files=[],
                attachments=[],
                chapters_source=None,
                output_path=Path("output.mkv"),
            )
        assert captured_kwargs["log_path"] == tmp_path / "mkvmerge.log"

    def test_mux_progress_callback(self) -> None:
        samples: list[ProgressSample] = []

        def fake(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            on_progress_line("Progress: 50%")
            return 0, ""

        adapter = MkvmergeAdapter(Path("mkvmerge.exe"))
        with patch("furnace.adapters.mkvmerge.run_tool", side_effect=fake):
            adapter.mux(
                video_path=Path("video.obu"),
                audio_files=[],
                subtitle_files=[],
                attachments=[],
                chapters_source=None,
                output_path=Path("output.mkv"),
                on_progress=samples.append,
            )
        assert len(samples) == 1
        assert samples[0].fraction == 0.5

    def test_mux_progress_non_progress_line(self) -> None:
        progress_results: list[bool] = []

        def fake(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            progress_results.append(on_progress_line("Not a progress line"))
            return 0, ""

        adapter = MkvmergeAdapter(Path("mkvmerge.exe"))
        with patch("furnace.adapters.mkvmerge.run_tool", side_effect=fake):
            adapter.mux(
                video_path=Path("video.obu"),
                audio_files=[],
                subtitle_files=[],
                attachments=[],
                chapters_source=None,
                output_path=Path("output.mkv"),
            )
        assert progress_results == [False]

    def test_mux_progress_without_callback(self) -> None:
        progress_results: list[bool] = []

        def fake(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            progress_results.append(on_progress_line("Progress: 50%"))
            return 0, ""

        adapter = MkvmergeAdapter(Path("mkvmerge.exe"))
        with patch("furnace.adapters.mkvmerge.run_tool", side_effect=fake):
            adapter.mux(
                video_path=Path("video.obu"),
                audio_files=[],
                subtitle_files=[],
                attachments=[],
                chapters_source=None,
                output_path=Path("output.mkv"),
                on_progress=None,
            )
        assert progress_results == [True]

    def test_mux_no_log_dir(self) -> None:
        captured_kwargs: dict[str, Any] = {}

        def fake(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            captured_kwargs["log_path"] = log_path
            return 0, ""

        adapter = MkvmergeAdapter(Path("mkvmerge.exe"), log_dir=None)
        with patch("furnace.adapters.mkvmerge.run_tool", side_effect=fake):
            adapter.mux(
                video_path=Path("video.obu"),
                audio_files=[],
                subtitle_files=[],
                attachments=[],
                chapters_source=None,
                output_path=Path("output.mkv"),
            )
        assert captured_kwargs["log_path"] is None


class TestMkvmergeDefaultDuration:
    def _video_idx(self, cmd: list[str]) -> int:
        return cmd.index("video.obu")

    def test_default_duration_emitted_for_integer_fps(self) -> None:
        cmd = _build_cmd({"fps_num": 24, "fps_den": 1})
        idx = cmd.index("--default-duration")
        assert cmd[idx + 1] == "0:24/1p"
        assert idx < self._video_idx(cmd)

    def test_default_duration_emitted_for_fractional_fps(self) -> None:
        cmd = _build_cmd({"fps_num": 24000, "fps_den": 1001})
        idx = cmd.index("--default-duration")
        assert cmd[idx + 1] == "0:24000/1001p"
        assert idx < self._video_idx(cmd)

    def test_default_duration_with_color_meta(self) -> None:
        cmd = _build_cmd({"fps_num": 24, "fps_den": 1, "color_range": "tv"})
        dd = cmd.index("--default-duration")
        cr = cmd.index("--color-range")
        assert cmd[dd + 1] == "0:24/1p"
        assert cmd[cr + 1] == "0:1"
        assert dd < self._video_idx(cmd)
        assert cr < self._video_idx(cmd)

    def test_no_default_duration_without_fps(self) -> None:
        assert "--default-duration" not in _build_cmd({"color_range": "tv"})

    def test_no_default_duration_when_video_meta_none(self) -> None:
        assert "--default-duration" not in _build_cmd(None)

    def test_no_default_duration_when_fps_den_missing(self) -> None:
        assert "--default-duration" not in _build_cmd({"fps_num": 24})

    def test_no_default_duration_when_fps_den_zero(self) -> None:
        assert "--default-duration" not in _build_cmd({"fps_num": 24, "fps_den": 0})
