"""Tests for mkvmerge color/HDR metadata flags, audio/sub/attach/chapters args, and mux execution."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from furnace.adapters.mkvmerge import MkvmergeAdapter
from furnace.core.progress import ProgressSample


def _build_cmd(
    video_meta: dict[str, Any] | None = None,
    audio_files: list[tuple[Path, dict[str, Any]]] | None = None,
    subtitle_files: list[tuple[Path, dict[str, Any]]] | None = None,
    attachments: list[tuple[Path, str, str]] | None = None,
    chapters_source: Path | None = None,
) -> list[str]:
    """Helper: build mkvmerge command with minimal args and optional video_meta."""
    adapter = MkvmergeAdapter(Path("mkvmerge.exe"))
    return adapter._build_mux_cmd(
        video_path=Path("video.hevc"),
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
    """Test --title and --normalize-language-ietf are always present."""

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


class TestMkvmergeFullHdrPipeline:
    """Integration: all color+HDR flags together as they would appear for HDR10."""

    def test_hdr10_full_metadata(self) -> None:
        cmd = _build_cmd({
            "color_range": "tv",
            "color_primaries": "bt2020",
            "color_transfer": "smpte2084",
            "hdr_max_cll": "1000",
            "hdr_max_fall": "400",
        })
        assert "--color-range" in cmd
        assert "--color-primaries" in cmd
        assert "--color-transfer-characteristics" in cmd
        assert "--max-content-light" in cmd
        assert "--max-frame-light" in cmd
        # Verify values
        assert cmd[cmd.index("--color-range") + 1] == "0:1"
        assert cmd[cmd.index("--color-primaries") + 1] == "0:9"
        assert cmd[cmd.index("--color-transfer-characteristics") + 1] == "0:16"

    def test_sdr_bt709(self) -> None:
        cmd = _build_cmd({
            "color_range": "tv",
            "color_primaries": "bt709",
            "color_transfer": "bt709",
        })
        assert "--color-range" in cmd
        assert "--color-primaries" in cmd
        assert "--color-transfer-characteristics" in cmd
        assert "--max-content-light" not in cmd
        assert "--max-frame-light" not in cmd


class TestMkvmergeAudioArgs:
    """Audio track flags: language, default, delay, no-chapters."""

    def test_language_flag(self) -> None:
        cmd = _build_cmd(audio_files=[(Path("a.flac"), {"language": "rus"})])
        # First --language is for video (0:und), second is for audio
        # Find audio --language after the audio track separator
        audio_lang_indices = [i for i, x in enumerate(cmd) if x == "--language" and cmd[i + 1].startswith("0:rus")]
        assert len(audio_lang_indices) == 1
        assert cmd[audio_lang_indices[0] + 1] == "0:rus"

    def test_default_flag_yes(self) -> None:
        cmd = _build_cmd(audio_files=[(Path("a.flac"), {"language": "eng", "default": True})])
        # Find default-track-flag with value "0:yes" (audio)
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
        # --no-chapters appears at least twice: once for video, once for audio
        count = cmd.count("--no-chapters")
        assert count >= 2


class TestMkvmergeSubtitleArgs:
    """Subtitle track flags: language, default, forced, encoding, no-chapters."""

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
    """Attachment flags: name, mime-type, attach-file."""

    def test_attachment_flags(self) -> None:
        cmd = _build_cmd(attachments=[(Path("/fonts/Arial.ttf"), "Arial.ttf", "application/x-truetype-font")])
        idx_name = cmd.index("--attachment-name")
        assert cmd[idx_name + 1] == "Arial.ttf"
        idx_mime = cmd.index("--attachment-mime-type")
        assert cmd[idx_mime + 1] == "application/x-truetype-font"
        idx_file = cmd.index("--attach-file")
        assert Path(cmd[idx_file + 1]) == Path("/fonts/Arial.ttf")

    def test_multiple_attachments(self) -> None:
        cmd = _build_cmd(attachments=[
            (Path("/fonts/A.ttf"), "A.ttf", "application/x-truetype-font"),
            (Path("/fonts/B.otf"), "B.otf", "font/otf"),
        ])
        assert cmd.count("--attach-file") == 2
        assert cmd.count("--attachment-name") == 2
        assert cmd.count("--attachment-mime-type") == 2


class TestMkvmergeChapters:
    """Chapters arg: --chapters path."""

    def test_chapters_present(self) -> None:
        chap = Path("/work/chapters.txt")
        cmd = _build_cmd(chapters_source=chap)
        idx = cmd.index("--chapters")
        assert cmd[idx + 1] == str(chap)

    def test_no_chapters_when_none(self) -> None:
        cmd = _build_cmd(chapters_source=None)
        assert "--chapters" not in cmd


class TestMkvmergeTrackOrder:
    """Track order: video first, then audio, then subtitles."""

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
    """Test mux() method by mocking run_tool."""

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
                video_path=Path("video.hevc"),
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
                video_path=Path("video.hevc"),
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
                video_path=Path("video.hevc"),
                audio_files=[],
                subtitle_files=[],
                attachments=[],
                chapters_source=None,
                output_path=Path("output.mkv"),
            )
        assert rc == 2

    def test_mux_log_path_from_log_dir(self, tmp_path: Path) -> None:
        """When log_dir is set, log_path is passed to run_tool."""
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
                video_path=Path("video.hevc"),
                audio_files=[],
                subtitle_files=[],
                attachments=[],
                chapters_source=None,
                output_path=Path("output.mkv"),
            )
        assert captured_kwargs["log_path"] == tmp_path / "mkvmerge.log"

    def test_mux_progress_callback(self) -> None:
        """Progress lines from run_tool are forwarded to on_progress."""
        samples: list[ProgressSample] = []

        def fake(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            # Simulate a progress line
            if on_progress_line is not None:
                on_progress_line("Progress: 50%")
            return 0, ""

        adapter = MkvmergeAdapter(Path("mkvmerge.exe"))
        with patch("furnace.adapters.mkvmerge.run_tool", side_effect=fake):
            adapter.mux(
                video_path=Path("video.hevc"),
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
        """Non-progress lines return False from on_progress_line."""
        progress_results: list[bool] = []

        def fake(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                progress_results.append(on_progress_line("Not a progress line"))
            return 0, ""

        adapter = MkvmergeAdapter(Path("mkvmerge.exe"))
        with patch("furnace.adapters.mkvmerge.run_tool", side_effect=fake):
            adapter.mux(
                video_path=Path("video.hevc"),
                audio_files=[],
                subtitle_files=[],
                attachments=[],
                chapters_source=None,
                output_path=Path("output.mkv"),
            )
        assert progress_results == [False]

    def test_mux_progress_without_callback(self) -> None:
        """Progress lines consumed even when on_progress is None."""
        progress_results: list[bool] = []

        def fake(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                progress_results.append(on_progress_line("Progress: 50%"))
            return 0, ""

        adapter = MkvmergeAdapter(Path("mkvmerge.exe"))
        with patch("furnace.adapters.mkvmerge.run_tool", side_effect=fake):
            adapter.mux(
                video_path=Path("video.hevc"),
                audio_files=[],
                subtitle_files=[],
                attachments=[],
                chapters_source=None,
                output_path=Path("output.mkv"),
                on_progress=None,
            )
        assert progress_results == [True]

    def test_mux_no_log_dir(self) -> None:
        """When log_dir is None, log_path=None is passed."""
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
                video_path=Path("video.hevc"),
                audio_files=[],
                subtitle_files=[],
                attachments=[],
                chapters_source=None,
                output_path=Path("output.mkv"),
            )
        assert captured_kwargs["log_path"] is None
