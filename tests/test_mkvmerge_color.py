"""Tests for mkvmerge color/HDR metadata flags and new options."""
from __future__ import annotations

from pathlib import Path

from furnace.adapters.mkvmerge import MkvmergeAdapter


def _build_cmd(
    video_meta: dict | None = None,
) -> list[str]:
    """Helper: build mkvmerge command with minimal args and optional video_meta."""
    adapter = MkvmergeAdapter(Path("mkvmerge.exe"))
    return adapter._build_mux_cmd(
        video_path=Path("video.hevc"),
        audio_files=[],
        subtitle_files=[],
        attachments=[],
        chapters_source=None,
        output_path=Path("output.mkv"),
        furnace_version="0.1.0",
        video_meta=video_meta,
    )


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
