from __future__ import annotations

import binascii
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from furnace.adapters.mkclean import (
    MkcleanAdapter,
    _parse_mkclean_progress_line,
    _patch_doctype_read_version,
)
from furnace.core.progress import ProgressSample


def _build_ebml_header(dt_version: int, dt_read_version: int, with_crc: bool = True) -> bytes:
    """Synthesize a minimal EBML header for tests (DocType=matroska)."""
    content = bytearray()
    content += b"\x42\x82\x88matroska"
    content += bytes([0x42, 0x87, 0x81, dt_version])
    content += bytes([0x42, 0x85, 0x81, dt_read_version])
    if with_crc:
        crc = binascii.crc32(bytes(content)) & 0xFFFFFFFF
        content = bytearray(b"\xbf\x84" + crc.to_bytes(4, "little")) + content
    size = len(content)
    assert size <= 0x7F, "test fixture must use 1-byte VINT size"
    return b"\x1a\x45\xdf\xa3" + bytes([0x80 | size]) + bytes(content)


class TestParseMkcleanProgressLine:
    def test_stage_1_zero(self) -> None:
        sample = _parse_mkclean_progress_line("Progress 1/3:   0%")
        assert sample is not None
        assert sample.fraction == pytest.approx(0.0)

    def test_stage_1_fortytwo(self) -> None:
        sample = _parse_mkclean_progress_line("Progress 1/3:  42%")
        assert sample is not None
        # (0 + 0.42) / 3 = 0.14
        assert sample.fraction == pytest.approx(0.14)

    def test_stage_2_fifty(self) -> None:
        sample = _parse_mkclean_progress_line("Progress 2/3:  50%")
        assert sample is not None
        # (1 + 0.5) / 3 = 0.5
        assert sample.fraction == pytest.approx(0.5)

    def test_stage_3_hundred(self) -> None:
        sample = _parse_mkclean_progress_line("Progress 3/3: 100%")
        assert sample is not None
        assert sample.fraction == pytest.approx(1.0)

    def test_stage_out_of_range(self) -> None:
        assert _parse_mkclean_progress_line("Progress 4/3:  10%") is None

    def test_plain_text(self) -> None:
        assert _parse_mkclean_progress_line("mkclean v0.8.7") is None


class TestMkcleanClean:
    """Test clean() execution with mocked run_tool."""

    @pytest.fixture(autouse=True)
    def _stub_dtrv_patch(self) -> Any:
        # These tests pass relative paths that never exist on disk; stub the
        # post-mkclean header patch so it doesn't try to open them.
        with patch("furnace.adapters.mkclean._patch_doctype_read_version"):
            yield

    def test_clean_cmd(self) -> None:
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

        adapter = MkcleanAdapter(Path("mkclean.exe"))
        with patch("furnace.adapters.mkclean.run_tool", side_effect=fake_run_tool):
            rc = adapter.clean(Path("input.mkv"), Path("output.mkv"))
        assert rc == 0
        assert "mkclean.exe" in captured
        assert "input.mkv" in captured
        assert "output.mkv" in captured

    def test_clean_forces_matroska_v4_doctype(self) -> None:
        # Without --doctype 6 mkclean defaults to matroska v2 and strips
        # BlockAdditionMapping / BlockAddIDName / BlockAddIDExtraData —
        # exactly the container elements that carry the Dolby Vision
        # configuration record. v4 keeps them.
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

        adapter = MkcleanAdapter(Path("mkclean.exe"))
        with patch("furnace.adapters.mkclean.run_tool", side_effect=fake_run_tool):
            adapter.clean(Path("input.mkv"), Path("output.mkv"))
        dt_idx = captured.index("--doctype")
        assert captured[dt_idx + 1] == "6"
        in_idx = captured.index("input.mkv")
        out_idx = captured.index("output.mkv")
        assert dt_idx < in_idx < out_idx

    def test_clean_progress_callback(self) -> None:
        samples: list[ProgressSample] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                on_progress_line("Progress 2/3:  50%")
            return 0, ""

        adapter = MkcleanAdapter(Path("mkclean.exe"))
        with patch("furnace.adapters.mkclean.run_tool", side_effect=fake_run_tool):
            adapter.clean(Path("input.mkv"), Path("output.mkv"), on_progress=samples.append)
        assert len(samples) == 1
        assert samples[0].fraction == pytest.approx(0.5)

    def test_clean_log_path(self, tmp_path: Path) -> None:
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

        adapter = MkcleanAdapter(Path("mkclean.exe"), log_dir=tmp_path)
        with patch("furnace.adapters.mkclean.run_tool", side_effect=fake_run_tool):
            adapter.clean(Path("input.mkv"), Path("output.mkv"))
        assert captured_kwargs["log_path"] == tmp_path / "mkclean.log"

    def test_clean_non_progress_line(self) -> None:
        results: list[bool] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                results.append(on_progress_line("mkclean v0.8.7"))
            return 0, ""

        adapter = MkcleanAdapter(Path("mkclean.exe"))
        with patch("furnace.adapters.mkclean.run_tool", side_effect=fake_run_tool):
            adapter.clean(Path("input.mkv"), Path("output.mkv"))
        assert results == [False]


class TestMkcleanCleanWithoutProgressCallback:
    """Test that progress lines are consumed even without on_progress."""

    @pytest.fixture(autouse=True)
    def _stub_dtrv_patch(self) -> Any:
        with patch("furnace.adapters.mkclean._patch_doctype_read_version"):
            yield

    def test_progress_consumed_without_callback(self) -> None:
        results: list[bool] = []

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            if on_progress_line is not None:
                results.append(on_progress_line("Progress 1/3:  50%"))
            return 0, ""

        adapter = MkcleanAdapter(Path("mkclean.exe"))
        with patch("furnace.adapters.mkclean.run_tool", side_effect=fake_run_tool):
            adapter.clean(Path("input.mkv"), Path("output.mkv"), on_progress=None)
        assert results == [True]

    def test_no_log_dir(self) -> None:
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

        adapter = MkcleanAdapter(Path("mkclean.exe"), log_dir=None)
        with patch("furnace.adapters.mkclean.run_tool", side_effect=fake_run_tool):
            adapter.clean(Path("input.mkv"), Path("output.mkv"))
        assert captured_kwargs["log_path"] is None


class TestMkcleanSetLogDir:
    def test_set_log_dir(self, tmp_path: Path) -> None:
        adapter = MkcleanAdapter(Path("mkclean.exe"))
        adapter.set_log_dir(tmp_path)
        assert adapter._log_dir == tmp_path


class TestPatchDocTypeReadVersion:
    """Patches mkclean v0.9.0's EBML header so FFmpeg/lavf accepts the file.

    mkclean writes DocTypeReadVersion equal to DocTypeVersion (4 with
    --doctype 6). FFmpeg refuses anything > 2. mkvmerge always writes
    DocTypeReadVersion=2 alongside DocTypeVersion=4. We mimic mkvmerge.
    """

    def test_patches_read_version_from_4_to_2(self, tmp_path: Path) -> None:
        header_before = _build_ebml_header(dt_version=4, dt_read_version=4)
        header_after = _build_ebml_header(dt_version=4, dt_read_version=2)
        payload = b"AAAA" * 16  # arbitrary post-header bytes
        target = tmp_path / "f.mkv"
        target.write_bytes(header_before + payload)

        _patch_doctype_read_version(target)

        new = target.read_bytes()
        # Patched header byte-identical to what mkvmerge would emit; the
        # payload after the header is untouched.
        assert new[: len(header_after)] == header_after
        assert new[len(header_after) :] == payload

    def test_recomputes_crc_after_patch(self, tmp_path: Path) -> None:
        header = _build_ebml_header(dt_version=4, dt_read_version=4)
        target = tmp_path / "f.mkv"
        target.write_bytes(header + b"\x00")

        _patch_doctype_read_version(target)

        new = target.read_bytes()
        # CRC element sits at offsets 5-10 (after 1A 45 DF A3 <size>)
        # and covers everything after itself, up to end of EBML header.
        stored_crc = int.from_bytes(new[7:11], "little")
        # Header size is the byte at offset 4 (1-byte VINT, low 7 bits)
        header_content_len = new[4] & 0x7F
        header_end = 5 + header_content_len
        expected_crc = binascii.crc32(bytes(new[11:header_end])) & 0xFFFFFFFF
        assert stored_crc == expected_crc

    def test_noop_when_already_2(self, tmp_path: Path) -> None:
        header = _build_ebml_header(dt_version=4, dt_read_version=2)
        target = tmp_path / "f.mkv"
        original = header + b"DATA"
        target.write_bytes(original)

        _patch_doctype_read_version(target)

        assert target.read_bytes() == original

    def test_handles_header_without_crc(self, tmp_path: Path) -> None:
        header = _build_ebml_header(dt_version=4, dt_read_version=4, with_crc=False)
        target = tmp_path / "f.mkv"
        target.write_bytes(header + b"\xff" * 8)

        _patch_doctype_read_version(target)

        new = target.read_bytes()
        assert new[len(header) - 1] == 0x02
        # No CRC bytes in the header range — nothing to validate beyond DTRV.

    def test_raises_when_not_ebml(self, tmp_path: Path) -> None:
        target = tmp_path / "f.mkv"
        target.write_bytes(b"NOT-AN-EBML-FILE")
        with pytest.raises(ValueError, match="EBML"):
            _patch_doctype_read_version(target)

    def test_noop_when_dtrv_element_missing(self, tmp_path: Path) -> None:
        # Strip DocTypeReadVersion element from the synthetic header.
        with_dtrv = _build_ebml_header(dt_version=4, dt_read_version=4, with_crc=False)
        # The DTRV element is the last 4 bytes of the content; remove them
        # and fix up the VINT size so the header stays parseable.
        without_dtrv = bytearray(with_dtrv[:-4])
        without_dtrv[4] = 0x80 | (without_dtrv[4] & 0x7F) - 4
        target = tmp_path / "f.mkv"
        original = bytes(without_dtrv) + b"PAYLOAD"
        target.write_bytes(original)

        _patch_doctype_read_version(target)

        assert target.read_bytes() == original


class TestMkcleanCleanPatchesDoctype:
    """clean() applies the DocTypeReadVersion patch when mkclean succeeds."""

    def test_patches_after_success(self, tmp_path: Path) -> None:
        out = tmp_path / "out.mkv"

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            out.write_bytes(_build_ebml_header(dt_version=4, dt_read_version=4) + b"X")
            return 0, ""

        adapter = MkcleanAdapter(Path("mkclean.exe"))
        with patch("furnace.adapters.mkclean.run_tool", side_effect=fake_run_tool):
            rc = adapter.clean(Path("in.mkv"), out)

        assert rc == 0
        data = out.read_bytes()
        # The DTRV element is the last 4 bytes of the EBML header; its value
        # byte is the very last byte before the X payload marker.
        assert data[-2] == 0x02

    def test_skip_patch_when_rc_nonzero(self, tmp_path: Path) -> None:
        out = tmp_path / "out.mkv"

        def fake_run_tool(
            cmd: Any,
            on_output: Any = None,
            on_progress_line: Any = None,
            log_path: Any = None,
            cwd: Any = None,
        ) -> tuple[int, str]:
            out.write_bytes(_build_ebml_header(dt_version=4, dt_read_version=4) + b"X")
            return 1, ""

        adapter = MkcleanAdapter(Path("mkclean.exe"))
        with patch("furnace.adapters.mkclean.run_tool", side_effect=fake_run_tool):
            rc = adapter.clean(Path("in.mkv"), out)

        assert rc == 1
        data = out.read_bytes()
        # Patch must not have run — DTRV stays at 4.
        assert data[-2] == 0x04
