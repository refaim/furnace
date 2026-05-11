from __future__ import annotations

import binascii
import logging
import re
from collections.abc import Callable
from pathlib import Path

from furnace.core.progress import ProgressSample

from ._subprocess import OutputCallback, run_tool

logger = logging.getLogger(__name__)

MKCLEAN_STAGE_COUNT = 3  # mkclean reports "Progress N/3:" — three stages, each 0-100%

_MKCLEAN_PROGRESS_RE = re.compile(r"^Progress\s+(\d)/3:\s*(\d+)%\s*$")

_EBML_MAGIC = b"\x1a\x45\xdf\xa3"
_DTRV_ID = b"\x42\x85\x81"  # DocTypeReadVersion ID 0x4285 + 1-byte VINT size 0x81
_CRC_HEADER = b"\xbf\x84"  # CRC-32 element ID 0xBF + fixed size 4
_EBML_HEAD_PEEK = 128  # mkclean/mkvmerge headers are ~30 bytes; 128 is generous
_FFMPEG_MAX_DTRV = 2  # highest DocTypeReadVersion FFmpeg/lavf will open


def _patch_doctype_read_version(path: Path) -> None:
    """Force EBML DocTypeReadVersion <= 2 in place.

    mkclean v0.9.0 with ``--doctype 6`` writes DocTypeVersion=4 *and*
    DocTypeReadVersion=4. FFmpeg/lavf refuses to open a Matroska file
    whose DocTypeReadVersion > 2 — even though the body is otherwise
    standard matroska v4. mkvmerge always writes DocTypeReadVersion=2
    next to DocTypeVersion=4, which the spec endorses ("a v2 reader can
    read this; v4 features are optional"). We match mkvmerge.

    No-op if DocTypeReadVersion is already <= 2 or the element is absent.
    Raises ValueError if the file is not an EBML/Matroska stream.
    """
    with path.open("r+b") as f:
        head = bytearray(f.read(_EBML_HEAD_PEEK))
        if not head.startswith(_EBML_MAGIC):
            msg = f"{path}: not an EBML/Matroska file (no 1A 45 DF A3 magic)"
            raise ValueError(msg)
        dtrv_idx = head.find(_DTRV_ID)
        if dtrv_idx < 0:
            return
        value_idx = dtrv_idx + len(_DTRV_ID)
        if head[value_idx] <= _FFMPEG_MAX_DTRV:
            return
        head[value_idx] = _FFMPEG_MAX_DTRV
        # If a CRC-32 element precedes DTRV in the header, recompute it.
        # Strict EBML parsers reject the file otherwise.
        crc_idx = head.find(_CRC_HEADER)
        if 0 <= crc_idx < dtrv_idx:
            header_end = 5 + (head[4] & 0x7F)  # 1-byte VINT size descriptor
            crc_value_idx = crc_idx + len(_CRC_HEADER)
            new_crc = binascii.crc32(bytes(head[crc_value_idx + 4 : header_end])) & 0xFFFFFFFF
            head[crc_value_idx : crc_value_idx + 4] = new_crc.to_bytes(4, "little")
        f.seek(0)
        f.write(head)


def _parse_mkclean_progress_line(line: str) -> ProgressSample | None:
    """Parse an mkclean staged progress line into a sample.

    Format (confirmed by existing UI regex at run_tui.py:49):

        Progress 1/3:   42%
        Progress 2/3:   85%
        Progress 3/3:  100%

    Three phases, each 0-100%. Maps to a single continuous 0-100% fraction
    (same mapping the UI hack used at run_tui.py:535).
    """
    m = _MKCLEAN_PROGRESS_RE.match(line.strip())
    if not m:
        return None
    stage = int(m.group(1))
    stage_pct = int(m.group(2))
    if not 1 <= stage <= MKCLEAN_STAGE_COUNT:
        return None
    fraction = ((stage - 1) + stage_pct / 100.0) / MKCLEAN_STAGE_COUNT
    return ProgressSample(fraction=max(0.0, min(1.0, fraction)))


class MkcleanAdapter:
    """Implements Cleaner."""

    def __init__(self, mkclean_path: Path, on_output: OutputCallback = None, log_dir: Path | None = None) -> None:
        self._mkclean = mkclean_path
        self._on_output = on_output
        self._log_dir = log_dir

    def set_log_dir(self, log_dir: Path | None) -> None:
        self._log_dir = log_dir

    def clean(
        self,
        input_path: Path,
        output_path: Path,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int:
        """mkclean input.mkv output.mkv"""
        # --doctype 6 forces 'matroska v4' output. The default 'matroska v2'
        # silently drops BlockAdditionMapping / BlockAddIDName /
        # BlockAddIDExtraData — the container elements that flag a track as
        # Dolby Vision. With v4 those survive and DV is preserved end-to-end.
        cmd = [str(self._mkclean), "--doctype", "6", str(input_path), str(output_path)]
        log_path = self._log_dir / "mkclean.log" if self._log_dir else None

        def _on_progress_line(line: str) -> bool:
            sample = _parse_mkclean_progress_line(line)
            if sample is None:
                return False
            if on_progress is not None:
                on_progress(sample)
            return True

        rc, _out = run_tool(
            cmd,
            on_output=self._on_output,
            log_path=log_path,
            on_progress_line=_on_progress_line,
        )
        if rc == 0:
            _patch_doctype_read_version(output_path)
        return rc
