from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from furnace.core.models import DiscTitle, DownmixMode
from furnace.core.progress import ProgressSample

from ._subprocess import OutputCallback, run_tool

logger = logging.getLogger(__name__)

_PLAYLIST_RE = re.compile(r"^(\d+)\)\s+(.+),\s+(\d+:\d{2}(?::\d{2})?)$")
_TRACK_RE = re.compile(r"^(\d+):\s+(.+)$")
_LANG_RE = re.compile(r"\[(\w{3})\]")
_EAC3TO_PROGRESS_RE = re.compile(r"^process:\s*(\d+)%\s*$")
# Broader pattern used for log suppression: matches both `process:` and
# `analyze:` phases. The parser itself only emits samples for `process:`,
# but `analyze:` lines should also be kept out of the log.
_EAC3TO_ANY_PROGRESS_RE = re.compile(r"^(?:process|analyze):\s*\d+%\s*$")


def _parse_eac3to_progress_line(line: str) -> ProgressSample | None:
    """Parse an eac3to ``-progressnumbers`` line into a sample.

    Format: ``process: NN%`` (integer percent, trailing %). During multi-phase
    operations eac3to restarts from 0 for each phase; the executor handles
    phase transitions explicitly via ``tracker.reset()``, so this parser only
    captures the ``process:`` lines and ignores any ``analyze:`` prefix.
    """
    m = _EAC3TO_PROGRESS_RE.match(line.strip())
    if not m:
        return None
    try:
        pct = int(m.group(1))
    except ValueError:
        return None
    return ProgressSample(fraction=pct / 100.0)


def _is_eac3to_progress_line(line: str) -> bool:
    """Return True for any ``process: NN%`` or ``analyze: NN%`` line.

    Used by the adapter closure to suppress both phases' lines from the log
    even though only ``process:`` contributes to structured progress.
    """
    return _EAC3TO_ANY_PROGRESS_RE.match(line.strip()) is not None


@dataclass(frozen=True)
class Eac3toTrack:
    """A track parsed from eac3to title listing."""
    number: int
    description: str
    language: str | None   # e.g. "rus", "eng", None for video/chapters
    extension: str         # e.g. ".mkv", ".dts", ".ac3", ".txt"

# Map eac3to codec descriptions to file extensions (raw copy, no re-encode)
_CODEC_EXT_MAP: dict[str, str] = {
    "mpeg2": ".mkv",
    "h264": ".mkv",
    "avc": ".mkv",
    "hevc": ".mkv",
    "vc-1": ".mkv",
    "dts": ".dts",
    "dts-hd": ".dtshd",
    "dts-hd master audio": ".dtshd",
    "dts master audio": ".dtshd",
    "dts hi-res": ".dtshd",
    "ac3": ".ac3",
    "e-ac3": ".eac3",
    "truehd": ".thd",
    "truehd/ac3": ".thd",
    "pcm": ".wav",
    "lpcm": ".wav",
    "flac": ".flac",
    "aac": ".m4a",
    "pgs": ".sup",
    "chapters": ".txt",
}


def _ext_for_track(description: str) -> str:
    """Determine file extension from eac3to track description."""
    desc_lower = description.lower()
    for key, ext in _CODEC_EXT_MAP.items():
        if desc_lower.startswith(key):
            return ext
    # Check for subtitle types that may not be at start
    if "pgs" in desc_lower:
        return ".sup"
    if "chapters" in desc_lower:
        return ".txt"
    return ".bin"


def _parse_duration(s: str) -> float:
    """Parse 'H:MM:SS' or 'M:SS' into total seconds."""
    parts = s.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return 0.0


class Eac3toAdapter:
    """Implements AudioDecoder and DiscDemuxerPort via eac3to."""

    def __init__(
        self,
        eac3to_path: Path,
        on_output: OutputCallback = None,
        log_dir: Path | None = None,
    ) -> None:
        self._eac3to = eac3to_path
        self._on_output = on_output
        self._log_dir = log_dir

    def set_log_dir(self, log_dir: Path | None) -> None:
        self._log_dir = log_dir

    def _log_path(self, label: str) -> Path | None:
        if self._log_dir is None:
            return None
        return self._log_dir / f"eac3to_{label}.log"

    @staticmethod
    def _delay_arg(delay_ms: int) -> list[str]:
        if delay_ms == 0:
            return []
        if delay_ms > 0:
            return [f"+{delay_ms}ms"]
        return [f"{delay_ms}ms"]

    def _run(
        self,
        args: list[str],
        log_label: str,
        on_output: OutputCallback = None,
        on_progress: Callable[[ProgressSample], None] | None = None,
        cwd: Path | None = None,
    ) -> tuple[int, str]:
        """Common eac3to invocation with logging and progress."""
        cmd = [str(self._eac3to), *args, "-progressnumbers"]

        def _on_progress_line(line: str) -> bool:
            # Suppress both `process: N%` and `analyze: N%` from the log,
            # but only emit samples for the `process:` phase (the parser
            # already filters). Unmatched lines pass through.
            if not _is_eac3to_progress_line(line):
                return False
            sample = _parse_eac3to_progress_line(line)
            if sample is not None and on_progress is not None:
                on_progress(sample)
            return True

        rc, output = run_tool(
            cmd,
            on_output=on_output or self._on_output,
            on_progress_line=_on_progress_line,
            log_path=self._log_path(log_label),
            cwd=cwd,
        )
        return rc, output

    # -- AudioDecoder protocol -------------------------------------------------

    def denormalize(
        self,
        input_path: Path,
        output_path: Path,
        delay_ms: int,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int:
        rc, _output = self._run(
            [str(input_path), str(output_path), "-removeDialnorm",
             *self._delay_arg(delay_ms)],
            "denorm",
            on_progress=on_progress,
        )
        return rc

    def decode_lossless(
        self,
        input_path: Path,
        output_path: Path,
        delay_ms: int,
        on_progress: Callable[[ProgressSample], None] | None = None,
        *,
        downmix: DownmixMode | None = None,
    ) -> int:
        downmix_args: list[str] = []
        if downmix == DownmixMode.STEREO:
            downmix_args.append("-downStereo")
        elif downmix == DownmixMode.DOWN6:
            downmix_args.append("-down6")

        rc, _output = self._run(
            [str(input_path), str(output_path), "-removeDialnorm",
             *self._delay_arg(delay_ms), *downmix_args],
            "decode",
            on_progress=on_progress,
        )
        return rc

    # -- DiscDemuxerPort protocol ----------------------------------------------

    def list_titles(self, disc_path: Path) -> list[DiscTitle]:
        """Run eac3to on disc path, parse playlist listing."""
        cmd = [str(self._eac3to), str(disc_path)]
        rc, output = run_tool(cmd, on_output=self._on_output, log_path=self._log_path("list_titles"))
        if rc != 0:
            raise RuntimeError(
                f"eac3to listing failed for {disc_path} (rc={rc})"
            )
        return self._parse_playlist_output(output)

    def demux_title(
        self,
        disc_path: Path,
        title_num: int,
        output_dir: Path,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> list[Path]:
        """Demux one BD playlist to separate files in output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve to absolute: we set cwd=output_dir below, so any relative
        # disc_path would no longer resolve from the subprocess's new cwd,
        # making eac3to fail with "HD DVD / Blu-Ray disc structure not found".
        disc_path = disc_path.resolve()

        rc, _output = self._run(
            [str(disc_path), f"{title_num})", "-demux"],
            f"demux_t{title_num}",
            on_progress=on_progress,
            cwd=output_dir,
        )
        if rc != 0:
            raise RuntimeError(
                f"eac3to demux failed for {disc_path} title {title_num} (rc={rc})"
            )
        return sorted(p for p in output_dir.iterdir() if p.is_file())

    # -- Parsing ---------------------------------------------------------------

    @staticmethod
    def _parse_track_listing(output: str) -> list[Eac3toTrack]:
        """Parse eac3to track listing (from running eac3to BDMV N)).

        Example line: "3: DTS-HD Master Audio, [rus], 5.1 channels, 16 bits, 48kHz"
        """
        results: list[Eac3toTrack] = []
        for line in output.splitlines():
            line = line.strip()
            m = _TRACK_RE.match(line)
            if not m:
                continue
            track_num = int(m.group(1))
            description = m.group(2).strip()
            lang_match = _LANG_RE.search(description)
            language = lang_match.group(1) if lang_match else None
            ext = _ext_for_track(description)
            results.append(Eac3toTrack(
                number=track_num,
                description=description,
                language=language,
                extension=ext,
            ))
        return results

    @staticmethod
    def _parse_playlist_output(output: str) -> list[DiscTitle]:
        """Parse eac3to listing output into DiscTitle objects."""
        results: list[DiscTitle] = []
        for line in output.splitlines():
            line = line.strip()
            m = _PLAYLIST_RE.match(line)
            if not m:
                continue
            number = int(m.group(1))
            label = m.group(2).strip()
            duration_str = m.group(3)
            duration_s = _parse_duration(duration_str)
            results.append(DiscTitle(
                number=number,
                duration_s=duration_s,
                raw_label=f"{number}) {label}, {duration_str}",
            ))
        return results
