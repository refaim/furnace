from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Callable
from pathlib import Path

from furnace.adapters._subprocess import run_tool
from furnace.adapters.mkvmerge import _parse_mkvmerge_progress_line
from furnace.core.chapters import fix_chapters_file
from furnace.core.models import DiscSource, DiscTitle, DiscType
from furnace.core.ports import DiscDemuxerPort, PcmTranscoder, PlanReporter, Prober
from furnace.core.progress import ProgressSample

logger = logging.getLogger(__name__)

_DISC_DIR_NAMES: dict[str, DiscType] = {
    "VIDEO_TS": DiscType.DVD,
    "BDMV": DiscType.BLURAY,
}

# Extensions that mkvmerge can mux, grouped by track type. Update both the
# group below AND nothing else — _MKV_TRACK_EXTS is derived so a new codec
# can never silently disable the audio-desync check (the count-mismatch
# guard in _compute_audio_sync_offsets relies on these being in sync).
_VIDEO_EXTS = {".mkv", ".m2v", ".h264", ".h265"}
_AUDIO_EXTS = {
    ".flac",
    ".ac3",
    ".eac3",
    ".dts",
    ".dtsma",
    ".dtshr",
    ".thd",
    ".wav",
    ".m4a",
}
_SUBTITLE_EXTS = {".sup"}
_MKV_TRACK_EXTS = _VIDEO_EXTS | _AUDIO_EXTS | _SUBTITLE_EXTS

_CHAPTERS_EXT = ".txt"
# Minimum audio-vs-video duration gap (ms) that triggers --sync correction.
# Below this, the gap falls within natural BD audio end-trim and is ignored.
_AUDIO_DESYNC_THRESHOLD_MS = 500

_DURATION_TAG_RE = re.compile(r"^(\d+):(\d{2}):(\d{2})(?:\.(\d{1,9}))?$")


def _parse_mkv_duration_tag(value: str | None) -> int | None:
    """Parse a mkvmerge DURATION tag (``HH:MM:SS.nnnnnnnnn``) to milliseconds.

    Returns None on missing, empty, or unparseable input.
    """
    if not value:
        return None
    m = _DURATION_TAG_RE.match(value)
    if m is None:
        return None
    h, minutes, seconds, frac = m.groups()
    total_ms = (int(h) * 3600 + int(minutes) * 60 + int(seconds)) * 1000
    if frac:
        total_ms += int(frac.ljust(9, "0")) // 1_000_000
    return total_ms


class DiscDemuxer:
    """Detect disc structures and orchestrate demux via appropriate adapter."""

    def __init__(
        self,
        bd_port: DiscDemuxerPort,
        dvd_port: DiscDemuxerPort,
        mkvmerge_path: Path | None = None,
        pcm_transcoder: PcmTranscoder | None = None,
        prober: Prober | None = None,
    ) -> None:
        self._ports: dict[DiscType, DiscDemuxerPort] = {
            DiscType.BLURAY: bd_port,
            DiscType.DVD: dvd_port,
        }
        self._mkvmerge = mkvmerge_path
        self._pcm_transcoder = pcm_transcoder
        self._prober = prober

    def _port_for(self, disc: DiscSource) -> DiscDemuxerPort:
        return self._ports[disc.disc_type]

    def detect(self, source: Path) -> list[DiscSource]:
        """Recursively search source for VIDEO_TS/ and BDMV/ directories."""
        results: list[DiscSource] = []
        for path in sorted(source.rglob("*")):
            if not path.is_dir():
                continue
            if ".furnace_demux" in path.parts:
                continue
            disc_type = _DISC_DIR_NAMES.get(path.name)
            if disc_type is not None:
                results.append(DiscSource(path=path, disc_type=disc_type))
                logger.info("Detected %s at %s", disc_type.value.upper(), path)
        return results

    def list_titles(self, disc: DiscSource) -> list[DiscTitle]:
        """List titles from a disc via the appropriate adapter."""
        return self._port_for(disc).list_titles(disc.path)

    def demux(
        self,
        discs: list[DiscSource],
        selected_titles: dict[DiscSource, list[DiscTitle]],
        demux_dir: Path,
        on_output: Callable[[str], None] | None = None,
        reporter: PlanReporter | None = None,
    ) -> list[Path]:
        """Demux selected titles to MKV files.

        `on_output` is a raw line callback used for live console echoing of
        the underlying tool output (eac3to, makemkv, mkvmerge). It is not the
        new structured progress channel — adapter-level progress is wired
        through the executor in the run phase.

        `reporter`, when provided, receives structured per-disc / per-title
        events:

        - `demux_disc_cached(label)` — every selected title is already on disk
        - `demux_disc_start(label)` — at least one title needs work
        - `demux_title_start(n)` — fresh title begins
        - `demux_title_substep("rip"|"transcode N/M"|"remux", has_progress=True)`
        - `demux_title_progress(fraction)` — forwarded from adapter
        - `demux_title_done()` / `demux_title_failed(reason)`

        Behaviour:

        - Skips titles with existing .done marker.
        - Deletes MKV without .done marker (partial) and re-demuxes.
        - Raises RuntimeError on demux failure (after reporting).
        - For BD (eac3to): demuxes to separate files, then muxes via mkvmerge.
        - For DVD (MakeMKV): single MKV output directly.
        """
        demux_dir.mkdir(parents=True, exist_ok=True)
        result_paths: list[Path] = []

        for disc in discs:
            titles = selected_titles.get(disc, [])
            disc_label = disc.path.parent.name
            port = self._port_for(disc)

            # Determine cached-ness up front: a disc is cached only if EVERY
            # selected title already has a .done marker AND at least one MKV
            # file matching that title. A bare .done with no MKV (e.g. partial
            # cleanup that left only the marker) does NOT count as cached and
            # must fall through to the per-title rip path so the title is
            # re-demuxed.
            all_cached = all(
                (demux_dir / f"{disc_label}_title_{t.number}.done").exists()
                and self._find_done_files(demux_dir, disc_label, t.number)
                for t in titles
            )
            if titles and all_cached:
                if reporter is not None:
                    reporter.demux_disc_cached(disc_label)
                for title in titles:
                    existing = self._find_done_files(demux_dir, disc_label, title.number)
                    result_paths.extend(existing)
                continue

            if titles and reporter is not None:
                reporter.demux_disc_start(disc_label)

            for title in titles:
                done_name = f"{disc_label}_title_{title.number}.done"
                done_marker = demux_dir / done_name

                # Check for already-demuxed files
                if done_marker.exists():
                    existing = self._find_done_files(demux_dir, disc_label, title.number)
                    if existing:
                        logger.info("Already demuxed, skipping: title %d", title.number)
                        result_paths.extend(existing)
                        continue

                if reporter is not None:
                    reporter.demux_title_start(title.number)

                # Clean up partial demux (no done marker)
                self._clean_partial(demux_dir, disc_label, title.number)

                logger.info(
                    "Demuxing title %d from %s",
                    title.number,
                    disc.path,
                )

                # Each title gets its own subdir to isolate adapter output
                title_dir = demux_dir / f"{disc_label}_title_{title.number}"
                if title_dir.exists():
                    shutil.rmtree(title_dir)
                title_dir.mkdir()

                def _rip_progress(s: ProgressSample) -> None:
                    if reporter is not None and s.fraction is not None:
                        reporter.demux_title_progress(s.fraction)

                try:
                    if reporter is not None:
                        reporter.demux_title_substep("rip", has_progress=True)
                    created_files = port.demux_title(
                        disc.path,
                        title.number,
                        title_dir,
                        on_progress=_rip_progress,
                    )

                    created_files = self._transcode_w64_files(
                        created_files, reporter=reporter,
                    )

                    # If multiple files (BD/eac3to), mux into single MKV
                    final_mkv = demux_dir / f"{disc_label}_title_{title.number}.mkv"
                    if self._needs_muxing(created_files):
                        if reporter is not None:
                            reporter.demux_title_substep("remux", has_progress=True)
                        self._mux_to_mkv(
                            created_files,
                            final_mkv,
                            on_output,
                            on_progress=_rip_progress,
                        )
                    else:
                        # Single MKV (DVD/MakeMKV) — just move it
                        src_mkv = next(f for f in created_files if f.suffix.lower() == ".mkv")
                        shutil.move(str(src_mkv), str(final_mkv))

                    # Clean up title subdir
                    shutil.rmtree(title_dir, ignore_errors=True)

                    done_marker.touch()
                    result_paths.append(final_mkv)

                    if reporter is not None:
                        reporter.demux_title_done()
                except Exception as exc:
                    if reporter is not None:
                        reporter.demux_title_failed(str(exc))
                    raise

        return result_paths

    @staticmethod
    def _needs_muxing(files: list[Path]) -> bool:
        """Check if demux output needs muxing (multiple files, not a single MKV)."""
        mkv_files = [f for f in files if f.suffix.lower() == ".mkv"]
        non_mkv = [f for f in files if f.suffix.lower() != ".mkv"]
        return len(mkv_files) != 1 or len(non_mkv) > 0

    def _transcode_w64_files(
        self,
        files: list[Path],
        reporter: PlanReporter | None = None,
    ) -> list[Path]:
        """Replace any Wave64 (.w64) entries in ``files`` with transcoded FLAC.

        eac3to sometimes writes PCM to Wave64 when the stream exceeds the 4 GB
        WAV limit; mkvmerge cannot read Wave64, so we transcode to FLAC
        (lossless) before muxing. The original .w64 is deleted on success.

        - If no .w64 is present: return ``files`` unchanged.
        - If pcm_transcoder is None and at least one .w64 is present: raise
          RuntimeError (fail fast rather than silently dropping the track).
        - If the transcoder returns non-zero rc for any file: raise
          RuntimeError, leaving the .w64 on disk for inspection.

        When ``reporter`` is provided, each transcode emits a
        ``demux_title_substep("transcode N/M", has_progress=True)`` followed
        by per-file ``demux_title_progress`` events forwarded from the
        underlying transcoder.
        """
        if not any(f.suffix.lower() == ".w64" for f in files):
            return files
        if self._pcm_transcoder is None:
            w64_names = [f.name for f in files if f.suffix.lower() == ".w64"]
            msg = (
                "pcm_transcoder not configured; cannot handle Wave64 demux "
                f"output: {w64_names}"
            )
            raise RuntimeError(msg)

        w64_files = [f for f in files if f.suffix.lower() == ".w64"]
        total = len(w64_files)
        result: list[Path] = []
        w64_seen = 0
        for f in files:
            if f.suffix.lower() != ".w64":
                result.append(f)
                continue
            w64_seen += 1
            if reporter is not None:
                label = (
                    "transcode"
                    if total == 1
                    else f"transcode {w64_seen}/{total}"
                )
                reporter.demux_title_substep(label, has_progress=True)

            def _tr_progress(s: ProgressSample) -> None:
                if reporter is not None and s.fraction is not None:
                    reporter.demux_title_progress(s.fraction)

            flac_path = f.with_suffix(".flac")
            logger.info("Transcoding Wave64 to FLAC: %s -> %s", f.name, flac_path.name)
            rc = self._pcm_transcoder.transcode_to_flac(
                f, flac_path, on_progress=_tr_progress,
            )
            if rc != 0:
                msg = f"eac3to transcode of {f.name} to FLAC failed (rc={rc})"
                raise RuntimeError(msg)
            f.unlink()
            result.append(flac_path)
        return result

    def _mux_to_mkv(
        self,
        files: list[Path],
        output_mkv: Path,
        on_output: Callable[[str], None] | None = None,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> None:
        """Mux separate track files into a single MKV via mkvmerge.

        After the initial mux, when a prober is configured we probe the
        output for per-track DURATION tags. If any audio track is shorter
        than the video track by more than ``_AUDIO_DESYNC_THRESHOLD_MS``
        — typical of multi-segment BD titles whose intro segment carries
        no PCM audio — the audio is at PTS 0 instead of the intended
        offset, so we re-mux with ``--sync`` to push each short audio
        track later by the missing-prefix duration. The fix assumes the
        gap is at the START (the common BD authoring pattern); an OUTRO-
        only gap would be wrongly shifted by this heuristic.
        """
        if self._mkvmerge is None:
            msg = "mkvmerge path not configured, cannot mux BD demux output"
            raise RuntimeError(msg)

        self._run_mkvmerge_mux(files, output_mkv, {}, on_output, on_progress)

        if self._prober is None:
            return
        offsets = self._compute_audio_sync_offsets(output_mkv, files)
        if not offsets:
            return

        summary = ", ".join(f"{p.name}: +{d}ms" for p, d in offsets.items())
        logger.warning(
            "Audio/video desync detected in %s (%s); re-muxing with --sync",
            output_mkv.name, summary,
        )
        output_mkv.unlink(missing_ok=True)
        self._run_mkvmerge_mux(files, output_mkv, offsets, on_output, on_progress)

    def _run_mkvmerge_mux(
        self,
        files: list[Path],
        output_mkv: Path,
        audio_sync_offsets: dict[Path, int],
        on_output: Callable[[str], None] | None,
        on_progress: Callable[[ProgressSample], None] | None,
    ) -> None:
        assert self._mkvmerge is not None  # noqa: S101 — narrowed by caller

        lang_re = re.compile(r"\[(\w{3})\]")
        retry_note = " (--sync correction)" if audio_sync_offsets else ""
        cmd: list[str] = [str(self._mkvmerge), "-o", str(output_mkv)]

        # Find chapters file
        chapters_file: Path | None = None
        for f in files:
            if f.suffix.lower() == _CHAPTERS_EXT:
                chapters_file = f

        # Add track files. eac3to encodes language in the filename as
        # ``[rus]`` etc.; --sync gets prefixed for audio tracks that need
        # delay correction (single track per file → TID 0).
        for f in files:
            if f.suffix.lower() not in _MKV_TRACK_EXTS:
                continue
            delay = audio_sync_offsets.get(f)
            if delay is not None:
                cmd += ["--sync", f"0:{delay}"]
            lang_match = lang_re.search(f.name)
            if lang_match:
                cmd += ["--language", f"0:{lang_match.group(1)}"]
            cmd.append(str(f))

        # Add chapters (fix mojibake if needed)
        if chapters_file is not None:
            if fix_chapters_file(chapters_file):
                logger.info("Fixed mojibake in chapters file %s", chapters_file.name)
            cmd += ["--chapters", str(chapters_file)]

        logger.info("Muxing demuxed tracks into %s%s", output_mkv.name, retry_note)
        logger.debug("mkvmerge cmd: %s", " ".join(cmd))

        def _on_progress_line(line: str) -> bool:
            sample = _parse_mkvmerge_progress_line(line)
            if sample is None:
                return False
            if on_progress is not None:
                on_progress(sample)
            return True

        rc, output = run_tool(
            cmd, on_output=on_output, on_progress_line=_on_progress_line,
        )
        if rc not in (0, 1):  # mkvmerge returns 1 for warnings
            raise RuntimeError(f"mkvmerge failed (rc={rc}): {output[-500:]}")

    def _compute_audio_sync_offsets(
        self,
        mkv_path: Path,
        files: list[Path],
    ) -> dict[Path, int]:
        """Probe ``mkv_path`` and return audio-file → positive delay (ms)
        for any audio track shorter than the video track by more than
        ``_AUDIO_DESYNC_THRESHOLD_MS``. Empty dict when no fix is needed
        or detection is not possible (no DURATION tags, probe error,
        audio/source-file count mismatch).
        """
        assert self._prober is not None  # noqa: S101 — narrowed by caller
        try:
            data = self._prober.probe(mkv_path)
        except Exception as exc:  # noqa: BLE001 — graceful degradation
            logger.warning("desync probe failed for %s: %s", mkv_path.name, exc)
            return {}

        video_ms: int | None = None
        audio_durations: list[int | None] = []
        for s in data.get("streams", []):
            ctype = s.get("codec_type")
            dur = _parse_mkv_duration_tag(s.get("tags", {}).get("DURATION"))
            if ctype == "video" and video_ms is None:
                video_ms = dur
            elif ctype == "audio":
                audio_durations.append(dur)

        if video_ms is None or not audio_durations:
            return {}

        audio_source_files = [
            f for f in files if f.suffix.lower() in _AUDIO_EXTS
        ]
        if len(audio_source_files) != len(audio_durations):
            logger.debug(
                "audio file/stream count mismatch (%d vs %d); skip desync fix",
                len(audio_source_files), len(audio_durations),
            )
            return {}

        offsets: dict[Path, int] = {}
        for f, a_ms in zip(audio_source_files, audio_durations, strict=True):
            if a_ms is None:
                continue
            delta = video_ms - a_ms
            if delta >= _AUDIO_DESYNC_THRESHOLD_MS:
                offsets[f] = delta
        return offsets

    @staticmethod
    def _find_done_files(demux_dir: Path, disc_label: str, title_num: int) -> list[Path]:
        """Find MKV files for an already-demuxed title."""
        prefix = f"{disc_label}_title_{title_num}"
        return sorted(p for p in demux_dir.glob(f"{prefix}*.mkv") if p.is_file())

    @staticmethod
    def _clean_partial(demux_dir: Path, disc_label: str, title_num: int) -> None:
        """Remove partial MKV files and stale done markers for a title."""
        prefix = f"{disc_label}_title_{title_num}"
        for p in demux_dir.glob(f"{prefix}*.mkv"):
            logger.warning("Deleting partial demux: %s", p.name)
            p.unlink()
        done = demux_dir / f"{prefix}.done"
        if done.exists():
            done.unlink()
