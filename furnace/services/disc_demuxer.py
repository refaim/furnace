from __future__ import annotations

import logging
import math
import re
import shutil
from collections.abc import Callable
from pathlib import Path

from furnace.adapters._subprocess import run_tool
from furnace.adapters.mkvmerge import _parse_mkvmerge_progress_line
from furnace.core.chapters import fix_chapters_file
from furnace.core.models import DiscSource, DiscTitle, DiscType
from furnace.core.ports import (
    DiscDemuxerPort,
    PcmTranscoder,
    PlanReporter,
    Prober,
)
from furnace.core.progress import ProgressSample

logger = logging.getLogger(__name__)

_DISC_DIR_NAMES: dict[str, DiscType] = {
    "VIDEO_TS": DiscType.DVD,
    "BDMV": DiscType.BLURAY,
}

# Extensions that mkvmerge can mux, grouped by track type. Update both the
# group below AND nothing else — _MKV_TRACK_EXTS is derived so a new codec
# can never silently disable the audio-sync check (the audio stream/file
# count guard in _compute_audio_sync_offsets relies on these being in sync).
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
# Minimum real source audio-vs-video start offset (ms) that triggers --sync
# correction. Below this, the offset is within ordinary BD frame-alignment
# jitter and the audio is treated as starting together with the video.
_AUDIO_DESYNC_THRESHOLD_MS = 500
# eac3to title labels embed the source m2ts segment name(s), e.g.
# "1) 00001.mpls, 00001.m2ts, 1:36:32" or multi-segment "00800.m2ts+00801.m2ts".
_M2TS_RE = re.compile(r"\d+\.m2ts", re.IGNORECASE)


def _parse_start_time(value: object) -> float | None:
    """Parse an ffprobe ``start_time`` field (a string like ``"5.866667"``,
    or ``"N/A"``/missing) to a finite float. ``None`` when absent, non-numeric,
    or non-finite (``inf``/``nan`` — which ``float()`` accepts but which would
    blow up the later offset arithmetic).
    """
    if value is None:
        return None
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


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
                            source_segments=self._bd_source_segments(disc, title),
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
        source_segments: list[Path] | None = None,
    ) -> None:
        """Mux separate track files into a single MKV via mkvmerge.

        After the initial mux, when a prober is configured and
        ``source_segments`` is a single BD ``.m2ts``, we probe that source
        and apply ``--sync`` only by the REAL audio-vs-video start offset
        found there. An aligned source (the common case) yields offset 0 →
        no ``--sync``, regardless of any output duration mismatch caused by
        trailing/outro silence. When the source cannot be probed
        authoritatively (multi-segment title, missing/failed probe, or an
        ambiguous stream layout) no ``--sync`` is applied — the audio stays
        at PTS 0, which is where eac3to already aligned it.
        """
        if self._mkvmerge is None:
            msg = "mkvmerge path not configured, cannot mux BD demux output"
            raise RuntimeError(msg)

        self._run_mkvmerge_mux(files, output_mkv, {}, on_output, on_progress)

        if self._prober is None:
            return
        offsets = self._compute_audio_sync_offsets(files, source_segments)
        if not offsets:
            return

        summary = ", ".join(f"{p.name}: +{d}ms" for p, d in offsets.items())
        logger.warning(
            "Source audio delay in %s (%s); re-muxing with --sync",
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
        files: list[Path],
        source_segments: list[Path] | None = None,
    ) -> dict[Path, int]:
        """Return audio-file → positive ``--sync`` delay (ms) derived from
        the REAL source audio-vs-video start offset.

        ``source_segments`` must be a single BD ``.m2ts`` for source-truth to
        apply; we probe it and, for each demuxed audio track, compute
        ``audio.start_time - video.start_time`` and apply ``--sync`` only when
        that offset is at least ``_AUDIO_DESYNC_THRESHOLD_MS``. An aligned
        source (the common case) yields offset 0 → no ``--sync``, regardless
        of any output duration mismatch caused by trailing/outro silence.

        Returns an empty dict (no ``--sync``) whenever the offset cannot be
        determined authoritatively: no single source segment (multi-segment
        title), probe error, no video stream / no video start_time, an audio
        stream missing its start_time, or an audio stream/file count
        mismatch. In every such case the audio is left at PTS 0, where eac3to
        already aligned it.
        """
        assert self._prober is not None  # noqa: S101 — narrowed by caller
        if source_segments is None or len(source_segments) != 1:
            return {}
        m2ts = source_segments[0]
        try:
            data = self._prober.probe(m2ts)
        except Exception as exc:  # noqa: BLE001 — graceful degradation
            logger.warning("source A/V offset probe failed for %s: %s", m2ts.name, exc)
            return {}

        video_start: float | None = None
        audio_starts: list[float | None] = []
        for s in data.get("streams", []):
            ctype = s.get("codec_type")
            start = _parse_start_time(s.get("start_time"))
            if ctype == "video" and video_start is None:
                video_start = start
            elif ctype == "audio":
                audio_starts.append(start)

        if video_start is None:
            logger.info(
                "source %s has no usable video start_time; no --sync applied",
                m2ts.name,
            )
            return {}

        # Pair each demuxed audio file with its source stream positionally.
        # Both lists are in eac3to track order: ``files`` is eac3to's demux
        # output (named "N) ...") and ``audio_starts`` is ffprobe's container
        # order of the same m2ts — the two agree numerically. ``files`` is
        # sorted lexicographically, which only diverges from numeric order once
        # a track is numbered >=10 (a title with 10+ demuxed tracks), which no
        # real BD title reaches; and even then every audio stream in a single
        # m2ts shares the same start PTS, so a mispairing would assign identical
        # offsets — harmless.
        audio_source_files = [f for f in files if f.suffix.lower() in _AUDIO_EXTS]
        if len(audio_source_files) != len(audio_starts):
            logger.debug(
                "source audio stream/file count mismatch (%d vs %d); "
                "no --sync applied",
                len(audio_starts), len(audio_source_files),
            )
            return {}

        offsets: dict[Path, int] = {}
        for f, a_start in zip(audio_source_files, audio_starts, strict=True):
            if a_start is None:
                logger.info(
                    "source %s audio track has no start_time; no --sync applied",
                    m2ts.name,
                )
                return {}
            offset_ms = round((a_start - video_start) * 1000)
            if offset_ms >= _AUDIO_DESYNC_THRESHOLD_MS:
                logger.info(
                    "source %s: %s starts %dms after video; applying --sync",
                    m2ts.name, f.name, offset_ms,
                )
                offsets[f] = offset_ms
        return offsets

    @staticmethod
    def _bd_source_segments(disc: DiscSource, title: DiscTitle) -> list[Path] | None:
        """Map a BD title's ``raw_label`` to its source ``.m2ts`` segment
        paths under ``<BDMV>/STREAM``.

        Returns ``None`` for non-BD discs or when the label carries no m2ts
        name; in that case source-truth offset detection is disabled and no
        ``--sync`` is applied (the audio stays at PTS 0).
        """
        if disc.disc_type != DiscType.BLURAY:
            return None
        names = _M2TS_RE.findall(title.raw_label)
        if not names:
            return None
        stream_dir = disc.path / "STREAM"
        return [stream_dir / name for name in names]

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
