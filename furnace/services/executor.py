from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..core.models import (
    AudioAction,
    AudioInstruction,
    Job,
    JobStatus,
    Plan,
    SubtitleAction,
    SubtitleInstruction,
)
from ..core.ports import (
    AacEncoder,
    AudioDecoder,
    AudioExtractor,
    Cleaner,
    DoviProcessor,
    Encoder,
    Muxer,
    Prober,
    Tagger,
)
from ..core.progress import ProgressSample, ProgressTracker
from ..plan import update_job_status
from furnace import VERSION as FURNACE_VERSION

logger = logging.getLogger(__name__)
MAX_STDERR_LINES = 6

# Extension mapping for audio codec names to file extensions
_AUDIO_CODEC_EXT: dict[str, str] = {
    "aac":      ".m4a",
    "ac3":      ".ac3",
    "eac3":     ".eac3",
    "dts":      ".dts",
    "truehd":   ".thd",
    "flac":     ".flac",
    "pcm_s16le": ".wav",
    "pcm_s24le": ".wav",
    "pcm_s16be": ".wav",
    "mp2":      ".mp2",
    "mp3":      ".mp3",
    "vorbis":   ".ogg",
    "opus":     ".opus",
    "wmav2":    ".wma",
    "wmapro":   ".wma",
    "amr_nb":   ".amr",
}

# Extension mapping for subtitle codec names
_SUBTITLE_CODEC_EXT: dict[str, str] = {
    "subrip":             ".srt",
    "ass":                ".ass",
    "hdmv_pgs_subtitle":  ".sup",
    "dvd_subtitle":       ".mkv",  # VOBSUB can't be extracted raw; wrap in MKV
}


class Executor:
    def __init__(
        self,
        encoder: Encoder,
        audio_extractor: AudioExtractor,
        audio_decoder: AudioDecoder,
        aac_encoder: AacEncoder,
        muxer: Muxer,
        tagger: Tagger,
        cleaner: Cleaner,
        prober: Prober,
        dovi_processor: DoviProcessor | None = None,
        progress: Any | None = None,  # RunApp or similar (optional, avoids circular import)
        log_dir: Path | None = None,
    ) -> None:
        self._encoder = encoder
        self._audio_extractor = audio_extractor
        self._audio_decoder = audio_decoder
        self._aac_encoder = aac_encoder
        self._muxer = muxer
        self._tagger = tagger
        self._cleaner = cleaner
        self._prober = prober
        self._dovi_processor = dovi_processor
        self._progress = progress
        self._log_dir = log_dir
        self._shutdown_event = threading.Event()
        self._adapters: list[Any] = [encoder, audio_extractor, audio_decoder, aac_encoder, muxer, tagger, cleaner]
        if dovi_processor is not None:
            self._adapters.append(dovi_processor)

    def _make_progress_callback(
        self,
        total_s: float | None = None,
    ) -> tuple[ProgressTracker, Callable[[ProgressSample], None]]:
        """Create a tracker + wrapper callback for a long-running step.

        Returns the tracker (so caller can `reset()` it between sub-phases) and
        a callback that the adapter receives as `on_progress`. The callback adds
        each sample to the tracker and pushes the snapshot to the TUI.
        """
        tracker = ProgressTracker(total_s=total_s)

        def _on_progress(sample: ProgressSample) -> None:
            tracker.add(sample, time.monotonic())
            if self._progress is not None:
                self._progress.update_progress(tracker.snapshot())

        return tracker, _on_progress

    def _set_adapters_log_dir(self, job_name: str) -> None:
        """Create per-job log subdirectory and update all adapters."""
        if self._log_dir is None:
            return
        job_log_dir = self._log_dir / job_name
        job_log_dir.mkdir(parents=True, exist_ok=True)
        for adapter in self._adapters:
            set_fn = getattr(adapter, "set_log_dir", None)
            if set_fn is not None:
                set_fn(job_log_dir)

    def run(
        self,
        plan: Plan,
        plan_path: Path,
    ) -> None:
        """Execute all pending/error jobs sequentially.
        Update JSON after each via update_job_status.
        Check _shutdown_event between jobs.
        """
        self._vmaf_enabled = plan.vmaf_enabled

        pending_jobs = [
            job for job in plan.jobs
            if job.status in (JobStatus.PENDING, JobStatus.ERROR)
        ]

        logger.debug(
            "Starting execution: %d jobs to process (total: %d)",
            len(pending_jobs),
            len(plan.jobs),
        )

        for i, job in enumerate(pending_jobs):
            if self._shutdown_event.is_set():
                logger.debug("Shutdown requested, stopping before job %s", job.id)
                break

            logger.debug("Starting job %s -> %s", job.id, job.output_file)

            # Set per-job log directory for all adapters
            job_name = Path(job.output_file).stem
            self._set_adapters_log_dir(job_name)

            if self._progress is not None:
                self._progress.start_job(job, i)

            try:
                self._execute_job(job)
                output_size: int | None = None
                output_path = Path(job.output_file)
                if output_path.exists():
                    output_size = output_path.stat().st_size
                update_job_status(
                    plan_path,
                    job.id,
                    JobStatus.DONE,
                    error=None,
                    output_size=output_size,
                    vmaf_score=job.vmaf_score,
                    ssim_score=job.ssim_score,
                )
                logger.debug("Job %s completed successfully", job.id)
                if self._progress is not None:
                    job.output_size = output_size
                    self._progress.finish_job(job)
            except Exception as exc:
                error_msg = str(exc)
                logger.error("Job %s failed: %s", job.id, error_msg)
                update_job_status(
                    plan_path,
                    job.id,
                    JobStatus.ERROR,
                    error=error_msg,
                )

    def _execute_job(self, job: Job) -> None:
        """Full pipeline for one job."""
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        temp_dir = Path(tempfile.mkdtemp(prefix="furnace_"))
        try:
            self._run_pipeline(job, output_path, temp_dir)
        finally:
            # Always clean up temp files
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as exc:
                logger.warning("Failed to clean temp dir %s: %s", temp_dir, exc)

    def _run_pipeline(
        self,
        job: Job,
        output_path: Path,
        temp_dir: Path,
    ) -> None:
        """Inner pipeline logic (separated so finally in _execute_job always runs).

        Order: audio -> subtitles -> video -> mux -> tag -> mkclean
        Audio/subs are fast (seconds), processed first so everything is ready
        when the long video encode finishes.
        """
        main_source = Path(job.source_files[0])
        self._cumulative_audio_size = 0

        # Step 1: Process audio tracks (fast)
        if self._shutdown_event.is_set():
            return

        audio_files: list[tuple[Path, dict[str, Any]]] = []
        for i, audio_instr in enumerate(job.audio):
            if self._shutdown_event.is_set():
                return
            status_msg = (
                f"Processing audio {i + 1}/{len(job.audio)} "
                f"({audio_instr.codec_name} {audio_instr.language})"
            )
            logger.info(status_msg)
            if self._progress is not None:
                self._progress.update_status(status_msg)
                self._progress.add_tool_line(f"[furnace] {status_msg}")
            audio_path = self._process_audio_track(audio_instr, temp_dir, job)
            audio_meta = {
                "language": audio_instr.language,
                "default": audio_instr.is_default,
                "delay_ms": audio_instr.delay_ms if audio_instr.action == AudioAction.COPY else 0,
            }
            audio_files.append((audio_path, audio_meta))
            # Track cumulative output size
            if self._progress is not None and audio_path.exists():
                self._cumulative_audio_size += audio_path.stat().st_size
                self._progress.update_output_size(self._cumulative_audio_size)

        # Step 2: Process subtitle tracks (fast)
        if self._shutdown_event.is_set():
            return

        subtitle_files: list[tuple[Path, dict[str, Any]]] = []
        for i, sub_instr in enumerate(job.subtitles):
            if self._shutdown_event.is_set():
                return
            status_msg = (
                f"Processing subtitle {i + 1}/{len(job.subtitles)} "
                f"({sub_instr.codec_name} {sub_instr.language})"
            )
            logger.info(status_msg)
            if self._progress is not None:
                self._progress.update_status(status_msg)
                self._progress.add_tool_line(f"[furnace] {status_msg}")
            sub_path = self._process_subtitle_track(sub_instr, temp_dir, job)
            sub_meta = {
                "language": sub_instr.language,
                "default": sub_instr.is_default,
                "forced": sub_instr.is_forced,
                "encoding": "UTF-8",
            }
            subtitle_files.append((sub_path, sub_meta))

        # Step 3: DV RPU extraction (if needed)
        rpu_path: Path | None = None
        if job.video_params.dv_mode is not None:
            if self._shutdown_event.is_set():
                return
            if self._dovi_processor is None:
                msg = "DV content requires dovi_tool but it is not configured"
                raise RuntimeError(msg)
            rpu_path = temp_dir / "RPU.bin"
            status_msg = f"Extracting DV RPU (mode={job.video_params.dv_mode.name})"
            logger.info(status_msg)
            if self._progress is not None:
                self._progress.update_status(status_msg)
                self._progress.add_tool_line(f"[furnace] {status_msg}")
            rc = self._dovi_processor.extract_rpu(
                input_path=main_source,
                output_rpu=rpu_path,
                mode=job.video_params.dv_mode,
            )
            if rc != 0:
                raise RuntimeError(f"DV RPU extraction failed with return code {rc}")

        # Step 4: Encode video (slow — main bottleneck)
        if self._shutdown_event.is_set():
            return

        video_output = temp_dir / "video.mkv"
        logger.info("Encoding video: %s", main_source.name)
        if self._progress is not None:
            self._progress.update_status("Encoding video")
            self._progress.add_tool_line(f"[furnace] Encoding video: {main_source.name}")

        _, base_encode_on_progress = self._make_progress_callback(
            total_s=job.duration_s or None,
        )

        def encode_on_progress(sample: ProgressSample) -> None:
            base_encode_on_progress(sample)
            # Preserve the output-size update alongside encoding progress
            try:
                video_size = video_output.stat().st_size if video_output.exists() else 0
            except OSError:
                video_size = 0
            if self._progress is not None:
                self._progress.update_output_size(self._cumulative_audio_size + video_size)

        rc_result = self._encoder.encode(
            input_path=main_source,
            output_path=video_output,
            video_params=job.video_params,
            source_size=job.source_size,
            on_progress=encode_on_progress,
            vmaf_enabled=self._vmaf_enabled,
            rpu_path=rpu_path,
        )
        if rc_result.return_code != 0:
            raise RuntimeError(f"Video encoding failed with return code {rc_result.return_code}")

        # Store metrics from encode
        if rc_result.vmaf_score is not None:
            job.vmaf_score = rc_result.vmaf_score
        if rc_result.ssim_score is not None:
            job.ssim_score = rc_result.ssim_score
        encoder_settings = rc_result.encoder_settings

        # Step 5: Mux
        if self._shutdown_event.is_set():
            return

        muxed_path = temp_dir / "muxed.mkv"
        logger.info("Muxing tracks")
        if self._progress is not None:
            self._progress.update_status("Muxing...")
            self._progress.add_tool_line("[furnace] Muxing tracks")

        # Build attachments list: (path, filename, mime_type)
        attachments: list[tuple[Path, str, str]] = []
        for att_dict in job.attachments:
            att_path = Path(att_dict["source_file"])
            filename = att_dict["filename"]
            mime_type = att_dict["mime_type"]
            attachments.append((att_path, filename, mime_type))

        chapters_source: Path | None = None
        if job.copy_chapters and job.chapters_source:
            chapters_source = self._extract_chapters_file(
                Path(job.chapters_source), temp_dir,
            )

        # Build video metadata for container-level color/HDR flags
        video_meta: dict[str, Any] = {}
        vp = job.video_params
        if vp.color_range:
            video_meta["color_range"] = vp.color_range
        if vp.color_primaries:
            video_meta["color_primaries"] = vp.color_primaries
        if vp.color_transfer:
            video_meta["color_transfer"] = vp.color_transfer
        if vp.hdr and vp.hdr.content_light:
            # content_light format: "MaxCLL=X,MaxFALL=Y"
            for part in vp.hdr.content_light.split(","):
                if part.startswith("MaxCLL="):
                    video_meta["hdr_max_cll"] = part.split("=", 1)[1]
                elif part.startswith("MaxFALL="):
                    video_meta["hdr_max_fall"] = part.split("=", 1)[1]

        _, mux_on_progress = self._make_progress_callback(total_s=None)
        rc = self._muxer.mux(
            video_path=video_output,
            audio_files=audio_files,
            subtitle_files=subtitle_files,
            attachments=attachments,
            chapters_source=chapters_source,
            output_path=muxed_path,
            furnace_version=FURNACE_VERSION,
            video_meta=video_meta or None,
            on_progress=mux_on_progress,
        )
        if rc != 0:
            raise RuntimeError(f"Muxing failed with return code {rc}")
        if self._progress is not None and muxed_path.exists():
            self._progress.update_output_size(muxed_path.stat().st_size)

        # Step 6: Set ENCODER tag
        if self._shutdown_event.is_set():
            return

        logger.info("Setting ENCODER tag")
        if self._progress is not None:
            self._progress.update_status("Setting metadata...")
            self._progress.add_tool_line("[furnace] Setting ENCODER tag")
        tag_value = f"Furnace v{FURNACE_VERSION}"
        rc = self._tagger.set_encoder_tag(muxed_path, tag_value, encoder_settings)
        if rc != 0:
            logger.warning("mkvpropedit returned %d for %s", rc, muxed_path)

        # Step 7: mkclean
        if self._shutdown_event.is_set():
            return

        cleaned_path = temp_dir / "cleaned.mkv"
        logger.info("Optimizing MKV index (mkclean)")
        if self._progress is not None:
            self._progress.update_status("Optimizing MKV index...")
            self._progress.add_tool_line("[furnace] Optimizing MKV index (mkclean)")
        _, clean_on_progress = self._make_progress_callback(total_s=None)
        rc = self._cleaner.clean(muxed_path, cleaned_path, on_progress=clean_on_progress)
        if rc != 0:
            logger.warning("mkclean returned %d, using muxed output", rc)
            cleaned_path = muxed_path
        if self._progress is not None and cleaned_path.exists():
            self._progress.update_output_size(cleaned_path.stat().st_size)

        # Move cleaned output to final destination
        shutil.move(str(cleaned_path), str(output_path))
        logger.debug("Job output written to %s", output_path)

    def _process_audio_track(
        self, instr: AudioInstruction, temp_dir: Path, job: Job
    ) -> Path:
        """Returns path to processed audio file.
        COPY: extract_track from container
        DENORM: extract_track -> denormalize (with delay)
        DECODE_ENCODE: extract_track -> decode_lossless (with delay) -> encode_aac
        FFMPEG_ENCODE: ffmpeg_to_wav -> encode_aac
        """
        source_path = Path(instr.source_file)
        track_idx = instr.stream_index

        ext = _AUDIO_CODEC_EXT.get(instr.codec_name, ".audio")

        if instr.action == AudioAction.COPY:
            logger.info("Extracting audio stream %d (copy)", track_idx)
            if self._progress is not None:
                self._progress.add_tool_line(f"[furnace] Extracting audio stream {track_idx} (copy)")
            out_path = temp_dir / f"audio_{track_idx}{ext}"
            _, on_progress = self._make_progress_callback(total_s=job.duration_s or None)
            rc = self._audio_extractor.extract_track(
                source_path, track_idx, out_path, instr.codec_name,
                on_progress=on_progress,
            )
            if rc != 0:
                raise RuntimeError(
                    f"Audio extract (COPY) failed with rc={rc} for stream {track_idx}"
                )
            return out_path

        if instr.action == AudioAction.DENORM:
            logger.info("Extracting audio stream %d", track_idx)
            if self._progress is not None:
                self._progress.add_tool_line(f"[furnace] Extracting audio stream {track_idx}")
            extracted = temp_dir / f"audio_{track_idx}_raw{ext}"
            _, on_progress = self._make_progress_callback(total_s=job.duration_s or None)
            rc = self._audio_extractor.extract_track(
                source_path, track_idx, extracted, instr.codec_name,
                on_progress=on_progress,
            )
            if rc != 0:
                raise RuntimeError(
                    f"Audio extract (DENORM) failed with rc={rc} for stream {track_idx}"
                )
            logger.info("Denormalizing with eac3to")
            if self._progress is not None:
                self._progress.add_tool_line(f"[furnace] Denormalizing audio stream {track_idx} with eac3to")
            denormed = temp_dir / f"audio_{track_idx}_denorm{ext}"
            _, on_progress = self._make_progress_callback(total_s=None)
            rc = self._audio_decoder.denormalize(
                extracted, denormed, instr.delay_ms, on_progress=on_progress,
            )
            if rc != 0:
                raise RuntimeError(
                    f"Audio denormalize failed with rc={rc} for stream {track_idx}"
                )
            return denormed

        if instr.action == AudioAction.DECODE_ENCODE:
            logger.info("Extracting audio stream %d", track_idx)
            if self._progress is not None:
                self._progress.add_tool_line(f"[furnace] Extracting audio stream {track_idx}")
            extracted = temp_dir / f"audio_{track_idx}_raw{ext}"
            _, on_progress = self._make_progress_callback(total_s=job.duration_s or None)
            rc = self._audio_extractor.extract_track(
                source_path, track_idx, extracted, instr.codec_name,
                on_progress=on_progress,
            )
            if rc != 0:
                raise RuntimeError(
                    f"Audio extract (DECODE_ENCODE) failed with rc={rc} for stream {track_idx}"
                )
            logger.info("Decoding lossless with eac3to")
            if self._progress is not None:
                self._progress.add_tool_line(f"[furnace] Decoding lossless audio stream {track_idx} with eac3to")
            wav_path = temp_dir / f"audio_{track_idx}_decoded.wav"
            _, on_progress = self._make_progress_callback(total_s=None)
            rc = self._audio_decoder.decode_lossless(
                extracted, wav_path, instr.delay_ms, on_progress=on_progress,
            )
            if rc != 0:
                raise RuntimeError(
                    f"Audio decode_lossless failed with rc={rc} for stream {track_idx}"
                )
            logger.info("Encoding AAC with qaac64")
            if self._progress is not None:
                self._progress.add_tool_line(f"[furnace] Encoding AAC for stream {track_idx} with qaac64")
            m4a_path = temp_dir / f"audio_{track_idx}.m4a"
            _, on_progress = self._make_progress_callback(total_s=None)
            rc = self._aac_encoder.encode_aac(wav_path, m4a_path, on_progress=on_progress)
            if rc != 0:
                raise RuntimeError(
                    f"AAC encode failed with rc={rc} for stream {track_idx}"
                )
            return m4a_path

        if instr.action == AudioAction.FFMPEG_ENCODE:
            logger.info("Decoding with ffmpeg to WAV")
            if self._progress is not None:
                self._progress.add_tool_line(f"[furnace] Decoding audio stream {track_idx} with ffmpeg to WAV")
            wav_path = temp_dir / f"audio_{track_idx}_ffmpeg.wav"
            _, on_progress = self._make_progress_callback(total_s=job.duration_s or None)
            rc = self._audio_extractor.ffmpeg_to_wav(
                source_path, track_idx, wav_path, on_progress=on_progress,
            )
            if rc != 0:
                raise RuntimeError(
                    f"ffmpeg_to_wav failed with rc={rc} for stream {track_idx}"
                )
            logger.info("  Encoding AAC with qaac64")
            if self._progress is not None:
                self._progress.add_tool_line(f"[furnace] Encoding AAC for stream {track_idx} with qaac64")
            m4a_path = temp_dir / f"audio_{track_idx}.m4a"
            _, on_progress = self._make_progress_callback(total_s=None)
            rc = self._aac_encoder.encode_aac(wav_path, m4a_path, on_progress=on_progress)
            if rc != 0:
                raise RuntimeError(
                    f"AAC encode (FFMPEG_ENCODE) failed with rc={rc} for stream {track_idx}"
                )
            return m4a_path

        raise ValueError(f"Unknown AudioAction: {instr.action}")

    def _process_subtitle_track(
        self, instr: SubtitleInstruction, temp_dir: Path, job: Job
    ) -> Path:
        """COPY: extract from container (ffmpeg).
        COPY_RECODE: extract + charset detection + recode to UTF-8.
        """
        source_path = Path(instr.source_file)
        track_idx = instr.stream_index

        ext = _SUBTITLE_CODEC_EXT.get(instr.codec_name, ".sub")

        if instr.action == SubtitleAction.COPY:
            # For external satellite files, just return as-is
            if source_path.suffix.lower() in {".srt", ".ass", ".ssa", ".sup", ".sub"}:
                return source_path
            # Extract from container
            out_path = temp_dir / f"sub_{track_idx}{ext}"
            _, on_progress = self._make_progress_callback(total_s=job.duration_s or None)
            rc = self._audio_extractor.extract_track(
                source_path, track_idx, out_path, instr.codec_name,
                on_progress=on_progress,
            )
            if rc != 0:
                raise RuntimeError(
                    f"Subtitle extract (COPY) failed with rc={rc} for stream {track_idx}"
                )
            return out_path

        if instr.action == SubtitleAction.COPY_RECODE:
            # For external satellite files, just recode them
            if source_path.suffix.lower() in {".srt", ".ass", ".ssa"}:
                extracted = source_path
            else:
                # Extract from container first
                extracted = temp_dir / f"sub_{track_idx}_raw{ext}"
                _, on_progress = self._make_progress_callback(total_s=job.duration_s or None)
                rc = self._audio_extractor.extract_track(
                    source_path, track_idx, extracted, instr.codec_name,
                    on_progress=on_progress,
                )
                if rc != 0:
                    raise RuntimeError(
                        f"Subtitle extract (COPY_RECODE) failed with rc={rc} for stream {track_idx}"
                    )

            out_path = temp_dir / f"sub_{track_idx}_utf8{ext}"

            # Recode to UTF-8
            source_encoding = instr.source_encoding or "utf-8"
            if source_encoding.lower().replace("-", "") == "utf8":
                # Already UTF-8, just copy
                shutil.copy2(str(extracted), str(out_path))
            else:
                try:
                    content = extracted.read_bytes()
                    text = content.decode(source_encoding)
                    out_path.write_text(text, encoding="utf-8")
                except Exception as exc:
                    logger.warning(
                        "Recode failed for stream %d (%s->utf-8): %s; copying as-is",
                        track_idx, source_encoding, exc,
                    )
                    shutil.copy2(str(extracted), str(out_path))

            return out_path

        raise ValueError(f"Unknown SubtitleAction: {instr.action}")

    def _extract_chapters_file(self, source: Path, job_dir: Path) -> Path | None:
        """Extract chapters from source MKV via ffprobe, fix mojibake, write OGM file."""
        from ..core.chapters import chapters_have_mojibake, write_ogm_chapters

        try:
            probe = self._prober.probe(source)
        except RuntimeError:
            logger.warning("Failed to probe %s for chapters", source)
            return None
        chapters: list[dict[str, Any]] = probe.get("chapters", [])
        if not chapters:
            return None
        if chapters_have_mojibake(chapters):
            logger.info("Mojibake detected in chapter titles, fixing encoding")
        ogm_path = job_dir / "chapters.txt"
        write_ogm_chapters(chapters, ogm_path)
        return ogm_path

    def graceful_shutdown(self) -> None:
        """Called on ESC. Kill ffmpeg process tree via psutil."""
        logger.debug("Graceful shutdown requested")
        self._shutdown_event.set()
        try:
            import os as _os

            import psutil
            parent = psutil.Process(_os.getpid())
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
        except ImportError:
            logger.warning("psutil not available; cannot kill child processes")
        except Exception as exc:
            logger.error("Error during graceful shutdown: %s", exc)
