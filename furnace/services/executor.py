from __future__ import annotations

import logging
import shutil
import tempfile
import threading
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
    Encoder,
    Muxer,
    Prober,
    Tagger,
)
from ..plan import update_job_status

logger = logging.getLogger(__name__)

FURNACE_VERSION = "0.1.0"

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
    "dvd_subtitle":       ".sub",
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
    ) -> None:
        self._encoder = encoder
        self._audio_extractor = audio_extractor
        self._audio_decoder = audio_decoder
        self._aac_encoder = aac_encoder
        self._muxer = muxer
        self._tagger = tagger
        self._cleaner = cleaner
        self._prober = prober
        self._shutdown_event = threading.Event()

    def run(
        self,
        plan: Plan,
        plan_path: Path,
        on_job_progress: Callable[..., None] | None = None,
    ) -> None:
        """Execute all pending/error jobs sequentially.
        Update JSON after each via update_job_status.
        Check _shutdown_event between jobs.
        """
        pending_jobs = [
            job for job in plan.jobs
            if job.status in (JobStatus.PENDING, JobStatus.ERROR)
        ]

        logger.info(
            "Starting execution: %d jobs to process (total: %d)",
            len(pending_jobs),
            len(plan.jobs),
        )

        for job in pending_jobs:
            if self._shutdown_event.is_set():
                logger.info("Shutdown requested, stopping before job %s", job.id)
                break

            logger.info("Starting job %s -> %s", job.id, job.output_file)

            try:
                self._execute_job(job, on_job_progress)
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
                )
                logger.info("Job %s completed successfully", job.id)
            except Exception as exc:
                error_msg = str(exc)
                logger.error("Job %s failed: %s", job.id, error_msg)
                update_job_status(
                    plan_path,
                    job.id,
                    JobStatus.ERROR,
                    error=error_msg,
                )

    def _execute_job(self, job: Job, on_progress: Callable[..., None] | None) -> None:
        """Full pipeline for one job:
        1. Encode video
        2. Process audio tracks
        3. Process subtitle tracks
        4. Mux
        5. Set ENCODER tag (mkvpropedit)
        6. mkclean
        7. Post-encoding bloat check
        8. Optional VMAF (placeholder)
        9. Cleanup temp (try/finally)
        """
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        temp_dir = Path(tempfile.mkdtemp(prefix="furnace_"))
        try:
            self._run_pipeline(job, output_path, temp_dir, on_progress)
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
        on_progress: Callable[..., None] | None,
    ) -> None:
        """Inner pipeline logic (separated so finally in _execute_job always runs)."""
        main_source = Path(job.source_files[0])

        # Step 1: Encode video
        if self._shutdown_event.is_set():
            return

        video_output = temp_dir / "video.mkv"
        logger.info("Encoding video: %s -> %s", main_source, video_output)
        rc = self._encoder.encode(
            input_path=main_source,
            output_path=video_output,
            video_params=job.video_params,
            source_size=job.source_size,
            on_progress=on_progress,
        )
        if rc != 0:
            raise RuntimeError(f"Video encoding failed with return code {rc}")

        # Step 2: Process audio tracks
        if self._shutdown_event.is_set():
            return

        audio_files: list[tuple[Path, dict[str, Any]]] = []
        for i, audio_instr in enumerate(job.audio):
            if self._shutdown_event.is_set():
                return
            logger.info(
                "Processing audio track %d/%d (lang=%s, action=%s)",
                i + 1, len(job.audio), audio_instr.language, audio_instr.action.value,
            )
            audio_path = self._process_audio_track(audio_instr, temp_dir)
            audio_meta = {
                "language": audio_instr.language,
                "default": audio_instr.is_default,
                "delay_ms": audio_instr.delay_ms if audio_instr.action == AudioAction.COPY else 0,
            }
            audio_files.append((audio_path, audio_meta))

        # Step 3: Process subtitle tracks
        if self._shutdown_event.is_set():
            return

        subtitle_files: list[tuple[Path, dict[str, Any]]] = []
        for i, sub_instr in enumerate(job.subtitles):
            if self._shutdown_event.is_set():
                return
            logger.info(
                "Processing subtitle track %d/%d (lang=%s, action=%s)",
                i + 1, len(job.subtitles), sub_instr.language, sub_instr.action.value,
            )
            sub_path = self._process_subtitle_track(sub_instr, temp_dir)
            sub_meta = {
                "language": sub_instr.language,
                "default": sub_instr.is_default,
                "forced": sub_instr.is_forced,
                "encoding": "UTF-8",
            }
            subtitle_files.append((sub_path, sub_meta))

        # Step 4: Mux
        if self._shutdown_event.is_set():
            return

        muxed_path = temp_dir / "muxed.mkv"
        logger.info("Muxing to %s", muxed_path)

        # Build attachments list: (path, filename, mime_type)
        attachments: list[tuple[Path, str, str]] = []
        for att_dict in job.attachments:
            att_path = Path(att_dict["source_file"])
            filename = att_dict["filename"]
            mime_type = att_dict["mime_type"]
            attachments.append((att_path, filename, mime_type))

        chapters_source: Path | None = None
        if job.copy_chapters and job.chapters_source:
            chapters_source = Path(job.chapters_source)

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

        rc = self._muxer.mux(
            video_path=video_output,
            audio_files=audio_files,
            subtitle_files=subtitle_files,
            attachments=attachments,
            chapters_source=chapters_source,
            output_path=muxed_path,
            furnace_version=FURNACE_VERSION,
            video_meta=video_meta or None,
        )
        if rc != 0:
            raise RuntimeError(f"Muxing failed with return code {rc}")

        # Step 5: Set ENCODER tag
        if self._shutdown_event.is_set():
            return

        logger.info("Setting ENCODER tag on %s", muxed_path)
        tag_value = f"Furnace/{FURNACE_VERSION}"
        rc = self._tagger.set_encoder_tag(muxed_path, tag_value)
        if rc != 0:
            logger.warning("mkvpropedit returned %d for %s", rc, muxed_path)

        # Step 6: mkclean
        if self._shutdown_event.is_set():
            return

        cleaned_path = temp_dir / "cleaned.mkv"
        logger.info("Running mkclean: %s -> %s", muxed_path, cleaned_path)
        rc = self._cleaner.clean(muxed_path, cleaned_path)
        if rc != 0:
            logger.warning("mkclean returned %d, using muxed output", rc)
            cleaned_path = muxed_path

        # Step 7: Post-encoding bloat check
        if self._check_bloat(job.source_size, cleaned_path):
            raise RuntimeError(
                f"Post-encoding bloat check failed: output exceeds source size "
                f"(source={job.source_size}, output={cleaned_path.stat().st_size if cleaned_path.exists() else 'missing'})"
            )

        # Step 8: Optional VMAF (placeholder for Phase 6)
        # VMAF scoring not implemented in Phase 2

        # Move cleaned output to final destination
        shutil.move(str(cleaned_path), str(output_path))
        logger.info("Job output written to %s", output_path)

    def _process_audio_track(self, instr: AudioInstruction, temp_dir: Path) -> Path:
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
            out_path = temp_dir / f"audio_{track_idx}{ext}"
            rc = self._audio_extractor.extract_track(
                source_path, track_idx, out_path, instr.codec_name
            )
            if rc != 0:
                raise RuntimeError(
                    f"Audio extract (COPY) failed with rc={rc} for stream {track_idx}"
                )
            return out_path

        if instr.action == AudioAction.DENORM:
            extracted = temp_dir / f"audio_{track_idx}_raw{ext}"
            rc = self._audio_extractor.extract_track(
                source_path, track_idx, extracted, instr.codec_name
            )
            if rc != 0:
                raise RuntimeError(
                    f"Audio extract (DENORM) failed with rc={rc} for stream {track_idx}"
                )
            denormed = temp_dir / f"audio_{track_idx}_denorm{ext}"
            rc = self._audio_decoder.denormalize(extracted, denormed, instr.delay_ms)
            if rc != 0:
                raise RuntimeError(
                    f"Audio denormalize failed with rc={rc} for stream {track_idx}"
                )
            return denormed

        if instr.action == AudioAction.DECODE_ENCODE:
            extracted = temp_dir / f"audio_{track_idx}_raw{ext}"
            rc = self._audio_extractor.extract_track(
                source_path, track_idx, extracted, instr.codec_name
            )
            if rc != 0:
                raise RuntimeError(
                    f"Audio extract (DECODE_ENCODE) failed with rc={rc} for stream {track_idx}"
                )
            wav_path = temp_dir / f"audio_{track_idx}_decoded.wav"
            rc = self._audio_decoder.decode_lossless(extracted, wav_path, instr.delay_ms)
            if rc != 0:
                raise RuntimeError(
                    f"Audio decode_lossless failed with rc={rc} for stream {track_idx}"
                )
            m4a_path = temp_dir / f"audio_{track_idx}.m4a"
            rc = self._aac_encoder.encode_aac(wav_path, m4a_path)
            if rc != 0:
                raise RuntimeError(
                    f"AAC encode failed with rc={rc} for stream {track_idx}"
                )
            return m4a_path

        if instr.action == AudioAction.FFMPEG_ENCODE:
            wav_path = temp_dir / f"audio_{track_idx}_ffmpeg.wav"
            rc = self._audio_extractor.ffmpeg_to_wav(source_path, track_idx, wav_path)
            if rc != 0:
                raise RuntimeError(
                    f"ffmpeg_to_wav failed with rc={rc} for stream {track_idx}"
                )
            m4a_path = temp_dir / f"audio_{track_idx}.m4a"
            rc = self._aac_encoder.encode_aac(wav_path, m4a_path)
            if rc != 0:
                raise RuntimeError(
                    f"AAC encode (FFMPEG_ENCODE) failed with rc={rc} for stream {track_idx}"
                )
            return m4a_path

        raise ValueError(f"Unknown AudioAction: {instr.action}")

    def _process_subtitle_track(self, instr: SubtitleInstruction, temp_dir: Path) -> Path:
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
            rc = self._audio_extractor.extract_track(
                source_path, track_idx, out_path, instr.codec_name
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
                rc = self._audio_extractor.extract_track(
                    source_path, track_idx, extracted, instr.codec_name
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

    def _check_bloat(self, source_size: int, output_path: Path) -> bool:
        """True if output > source -> delete output."""
        if source_size <= 0:
            return False
        try:
            output_size = output_path.stat().st_size
        except FileNotFoundError:
            return False
        if output_size > source_size:
            logger.warning(
                "Bloat check failed: output %d bytes > source %d bytes; deleting %s",
                output_size, source_size, output_path,
            )
            try:
                output_path.unlink()
            except OSError as exc:
                logger.error("Failed to delete bloated output %s: %s", output_path, exc)
            return True
        return False

    def graceful_shutdown(self) -> None:
        """Called on ESC. Kill ffmpeg process tree via psutil."""
        logger.info("Graceful shutdown requested")
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
