from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil

from furnace import VERSION as FURNACE_VERSION
from furnace.core.chapters import chapters_have_mojibake, write_ogm_chapters
from furnace.core.models import (
    STEREO_CHANNELS,
    AudioAction,
    AudioInstruction,
    DownmixMode,
    Job,
    JobStatus,
    Plan,
    SubtitleAction,
    SubtitleInstruction,
)
from furnace.core.ports import (
    AacEncoder,
    AudioDecoder,
    AudioExtractor,
    Cleaner,
    DoviProcessor,
    Encoder,
    Muxer,
    Prober,
    Tagger,
    VideoCopier,
)
from furnace.core.progress import ProgressSample, ProgressTracker
from furnace.core.target_quality import fixed_grain_knob, grain_uses_svt
from furnace.plan import update_job_status
from furnace.services.target_quality import TargetQualityService

logger = logging.getLogger(__name__)

MAX_STDERR_LINES = 6

_AUDIO_CODEC_EXT: dict[str, str] = {
    "aac": ".m4a",
    "ac3": ".ac3",
    "eac3": ".eac3",
    "dts": ".dts",
    "truehd": ".thd",
    "flac": ".flac",
    "pcm_s16le": ".wav",
    "pcm_s24le": ".wav",
    "pcm_s16be": ".wav",
    "mp2": ".mp2",
    "mp3": ".mp3",
    "vorbis": ".ogg",
    "opus": ".opus",
    "wmav2": ".wma",
    "wmapro": ".wma",
    "amr_nb": ".amr",
}

_SUBTITLE_CODEC_EXT: dict[str, str] = {
    "subrip": ".srt",
    "ass": ".ass",
    "hdmv_pgs_subtitle": ".sup",
    "dvd_subtitle": ".mkv",
}


_EAC3TO_SUPPORTED_SRC: frozenset[str] = frozenset(
    {
        "ac3",
        "eac3",
        "dts",
        "truehd",
        "flac",
        "pcm_s16le",
        "pcm_s24le",
        "pcm_s16be",
        "mp2",
        "mp3",
    }
)


def _codec_supported_by_eac3to(codec_name: str) -> bool:
    return codec_name.lower() in _EAC3TO_SUPPORTED_SRC


_FFMPEG_DRC_CODECS: frozenset[str] = frozenset({"ac3", "eac3"})


def _video_intermediate_name(*, passthrough: bool) -> str:
    return "video.mkv" if passthrough else "video.obu"


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
        video_copier: VideoCopier | None = None,
        grain_encoder: Encoder | None = None,
        target_quality: TargetQualityService | None = None,
        progress: Any | None = None,
        log_dir: Path | None = None,
    ) -> None:
        self._encoder = encoder
        self._grain_encoder = grain_encoder
        self._target_quality = target_quality
        self._audio_extractor = audio_extractor
        self._audio_decoder = audio_decoder
        self._aac_encoder = aac_encoder
        self._muxer = muxer
        self._tagger = tagger
        self._cleaner = cleaner
        self._prober = prober
        self._dovi_processor = dovi_processor
        self._video_copier = video_copier
        self._progress = progress
        self._log_dir = log_dir
        self._shutdown_event = threading.Event()
        self._adapters: list[Any] = [encoder, audio_extractor, audio_decoder, aac_encoder, muxer, tagger, cleaner]
        if dovi_processor is not None:
            self._adapters.append(dovi_processor)
        if video_copier is not None:
            self._adapters.append(video_copier)
        if grain_encoder is not None:
            self._adapters.append(grain_encoder)

    def _make_progress_callback(
        self,
        total_s: float | None = None,
    ) -> tuple[ProgressTracker, Callable[[ProgressSample], None]]:
        tracker = ProgressTracker(total_s=total_s)

        def _on_progress(sample: ProgressSample) -> None:
            tracker.add(sample, time.monotonic())
            if self._progress is not None:
                self._progress.update_progress(tracker.snapshot())

        return tracker, _on_progress

    def _set_adapters_log_dir(self, job_name: str) -> None:
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
        pending_jobs = [job for job in plan.jobs if job.status in (JobStatus.PENDING, JobStatus.ERROR)]

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
                    chosen_cq=job.chosen_cq,
                )
                logger.debug("Job %s completed successfully", job.id)
                if self._progress is not None:
                    job.output_size = output_size
                    self._progress.finish_job(job)
            except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                error_msg = str(exc)
                logger.exception("Job %s failed", job.id)
                update_job_status(
                    plan_path,
                    job.id,
                    JobStatus.ERROR,
                    error=error_msg,
                    chosen_cq=job.chosen_cq,
                )

    def _execute_job(self, job: Job) -> None:
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        temp_dir = Path(tempfile.mkdtemp(prefix="furnace_"))
        try:
            self._run_pipeline(job, output_path, temp_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _run_pipeline(
        self,
        job: Job,
        output_path: Path,
        temp_dir: Path,
    ) -> None:
        main_source = Path(job.source_files[0])
        self._cumulative_audio_size = 0

        if self._shutdown_event.is_set():
            return

        audio_files: list[tuple[Path, dict[str, Any]]] = []
        for i, audio_instr in enumerate(job.audio):
            if self._shutdown_event.is_set():
                return
            status_msg = f"Processing audio {i + 1}/{len(job.audio)} ({audio_instr.codec_name} {audio_instr.language})"
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
            if self._progress is not None and audio_path.exists():
                self._cumulative_audio_size += audio_path.stat().st_size
                self._progress.update_output_size(self._cumulative_audio_size)

        if self._shutdown_event.is_set():
            return

        subtitle_files: list[tuple[Path, dict[str, Any]]] = []
        for i, sub_instr in enumerate(job.subtitles):
            if self._shutdown_event.is_set():
                return
            status_msg = (
                f"Processing subtitle {i + 1}/{len(job.subtitles)} ({sub_instr.codec_name} {sub_instr.language})"
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

        passthrough = job.video_params.passthrough

        rpu_path: Path | None = None
        if job.video_params.dv_mode is not None and not passthrough:
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

        if self._shutdown_event.is_set():
            return

        video_output = temp_dir / _video_intermediate_name(passthrough=passthrough)

        _, base_video_on_progress = self._make_progress_callback(
            total_s=job.duration_s or None,
        )

        def video_on_progress(sample: ProgressSample) -> None:
            base_video_on_progress(sample)
            try:
                video_size = video_output.stat().st_size if video_output.exists() else 0
            except OSError:
                video_size = 0
            if self._progress is not None:
                self._progress.update_output_size(self._cumulative_audio_size + video_size)

        if passthrough:
            if self._video_copier is None:
                msg = "passthrough video requires a video_copier but it is not configured"
                raise RuntimeError(msg)
            logger.info("Copying video stream (passthrough): %s", main_source.name)
            if self._progress is not None:
                self._progress.update_status("Copying video (passthrough)")
                self._progress.add_tool_line(
                    f"[furnace] Copying video stream (passthrough): {main_source.name}",
                )
            rc = self._video_copier.copy_video(
                main_source,
                video_output,
                on_progress=video_on_progress,
            )
            if rc != 0:
                raise RuntimeError(f"Video passthrough copy failed with return code {rc}")
            encoder_settings = "video stream copied (passthrough)"
        else:
            enc = (
                self._grain_encoder
                if (grain_uses_svt(job.video_params) and self._grain_encoder is not None)
                else self._encoder
            )

            if self._progress is not None:
                self._progress.update_status("Searching quality...")
            fixed = fixed_grain_knob(job.video_params)
            if fixed is not None:
                cq_override: int | None = fixed
                job.chosen_cq = fixed
            else:
                cq_override = self._maybe_search_target_quality(job, main_source, temp_dir)
            if self._progress is not None and cq_override is not None:
                self._progress.set_chosen_quality(cq_override)

            logger.info("Encoding video: %s", main_source.name)
            if self._progress is not None:
                self._progress.update_status("Encoding video")
                self._progress.add_tool_line(f"[furnace] Encoding video: {main_source.name}")

            rc_result = enc.encode(
                input_path=main_source,
                output_path=video_output,
                video_params=job.video_params,
                on_progress=video_on_progress,
                rpu_path=rpu_path,
                cq_override=cq_override,
            )
            if rc_result.return_code != 0:
                raise RuntimeError(f"Video encoding failed with return code {rc_result.return_code}")

            encoder_settings = rc_result.encoder_settings

        if self._shutdown_event.is_set():
            return

        muxed_path = temp_dir / "muxed.mkv"
        logger.info("Muxing tracks")
        if self._progress is not None:
            self._progress.update_status("Muxing...")
            self._progress.add_tool_line("[furnace] Muxing tracks")

        attachments: list[tuple[Path, str, str]] = []
        for att_dict in job.attachments:
            att_path = Path(att_dict["source_file"])
            filename = att_dict["filename"]
            mime_type = att_dict["mime_type"]
            attachments.append((att_path, filename, mime_type))

        chapters_source: Path | None = None
        if job.copy_chapters and job.chapters_source:
            chapters_source = self._extract_chapters_file(
                Path(job.chapters_source),
                temp_dir,
            )

        video_meta: dict[str, Any] = {}
        vp = job.video_params
        if vp.color_range:
            video_meta["color_range"] = vp.color_range
        if vp.color_primaries:
            video_meta["color_primaries"] = vp.color_primaries
        if vp.color_transfer:
            video_meta["color_transfer"] = vp.color_transfer
        if vp.color_matrix:
            video_meta["color_matrix"] = vp.color_matrix
        if vp.hdr and vp.hdr.content_light:
            for part in vp.hdr.content_light.split(","):
                if part.startswith("MaxCLL="):
                    video_meta["hdr_max_cll"] = part.split("=", 1)[1]
                elif part.startswith("MaxFALL="):
                    video_meta["hdr_max_fall"] = part.split("=", 1)[1]

        if not passthrough:
            video_meta["fps_num"] = vp.fps_num
            video_meta["fps_den"] = vp.fps_den

        _, mux_on_progress = self._make_progress_callback(total_s=None)
        rc = self._muxer.mux(
            video_path=video_output,
            audio_files=audio_files,
            subtitle_files=subtitle_files,
            attachments=attachments,
            chapters_source=chapters_source,
            output_path=muxed_path,
            video_meta=video_meta or None,
            on_progress=mux_on_progress,
        )
        if rc != 0:
            raise RuntimeError(f"Muxing failed with return code {rc}")
        if self._progress is not None and muxed_path.exists():
            self._progress.update_output_size(muxed_path.stat().st_size)

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

        shutil.move(str(cleaned_path), str(output_path))
        logger.debug("Job output written to %s", output_path)

    def _maybe_search_target_quality(self, job: Job, source: Path, temp_dir: Path) -> int | None:
        if self._target_quality is None or not self._target_quality.can_search(job.video_params):
            return None

        knob = "CRF" if grain_uses_svt(job.video_params) else "QVBR"

        if job.chosen_cq is not None:
            logger.info("Reusing cached target-quality %s %d for %s", knob, job.chosen_cq, source.name)
            return job.chosen_cq

        if job.duration_s <= 0:
            logger.warning(
                "Target-quality skipped for %s: source duration unknown; encoding at the encoder's default knob",
                source.name,
            )
            if self._progress is not None:
                self._progress.add_tool_line(
                    "[furnace] WARNING: target-quality skipped (unknown duration); using default knob"
                )
            return None

        logger.info("Searching target quality (%s) for %s", knob, source.name)

        if self._progress is not None:
            self._progress.mute_tool_output()
        try:
            result = self._target_quality.search(
                source,
                job.video_params,
                job.duration_s,
                temp_dir,
                on_event=self._search_narration,
            )
        finally:
            if self._progress is not None:
                self._progress.unmute_tool_output()
        job.chosen_cq = result.knob
        if result.hit:
            logger.info("Target quality: %s %d (score %.3f)", knob, result.knob, result.score)
            if self._progress is not None:
                self._progress.add_tool_line(
                    f"[furnace] Target-quality {knob} {result.knob} (score {result.score:.3f})"
                )
        else:
            logger.warning(
                "Target-quality band not hit for %s; using closest %s %d (score %.3f)",
                source.name,
                knob,
                result.knob,
                result.score,
            )
            if self._progress is not None:
                self._progress.add_tool_line(
                    f"[furnace] WARNING: target-quality band not hit; using {knob} {result.knob} "
                    f"(score {result.score:.3f})"
                )
        return result.knob

    def _search_narration(self, message: str) -> None:
        if self._progress is not None:
            self._progress.add_tool_line(f"[furnace] {message}")

    def _process_audio_track(self, instr: AudioInstruction, temp_dir: Path, job: Job) -> Path:
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
                source_path,
                track_idx,
                out_path,
                on_progress=on_progress,
            )
            if rc != 0:
                raise RuntimeError(f"Audio extract (COPY) failed with rc={rc} for stream {track_idx}")
            return out_path

        if instr.action == AudioAction.DENORM:
            logger.info("Extracting audio stream %d", track_idx)
            if self._progress is not None:
                self._progress.add_tool_line(f"[furnace] Extracting audio stream {track_idx}")
            extracted = temp_dir / f"audio_{track_idx}_raw{ext}"
            _, on_progress = self._make_progress_callback(total_s=job.duration_s or None)
            rc = self._audio_extractor.extract_track(
                source_path,
                track_idx,
                extracted,
                on_progress=on_progress,
            )
            if rc != 0:
                raise RuntimeError(f"Audio extract (DENORM) failed with rc={rc} for stream {track_idx}")
            logger.info("Denormalizing with eac3to")
            if self._progress is not None:
                self._progress.add_tool_line(f"[furnace] Denormalizing audio stream {track_idx} with eac3to")
            denormed = temp_dir / f"audio_{track_idx}_denorm{ext}"
            _, on_progress = self._make_progress_callback(total_s=None)
            rc = self._audio_decoder.denormalize(
                extracted,
                denormed,
                instr.delay_ms,
                on_progress=on_progress,
            )
            if rc != 0:
                raise RuntimeError(f"Audio denormalize failed with rc={rc} for stream {track_idx}")
            return denormed

        if instr.action == AudioAction.DECODE_ENCODE:
            if instr.downmix == DownmixMode.MONO:
                if instr.channels is None:
                    raise RuntimeError(
                        f"MONO downmix without channel count for stream {track_idx}",
                    )

                is_stereo = instr.channels == STEREO_CHANNELS

                if is_stereo and instr.codec_name.lower() not in _FFMPEG_DRC_CODECS:
                    if self._progress is not None:
                        self._progress.add_tool_line(
                            f"[furnace] Averaging audio stream {track_idx} to mono with ffmpeg",
                        )
                    mono_wav = temp_dir / f"audio_{track_idx}_mono.wav"
                    _, on_progress = self._make_progress_callback(
                        total_s=job.duration_s or None,
                    )
                    rc = self._audio_extractor.stereo_to_mono_wav(
                        input_path=source_path,
                        stream_index=track_idx,
                        output_wav=mono_wav,
                        delay_ms=instr.delay_ms,
                        on_progress=on_progress,
                    )
                    if rc != 0:
                        raise RuntimeError(f"stereo_to_mono_wav failed: rc={rc}")

                    if self._progress is not None:
                        self._progress.add_tool_line(
                            f"[furnace] Encoding AAC for stream {track_idx} with qaac64",
                        )
                    m4a_path = temp_dir / f"audio_{track_idx}.m4a"
                    _, on_progress = self._make_progress_callback(total_s=None)
                    rc = self._aac_encoder.encode_aac(
                        mono_wav,
                        m4a_path,
                        on_progress=on_progress,
                    )
                    if rc != 0:
                        raise RuntimeError(f"encode_aac failed: rc={rc}")
                    return m4a_path

                if _codec_supported_by_eac3to(instr.codec_name):
                    if self._progress is not None:
                        self._progress.add_tool_line(
                            f"[furnace] Extracting audio stream {track_idx} for MONO downmix",
                        )
                    extracted = temp_dir / f"audio_{track_idx}_raw{ext}"
                    _, on_progress = self._make_progress_callback(
                        total_s=job.duration_s or None,
                    )
                    rc = self._audio_extractor.extract_track(
                        source_path,
                        track_idx,
                        extracted,
                        on_progress=on_progress,
                    )
                    if rc != 0:
                        raise RuntimeError(
                            f"Audio extract (MONO) failed with rc={rc} for stream {track_idx}",
                        )
                else:
                    if self._progress is not None:
                        self._progress.add_tool_line(
                            f"[furnace] Pre-decoding audio stream {track_idx} with ffmpeg for MONO downmix "
                            f"(source codec {instr.codec_name} not readable by eac3to)",
                        )
                    extracted = temp_dir / f"audio_{track_idx}_pre.wav"
                    _, on_progress = self._make_progress_callback(
                        total_s=job.duration_s or None,
                    )
                    rc = self._audio_extractor.ffmpeg_to_wav(
                        source_path,
                        track_idx,
                        extracted,
                        on_progress=on_progress,
                    )
                    if rc != 0:
                        raise RuntimeError(
                            f"ffmpeg pre-decode (MONO) failed with rc={rc} for stream {track_idx}",
                        )

                decode_downmix = None if is_stereo else DownmixMode.STEREO
                if self._progress is not None:
                    eac3to_step = (
                        f"[furnace] Decoding audio stream {track_idx} with eac3to"
                        if is_stereo
                        else f"[furnace] Downmixing audio stream {track_idx} to stereo with eac3to"
                    )
                    self._progress.add_tool_line(eac3to_step)
                stereo_wav = temp_dir / f"audio_{track_idx}_stereo.wav"
                _, on_progress = self._make_progress_callback(total_s=None)
                rc = self._audio_decoder.decode_lossless(
                    extracted,
                    stereo_wav,
                    instr.delay_ms,
                    on_progress=on_progress,
                    downmix=decode_downmix,
                )
                if rc != 0:
                    raise RuntimeError(f"eac3to decode failed: rc={rc}")

                if self._progress is not None:
                    self._progress.add_tool_line(
                        f"[furnace] Averaging audio stream {track_idx} to mono with ffmpeg",
                    )
                mono_wav = temp_dir / f"audio_{track_idx}_mono.wav"
                _, on_progress = self._make_progress_callback(total_s=None)
                rc = self._audio_extractor.stereo_to_mono_wav(
                    input_path=stereo_wav,
                    stream_index=0,
                    output_wav=mono_wav,
                    delay_ms=0,
                    on_progress=on_progress,
                )
                if rc != 0:
                    raise RuntimeError(f"stereo_to_mono_wav failed: rc={rc}")

                if self._progress is not None:
                    self._progress.add_tool_line(
                        f"[furnace] Encoding AAC for stream {track_idx} with qaac64",
                    )
                m4a_path = temp_dir / f"audio_{track_idx}.m4a"
                _, on_progress = self._make_progress_callback(total_s=None)
                rc = self._aac_encoder.encode_aac(
                    mono_wav,
                    m4a_path,
                    on_progress=on_progress,
                )
                if rc != 0:
                    raise RuntimeError(f"encode_aac failed: rc={rc}")
                return m4a_path

            if _codec_supported_by_eac3to(instr.codec_name):
                logger.info("Extracting audio stream %d", track_idx)
                if self._progress is not None:
                    self._progress.add_tool_line(f"[furnace] Extracting audio stream {track_idx}")
                extracted = temp_dir / f"audio_{track_idx}_raw{ext}"
                _, on_progress = self._make_progress_callback(
                    total_s=job.duration_s or None,
                )
                rc = self._audio_extractor.extract_track(
                    source_path,
                    track_idx,
                    extracted,
                    on_progress=on_progress,
                )
                if rc != 0:
                    raise RuntimeError(f"Audio extract (DECODE_ENCODE) failed with rc={rc} for stream {track_idx}")
            else:
                logger.info(
                    "Pre-decoding audio stream %d with ffmpeg to WAV",
                    track_idx,
                )
                if self._progress is not None:
                    self._progress.add_tool_line(
                        f"[furnace] Pre-decoding audio stream {track_idx} with ffmpeg "
                        f"(source codec {instr.codec_name} not readable by eac3to)"
                    )
                extracted = temp_dir / f"audio_{track_idx}_pre.wav"
                _, on_progress = self._make_progress_callback(
                    total_s=job.duration_s or None,
                )
                rc = self._audio_extractor.ffmpeg_to_wav(
                    source_path,
                    track_idx,
                    extracted,
                    on_progress=on_progress,
                )
                if rc != 0:
                    raise RuntimeError(f"ffmpeg pre-decode failed with rc={rc} for stream {track_idx}")

            logger.info("Decoding lossless with eac3to")
            if self._progress is not None:
                self._progress.add_tool_line(f"[furnace] Decoding lossless audio stream {track_idx} with eac3to")
            wav_path = temp_dir / f"audio_{track_idx}_decoded.wav"
            _, on_progress = self._make_progress_callback(total_s=None)
            rc = self._audio_decoder.decode_lossless(
                extracted,
                wav_path,
                instr.delay_ms,
                on_progress=on_progress,
                downmix=instr.downmix,
            )
            if rc != 0:
                raise RuntimeError(f"Audio decode_lossless failed with rc={rc} for stream {track_idx}")

            logger.info("Encoding AAC with qaac64")
            if self._progress is not None:
                self._progress.add_tool_line(f"[furnace] Encoding AAC for stream {track_idx} with qaac64")
            m4a_path = temp_dir / f"audio_{track_idx}.m4a"
            _, on_progress = self._make_progress_callback(total_s=None)
            rc = self._aac_encoder.encode_aac(
                wav_path,
                m4a_path,
                on_progress=on_progress,
            )
            if rc != 0:
                raise RuntimeError(f"AAC encode failed with rc={rc} for stream {track_idx}")
            return m4a_path

        if instr.action == AudioAction.FFMPEG_ENCODE:
            logger.info("Decoding with ffmpeg to WAV")
            if self._progress is not None:
                self._progress.add_tool_line(f"[furnace] Decoding audio stream {track_idx} with ffmpeg to WAV")
            wav_path = temp_dir / f"audio_{track_idx}_ffmpeg.wav"
            _, on_progress = self._make_progress_callback(total_s=job.duration_s or None)
            rc = self._audio_extractor.ffmpeg_to_wav(
                source_path,
                track_idx,
                wav_path,
                on_progress=on_progress,
            )
            if rc != 0:
                raise RuntimeError(f"ffmpeg_to_wav failed with rc={rc} for stream {track_idx}")
            logger.info("  Encoding AAC with qaac64")
            if self._progress is not None:
                self._progress.add_tool_line(f"[furnace] Encoding AAC for stream {track_idx} with qaac64")
            m4a_path = temp_dir / f"audio_{track_idx}.m4a"
            _, on_progress = self._make_progress_callback(total_s=None)
            rc = self._aac_encoder.encode_aac(wav_path, m4a_path, on_progress=on_progress)
            if rc != 0:
                raise RuntimeError(f"AAC encode (FFMPEG_ENCODE) failed with rc={rc} for stream {track_idx}")
            return m4a_path

        raise ValueError(f"Unknown AudioAction: {instr.action}")

    def _process_subtitle_track(self, instr: SubtitleInstruction, temp_dir: Path, job: Job) -> Path:
        source_path = Path(instr.source_file)
        track_idx = instr.stream_index

        ext = _SUBTITLE_CODEC_EXT.get(instr.codec_name, ".sub")

        if instr.action == SubtitleAction.COPY:
            if source_path.suffix.lower() in {".srt", ".ass", ".ssa", ".sup", ".sub"}:
                return source_path
            out_path = temp_dir / f"sub_{track_idx}{ext}"
            _, on_progress = self._make_progress_callback(total_s=job.duration_s or None)
            rc = self._audio_extractor.extract_track(
                source_path,
                track_idx,
                out_path,
                on_progress=on_progress,
            )
            if rc != 0:
                raise RuntimeError(f"Subtitle extract (COPY) failed with rc={rc} for stream {track_idx}")
            return out_path

        if instr.action == SubtitleAction.COPY_RECODE:
            if source_path.suffix.lower() in {".srt", ".ass", ".ssa"}:
                extracted = source_path
            else:
                extracted = temp_dir / f"sub_{track_idx}_raw{ext}"
                _, on_progress = self._make_progress_callback(total_s=job.duration_s or None)
                rc = self._audio_extractor.extract_track(
                    source_path,
                    track_idx,
                    extracted,
                    on_progress=on_progress,
                )
                if rc != 0:
                    raise RuntimeError(f"Subtitle extract (COPY_RECODE) failed with rc={rc} for stream {track_idx}")

            out_path = temp_dir / f"sub_{track_idx}_utf8{ext}"

            source_encoding = instr.source_encoding or "utf-8"
            if source_encoding.lower().replace("-", "") == "utf8":
                shutil.copy2(str(extracted), str(out_path))
            else:
                try:
                    content = extracted.read_bytes()
                    text = content.decode(source_encoding)
                    out_path.write_text(text, encoding="utf-8")
                except (OSError, ValueError, LookupError) as exc:
                    logger.warning(
                        "Recode failed for stream %d (%s->utf-8): %s; copying as-is",
                        track_idx,
                        source_encoding,
                        exc,
                    )
                    shutil.copy2(str(extracted), str(out_path))

            return out_path

        raise ValueError(f"Unknown SubtitleAction: {instr.action}")

    def _extract_chapters_file(self, source: Path, job_dir: Path) -> Path | None:
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
        logger.debug("Graceful shutdown requested")
        self._shutdown_event.set()
        try:
            parent = psutil.Process(os.getpid())
            for child in parent.children(recursive=True):
                with contextlib.suppress(psutil.NoSuchProcess):
                    child.kill()
        except (OSError, psutil.Error):
            logger.exception("Error during graceful shutdown")
