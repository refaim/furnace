from __future__ import annotations

import datetime
import logging
import uuid
from pathlib import Path
from typing import Any

from collections.abc import Callable

from ..core.models import (
    AudioInstruction,
    CropRect,
    Job,
    JobStatus,
    Movie,
    Plan,
    SubtitleInstruction,
    Track,
    TrackType,
    VideoInfo,
    VideoParams,
)
from ..core.ports import Previewer, Prober
from ..core.quality import (
    calculate_gop,
    determine_color_space,
    interpolate_cq,
)
from ..core.rules import get_audio_action, get_subtitle_action

from furnace import VERSION as FURNACE_VERSION

logger = logging.getLogger(__name__)

# Callback type: (movie, candidate_tracks, track_type) -> selected_tracks
TrackSelectorFn = Callable[[Movie, list[Track], TrackType], list[Track]]


class PlannerService:
    def __init__(
        self,
        prober: Prober,
        previewer: Previewer | None,  # None in --dry-run
        track_selector: TrackSelectorFn | None = None,  # None = include all (headless)
    ) -> None:
        self._prober = prober
        self._previewer = previewer
        self._track_selector = track_selector

    def create_plan(
        self,
        movies: list[tuple[Movie, Path]],
        audio_lang_filter: list[str],
        sub_lang_filter: list[str],
        vmaf_enabled: bool,
        dry_run: bool,
    ) -> Plan:
        """For each Movie:
        1. Skip logic
        2. Apply lang filter -> auto-select or TUI
        3. Detect forced subs
        4. Detect crop -> TUI confirm
        5. Calculate video params (CQ, deinterlace, colorspace, HDR)
        6. Determine audio/subtitle actions
        7. Build Job
        """
        jobs: list[Job] = []

        # Use the first movie's source path as plan source
        source = str(movies[0][0].main_file.parent) if movies else ""
        destination = str(movies[0][1].parent) if movies else ""

        for movie, output_path in movies:
            job = self._build_job(movie, output_path, audio_lang_filter, sub_lang_filter, vmaf_enabled, dry_run)
            if job is not None:
                jobs.append(job)

        now = datetime.datetime.now(datetime.UTC).isoformat()
        return Plan(
            version="2",
            furnace_version=FURNACE_VERSION,
            created_at=now,
            source=source,
            destination=destination,
            vmaf_enabled=vmaf_enabled,
            jobs=jobs,
        )

    def _build_job(
        self,
        movie: Movie,
        output_path: Path,
        audio_lang_filter: list[str],
        sub_lang_filter: list[str],
        vmaf_enabled: bool,
        dry_run: bool,
    ) -> Job | None:
        """Build a single Job for a Movie."""
        # Detect crop
        crop: CropRect | None = None
        if not dry_run:
            try:
                raw_crop = self._prober.detect_crop(
                    movie.main_file, movie.video.duration_s
                )
                if raw_crop is not None:
                    crop = raw_crop
                    # Skip crop if it equals full frame (no black bars)
                    if crop.w == movie.video.width and crop.h == movie.video.height:
                        logger.info(
                            "%s: no black bars detected (crop equals full frame %dx%d)",
                            movie.main_file.name, movie.video.width, movie.video.height,
                        )
                        crop = None
                    else:
                        logger.info(
                            "%s: crop detected %d:%d:%d:%d (source %dx%d)",
                            movie.main_file.name,
                            crop.w, crop.h, crop.x, crop.y,
                            movie.video.width, movie.video.height,
                        )
                else:
                    logger.warning("%s: cropdetect unable to determine crop", movie.main_file.name)
            except Exception as exc:
                logger.warning("Crop detection failed for %s: %s", movie.main_file.name, exc)

        # Build video params
        video_params = self._build_video_params(movie.video, crop)

        # Auto-select audio tracks
        audio_candidates = self._filter_audio_tracks_by_lang(movie.audio_tracks, audio_lang_filter)
        selected_audio = self._auto_select_from_candidates(audio_candidates)
        if selected_audio is None:
            if self._track_selector is not None:
                logger.debug(
                    "Multiple audio tracks per language for %s; showing TUI",
                    movie.main_file.name,
                )
                selected_audio = self._track_selector(movie, audio_candidates, TrackType.AUDIO)
            else:
                logger.warning(
                    "Multiple audio tracks per language for %s; no track_selector, including all",
                    movie.main_file.name,
                )
                selected_audio = audio_candidates

        # Auto-select subtitle tracks
        sub_candidates = self._filter_sub_tracks_by_lang(movie.subtitle_tracks, sub_lang_filter)
        selected_subs = self._auto_select_from_candidates(sub_candidates)
        if selected_subs is None:
            if self._track_selector is not None:
                logger.debug(
                    "Multiple subtitle tracks per language for %s; showing TUI",
                    movie.main_file.name,
                )
                selected_subs = self._track_selector(movie, sub_candidates, TrackType.SUBTITLE)
            else:
                logger.warning(
                    "Multiple subtitle tracks per language for %s; no track_selector, including all",
                    movie.main_file.name,
                )
                selected_subs = sub_candidates

        # Build audio instructions
        audio_instructions: list[AudioInstruction] = []
        for i, track in enumerate(selected_audio):
            is_default = i == 0
            audio_instr = self._build_audio_instruction(track, is_default)
            audio_instructions.append(audio_instr)

        # Build subtitle instructions
        sub_instructions: list[SubtitleInstruction] = []
        for i, track in enumerate(selected_subs):
            is_default = i == 0
            sub_instr = self._build_subtitle_instruction(track, is_default)
            sub_instructions.append(sub_instr)

        # Attachments as dicts
        attachments_dicts: list[dict[str, Any]] = [
            {
                "filename": att.filename,
                "mime_type": att.mime_type,
                "source_file": str(att.source_file),
            }
            for att in movie.attachments
        ]

        # Chapters
        copy_chapters = movie.has_chapters
        chapters_source: str | None = str(movie.main_file) if movie.has_chapters else None

        # Source files list
        source_files = [str(movie.main_file)] + [str(p) for p in movie.satellite_files]

        job = Job(
            id=str(uuid.uuid4()),
            source_files=source_files,
            output_file=str(output_path),
            video_params=video_params,
            audio=audio_instructions,
            subtitles=sub_instructions,
            attachments=attachments_dicts,
            copy_chapters=copy_chapters,
            chapters_source=chapters_source,
            status=JobStatus.PENDING,
            error=None,
            vmaf_score=None,
            source_size=movie.file_size,
            output_size=None,
        )
        return job

    def _filter_audio_tracks_by_lang(
        self, tracks: list[Track], lang_filter: list[str],
    ) -> list[Track]:
        """Filter audio tracks: keep matching languages + 'und', sort by lang_filter order."""
        filtered = [t for t in tracks if t.language in lang_filter or t.language == "und"]
        lang_order = {lang: i for i, lang in enumerate(lang_filter)}
        filtered.sort(key=lambda t: lang_order.get(t.language, len(lang_filter)))
        return filtered

    def _filter_sub_tracks_by_lang(
        self, tracks: list[Track], lang_filter: list[str],
    ) -> list[Track]:
        """Filter subtitle tracks: keep matching languages + 'und', discard forced, sort by lang_filter order."""
        filtered = [
            t for t in tracks
            if not t.is_forced and (t.language in lang_filter or t.language == "und")
        ]
        lang_order = {lang: i for i, lang in enumerate(lang_filter)}
        filtered.sort(key=lambda t: lang_order.get(t.language, len(lang_filter)))
        return filtered

    def _auto_select_from_candidates(
        self, candidates: list[Track],
    ) -> list[Track] | None:
        """If exactly one track per language -> auto-select.
        If multiple tracks for any language -> return None (caller shows TUI).
        """
        if not candidates:
            return candidates

        lang_groups: dict[str, list[Track]] = {}
        for track in candidates:
            lang_groups.setdefault(track.language, []).append(track)

        for group in lang_groups.values():
            if len(group) > 1:
                return None

        return candidates

    def _build_video_params(self, video: VideoInfo, crop: CropRect | None) -> VideoParams:
        """CQ interpolation, GOP calc, colorspace determination, deinterlace detection."""
        # Use cropped area for CQ if crop is applied
        if crop is not None:
            pixel_area = crop.w * crop.h
        else:
            pixel_area = video.pixel_area

        cq = interpolate_cq(pixel_area)
        gop = calculate_gop(video.fps_num, video.fps_den)

        color_space = determine_color_space(
            video.width, video.height,
            video.color_space.value if video.color_space is not None else None,
        )

        deinterlace = video.interlaced

        # HDR: only passthrough for HDR10 (not DV/HDR10+ which are already skipped)
        hdr = video.hdr if (video.hdr.mastering_display or video.hdr.content_light) else None

        # Color info passthrough
        color_transfer = video.color_transfer
        color_primaries = video.color_primaries

        return VideoParams(
            cq=cq,
            crop=crop,
            deinterlace=deinterlace,
            color_space=color_space,
            color_range="tv",
            color_transfer=color_transfer,
            color_primaries=color_primaries,
            hdr=hdr,
            gop=gop,
            fps_num=video.fps_num,
            fps_den=video.fps_den,
            source_width=video.width,
            source_height=video.height,
            source_codec=video.codec_name,
            source_bitrate=video.bitrate,
            sar_num=video.sar_num,
            sar_den=video.sar_den,
        )

    def _build_audio_instruction(self, track: Track, is_default: bool) -> AudioInstruction:
        """Route through rules.get_audio_action()."""
        from ..core.models import AudioAction, AudioCodecId

        if track.codec_id is not None and not isinstance(track.codec_id, AudioCodecId):
            # Should not happen for audio tracks, but guard
            action = AudioAction.FFMPEG_ENCODE
        elif track.codec_id is not None:
            maybe_action = get_audio_action(track.codec_id)
            action = maybe_action if maybe_action is not None else AudioAction.FFMPEG_ENCODE
        else:
            action = AudioAction.FFMPEG_ENCODE

        return AudioInstruction(
            source_file=str(track.source_file),
            stream_index=track.index,
            language=track.language,
            action=action,
            delay_ms=track.delay_ms,
            is_default=is_default,
            codec_name=track.codec_name,
            channels=track.channels,
            bitrate=track.bitrate,
        )

    def _build_subtitle_instruction(self, track: Track, is_default: bool) -> SubtitleInstruction:
        """Route through rules.get_subtitle_action()."""
        from ..core.models import SubtitleAction, SubtitleCodecId

        if track.codec_id is not None and not isinstance(track.codec_id, SubtitleCodecId):
            action = SubtitleAction.COPY
        elif track.codec_id is not None:
            maybe_action = get_subtitle_action(track.codec_id)
            action = maybe_action if maybe_action is not None else SubtitleAction.COPY
        else:
            action = SubtitleAction.COPY

        return SubtitleInstruction(
            source_file=str(track.source_file),
            stream_index=track.index,
            language=track.language,
            action=action,
            is_default=is_default,
            is_forced=track.is_forced,
            codec_name=track.codec_name,
            source_encoding=track.encoding,
        )
