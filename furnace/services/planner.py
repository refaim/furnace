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
    DownmixMode,
    DvMode,
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
    interpolate_cq,
)
from ..core.detect import detect_video_system, is_dvd_resolution, resolve_color_metadata
from ..core.rules import get_audio_action, get_subtitle_action

from furnace import VERSION as FURNACE_VERSION

logger = logging.getLogger(__name__)

# ITU-R BT.601 PAL 4:3 sample aspect ratio. Applied as a SAR override to DVD
# sources that ffprobe reports as square-pixel 720x480/720x576 — the correct
# display geometry for a standard NTSC/PAL DVD is 4:3, which requires
# non-square pixels at 64:45 (or 32:27 for 16:9, which we don't apply here).
DVD_SAR_NUM = 64
DVD_SAR_DEN = 45

# Callback type: (movie, candidate_tracks, track_type) -> selected_tracks
TrackSelectorFn = Callable[[Movie, list[Track], TrackType], list[Track]]

# Callback type: (movie, track, lang_list) -> chosen_language
UndLanguageResolverFn = Callable[[Movie, Track, list[str]], str]


class PlannerService:
    def __init__(
        self,
        prober: Prober,
        previewer: Previewer | None,  # None in --dry-run
        track_selector: TrackSelectorFn | None = None,  # None = include all (headless)
        und_resolver: UndLanguageResolverFn | None = None,
    ) -> None:
        self._prober = prober
        self._previewer = previewer
        self._track_selector = track_selector
        self._und_resolver = und_resolver

    def create_plan(
        self,
        movies: list[tuple[Movie, Path]],
        audio_lang_filter: list[str],
        sub_lang_filter: list[str],
        vmaf_enabled: bool,
        dry_run: bool,
        *,
        sar_overrides: set[Path] | None = None,
        downmix_overrides: dict[tuple[Path, int], DownmixMode] | None = None,
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

        # Preserve the caller's dict/set identity. Using `or {}` here would
        # silently swap an empty caller dict for a fresh literal and break the
        # reference, so any closure that mutates the caller's dict during
        # track selection (e.g. cli.py's _select_tracks_tui_for_planner) would
        # have its updates dropped on the floor.
        effective_overrides: dict[tuple[Path, int], DownmixMode] = (
            downmix_overrides if downmix_overrides is not None else {}
        )
        effective_sar_overrides: set[Path] = (
            sar_overrides if sar_overrides is not None else set()
        )

        for movie, output_path in movies:
            job = self._build_job(
                movie, output_path, audio_lang_filter, sub_lang_filter,
                vmaf_enabled, dry_run,
                sar_overrides=effective_sar_overrides,
                downmix_overrides=effective_overrides,
            )
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
        *,
        sar_overrides: set[Path],
        downmix_overrides: dict[tuple[Path, int], DownmixMode],
    ) -> Job | None:
        """Build a single Job for a Movie."""
        # Detect crop
        crop: CropRect | None = None
        if not dry_run:
            try:
                is_dvd = is_dvd_resolution(movie.video.width, movie.video.height)
                raw_crop = self._prober.detect_crop(
                    movie.main_file, movie.video.duration_s,
                    interlaced=movie.video.interlaced,
                    is_dvd=is_dvd,
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
        video_params = self._build_video_params(
            movie.video, crop,
            source_file=movie.main_file,
            sar_overrides=sar_overrides,
        )

        # Auto-select audio tracks
        audio_candidates = self._filter_audio_tracks_by_lang(movie.audio_tracks, audio_lang_filter)
        selected_audio = self._auto_select_from_candidates(audio_candidates, TrackType.AUDIO)
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
        selected_subs = self._auto_select_from_candidates(sub_candidates, TrackType.SUBTITLE)
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

        # Resolve und languages for selected audio
        if self._und_resolver is not None:
            selected_audio = self._resolve_und_languages(movie, selected_audio, audio_lang_filter, self._und_resolver)
            selected_audio = self._sort_and_set_default(selected_audio, audio_lang_filter)

        # Build audio instructions
        audio_instructions: list[AudioInstruction] = []
        for i, track in enumerate(selected_audio):
            is_default = i == 0
            track_key = (Path(track.source_file), track.index)
            track_downmix = downmix_overrides.get(track_key)
            audio_instr = self._build_audio_instruction(track, is_default, track_downmix)
            audio_instructions.append(audio_instr)

        # Resolve und languages for selected subs
        if self._und_resolver is not None:
            selected_subs = self._resolve_und_languages(movie, selected_subs, sub_lang_filter, self._und_resolver)
            selected_subs = self._sort_and_set_default(selected_subs, sub_lang_filter)

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
            duration_s=movie.video.duration_s,
        )
        return job

    def _filter_audio_tracks_by_lang(
        self, tracks: list[Track], lang_filter: list[str],
    ) -> list[Track]:
        """Filter audio tracks: keep matching languages + 'und', sort by lang_filter order."""
        filtered = [t for t in tracks if t.language in lang_filter or t.language == "und"]
        return self._sort_and_set_default(filtered, lang_filter)

    def _filter_sub_tracks_by_lang(
        self, tracks: list[Track], lang_filter: list[str],
    ) -> list[Track]:
        """Filter subtitle tracks: keep matching languages + 'und', discard forced, sort by lang_filter order."""
        filtered = [
            t for t in tracks
            if not t.is_forced and (t.language in lang_filter or t.language == "und")
        ]
        return self._sort_and_set_default(filtered, lang_filter)

    def _sort_and_set_default(
        self,
        tracks: list[Track],
        lang_filter: list[str],
    ) -> list[Track]:
        """Sort tracks by lang_filter order and set is_default on the first."""
        if not tracks:
            return tracks
        lang_order = {lang: i for i, lang in enumerate(lang_filter)}
        tracks.sort(key=lambda t: lang_order.get(t.language, len(lang_filter)))
        for i, t in enumerate(tracks):
            t.is_default = i == 0
        return tracks

    def _resolve_und_languages(
        self,
        movie: Movie,
        tracks: list[Track],
        lang_filter: list[str],
        resolve_cb: Callable[[Movie, Track, list[str]], str],
    ) -> list[Track]:
        """Assign real languages to 'und' tracks from lang_filter.

        - No und tracks: return unchanged.
        - Single lang in filter: auto-assign to all und tracks.
        - Multiple langs: call resolve_cb for each und track.
        """
        und_tracks = [t for t in tracks if t.language == "und"]
        if not und_tracks:
            return tracks
        if len(lang_filter) == 1:
            for t in und_tracks:
                t.language = lang_filter[0]
        else:
            for t in und_tracks:
                t.language = resolve_cb(movie, t, lang_filter)
        return tracks

    def _auto_select_from_candidates(
        self, candidates: list[Track], track_type: TrackType,
    ) -> list[Track] | None:
        """If exactly one track per language -> auto-select.
        For AUDIO only: additionally force TUI if any candidate has >2 channels
        (so the user can configure downmix even on unambiguous files).
        Returns None when the caller should invoke the track_selector.
        """
        if not candidates:
            return candidates

        lang_groups: dict[str, list[Track]] = {}
        for track in candidates:
            lang_groups.setdefault(track.language, []).append(track)

        for group in lang_groups.values():
            if len(group) > 1:
                return None

        # X-A: for audio only, any candidate with >2 channels forces the TUI
        if track_type == TrackType.AUDIO:
            for track in candidates:
                if track.channels is not None and track.channels > 2:
                    return None

        return candidates

    def _build_video_params(
        self,
        video: VideoInfo,
        crop: CropRect | None,
        *,
        source_file: Path,
        sar_overrides: set[Path],
    ) -> VideoParams:
        """CQ interpolation, GOP calc, colorspace determination, deinterlace detection."""
        # Use cropped area for CQ if crop is applied
        if crop is not None:
            pixel_area = crop.w * crop.h
        else:
            pixel_area = video.pixel_area

        cq = interpolate_cq(pixel_area)
        gop = calculate_gop(video.fps_num, video.fps_den)

        system = detect_video_system(video.height)
        has_hdr = bool(video.hdr.mastering_display or video.hdr.content_light)
        resolved = resolve_color_metadata(
            matrix_raw=video.color_matrix_raw,
            transfer_raw=video.color_transfer,
            primaries_raw=video.color_primaries,
            system=system,
            has_hdr=has_hdr,
        )

        deinterlace = video.interlaced

        # HDR10+ guard (should be caught by analyzer, but double-check)
        if video.hdr.is_hdr10_plus:
            raise ValueError(f"HDR10+ not supported: {video.source_file.name}")

        # DV mode
        dv_mode: DvMode | None = None
        if video.hdr.is_dolby_vision:
            if video.hdr.dv_profile == 7:
                dv_mode = DvMode.TO_8_1
            else:
                dv_mode = DvMode.COPY

        # HDR metadata passthrough
        hdr = video.hdr if (video.hdr.mastering_display or video.hdr.content_light) else None

        # SAR override: if the source file is flagged, force the DVD 4:3 SAR
        # (see DVD_SAR_NUM/DVD_SAR_DEN at module top for rationale).
        if source_file in sar_overrides:
            sar_num = DVD_SAR_NUM
            sar_den = DVD_SAR_DEN
        else:
            sar_num = video.sar_num
            sar_den = video.sar_den

        return VideoParams(
            cq=cq,
            crop=crop,
            deinterlace=deinterlace,
            color_matrix=resolved.matrix,
            color_range="tv",
            color_transfer=resolved.transfer,
            color_primaries=resolved.primaries,
            hdr=hdr,
            gop=gop,
            fps_num=video.fps_num,
            fps_den=video.fps_den,
            source_width=video.width,
            source_height=video.height,
            source_codec=video.codec_name,
            source_bitrate=video.bitrate,
            sar_num=sar_num,
            sar_den=sar_den,
            dv_mode=dv_mode,
        )

    def _build_audio_instruction(
        self,
        track: Track,
        is_default: bool,
        downmix: DownmixMode | None = None,
    ) -> AudioInstruction:
        """Route through rules.get_audio_action(), unless downmix forces
        DECODE_ENCODE. Validates downmix applicability."""
        from ..core.models import AudioAction, AudioCodecId

        if downmix is not None:
            if track.channels is None or track.channels <= 2:
                raise ValueError(
                    f"Downmix not applicable: track has {track.channels} channels "
                    f"({track.source_file} index {track.index})"
                )
            if downmix == DownmixMode.DOWN6 and track.channels <= 6:
                raise ValueError(
                    f"DOWN6 not applicable: track has {track.channels} channels "
                    f"({track.source_file} index {track.index})"
                )
            action = AudioAction.DECODE_ENCODE
        elif track.codec_id is not None and not isinstance(track.codec_id, AudioCodecId):
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
            downmix=downmix,
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
