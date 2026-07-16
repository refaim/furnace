from __future__ import annotations

import datetime
import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from furnace import VERSION as FURNACE_VERSION
from furnace.core.audio_profile import Verdict
from furnace.core.detect import (
    DV_PROFILE_FEL,
    classify_passthrough,
    detect_video_system,
    is_hdr_transfer,
    resolve_color_metadata,
)
from furnace.core.models import (
    STEREO_CHANNELS,
    SURROUND_5_1_CHANNELS,
    AudioAction,
    AudioCodecId,
    AudioInstruction,
    CropRect,
    DownmixMode,
    DvMode,
    Job,
    JobStatus,
    Movie,
    Plan,
    SubtitleAction,
    SubtitleCodecId,
    SubtitleInstruction,
    Track,
    TrackType,
    VideoInfo,
    VideoParams,
)
from furnace.core.ports import PlanReporter, Previewer
from furnace.core.quality import calculate_gop, final_output_dimensions, interpolate_cq
from furnace.core.rules import get_audio_action, get_subtitle_action

logger = logging.getLogger(__name__)


def _format_plan_summary(movie: Movie, job: Job, fallback_reason: str | None = None) -> str:
    """One-line per-movie summary shown after Plan completes for that movie.

    - Passthrough jobs render ``passthrough (copy video)``.
    - Encode jobs that *fell back* from a requested passthrough render
      ``encode (<reason>), <encode summary>`` — e.g. ``encode (interlaced), ...``
      or ``encode (DV P7 FEL), ...``.
    - Plain encode jobs render ``<SrcW>x<SrcH> to <DstW>x<DstH>[, deinterlace]``.

    No quality knob is shown: the real CRF/QVBR is target-quality-searched per
    film at ``furnace run`` time (and can differ wildly from any plan-time guess),
    so printing ``vp.cq`` — the pre-target-quality resolution-based anchor, unused
    by the grain encode and only a rare NVEnc fallback — would mislead.

    The ``DstWxDstH`` part is the *actual* encoded output (crop -> SAR ->
    mod-8 alignment), via :func:`final_output_dimensions`. The
    resolution separator is the word ``to`` (not ``->``) so it doesn't
    collide with the reporter's ``label -> status`` arrow.
    """
    if job.video_params.passthrough:
        return "passthrough (copy video)"
    src_w = movie.video.width
    src_h = movie.video.height
    dst_w, dst_h = final_output_dimensions(job.video_params)
    parts = [f"{src_w}x{src_h} to {dst_w}x{dst_h}"]
    if job.video_params.deinterlace:
        parts.append("deinterlace")
    encode_summary = ", ".join(parts)
    if fallback_reason is not None:
        return f"encode ({fallback_reason}), {encode_summary}"
    return encode_summary


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
        previewer: Previewer | None,  # None in --dry-run
        track_selector: TrackSelectorFn | None = None,  # None = include all (headless)
        und_resolver: UndLanguageResolverFn | None = None,
        reporter: PlanReporter | None = None,
        *,
        ignore_langs: bool = False,
    ) -> None:
        self._previewer = previewer
        self._track_selector = track_selector
        self._und_resolver = und_resolver
        self._reporter = reporter
        self._ignore_langs = ignore_langs

    def create_plan(
        self,
        movies: list[tuple[Movie, Path]],
        audio_lang_filter: list[str],
        sub_lang_filter: list[str],
        *,
        sar_overrides: set[Path] | None = None,
        downmix_overrides: dict[tuple[Path, int], DownmixMode] | None = None,
        lang_overrides: dict[tuple[Path, int], str] | None = None,
        precomputed_crops: dict[Path, CropRect] | None = None,
        grain_overrides: dict[Path, bool] | None = None,
        copy_video: bool = False,
    ) -> Plan:
        """For each Movie:
        1. Skip logic
        2. Apply lang filter -> auto-select or TUI
        3. Detect forced subs
        4. Apply precomputed crop (from ``precomputed_crops``)
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
        effective_lang_overrides: dict[tuple[Path, int], str] = (
            lang_overrides if lang_overrides is not None else {}
        )
        effective_sar_overrides: set[Path] = sar_overrides if sar_overrides is not None else set()
        effective_crops: dict[Path, CropRect] = precomputed_crops if precomputed_crops is not None else {}
        effective_grain_overrides: dict[Path, bool] = grain_overrides if grain_overrides is not None else {}

        for movie, output_path in movies:
            if self._reporter is not None:
                self._reporter.plan_file_start(movie.main_file.name)
            job, fallback_reason = self._build_job(
                movie,
                output_path,
                audio_lang_filter,
                sub_lang_filter,
                sar_overrides=effective_sar_overrides,
                downmix_overrides=effective_overrides,
                lang_overrides=effective_lang_overrides,
                precomputed_crops=effective_crops,
                grain_overrides=effective_grain_overrides,
                copy_video=copy_video,
            )
            if self._reporter is not None:
                # fallback_reason is the single source of truth from _build_job
                # (interlaced / DV P7 FEL) when a requested passthrough had to
                # fall back to a normal encode.
                summary = _format_plan_summary(movie, job, fallback_reason)
                self._reporter.plan_file_done(summary)
            jobs.append(job)

        now = datetime.datetime.now(datetime.UTC).isoformat()
        return Plan(
            version="2",
            furnace_version=FURNACE_VERSION,
            created_at=now,
            source=source,
            destination=destination,
            jobs=jobs,
        )

    def _build_job(
        self,
        movie: Movie,
        output_path: Path,
        audio_lang_filter: list[str],
        sub_lang_filter: list[str],
        *,
        sar_overrides: set[Path],
        downmix_overrides: dict[tuple[Path, int], DownmixMode],
        lang_overrides: dict[tuple[Path, int], str],
        precomputed_crops: dict[Path, CropRect],
        grain_overrides: dict[Path, bool],
        copy_video: bool = False,
    ) -> tuple[Job, str | None]:
        """Build a single Job for a Movie.

        Returns ``(job, fallback_reason)``: ``fallback_reason`` is the reason a
        requested passthrough had to fall back to a normal encode (``interlaced``
        / ``DV P7 FEL``), or ``None`` for passthrough jobs and plain encodes.
        """
        # Decide passthrough eligibility up front: an eligible video stream is
        # copied verbatim, so any precomputed crop is ignored downstream.
        passthrough, fallback_reason = classify_passthrough(movie.video, copy_video=copy_video)

        # Crop comes precomputed (None when no entry for this file).
        crop = precomputed_crops.get(movie.main_file)

        # Build video params
        video_params = self._build_video_params(
            movie.video,
            crop,
            source_file=movie.main_file,
            sar_overrides=sar_overrides,
            grain_overrides=grain_overrides,
            passthrough=passthrough,
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

        # Assign languages for selected audio: relabel under --ignore-langs,
        # otherwise resolve any 'und' tracks via the resolver.
        if self._ignore_langs:
            selected_audio = self._assign_languages_relabel(selected_audio, audio_lang_filter, lang_overrides)
            selected_audio = self._sort_and_set_default(selected_audio, audio_lang_filter, ignore_langs=False)
        elif self._und_resolver is not None:
            selected_audio = self._resolve_und_languages(movie, selected_audio, audio_lang_filter, self._und_resolver)
            selected_audio = self._sort_and_set_default(selected_audio, audio_lang_filter, ignore_langs=False)

        # Build audio instructions
        audio_instructions: list[AudioInstruction] = []
        for i, track in enumerate(selected_audio):
            is_default = i == 0
            track_key = (Path(track.source_file), track.index)
            track_downmix = downmix_overrides.get(track_key)
            audio_instr = self._build_audio_instruction(track, is_default=is_default, downmix=track_downmix)
            audio_instructions.append(audio_instr)

        # Assign languages for selected subs: relabel under --ignore-langs,
        # otherwise resolve any 'und' tracks via the resolver.
        if self._ignore_langs:
            selected_subs = self._assign_languages_relabel(selected_subs, sub_lang_filter, lang_overrides)
            selected_subs = self._sort_and_set_default(selected_subs, sub_lang_filter, ignore_langs=False)
        elif self._und_resolver is not None:
            selected_subs = self._resolve_und_languages(movie, selected_subs, sub_lang_filter, self._und_resolver)
            selected_subs = self._sort_and_set_default(selected_subs, sub_lang_filter, ignore_langs=False)

        # Build subtitle instructions
        sub_instructions: list[SubtitleInstruction] = []
        for i, track in enumerate(selected_subs):
            is_default = i == 0
            sub_instr = self._build_subtitle_instruction(track, is_default=is_default)
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
            source_size=movie.file_size,
            output_size=None,
            duration_s=movie.video.duration_s,
        )
        return job, fallback_reason

    def _eff_lang(self, track: Track) -> str:
        """Effective language for filtering/grouping. Under --ignore-langs every
        track is treated as 'und' so nothing is dropped and all tracks group together."""
        return "und" if self._ignore_langs else track.language

    def _filter_audio_tracks_by_lang(
        self,
        tracks: list[Track],
        lang_filter: list[str],
    ) -> list[Track]:
        """Filter audio tracks: keep matching languages + 'und', sort by lang_filter order."""
        filtered = [t for t in tracks if self._eff_lang(t) in lang_filter or self._eff_lang(t) == "und"]
        return self._sort_and_set_default(filtered, lang_filter, ignore_langs=self._ignore_langs)

    def _filter_sub_tracks_by_lang(
        self,
        tracks: list[Track],
        lang_filter: list[str],
    ) -> list[Track]:
        """Filter subtitle tracks: keep matching languages + 'und', discard forced, sort by lang_filter order."""
        filtered = [
            t for t in tracks if not t.is_forced and (self._eff_lang(t) in lang_filter or self._eff_lang(t) == "und")
        ]
        return self._sort_and_set_default(filtered, lang_filter, ignore_langs=self._ignore_langs)

    def _sort_and_set_default(
        self,
        tracks: list[Track],
        lang_filter: list[str],
        *,
        ignore_langs: bool,
    ) -> list[Track]:
        """Sort tracks by lang_filter order and set is_default on the first.

        Under ``ignore_langs`` the sort is skipped so source order is preserved
        (the TUI selector then shows tracks in their original order).
        """
        if not tracks:
            return tracks
        if not ignore_langs:
            lang_order = {lang: i for i, lang in enumerate(lang_filter)}
            tracks.sort(key=lambda t: lang_order.get(t.language, len(lang_filter)))
        for i, t in enumerate(tracks):
            t.is_default = i == 0
        return tracks

    def _assign_languages_relabel(
        self,
        tracks: list[Track],
        lang_filter: list[str],
        lang_overrides: dict[tuple[Path, int], str],
    ) -> list[Track]:
        """Under --ignore-langs, set each selected track's language to its explicit
        'l'-override, else the first target language (``lang_filter[0]``, or 'und'
        if the filter is empty)."""
        default = lang_filter[0] if lang_filter else "und"
        for t in tracks:
            key = (Path(t.source_file), t.index)
            t.language = lang_overrides.get(key, default)
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
        self,
        candidates: list[Track],
        track_type: TrackType,
    ) -> list[Track] | None:
        """If exactly one track per language -> auto-select.
        For AUDIO only: additionally force TUI if the fake-surround detector
        flagged any candidate as fake or possibly fake (verdict != REAL), so the
        user can pick a downmix. Tracks with no verdict (audio_profile=None) do
        not trigger the TUI on their own.
        Returns None when the caller should invoke the track_selector.
        """
        if not candidates:
            return candidates

        lang_groups: dict[str, list[Track]] = {}
        for track in candidates:
            lang_groups.setdefault(self._eff_lang(track), []).append(track)

        for group in lang_groups.values():
            if len(group) > 1:
                return None

        # For audio only, a fake/suspicious detector verdict forces the TUI
        if track_type == TrackType.AUDIO:
            for track in candidates:
                profile = track.audio_profile
                if profile is not None and profile.verdict != Verdict.REAL:
                    return None

        return candidates

    def _build_video_params(
        self,
        video: VideoInfo,
        crop: CropRect | None,
        *,
        source_file: Path,
        sar_overrides: set[Path],
        grain_overrides: dict[Path, bool],
        passthrough: bool = False,
    ) -> VideoParams:
        """CQ interpolation, GOP calc, colorspace determination, deinterlace detection.

        When ``passthrough`` is set, the video stream is copied verbatim:
        crop is forced off and deinterlace is disabled (``cq``/``gop`` become
        inert), while colour/HDR/SAR fields stay populated for container flags.
        """
        # Passthrough copies the stream as-is: no crop, no deinterlace.
        if passthrough:
            crop = None

        # Use cropped area for CQ if crop is applied
        pixel_area = crop.w * crop.h if crop is not None else video.pixel_area

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

        deinterlace = video.interlaced and not passthrough

        # HDR10+ guard (should be caught by analyzer, but double-check)
        if video.hdr.is_hdr10_plus:
            raise ValueError(f"HDR10+ not supported: {video.source_file.name}")

        # DV mode
        dv_mode: DvMode | None = None
        if video.hdr.is_dolby_vision:
            dv_mode = DvMode.TO_8_1 if video.hdr.dv_profile == DV_PROFILE_FEL else DvMode.COPY

        # HDR metadata passthrough
        hdr = video.hdr if has_hdr else None

        # SAR override: if the source file is flagged, force the DVD 4:3 SAR
        # (see DVD_SAR_NUM/DVD_SAR_DEN at module top for rationale).
        if source_file in sar_overrides:
            sar_num = DVD_SAR_NUM
            sar_den = DVD_SAR_DEN
        else:
            sar_num = video.sar_num
            sar_den = video.sar_den

        # Grain: a passthrough job copies the stream verbatim, so there is
        # nothing to tune -> always False. Otherwise an explicit per-file
        # override wins over the analyzer's ``grainy`` verdict.
        if passthrough:
            grain = False
        elif source_file in grain_overrides:
            grain = grain_overrides[source_file]
        else:
            grain = video.grainy

        # The grain path is SDR-only: its target-quality search scores with
        # SSIMULACRA2, which does not score PQ/HLG, so ``resolve_target`` refuses
        # grain+HDR loudly. This is where that invariant is enforced, because this
        # is the only point where the grain decision and the RESOLVED transfer are
        # both known: the analyzer's probe gate sees the source's RAW transfer, but
        # an untagged HDR remux only becomes PQ here (``resolve_color_metadata``
        # promotes an absent transfer + mastering-display metadata to 'smpte2084'),
        # and a manual grain override bypasses that gate entirely. Routing HDR to
        # NVEnc/CVVDP is the correct answer for it, not a degradation — but say so,
        # since it silently overrides an explicit user toggle.
        if grain and is_hdr_transfer(resolved.transfer):
            logger.info(
                "%s: HDR (%s) — grain path is SDR-only, encoding on the NVEnc/CVVDP path",
                source_file.name, resolved.transfer,
            )
            grain = False

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
            passthrough=passthrough,
            grain=grain,
        )

    def _build_audio_instruction(
        self,
        track: Track,
        *,
        is_default: bool,
        downmix: DownmixMode | None = None,
    ) -> AudioInstruction:
        """Route through rules.get_audio_action(), unless downmix forces
        DECODE_ENCODE. Validates downmix applicability."""
        if downmix is not None:
            if downmix == DownmixMode.MONO:
                if track.channels is None or track.channels < STEREO_CHANNELS:
                    raise ValueError(
                        f"MONO downmix requires >=2 channels, got {track.channels} "
                        f"({track.source_file} index {track.index})"
                    )
            else:
                if track.channels is None or track.channels <= STEREO_CHANNELS:
                    raise ValueError(
                        f"Downmix not applicable: track has {track.channels} channels "
                        f"({track.source_file} index {track.index})"
                    )
                if downmix == DownmixMode.DOWN6 and track.channels <= SURROUND_5_1_CHANNELS:
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

    def _build_subtitle_instruction(self, track: Track, *, is_default: bool) -> SubtitleInstruction:
        """Route through rules.get_subtitle_action()."""
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
