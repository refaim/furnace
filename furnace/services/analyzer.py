from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from charset_normalizer import from_path as _from_path

from furnace.core.audio_profile import classify_audio
from furnace.core.detect import (
    check_unsupported_codecs,
    detect_forced_subtitles,
    detect_hdr,
    needs_idet,
    should_deinterlace,
    should_skip_file,
)
from furnace.core.models import (
    AnalysisOutcome,
    AnalyzeStatus,
    Attachment,
    Movie,
    ScanResult,
    SubtitleCodecId,
    Track,
    TrackType,
    VideoInfo,
)
from furnace.core.ports import Prober
from furnace.core.rules import parse_audio_codec, parse_subtitle_codec

_PROFILEABLE_CHANNEL_COUNTS = frozenset({2, 6, 8})

_ISO_639_3_LENGTH = 3  # ISO 639-3 language codes are exactly three letters

logger = logging.getLogger(__name__)

# Text-based subtitle codecs that need encoding detection
_TEXT_SUBTITLE_CODECS: set[SubtitleCodecId] = {SubtitleCodecId.SRT, SubtitleCodecId.ASS}


def _hdr_class(video: VideoInfo) -> str:
    """Short HDR class label used in the per-file analyze summary."""
    if video.hdr.is_dolby_vision:
        bl_map = {1: "HDR10", 2: "SDR", 4: "HLG"}
        compat = (
            int(video.hdr.dv_bl_compatibility) if video.hdr.dv_bl_compatibility else 0
        )
        bl = bl_map.get(compat, "none")
        prof = video.hdr.dv_profile if video.hdr.dv_profile is not None else "?"
        return f"DV P{prof} (BL={bl})"
    if video.color_transfer == "smpte2084":
        return "HDR10"
    if video.color_transfer == "arib-std-b67":
        return "HLG"
    return "SDR"


def _format_analyze_summary(
    video: VideoInfo,
    audio_tracks: list[Track],
    subtitle_tracks: list[Track],
) -> str:
    """Build the one-line summary shown for a successfully analyzed file.

    Format: ``codec WxH FPSfps HDR-CLASS [interlaced], N audio (langs), N subs``
    """
    fps = video.fps_num // video.fps_den if video.fps_den else 0
    hdr = _hdr_class(video)
    parts = [
        video.codec_name,
        f"{video.width}x{video.height}",
        f"{fps}fps",
        hdr,
    ]
    if video.interlaced:
        parts.append("(interlaced)")
    head = " ".join(parts)

    audio_langs = sorted({t.language for t in audio_tracks if t.language})
    audio = (
        f"{len(audio_tracks)} audio ({','.join(audio_langs)})"
        if audio_langs
        else f"{len(audio_tracks)} audio"
    )
    subs = f"{len(subtitle_tracks)} subs"
    return f"{head}, {audio}, {subs}"


class Analyzer:
    def __init__(self, prober: Prober, *, force: bool = False) -> None:
        self._prober = prober
        self._force = force

    def analyze(
        self,
        scan_result: ScanResult,
        *,
        on_progress: Callable[[float], None] | None = None,
    ) -> AnalysisOutcome:
        """Probe main file + satellites. Parse video/audio/subtitle/attachments.

        Returns an ``AnalysisOutcome``:
        - ``DONE`` with the built ``Movie`` and a one-line summary on success.
        - ``SKIPPED`` (movie ``None``) when the file is already encoded, has no
          video stream, or uses unsupported codecs.
        - ``FAILED`` (movie ``None``) when probing/parsing fails or the content
          is HDR10+.

        ``on_progress`` (when supplied) receives the analyze-phase fraction in
        ``[0, 1]``: it advances by one step per completed heavy stage (idet, then
        one per profileable audio track) and is called once more with ``1.0``
        when analysis finishes. Files with no heavy stages report only the final
        ``1.0``. Used by the parallel pipeline to drive a smooth batch bar.
        """
        main_file = scan_result.main_file
        output_path = scan_result.output_path
        name = main_file.name

        # Check skip conditions on the output path
        encoder_tag = self._prober.get_encoder_tag(main_file)
        skip, reason = should_skip_file(output_path, encoder_tag, force=self._force)
        if skip:
            logger.info("Skipping %s: %s", name, reason)
            return AnalysisOutcome(None, AnalyzeStatus.SKIPPED, reason)

        # Probe the main file
        try:
            probe_data = self._prober.probe(main_file)
        except (OSError, RuntimeError, ValueError):
            logger.exception("Failed to probe %s", main_file)
            return AnalysisOutcome(None, AnalyzeStatus.FAILED, "probe failed")

        streams = probe_data.get("streams", [])
        format_data = probe_data.get("format", {})
        chapters = probe_data.get("chapters", [])

        # Find the video stream
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        if not video_streams:
            logger.warning("No video stream found in %s, skipping", name)
            return AnalysisOutcome(None, AnalyzeStatus.SKIPPED, "no video stream")

        video_stream = video_streams[0]
        try:
            video_info = self._parse_video_info(video_stream, format_data, main_file)
        except (KeyError, ValueError, IndexError, TypeError):
            logger.exception("Failed to parse video info for %s", main_file)
            return AnalysisOutcome(None, AnalyzeStatus.FAILED, "parse failed")

        # HDR10+ not supported
        if video_info.hdr.is_hdr10_plus:
            return AnalysisOutcome(None, AnalyzeStatus.FAILED, "HDR10+ not supported")
        # DV content proceeds to planning (no skip)

        # Parse tracks from main file
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
        subtitle_streams = [s for s in streams if s.get("codec_type") == "subtitle"]
        attachment_streams = [s for s in streams if s.get("codec_type") == "attachment"]

        audio_tracks = self._parse_audio_tracks(audio_streams, main_file)
        subtitle_tracks = self._parse_subtitle_tracks(subtitle_streams, main_file)
        attachments = self._parse_attachments(attachment_streams, main_file)

        has_chapters = bool(chapters)

        # Process satellite files
        for sat_path in scan_result.satellite_files:
            ext = sat_path.suffix.lower()
            if ext in {".srt", ".ass", ".ssa", ".sup"}:
                # Treat as external subtitle
                sat_track = self._parse_external_subtitle(sat_path, len(subtitle_tracks))
                if sat_track is not None:
                    subtitle_tracks.append(sat_track)
            elif ext in {".ac3", ".dts", ".eac3", ".flac", ".m4a", ".mp3", ".wav"}:
                # Treat as external audio
                sat_track = self._parse_external_audio(sat_path, len(audio_tracks))
                if sat_track is not None:
                    audio_tracks.append(sat_track)

        # Check for unknown codecs
        codec_warning = check_unsupported_codecs(audio_tracks, subtitle_tracks)
        if codec_warning:
            logger.warning("Skipping %s: %s", main_file.name, codec_warning)
            return AnalysisOutcome(None, AnalyzeStatus.SKIPPED, codec_warning)

        # Detect interlace: ffprobe field_order + idet when ambiguous
        # Use r_frame_rate (field rate) for interlace detection, not avg_frame_rate
        field_order_raw = video_stream.get("field_order")
        r_fps_str = video_stream.get("r_frame_rate", "0/1")
        if "/" in r_fps_str:
            r_parts = r_fps_str.split("/")
            r_num = int(r_parts[0])
            r_den = int(r_parts[1]) if len(r_parts) > 1 and int(r_parts[1]) != 0 else 1
        else:
            r_num = int(float(r_fps_str))
            r_den = 1
        fps = r_num / r_den if r_den else 0.0

        # Plan the heavy analysis stages so the batch progress bar advances across
        # them: idet (when needed) plus one per profileable audio track. ``_emit``
        # reports the running fraction after each stage completes.
        idet_will_run = needs_idet(field_order_raw, fps, video_info.height)
        n_profileable = sum(1 for t in audio_tracks if t.channels in _PROFILEABLE_CHANNEL_COUNTS)
        total_stages = (1 if idet_will_run else 0) + n_profileable
        stages_done = 0

        def _emit() -> None:
            if on_progress is not None:
                on_progress(stages_done / total_stages)

        idet_ratio = 0.0
        if idet_will_run:
            try:
                idet_ratio = self._prober.run_idet(main_file, video_info.duration_s)
                logger.debug("%s: idet ratio %.3f", name, idet_ratio)
            except (OSError, RuntimeError, ValueError) as exc:
                logger.warning("idet failed for %s: %s", name, exc)
            stages_done += 1
            _emit()
        video_info.interlaced = should_deinterlace(field_order_raw, fps, idet_ratio, video_info.height)
        if video_info.interlaced:
            logger.info("%s: interlaced content detected", name)
        else:
            logger.debug("%s: progressive content", name)

        # Detect forced subtitles (in-place mutation)
        detect_forced_subtitles(subtitle_tracks)

        # Profile audio tracks with supported channel layouts; attach a
        # classification verdict. Errors are swallowed and leave audio_profile=None.
        for track in audio_tracks:
            if track.channels not in _PROFILEABLE_CHANNEL_COUNTS:
                continue
            logger.info(
                "Profiling audio track %d (%s %s %dch)",
                track.index, track.codec_name, track.language, track.channels,
            )
            try:
                metrics = self._prober.profile_audio_track(
                    path=main_file,
                    stream_index=track.index,
                    channels=track.channels,
                    duration_s=video_info.duration_s,
                )
                track.audio_profile = classify_audio(metrics)
            except Exception as exc:  # noqa: BLE001 -- fail-soft by design
                logger.warning(
                    "profile_audio_track failed for track %d: %s", track.index, exc,
                )
            else:
                logger.info(
                    "Profiled track %d: %s (score %d)",
                    track.index, track.audio_profile.verdict.value, track.audio_profile.score,
                )
            stages_done += 1
            _emit()

        if on_progress is not None:
            on_progress(1.0)

        file_size = main_file.stat().st_size

        movie = Movie(
            main_file=main_file,
            satellite_files=scan_result.satellite_files,
            video=video_info,
            audio_tracks=audio_tracks,
            subtitle_tracks=subtitle_tracks,
            attachments=attachments,
            has_chapters=has_chapters,
            file_size=file_size,
        )
        summary = _format_analyze_summary(video_info, audio_tracks, subtitle_tracks)
        return AnalysisOutcome(movie, AnalyzeStatus.DONE, summary)

    def _parse_video_info(self, stream: dict[str, Any], format_data: dict[str, Any], path: Path) -> VideoInfo:
        """Extract VideoInfo from ffprobe stream data."""
        index = stream.get("index", 0)
        codec_name = stream.get("codec_name", "unknown")
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        pixel_area = width * height

        # FPS: prefer avg_frame_rate, fall back to r_frame_rate. ffprobe
        # reports an unknown rate as "0/0"; treat that as absent so a re-encode
        # never carries fps_num=0 (which would make mkvmerge default the muxed
        # track to 25 fps and drift the audio out of sync).
        fps_str = stream.get("avg_frame_rate")
        if not fps_str or fps_str == "0/0":
            fps_str = stream.get("r_frame_rate", "25/1")
        if not fps_str or fps_str == "0/0":
            fps_str = "25/1"
        if "/" in fps_str:
            parts = fps_str.split("/")
            fps_num = int(parts[0])
            fps_den = int(parts[1]) if len(parts) > 1 and int(parts[1]) != 0 else 1
        else:
            fps_num = int(float(fps_str))
            fps_den = 1

        # Duration: from stream, fallback to format
        duration_s = 0.0
        if "duration" in stream:
            with contextlib.suppress(ValueError, TypeError):
                duration_s = float(stream["duration"])
        if duration_s == 0.0 and "duration" in format_data:
            with contextlib.suppress(ValueError, TypeError):
                duration_s = float(format_data["duration"])

        # Interlace — set to False here, real detection via idet in analyze()
        interlaced = False

        # Color info
        color_primaries_raw = stream.get("color_primaries")
        color_transfer_raw = stream.get("color_transfer")
        color_matrix_raw = stream.get("color_space")
        color_range_raw = stream.get("color_range")
        pix_fmt = stream.get("pix_fmt", "yuv420p")

        # Bitrate: from stream, fallback to format
        bitrate = 0
        if "bit_rate" in stream:
            with contextlib.suppress(ValueError, TypeError):
                bitrate = int(stream["bit_rate"])
        if bitrate == 0 and "bit_rate" in format_data:
            with contextlib.suppress(ValueError, TypeError):
                bitrate = int(format_data["bit_rate"])

        # HDR metadata — merge stream-level (DOVI configuration record) and
        # frame-level (Mastering display / Content light / DV RPU) side data.
        # For PQ/HLG content the two layers are complementary: MKV remuxes of
        # UHD Blu-Ray DV P7 carry DOVI config at packet level but MDCV/CLL only
        # at frame level. SDR content skips the frame probe as an optimization.
        stream_side_data: list[dict[str, Any]] = stream.get("side_data_list") or []
        frame_side_data: list[dict[str, Any]] = []
        if color_transfer_raw in ("smpte2084", "arib-std-b67"):
            frame_side_data = self._prober.probe_hdr_side_data(path)
        side_data = [*stream_side_data, *frame_side_data]
        hdr = detect_hdr(stream, side_data)

        # SAR (sample aspect ratio)
        sar_num, sar_den = 1, 1
        sar_raw = stream.get("sample_aspect_ratio", "1:1")
        if sar_raw and ":" in sar_raw:
            sar_parts = sar_raw.split(":")
            try:
                sar_num = int(sar_parts[0])
                sar_den = int(sar_parts[1])
            except (ValueError, IndexError):
                sar_num, sar_den = 1, 1

        return VideoInfo(
            index=index,
            codec_name=codec_name,
            width=width,
            height=height,
            pixel_area=pixel_area,
            fps_num=fps_num,
            fps_den=fps_den,
            duration_s=duration_s,
            interlaced=interlaced,
            color_matrix_raw=color_matrix_raw,
            color_range=color_range_raw,
            color_transfer=color_transfer_raw,
            color_primaries=color_primaries_raw,
            pix_fmt=pix_fmt,
            hdr=hdr,
            source_file=path,
            bitrate=bitrate,
            sar_num=sar_num,
            sar_den=sar_den,
        )

    def _parse_audio_tracks(self, streams: list[dict[str, Any]], path: Path) -> list[Track]:
        """Parse audio tracks. codec_name + profile -> AudioCodecId.
        delay_ms from start_pts (see _detect_audio_delay).
        """
        tracks: list[Track] = []
        for stream in streams:
            index = stream.get("index", 0)
            codec_name = stream.get("codec_name", "unknown")
            profile = stream.get("profile")
            codec_id = parse_audio_codec(codec_name, profile)

            tags = stream.get("tags", {})
            language = tags.get("language", "und")
            title = tags.get("title", "") or ""

            disposition = stream.get("disposition", {})
            is_default = bool(disposition.get("default", 0))
            is_forced = bool(disposition.get("forced", 0))

            channels = stream.get("channels")
            channel_layout = stream.get("channel_layout")
            sample_rate = stream.get("sample_rate")
            if sample_rate is not None:
                try:
                    sample_rate = int(sample_rate)
                except (ValueError, TypeError):
                    sample_rate = None

            # Bitrate: from stream tags or stream bit_rate
            bitrate: int | None = None
            raw_bitrate = stream.get("bit_rate") or tags.get("BPS") or tags.get("BPS-eng")
            if raw_bitrate is not None:
                with contextlib.suppress(ValueError, TypeError):
                    bitrate = int(raw_bitrate)

            delay_ms = self._detect_audio_delay(stream)

            track = Track(
                index=index,
                track_type=TrackType.AUDIO,
                codec_name=codec_name,
                codec_id=codec_id,
                language=language,
                title=title,
                is_default=is_default,
                is_forced=is_forced,
                source_file=path,
                channels=channels,
                channel_layout=channel_layout,
                bitrate=bitrate,
                sample_rate=sample_rate,
                delay_ms=delay_ms,
                profile=profile,
            )
            tracks.append(track)
        return tracks

    def _parse_subtitle_tracks(self, streams: list[dict[str, Any]], path: Path) -> list[Track]:
        """Parse subtitles. Detect encoding via charset_normalizer for text subs."""
        tracks: list[Track] = []
        for stream in streams:
            index = stream.get("index", 0)
            codec_name = stream.get("codec_name", "unknown")
            codec_id = parse_subtitle_codec(codec_name)

            tags = stream.get("tags", {})
            language = tags.get("language", "und")
            title = tags.get("title", "") or ""

            disposition = stream.get("disposition", {})
            is_default = bool(disposition.get("default", 0))
            is_forced = bool(disposition.get("forced", 0))

            # Frame/caption count for forced detection
            num_frames: int | None = None
            raw_frames = tags.get("NUMBER_OF_FRAMES") or tags.get("NUMBER_OF_FRAMES-eng")
            if raw_frames is not None:
                with contextlib.suppress(ValueError, TypeError):
                    num_frames = int(raw_frames)

            num_captions: int | None = None

            track = Track(
                index=index,
                track_type=TrackType.SUBTITLE,
                codec_name=codec_name,
                codec_id=codec_id,
                language=language,
                title=title,
                is_default=is_default,
                is_forced=is_forced,
                source_file=path,
                num_frames=num_frames,
                num_captions=num_captions,
                encoding=None,
            )
            tracks.append(track)
        return tracks

    def _parse_external_subtitle(self, path: Path, base_index: int) -> Track | None:
        """Parse an external subtitle file (satellite). Detect encoding for text subs."""
        ext = path.suffix.lower()
        codec_name_map = {
            ".srt": "subrip",
            ".ass": "ass",
            ".ssa": "ass",
            ".sup": "hdmv_pgs_subtitle",
        }
        codec_name = codec_name_map.get(ext, "unknown")
        codec_id = parse_subtitle_codec(codec_name)

        # Infer language from filename stem (e.g. movie.rus.srt -> rus)
        stem = path.stem
        # stem may have video stem as prefix; look for language code after dot
        language = "und"
        parts = stem.split(".")
        if len(parts) >= len(["stem", "lang"]):
            # last part or second-to-last may be language
            for part in reversed(parts[1:]):
                if len(part) == _ISO_639_3_LENGTH and part.isalpha():
                    language = part.lower()
                    break

        # Is_forced from filename
        name_lower = path.name.lower()
        is_forced = any(kw in name_lower for kw in ["forced", "форсир", "forsed"])

        # Encoding detection for text subs
        encoding: str | None = None
        if codec_id in _TEXT_SUBTITLE_CODECS:
            encoding = self._detect_text_encoding(path)

        return Track(
            index=base_index,
            track_type=TrackType.SUBTITLE,
            codec_name=codec_name,
            codec_id=codec_id,
            language=language,
            title="",
            is_default=False,
            is_forced=is_forced,
            source_file=path,
            num_frames=None,
            num_captions=None,
            encoding=encoding,
        )

    def _parse_external_audio(self, path: Path, base_index: int) -> Track | None:
        """Parse an external audio satellite file."""
        try:
            probe_data = self._prober.probe(path)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Failed to probe satellite audio %s: %s", path, exc)
            return None

        streams = probe_data.get("streams", [])
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
        if not audio_streams:
            return None

        tracks = self._parse_audio_tracks(audio_streams, path)
        if not tracks:
            return None

        # Use the first track but override the index with base_index
        track = tracks[0]
        track.index = base_index
        return track

    def _parse_attachments(self, streams: list[dict[str, Any]], path: Path) -> list[Attachment]:
        """Extract attachments (fonts) from streams."""
        attachments: list[Attachment] = []
        for stream in streams:
            tags = stream.get("tags", {})
            filename = tags.get("filename", "")
            mime_type = tags.get("mimetype", "") or tags.get("mime_type", "")
            if filename:
                attachments.append(
                    Attachment(
                        filename=filename,
                        mime_type=mime_type,
                        source_file=path,
                    )
                )
        return attachments

    def _detect_audio_delay(self, stream: dict[str, Any]) -> int:
        """Determine audio delay in milliseconds.

        For MKV containers start_pts is already in ms (time_base=1/1000),
        so use it directly as integer ms.
        Fallback: start_time (in seconds) * 1000.
        Default: 0.
        """
        if "start_pts" in stream:
            return int(stream["start_pts"])
        if "start_time" in stream:
            return int(float(stream["start_time"]) * 1000)
        return 0

    def _detect_text_encoding(self, path: Path) -> str | None:
        """Detect encoding of a text subtitle file using charset_normalizer."""
        try:
            result = _from_path(path)
            best = result.best()
        except (OSError, ValueError) as exc:
            logger.debug("Encoding detection failed for %s: %s", path, exc)
            return None
        else:
            if best is None:
                return None
            return best.encoding
