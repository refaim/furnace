from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .models import CropRect, VideoParams


@runtime_checkable
class Prober(Protocol):
    """Extract metadata from media files."""

    def probe(self, path: Path) -> dict[str, Any]:
        """Return raw ffprobe JSON (streams + format + chapters)."""
        ...

    def detect_crop(self, path: Path, duration_s: float) -> CropRect | None:
        """Run cropdetect, return detected values (before alignment)."""
        ...

    def get_encoder_tag(self, path: Path) -> str | None:
        """Read MKV tag ENCODER. None if absent."""
        ...


@runtime_checkable
class Encoder(Protocol):
    """Video encoding via ffmpeg/NVENC.

    Note: compute_vmaf() will be added in Phase 6, not Phase 2.
    In Phase 2 the Encoder Protocol contains only encode().
    """

    def encode(
        self,
        input_path: Path,
        output_path: Path,
        video_params: VideoParams,
        source_size: int,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> int:
        """Encode video. Returns return code (0 = ok).

        source_size is passed for mid-encoding bloat check (see 12.11).
        on_progress callback receives (progress_pct, status_line).
        """
        ...


@runtime_checkable
class AudioExtractor(Protocol):
    """Extract audio tracks from container and decode exotic codecs.
    Implemented by FFmpegAdapter."""

    def extract_track(
        self,
        input_path: Path,
        stream_index: int,
        output_path: Path,
        codec: str,
    ) -> int:
        """Extract audio track from container to a separate file.
        ffmpeg -i input -map 0:{index} -c:a copy output
        """
        ...

    def ffmpeg_to_wav(
        self,
        input_path: Path,
        stream_index: int,
        output_wav: Path,
    ) -> int:
        """Decode exotic codec to WAV via ffmpeg.
        ffmpeg -i input -map 0:{index} -f wav -rf64 auto output.wav
        """
        ...


@runtime_checkable
class AudioDecoder(Protocol):
    """Denormalization and lossless audio decoding via eac3to.
    Implemented by Eac3toAdapter."""

    def denormalize(self, input_path: Path, output_path: Path, delay_ms: int) -> int:
        """eac3to denormalize (AC3/EAC3/DTS core)."""
        ...

    def decode_lossless(self, input_path: Path, output_path: Path, delay_ms: int) -> int:
        """eac3to decode lossless -> WAV."""
        ...


@runtime_checkable
class AacEncoder(Protocol):
    """Encode WAV to AAC via qaac64.
    Implemented by QaacAdapter."""

    def encode_aac(self, input_wav: Path, output_m4a: Path) -> int:
        """qaac64 encode WAV -> AAC."""
        ...


@runtime_checkable
class Muxer(Protocol):
    """Assemble the final MKV."""

    def mux(
        self,
        video_path: Path,
        audio_files: list[tuple[Path, dict[str, Any]]],
        subtitle_files: list[tuple[Path, dict[str, Any]]],
        attachments: list[tuple[Path, str, str]],
        chapters_source: Path | None,
        output_path: Path,
        furnace_version: str,
        video_meta: dict[str, Any] | None = None,
    ) -> int:
        """Assemble MKV. Returns return code.

        audio_files: list of (path, {language, default, delay_ms})
        subtitle_files: list of (path, {language, default, forced, encoding})
        attachments: list of (path, filename, mime_type)
        video_meta: optional dict with color/HDR metadata for container-level flags
            {color_range, color_primaries, color_transfer, hdr_max_cll, hdr_max_fall}
        """
        ...


@runtime_checkable
class Tagger(Protocol):
    """Set MKV tags via mkvpropedit."""

    def set_encoder_tag(self, mkv_path: Path, tag_value: str) -> int:
        """Set global ENCODER tag. Returns return code."""
        ...


@runtime_checkable
class Cleaner(Protocol):
    """Optimize MKV index."""

    def clean(self, input_path: Path, output_path: Path) -> int:
        """mkclean. Returns return code."""
        ...


@runtime_checkable
class Previewer(Protocol):
    """Preview tracks in mpv."""

    def preview_audio(self, video_path: Path, audio_path: Path, stream_index: int) -> None:
        """Open mpv with the specified audio."""
        ...

    def preview_subtitle(self, video_path: Path, sub_path: Path, stream_index: int) -> None:
        """Open mpv with the specified subtitles."""
        ...
