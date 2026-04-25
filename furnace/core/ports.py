from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from furnace.core.progress import ProgressSample

from .audio_profile import AudioMetrics
from .models import CropRect, DiscTitle, DiscType, DownmixMode, DvMode, EncodeResult, VideoParams


@runtime_checkable
class Prober(Protocol):
    """Extract metadata from media files."""

    def probe(self, path: Path) -> dict[str, Any]:
        """Return raw ffprobe JSON (streams + format + chapters)."""
        ...

    def detect_crop(
        self,
        path: Path,
        duration_s: float,
        *,
        interlaced: bool = False,
        is_dvd: bool = False,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> CropRect | None:
        """Run cropdetect, return detected values (before alignment).

        ``on_progress`` is called after each sample point.
        """
        ...

    def get_encoder_tag(self, path: Path) -> str | None:
        """Read MKV tag ENCODER. None if absent."""
        ...

    def run_idet(
        self,
        path: Path,
        duration_s: float,
        *,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> float:
        """Run idet analysis. Returns interlaced frame ratio (0.0 to 1.0).

        ``on_progress`` is called after each sample point with a fraction
        (``points_done / total_points``).
        """
        ...

    def probe_hdr_side_data(self, path: Path) -> list[dict[str, Any]]:
        """Read side_data_list from the first video frame."""
        ...

    def profile_audio_track(
        self,
        path: Path,
        stream_index: int,
        channels: int,
        duration_s: float,
        *,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> AudioMetrics:
        """Sample PCM windows from an audio stream, compute per-channel RMS
        and pairwise correlations, and return raw measurements.

        channels must be 2, 6, or 8; other counts raise ValueError.
        duration_s is used to pick sample offsets.
        ``on_progress`` is called after each window decode with a fraction.

        Raises RuntimeError if no windows decoded successfully.
        """
        ...


@runtime_checkable
class Encoder(Protocol):
    """Video encoding via NVEncC."""

    def encode(
        self,
        input_path: Path,
        output_path: Path,
        video_params: VideoParams,
        *,
        on_progress: Callable[[ProgressSample], None] | None = None,
        vmaf_enabled: bool = False,
        rpu_path: Path | None = None,
    ) -> EncodeResult:
        """Encode video. Returns EncodeResult with return code, settings, and optional metrics."""
        ...


@runtime_checkable
class DoviProcessor(Protocol):
    """Extract/convert Dolby Vision RPU metadata via dovi_tool."""

    def extract_rpu(
        self,
        input_path: Path,
        output_rpu: Path,
        mode: DvMode,
    ) -> int:
        """Extract RPU from HEVC stream.

        mode=COPY: extract as-is (no -m flag).
        mode=TO_8_1: convert P7 FEL -> P8.1 (-m 2).
        Returns exit code.
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
        on_progress: Callable[[ProgressSample], None] | None = None,
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
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int:
        """Decode exotic codec to WAV via ffmpeg.
        ffmpeg -i input -map 0:{index} -f wav -rf64 auto output.wav
        """
        ...

    def downmix_to_mono_wav(
        self,
        input_path: Path,
        stream_index: int,
        channels: int,
        output_wav: Path,
        delay_ms: int,
    ) -> int:
        """Produce a mono WAV from a 2/6/8-channel audio stream via ffmpeg's
        pan filter. Multichannel sources use an ITU-R BS.775 / Dolby Lo
        downmix (FC=0.707, FL/FR=0.5, surrounds=0.354) with alimiter peak
        protection; LFE is excluded. Stereo averages L and R.

        delay_ms is applied via -af adelay (pad leading silence) when positive
        or by trimming when negative. Returns ffmpeg exit code.
        """
        ...


@runtime_checkable
class AudioDecoder(Protocol):
    """Denormalization and lossless audio decoding via eac3to.
    Implemented by Eac3toAdapter."""

    def denormalize(
        self,
        input_path: Path,
        output_path: Path,
        delay_ms: int,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int:
        """eac3to denormalize (AC3/EAC3/DTS core)."""
        ...

    def decode_lossless(
        self,
        input_path: Path,
        output_path: Path,
        delay_ms: int,
        on_progress: Callable[[ProgressSample], None] | None = None,
        *,
        downmix: DownmixMode | None = None,
    ) -> int:
        """eac3to decode lossless -> WAV. With downmix set, also emits the
        corresponding eac3to flags."""
        ...


@runtime_checkable
class AacEncoder(Protocol):
    """Encode WAV to AAC via qaac64.
    Implemented by QaacAdapter."""

    def encode_aac(
        self,
        input_wav: Path,
        output_m4a: Path,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int:
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
        video_meta: dict[str, Any] | None = None,
        on_progress: Callable[[ProgressSample], None] | None = None,
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

    def set_encoder_tag(self, mkv_path: Path, tag_value: str, encoder_settings: str | None = None) -> int:
        """Set global ENCODER tag (and ENCODER_SETTINGS if provided). Returns return code."""
        ...


@runtime_checkable
class Cleaner(Protocol):
    """Optimize MKV index."""

    def clean(
        self,
        input_path: Path,
        output_path: Path,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int:
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


@runtime_checkable
class DiscDemuxerPort(Protocol):
    """Demux disc structures (DVD/Blu-ray) to MKV."""

    def list_titles(self, disc_path: Path) -> list[DiscTitle]:
        """List titles from a disc structure."""
        ...

    def demux_title(
        self,
        disc_path: Path,
        title_num: int,
        output_dir: Path,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> list[Path]:
        """Demux one title to MKV file(s) in output_dir. Returns paths to created files."""
        ...


@runtime_checkable
class PcmTranscoder(Protocol):
    """Transcode uncompressed PCM (Wave64) to FLAC.

    Used by DiscDemuxer to normalize eac3to's Wave64 output to a format
    mkvmerge can mux (FLAC). Lossless — the resulting stream decodes
    bit-identical to the source PCM. Implemented by Eac3toAdapter.
    """

    def transcode_to_flac(
        self,
        input_path: Path,
        output_path: Path,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int:
        """Transcode a PCM input (Wave64 or WAV) to FLAC. Returns exit code."""
        ...


@runtime_checkable
class PlanReporter(Protocol):
    """Structured terminal output for ``furnace plan``.

    State is implicit: after ``*_file_start(name)`` or ``demux_title_start(n)``,
    all subsequent micro-op / progress / done calls apply to that latest-started
    item. The ``plan`` pipeline is strictly serial — only one file/title is
    active at a time — so this is unambiguous.
    """

    # Detect
    def detect_disc(self, disc_type: DiscType, rel_path: str) -> None: ...

    # Demux
    def demux_disc_cached(self, label: str) -> None: ...
    def demux_disc_start(self, label: str) -> None: ...
    def demux_title_start(self, title_num: int) -> None: ...
    def demux_title_substep(self, label: str, *, has_progress: bool) -> None: ...
    def demux_title_progress(self, fraction: float) -> None: ...
    def demux_title_done(self) -> None: ...
    def demux_title_failed(self, reason: str) -> None: ...

    # Scan
    def scan_file(self, name: str) -> None: ...
    def scan_skipped(self, name: str, reason: str) -> None: ...

    # Analyze
    def analyze_file_start(self, name: str) -> None: ...
    def analyze_microop(self, label: str, *, has_progress: bool) -> None: ...
    def analyze_progress(self, fraction: float) -> None: ...
    def analyze_file_done(self, summary: str) -> None: ...
    def analyze_file_failed(self, reason: str) -> None: ...
    def analyze_file_skipped(self, reason: str) -> None: ...

    # Plan
    def plan_file_start(self, name: str) -> None: ...
    def plan_microop(self, label: str, *, has_progress: bool) -> None: ...
    def plan_progress(self, fraction: float) -> None: ...
    def plan_file_done(self, summary: str) -> None: ...

    # Final
    def plan_saved(self, path: Path, n_jobs: int) -> None: ...
    def interrupted(self) -> None: ...

    # Lifecycle (for interactive Textual TUI pauses)
    def pause(self) -> None: ...
    def resume(self) -> None: ...
