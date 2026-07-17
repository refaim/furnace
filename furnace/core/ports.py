from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from furnace.core.progress import ProgressSample

from .audio_profile import AudioMetrics
from .models import (
    METRIC_NAMES,
    AnalyzeStatus,
    CropRect,
    DiscTitle,
    DiscType,
    DownmixMode,
    DvMode,
    EncodeResult,
    MetricPool,
    MetricScores,
    VideoParams,
)


@runtime_checkable
class Prober(Protocol):
    def probe(self, path: Path) -> dict[str, Any]: ...

    def detect_crop(
        self,
        path: Path,
        duration_s: float,
        *,
        interlaced: bool = False,
        is_dvd: bool = False,
        hdr_transfer: str | None = None,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> CropRect | None: ...

    def get_encoder_tag(self, path: Path) -> str | None: ...

    def run_idet(
        self,
        path: Path,
        duration_s: float,
        *,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> float: ...

    def probe_hdr_side_data(self, path: Path) -> list[dict[str, Any]]: ...

    def sample_repeat_pict(self, path: Path, duration_s: float) -> list[int]: ...

    def sample_field_pairing(self, path: Path) -> tuple[int, int]: ...

    def sample_grain(self, path: Path, duration_s: float) -> list[float]: ...

    def profile_audio_track(
        self,
        path: Path,
        stream_index: int,
        channels: int,
        duration_s: float,
        *,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> AudioMetrics: ...


@runtime_checkable
class Encoder(Protocol):
    def encode(
        self,
        input_path: Path,
        output_path: Path,
        video_params: VideoParams,
        *,
        on_progress: Callable[[ProgressSample], None] | None = None,
        rpu_path: Path | None = None,
        cq_override: int | None = None,
    ) -> EncodeResult: ...


@runtime_checkable
class InlineQualityProbe(Protocol):
    def probe(
        self,
        input_path: Path,
        output_path: Path,
        video_params: VideoParams,
        *,
        qvbr: int,
        metric: str,
    ) -> float: ...


@runtime_checkable
class PerceptualMetrics(Protocol):
    def measure(
        self,
        reference: Path,
        distorted: Path,
        *,
        matrix: str,
        fps_num: int,
        fps_den: int,
        pool: MetricPool = MetricPool.MEAN,
        metrics: frozenset[str] = METRIC_NAMES,
    ) -> MetricScores: ...


@runtime_checkable
class VideoCopier(Protocol):
    def copy_video(
        self,
        input_path: Path,
        output_path: Path,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int: ...


@runtime_checkable
class WindowExtractor(Protocol):
    def extract_window(
        self,
        input_path: Path,
        output_path: Path,
        *,
        start_s: float,
        frames: int,
    ) -> int: ...

    def build_reference(
        self,
        input_path: Path,
        output_path: Path,
        video_params: VideoParams,
    ) -> int: ...

    def window_bitrates(self, source: Path, window_s: float) -> list[tuple[float, float]]: ...


@runtime_checkable
class DoviProcessor(Protocol):
    def extract_rpu(
        self,
        input_path: Path,
        output_rpu: Path,
        mode: DvMode,
    ) -> int: ...


@runtime_checkable
class AudioExtractor(Protocol):
    def extract_track(
        self,
        input_path: Path,
        stream_index: int,
        output_path: Path,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int: ...

    def ffmpeg_to_wav(
        self,
        input_path: Path,
        stream_index: int,
        output_wav: Path,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int: ...

    def stereo_to_mono_wav(
        self,
        input_path: Path,
        stream_index: int,
        output_wav: Path,
        delay_ms: int,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int: ...


@runtime_checkable
class AudioDecoder(Protocol):
    def denormalize(
        self,
        input_path: Path,
        output_path: Path,
        delay_ms: int,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int: ...

    def decode_lossless(
        self,
        input_path: Path,
        output_path: Path,
        delay_ms: int,
        on_progress: Callable[[ProgressSample], None] | None = None,
        *,
        downmix: DownmixMode | None = None,
    ) -> int: ...


@runtime_checkable
class AacEncoder(Protocol):
    def encode_aac(
        self,
        input_wav: Path,
        output_m4a: Path,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int: ...


@runtime_checkable
class Muxer(Protocol):
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
    ) -> int: ...


@runtime_checkable
class Tagger(Protocol):
    def set_encoder_tag(self, mkv_path: Path, tag_value: str, encoder_settings: str | None = None) -> int: ...


@runtime_checkable
class Cleaner(Protocol):
    def clean(
        self,
        input_path: Path,
        output_path: Path,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int: ...


@runtime_checkable
class Previewer(Protocol):
    def preview_audio(self, video_path: Path, audio_path: Path, stream_index: int) -> None: ...

    def preview_subtitle(self, video_path: Path, sub_path: Path, stream_index: int) -> None: ...


@runtime_checkable
class DiscDemuxerPort(Protocol):
    def list_titles(self, disc_path: Path) -> list[DiscTitle]: ...

    def demux_title(
        self,
        disc_path: Path,
        title_num: int,
        output_dir: Path,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> list[Path]: ...


@runtime_checkable
class PcmTranscoder(Protocol):
    def transcode_to_flac(
        self,
        input_path: Path,
        output_path: Path,
        on_progress: Callable[[ProgressSample], None] | None = None,
    ) -> int: ...


@runtime_checkable
class PlanReporter(Protocol):
    def detect_disc(self, disc_type: DiscType, rel_path: str) -> None: ...
    def detect_disc_titles_done(self, n_titles: int) -> None: ...

    def demux_disc_cached(self, label: str) -> None: ...
    def demux_disc_start(self, label: str) -> None: ...
    def demux_title_start(self, title_num: int) -> None: ...
    def demux_title_substep(self, label: str, *, has_progress: bool) -> None: ...
    def demux_title_progress(self, fraction: float) -> None: ...
    def demux_title_done(self) -> None: ...
    def demux_title_failed(self, reason: str) -> None: ...

    def scan_file(self, name: str) -> None: ...
    def scan_skipped(self, name: str, reason: str) -> None: ...

    def analyze_batch_start(self, total: int) -> None: ...
    def analyze_batch_progress(self, completed: float) -> None: ...
    def analyze_batch_item(self, name: str, detail: str, *, status: AnalyzeStatus) -> None: ...
    def analyze_batch_finish(self) -> None: ...

    def plan_file_start(self, name: str) -> None: ...
    def plan_file_done(self, summary: str) -> None: ...

    def plan_saved(self, path: Path, n_jobs: int) -> None: ...
    def interrupted(self) -> None: ...

    def pause(self) -> None: ...
    def resume(self) -> None: ...
