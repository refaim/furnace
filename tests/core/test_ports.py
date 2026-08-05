from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any

from furnace.core.audio_profile import AudioMetrics
from furnace.core.models import CropRect
from furnace.core.ports import AudioExtractor, Prober
from furnace.core.progress import ProgressSample


class _MinimalProber:
    def probe(self, path: Path) -> dict[str, Any]:  # noqa: ARG002
        return {}

    def detect_crop(
        self,
        path: Path,  # noqa: ARG002
        duration_s: float,  # noqa: ARG002
        *,
        interlaced: bool = False,  # noqa: ARG002
        is_dvd: bool = False,  # noqa: ARG002
        hdr_transfer: str | None = None,  # noqa: ARG002
    ) -> CropRect | None:
        return None

    def get_encoder_tag(self, path: Path) -> str | None:  # noqa: ARG002
        return None

    def run_idet(self, path: Path, duration_s: float) -> float:  # noqa: ARG002
        return 0.0

    def probe_hdr_side_data(self, path: Path) -> list[dict[str, Any]]:  # noqa: ARG002
        return []

    def sample_repeat_pict(self, path: Path, duration_s: float) -> list[int]:  # noqa: ARG002
        return []

    def sample_grain(
        self,
        path: Path,  # noqa: ARG002
        duration_s: float,  # noqa: ARG002
        *,
        hdr_transfer: str | None = None,  # noqa: ARG002
    ) -> list[float]:
        return []

    def sample_field_pairing(self, path: Path) -> tuple[int, int]:  # noqa: ARG002
        return (0, 0)

    def profile_audio_track(
        self,
        path: Path,  # noqa: ARG002
        stream_index: int,  # noqa: ARG002
        channels: int,
        duration_s: float,  # noqa: ARG002
    ) -> AudioMetrics:
        return AudioMetrics(
            channels=channels,
            rms_l=-20.0,
            rms_r=-20.0,
            rms_c=None,
            rms_lfe=None,
            rms_ls=None,
            rms_rs=None,
            rms_lb=None,
            rms_rb=None,
            corr_lr=0.0,
            corr_ls_l=None,
            corr_rs_r=None,
            corr_ls_rs=None,
            corr_lb_ls=None,
            corr_rb_rs=None,
        )


def test_prober_has_profile_audio_track() -> None:
    assert hasattr(Prober, "profile_audio_track")
    assert callable(Prober.profile_audio_track)


def test_prober_profile_audio_track_signature() -> None:
    sig = inspect.signature(Prober.profile_audio_track)
    params = sig.parameters
    assert list(params) == [
        "self",
        "path",
        "stream_index",
        "channels",
        "duration_s",
        "channel_layout",
        "on_progress",
    ]
    assert params["on_progress"].default is None
    assert params["channel_layout"].default is None

    hints = typing.get_type_hints(Prober.profile_audio_track)
    assert hints["path"] is Path
    assert hints["stream_index"] is int
    assert hints["channels"] is int
    assert hints["channel_layout"] == str | None
    assert hints["duration_s"] is float
    assert hints["on_progress"] == Callable[[ProgressSample], None] | None
    assert hints["return"] is AudioMetrics


def test_prober_has_sample_repeat_pict() -> None:
    assert hasattr(Prober, "sample_repeat_pict")
    assert callable(Prober.sample_repeat_pict)


def test_prober_sample_repeat_pict_signature() -> None:
    sig = inspect.signature(Prober.sample_repeat_pict)
    assert list(sig.parameters) == ["self", "path", "duration_s"]

    hints = typing.get_type_hints(Prober.sample_repeat_pict)
    assert hints["path"] is Path
    assert hints["duration_s"] is float
    assert hints["return"] == list[int]


def test_prober_has_sample_grain() -> None:
    assert hasattr(Prober, "sample_grain")
    assert callable(Prober.sample_grain)


def test_prober_sample_grain_signature() -> None:
    sig = inspect.signature(Prober.sample_grain)
    assert list(sig.parameters) == ["self", "path", "duration_s", "hdr_transfer"]
    assert sig.parameters["hdr_transfer"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["hdr_transfer"].default is None

    hints = typing.get_type_hints(Prober.sample_grain)
    assert hints["path"] is Path
    assert hints["duration_s"] is float
    assert hints["hdr_transfer"] == (str | None)
    assert hints["return"] == list[float]


def test_prober_detect_crop_signature_includes_hdr_transfer() -> None:
    sig = inspect.signature(Prober.detect_crop)
    params = sig.parameters
    assert list(params) == [
        "self",
        "path",
        "duration_s",
        "interlaced",
        "is_dvd",
        "hdr_transfer",
        "on_progress",
    ]
    assert params["hdr_transfer"].default is None

    hints = typing.get_type_hints(Prober.detect_crop)
    assert hints["hdr_transfer"] == str | None


def test_minimal_prober_satisfies_runtime_checkable_protocol() -> None:
    stub = _MinimalProber()
    assert isinstance(stub, Prober)
    metrics = stub.profile_audio_track(Path("/dev/null"), 0, 6, 60.0)
    assert metrics.channels == 6


def test_minimal_prober_method_surface() -> None:
    stub = _MinimalProber()
    assert stub.probe(Path("/dev/null")) == {}
    assert stub.detect_crop(Path("/dev/null"), 60.0) is None
    assert stub.detect_crop(Path("/dev/null"), 60.0, interlaced=True, is_dvd=True) is None
    assert (
        stub.detect_crop(
            Path("/dev/null"),
            60.0,
            interlaced=True,
            is_dvd=True,
            hdr_transfer="smpte2084",
        )
        is None
    )
    assert stub.get_encoder_tag(Path("/dev/null")) is None
    assert stub.run_idet(Path("/dev/null"), 60.0) == 0.0
    assert stub.probe_hdr_side_data(Path("/dev/null")) == []
    assert stub.sample_repeat_pict(Path("/dev/null"), 60.0) == []
    assert stub.sample_grain(Path("/dev/null"), 60.0) == []
    assert stub.sample_field_pairing(Path("/dev/null")) == (0, 0)


class _MinimalAudioExtractor:
    def extract_track(
        self,
        input_path: Path,  # noqa: ARG002
        stream_index: int,  # noqa: ARG002
        output_path: Path,  # noqa: ARG002
        on_progress: Callable[[ProgressSample], None] | None = None,  # noqa: ARG002
    ) -> int:
        return 0

    def ffmpeg_to_wav(
        self,
        input_path: Path,  # noqa: ARG002
        stream_index: int,  # noqa: ARG002
        output_wav: Path,  # noqa: ARG002
        on_progress: Callable[[ProgressSample], None] | None = None,  # noqa: ARG002
    ) -> int:
        return 0

    def decode_full_wav(
        self,
        input_path: Path,  # noqa: ARG002
        stream_index: int,  # noqa: ARG002
        output_wav: Path,  # noqa: ARG002
        *,
        disable_drc: bool = False,  # noqa: ARG002
        on_progress: Callable[[ProgressSample], None] | None = None,  # noqa: ARG002
    ) -> int:
        return 0

    def stereo_to_mono_wav(
        self,
        input_path: Path,  # noqa: ARG002
        stream_index: int,  # noqa: ARG002
        output_wav: Path,  # noqa: ARG002
        delay_ms: int,  # noqa: ARG002
        on_progress: Callable[[ProgressSample], None] | None = None,  # noqa: ARG002
    ) -> int:
        return 0


def test_audio_extractor_has_stereo_to_mono_wav() -> None:
    assert hasattr(AudioExtractor, "stereo_to_mono_wav")
    assert callable(AudioExtractor.stereo_to_mono_wav)


def test_audio_extractor_stereo_to_mono_wav_signature() -> None:
    sig = inspect.signature(AudioExtractor.stereo_to_mono_wav)
    params = sig.parameters
    assert list(params) == [
        "self",
        "input_path",
        "stream_index",
        "output_wav",
        "delay_ms",
        "on_progress",
    ]

    hints = typing.get_type_hints(AudioExtractor.stereo_to_mono_wav)
    assert hints["input_path"] is Path
    assert hints["stream_index"] is int
    assert hints["output_wav"] is Path
    assert hints["delay_ms"] is int
    assert hints["return"] is int


def test_minimal_audio_extractor_satisfies_runtime_checkable_protocol(
    tmp_path: Path,
) -> None:
    stub = _MinimalAudioExtractor()
    assert isinstance(stub, AudioExtractor)
    rc = stub.stereo_to_mono_wav(Path("/dev/null"), 1, tmp_path / "out.wav", 0)
    assert rc == 0


def test_minimal_audio_extractor_method_surface(tmp_path: Path) -> None:
    stub = _MinimalAudioExtractor()
    assert stub.extract_track(Path("/dev/null"), 0, tmp_path / "o.thd") == 0
    assert stub.ffmpeg_to_wav(Path("/dev/null"), 0, tmp_path / "o.wav") == 0
    assert stub.decode_full_wav(Path("/dev/null"), 0, tmp_path / "o.wav", disable_drc=True) == 0
    assert stub.stereo_to_mono_wav(Path("/dev/null"), 0, tmp_path / "o.wav", -50) == 0
