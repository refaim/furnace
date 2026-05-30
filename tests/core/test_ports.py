"""Protocol-shape tests for ``furnace.core.ports``.

These tests pin down the public surface of ``Prober`` so that adapters and
fakes can rely on the method set without importing the concrete adapter.
They also assert runtime_checkable conformance for a minimal in-test stub —
any drift in the Protocol signature shows up here immediately.
"""
from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any

from furnace.core.audio_profile import AudioMetrics
from furnace.core.models import CropRect
from furnace.core.ports import AudioAnalyzer, AudioExtractor, Prober
from furnace.core.progress import ProgressSample


class _MinimalProber:
    """Concrete no-op implementation of every Prober method.

    Used to verify that the ``@runtime_checkable`` Protocol accepts an
    independently-declared class that merely provides the expected methods.
    """

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

    def profile_audio_track(
        self,
        path: Path,  # noqa: ARG002
        stream_index: int,  # noqa: ARG002
        channels: int,
        duration_s: float,  # noqa: ARG002
    ) -> AudioMetrics:
        return AudioMetrics(
            channels=channels,
            rms_l=-20.0, rms_r=-20.0,
            rms_c=None, rms_lfe=None, rms_ls=None, rms_rs=None,
            rms_lb=None, rms_rb=None,
            corr_lr=0.0,
            corr_ls_l=None, corr_rs_r=None, corr_ls_rs=None,
            corr_lb_ls=None, corr_rb_rs=None,
        )


def test_prober_has_profile_audio_track() -> None:
    assert hasattr(Prober, "profile_audio_track")
    assert callable(Prober.profile_audio_track)


def test_prober_profile_audio_track_signature() -> None:
    sig = inspect.signature(Prober.profile_audio_track)
    params = sig.parameters
    # self + four positional args + on_progress (positional-with-default)
    assert list(params) == [
        "self",
        "path",
        "stream_index",
        "channels",
        "duration_s",
        "on_progress",
    ]
    # ``on_progress`` is opt-in: defaults to None so existing callers keep working.
    assert params["on_progress"].default is None

    # Annotations are stringified by `from __future__ import annotations`, so
    # resolve them with `typing.get_type_hints` before comparing identities.
    hints = typing.get_type_hints(Prober.profile_audio_track)
    assert hints["path"] is Path
    assert hints["stream_index"] is int
    assert hints["channels"] is int
    assert hints["duration_s"] is float
    assert hints["on_progress"] == Callable[[ProgressSample], None] | None
    assert hints["return"] is AudioMetrics


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
    """Exercise every method of the stub so coverage stays at 100%.

    The stub exists purely to demonstrate Protocol conformance; if a new
    method lands on Prober and the stub forgets to implement it, this test
    is where it gets caught.
    """
    stub = _MinimalProber()
    assert stub.probe(Path("/dev/null")) == {}
    assert stub.detect_crop(Path("/dev/null"), 60.0) is None
    assert stub.detect_crop(Path("/dev/null"), 60.0, interlaced=True, is_dvd=True) is None
    assert stub.detect_crop(
        Path("/dev/null"), 60.0,
        interlaced=True, is_dvd=True, hdr_transfer="smpte2084",
    ) is None
    assert stub.get_encoder_tag(Path("/dev/null")) is None
    assert stub.run_idet(Path("/dev/null"), 60.0) == 0.0
    assert stub.probe_hdr_side_data(Path("/dev/null")) == []


class _MinimalAudioExtractor:
    """Concrete no-op implementation of every AudioExtractor method.

    Mirrors ``_MinimalProber`` — proves that the runtime_checkable Protocol
    accepts an independently-declared class with the expected surface, and
    locks the signature of ``stereo_to_mono_wav`` in place.
    """

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
    # self + four positional args + optional on_progress callback
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
    """Exercise every method of the AudioExtractor stub so coverage stays
    at 100% — same rationale as ``test_minimal_prober_method_surface``.
    """
    stub = _MinimalAudioExtractor()
    assert stub.extract_track(Path("/dev/null"), 0, tmp_path / "o.thd") == 0
    assert stub.ffmpeg_to_wav(Path("/dev/null"), 0, tmp_path / "o.wav") == 0
    assert stub.stereo_to_mono_wav(Path("/dev/null"), 0, tmp_path / "o.wav", -50) == 0


class _MinimalAudioAnalyzer:
    """Concrete no-op implementation of every AudioAnalyzer method.

    Mirrors the other minimal stubs — proves runtime_checkable Protocol
    conformance and locks the signature in place.
    """

    def first_second_rms_db(self, audio_file: Path) -> float | None:  # noqa: ARG002
        return -100.0


def test_audio_analyzer_has_first_second_rms_db() -> None:
    assert hasattr(AudioAnalyzer, "first_second_rms_db")
    assert callable(AudioAnalyzer.first_second_rms_db)


def test_audio_analyzer_first_second_rms_db_signature() -> None:
    sig = inspect.signature(AudioAnalyzer.first_second_rms_db)
    params = sig.parameters
    assert list(params) == ["self", "audio_file"]

    hints = typing.get_type_hints(AudioAnalyzer.first_second_rms_db)
    assert hints["audio_file"] is Path
    assert hints["return"] == float | None


def test_minimal_audio_analyzer_satisfies_runtime_checkable_protocol() -> None:
    stub = _MinimalAudioAnalyzer()
    assert isinstance(stub, AudioAnalyzer)
    assert stub.first_second_rms_db(Path("/dev/null")) == -100.0


def test_minimal_audio_analyzer_method_surface() -> None:
    stub = _MinimalAudioAnalyzer()
    assert stub.first_second_rms_db(Path("audio.dts")) == -100.0
