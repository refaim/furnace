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
from furnace.core.ports import AudioExtractor, Prober
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
    # self + four positional args
    assert list(params) == ["self", "path", "stream_index", "channels", "duration_s"]

    # Annotations are stringified by `from __future__ import annotations`, so
    # resolve them with `typing.get_type_hints` before comparing identities.
    hints = typing.get_type_hints(Prober.profile_audio_track)
    assert hints["path"] is Path
    assert hints["stream_index"] is int
    assert hints["channels"] is int
    assert hints["duration_s"] is float
    assert hints["return"] is AudioMetrics


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
    assert stub.get_encoder_tag(Path("/dev/null")) is None
    assert stub.run_idet(Path("/dev/null"), 60.0) == 0.0
    assert stub.probe_hdr_side_data(Path("/dev/null")) == []


class _MinimalAudioExtractor:
    """Concrete no-op implementation of every AudioExtractor method.

    Mirrors ``_MinimalProber`` — proves that the runtime_checkable Protocol
    accepts an independently-declared class with the expected surface, and
    locks the signature of ``downmix_to_mono_wav`` (Task 10) in place.
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

    def downmix_to_mono_wav(
        self,
        input_path: Path,  # noqa: ARG002
        stream_index: int,  # noqa: ARG002
        channels: int,  # noqa: ARG002
        output_wav: Path,  # noqa: ARG002
        delay_ms: int,  # noqa: ARG002
    ) -> int:
        return 0


def test_audio_extractor_has_downmix_to_mono_wav() -> None:
    assert hasattr(AudioExtractor, "downmix_to_mono_wav")
    assert callable(AudioExtractor.downmix_to_mono_wav)


def test_audio_extractor_downmix_to_mono_wav_signature() -> None:
    sig = inspect.signature(AudioExtractor.downmix_to_mono_wav)
    params = sig.parameters
    # self + five positional args, in fixed order
    assert list(params) == [
        "self",
        "input_path",
        "stream_index",
        "channels",
        "output_wav",
        "delay_ms",
    ]

    hints = typing.get_type_hints(AudioExtractor.downmix_to_mono_wav)
    assert hints["input_path"] is Path
    assert hints["stream_index"] is int
    assert hints["channels"] is int
    assert hints["output_wav"] is Path
    assert hints["delay_ms"] is int
    assert hints["return"] is int


def test_minimal_audio_extractor_satisfies_runtime_checkable_protocol(
    tmp_path: Path,
) -> None:
    stub = _MinimalAudioExtractor()
    assert isinstance(stub, AudioExtractor)
    rc = stub.downmix_to_mono_wav(Path("/dev/null"), 1, 6, tmp_path / "out.wav", 0)
    assert rc == 0


def test_minimal_audio_extractor_method_surface(tmp_path: Path) -> None:
    """Exercise every method of the AudioExtractor stub so coverage stays
    at 100% — same rationale as ``test_minimal_prober_method_surface``.
    """
    stub = _MinimalAudioExtractor()
    assert stub.extract_track(Path("/dev/null"), 0, tmp_path / "o.thd") == 0
    assert stub.ffmpeg_to_wav(Path("/dev/null"), 0, tmp_path / "o.wav") == 0
    assert stub.downmix_to_mono_wav(Path("/dev/null"), 0, 2, tmp_path / "o.wav", -50) == 0
