"""Pure progress-tracking primitives shared by all long-running adapters.

`ProgressSample` is the raw DTO returned by adapter parsers. `ProgressTracker`
accumulates samples, normalizes them against an optional total duration, and
computes a `TrackerSnapshot` (immutable view) that the UI consumes. No I/O;
wall time is passed as a parameter on every `add()` so the tracker is fully
unit-testable against synthetic clocks.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressSample:
    """One progress update decoded from a tool's output.

    At least one of `processed_s` or `fraction` should be set; if both are
    set, `fraction` takes priority. `speed` is optional and copied through
    only when the tool emits it directly (e.g., ffmpeg `speed=2.5x`,
    qaac `(30.5x)`).
    """
    processed_s: float | None = None
    fraction: float | None = None
    speed: float | None = None


@dataclass(frozen=True)
class TrackerSnapshot:
    """Immutable view of tracker state, passed thread-safely to the UI."""
    fraction: float
    speed: float | None
    eta_s: float | None


class ProgressTracker:
    """Stateful progress accumulator. Time-pure: wall time is a parameter."""

    def __init__(self, total_s: float | None = None) -> None:
        self._total_s = total_s
        self._samples: deque[tuple[float, float, float | None]] = deque(maxlen=20)

    def reset(self) -> None:
        """Drop all samples. Call at sub-phase boundaries so the next set of
        samples starts a fresh 0 → 100% bar without backwards jumps."""
        self._samples.clear()

    def add(self, sample: ProgressSample, wall_time: float) -> None:
        if sample.fraction is not None:
            f = max(0.0, min(1.0, sample.fraction))
        elif sample.processed_s is not None and self._total_s:
            f = max(0.0, min(1.0, sample.processed_s / self._total_s))
        else:
            return
        self._samples.append((wall_time, f, sample.speed))

    def snapshot(self) -> TrackerSnapshot:
        if not self._samples:
            return TrackerSnapshot(fraction=0.0, speed=None, eta_s=None)
        _, frac, _ = self._samples[-1]
        return TrackerSnapshot(
            fraction=frac,
            speed=self._smoothed_speed(),
            eta_s=self._eta(),
        )

    def _recent(self) -> list[tuple[float, float, float | None]]:
        """Return the last 5 samples as a list."""
        return list(self._samples)[-5:]

    def _smoothed_speed(self) -> float | None:
        recent = [s for _, _, s in self._recent() if s is not None]
        if not recent:
            return None
        return sum(recent) / len(recent)

    def _eta(self) -> float | None:
        """ETA in seconds, derived from the wall-time rate of change of
        fraction over the last 5 samples. Returns None when history is
        insufficient (<2 samples) or progress has stalled."""
        recent = self._recent()
        if len(recent) < 2:
            return None
        start_wall, start_frac, _ = recent[0]
        now_wall, now_frac, _ = recent[-1]
        d_wall = now_wall - start_wall
        d_frac = now_frac - start_frac
        if d_wall <= 0 or d_frac <= 0:
            return None
        rate = d_frac / d_wall
        remaining = 1.0 - now_frac
        return remaining / rate
