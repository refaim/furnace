from __future__ import annotations

from collections import deque
from dataclasses import dataclass

_MIN_SAMPLES_FOR_RATE = 2


@dataclass(frozen=True)
class ProgressSample:
    processed_s: float | None = None
    fraction: float | None = None
    speed: float | None = None


@dataclass(frozen=True)
class TrackerSnapshot:
    fraction: float
    speed: float | None
    eta_s: float | None


class ProgressTracker:
    def __init__(self, total_s: float | None = None) -> None:
        self._total_s = total_s
        self._samples: deque[tuple[float, float, float | None]] = deque(maxlen=20)

    def reset(self) -> None:
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
        return list(self._samples)[-5:]

    def _smoothed_speed(self) -> float | None:
        recent = [s for _, _, s in self._recent() if s is not None]
        if not recent:
            return None
        return sum(recent) / len(recent)

    def _eta(self) -> float | None:
        recent = self._recent()
        if len(recent) < _MIN_SAMPLES_FOR_RATE:
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
