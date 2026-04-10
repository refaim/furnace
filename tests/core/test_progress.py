from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from furnace.core.progress import ProgressSample, ProgressTracker, TrackerSnapshot


class TestProgressSample:
    def test_fraction_only(self) -> None:
        s = ProgressSample(fraction=0.5)
        assert s.fraction == 0.5
        assert s.processed_s is None
        assert s.speed is None

    def test_processed_s_only(self) -> None:
        s = ProgressSample(processed_s=60.0)
        assert s.processed_s == 60.0
        assert s.fraction is None

    def test_speed_is_optional(self) -> None:
        s = ProgressSample(fraction=0.5, speed=2.5)
        assert s.speed == 2.5

    def test_frozen(self) -> None:
        s = ProgressSample(fraction=0.5)
        with pytest.raises(FrozenInstanceError):
            s.fraction = 0.6  # type: ignore[misc]


class TestTrackerSnapshot:
    def test_defaults(self) -> None:
        snap = TrackerSnapshot(fraction=0.0, speed=None, eta_s=None)
        assert snap.fraction == 0.0
        assert snap.speed is None
        assert snap.eta_s is None

    def test_frozen(self) -> None:
        snap = TrackerSnapshot(fraction=0.5, speed=None, eta_s=None)
        with pytest.raises(FrozenInstanceError):
            snap.fraction = 0.6  # type: ignore[misc]


class TestProgressTrackerEmpty:
    def test_empty_snapshot(self) -> None:
        t = ProgressTracker()
        snap = t.snapshot()
        assert snap.fraction == 0.0
        assert snap.speed is None
        assert snap.eta_s is None


class TestProgressTrackerFractionSamples:
    def test_single_fraction_sample(self) -> None:
        t = ProgressTracker()
        t.add(ProgressSample(fraction=0.5), wall_time=0.0)
        assert t.snapshot().fraction == 0.5

    def test_fraction_clamped_low(self) -> None:
        t = ProgressTracker()
        t.add(ProgressSample(fraction=-0.1), wall_time=0.0)
        assert t.snapshot().fraction == 0.0

    def test_fraction_clamped_high(self) -> None:
        t = ProgressTracker()
        t.add(ProgressSample(fraction=1.5), wall_time=0.0)
        assert t.snapshot().fraction == 1.0


class TestProgressTrackerProcessedSamples:
    def test_processed_with_total(self) -> None:
        t = ProgressTracker(total_s=120.0)
        t.add(ProgressSample(processed_s=60.0), wall_time=0.0)
        assert t.snapshot().fraction == 0.5

    def test_processed_without_total_dropped(self) -> None:
        t = ProgressTracker(total_s=None)
        t.add(ProgressSample(processed_s=60.0), wall_time=0.0)
        # Sample has no recoverable fraction → silently dropped
        assert t.snapshot().fraction == 0.0

    def test_empty_sample_dropped(self) -> None:
        t = ProgressTracker(total_s=120.0)
        t.add(ProgressSample(), wall_time=0.0)
        assert t.snapshot().fraction == 0.0

    def test_processed_with_zero_total_dropped(self) -> None:
        t = ProgressTracker(total_s=0.0)
        t.add(ProgressSample(processed_s=60.0), wall_time=0.0)
        assert t.snapshot().fraction == 0.0


class TestProgressTrackerReset:
    def test_reset_clears_samples(self) -> None:
        t = ProgressTracker(total_s=120.0)
        t.add(ProgressSample(processed_s=60.0), wall_time=0.0)
        t.reset()
        assert t.snapshot().fraction == 0.0

    def test_reset_allows_fresh_start(self) -> None:
        t = ProgressTracker()
        t.add(ProgressSample(fraction=0.8), wall_time=0.0)
        t.reset()
        t.add(ProgressSample(fraction=0.1), wall_time=10.0)
        assert t.snapshot().fraction == 0.1


class TestProgressTrackerSmoothedSpeed:
    def test_no_speed_samples(self) -> None:
        t = ProgressTracker()
        t.add(ProgressSample(fraction=0.2), wall_time=0.0)
        assert t.snapshot().speed is None

    def test_single_speed_sample(self) -> None:
        t = ProgressTracker()
        t.add(ProgressSample(fraction=0.2, speed=2.0), wall_time=0.0)
        assert t.snapshot().speed == 2.0

    def test_average_of_last_five(self) -> None:
        t = ProgressTracker()
        speeds = [1.0, 2.0, 3.0, 4.0, 5.0]
        for i, s in enumerate(speeds):
            t.add(ProgressSample(fraction=(i + 1) / 10, speed=s), wall_time=float(i))
        # Average of 1..5 = 3.0
        assert t.snapshot().speed == 3.0

    def test_older_samples_dropped_from_average(self) -> None:
        t = ProgressTracker()
        # Add six samples with speed 1..6; only last five should count (avg = 4.0)
        for i in range(6):
            t.add(ProgressSample(fraction=(i + 1) / 10, speed=float(i + 1)), wall_time=float(i))
        assert t.snapshot().speed == 4.0

    def test_none_speed_samples_ignored(self) -> None:
        t = ProgressTracker()
        t.add(ProgressSample(fraction=0.1), wall_time=0.0)
        t.add(ProgressSample(fraction=0.2, speed=3.0), wall_time=1.0)
        t.add(ProgressSample(fraction=0.3), wall_time=2.0)
        # Only one speed sample exists → average of that one
        assert t.snapshot().speed == 3.0


class TestProgressTrackerEta:
    def test_empty_eta(self) -> None:
        t = ProgressTracker()
        assert t.snapshot().eta_s is None

    def test_single_sample_no_eta(self) -> None:
        t = ProgressTracker()
        t.add(ProgressSample(fraction=0.5), wall_time=0.0)
        # Need ≥2 samples to compute rate → None
        assert t.snapshot().eta_s is None

    def test_linear_progress_eta(self) -> None:
        t = ProgressTracker()
        t.add(ProgressSample(fraction=0.0), wall_time=0.0)
        t.add(ProgressSample(fraction=0.5), wall_time=10.0)
        # Rate = 0.05/s, remaining = 0.5, ETA = 10s
        eta = t.snapshot().eta_s
        assert eta is not None
        assert abs(eta - 10.0) < 1e-9

    def test_stalled_progress_no_eta(self) -> None:
        t = ProgressTracker()
        t.add(ProgressSample(fraction=0.5), wall_time=0.0)
        t.add(ProgressSample(fraction=0.5), wall_time=10.0)
        # No forward progress → None
        assert t.snapshot().eta_s is None

    def test_fraction_at_one_eta_zero(self) -> None:
        t = ProgressTracker()
        t.add(ProgressSample(fraction=0.0), wall_time=0.0)
        t.add(ProgressSample(fraction=1.0), wall_time=5.0)
        eta = t.snapshot().eta_s
        assert eta is not None
        assert eta == 0.0

    def test_eta_uses_last_five_samples_window(self) -> None:
        """ETA rate should be computed from the window of last 5 samples,
        not from the first sample ever added."""
        t = ProgressTracker()
        # Add a slow first sample, then 5 fast samples.
        # If _eta used the first sample, rate = 0.6 / 5 = 0.12/s, ETA ≈ 3.33s
        # If _eta uses the last-5 window (indices 1..5),
        #   rate = (0.6 - 0.2) / (5 - 1) = 0.1/s
        # Remaining = 0.4, so ETA = 4.0 seconds.
        t.add(ProgressSample(fraction=0.0), wall_time=0.0)
        t.add(ProgressSample(fraction=0.2), wall_time=1.0)
        t.add(ProgressSample(fraction=0.3), wall_time=2.0)
        t.add(ProgressSample(fraction=0.4), wall_time=3.0)
        t.add(ProgressSample(fraction=0.5), wall_time=4.0)
        t.add(ProgressSample(fraction=0.6), wall_time=5.0)
        eta = t.snapshot().eta_s
        assert eta is not None
        assert abs(eta - 4.0) < 1e-9
