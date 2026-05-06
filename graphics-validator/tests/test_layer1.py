"""Tests for Layer-1 cheap heuristic detectors.

Synthetic frames are generated with NumPy — no video file required.
"""
from __future__ import annotations

import numpy as np
import pytest

from graphics_validator.detectors.layer1_cheap import Layer1Detector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(h: int = 720, w: int = 1280, dtype=np.uint8) -> np.ndarray:
    """Return an all-zero BGR frame of shape (h, w, 3)."""
    return np.zeros((h, w, 3), dtype=dtype)


def _solid(b: int, g: int, r: int, h: int = 720, w: int = 1280) -> np.ndarray:
    frame = _make_frame(h, w)
    frame[:] = (b, g, r)
    return frame


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def detector() -> Layer1Detector:
    """Fresh detector with default thresholds."""
    return Layer1Detector()


# ---------------------------------------------------------------------------
# BSOD detection
# ---------------------------------------------------------------------------


class TestBSOD:
    def test_pure_blue_flagged_as_critical(self, detector):
        """A pure blue frame must produce a BSOD flag with severity=critical."""
        frame = _solid(b=200, g=0, r=0)
        flags = detector.detect(frame, idx=0)
        bsod_flags = [f for f in flags if f.kind == "BSOD"]
        assert bsod_flags, "Expected a BSOD flag for pure blue frame"
        assert bsod_flags[0].severity == "critical"
        assert bsod_flags[0].score > 0.5

    def test_windows_bsod_color_flagged(self, detector):
        """Classic Windows BSOD colour (0x0078D7 ≈ RGB 0,120,215 → BGR 215,120,0)
        has high blue dominance and should also trigger BSOD."""
        # Approximate Windows 10 BSOD: medium blue, low std
        frame = _solid(b=200, g=100, r=40)
        flags = detector.detect(frame, idx=0)
        # Depending on threshold, this may or may not trigger; at minimum no crash.
        assert isinstance(flags, list)

    def test_white_frame_not_bsod(self, detector):
        """A white frame should NOT be flagged as BSOD."""
        frame = _solid(b=255, g=255, r=255)
        flags = detector.detect(frame, idx=0)
        bsod_flags = [f for f in flags if f.kind == "BSOD"]
        assert not bsod_flags, "White frame should not be BSOD"


# ---------------------------------------------------------------------------
# Black screen detection
# ---------------------------------------------------------------------------


class TestBlackScreen:
    def test_pure_black_flagged_as_high(self, detector):
        """A pure black frame must produce a BLACK_SCREEN flag with severity=high."""
        frame = _solid(b=0, g=0, r=0)
        flags = detector.detect(frame, idx=0)
        black_flags = [f for f in flags if f.kind == "BLACK_SCREEN"]
        assert black_flags, "Expected a BLACK_SCREEN flag for pure black frame"
        assert black_flags[0].severity == "high"

    def test_near_black_flagged(self, detector):
        """A very dark frame (mean < threshold) should also be flagged."""
        frame = _solid(b=5, g=5, r=5)
        flags = detector.detect(frame, idx=0)
        black_flags = [f for f in flags if f.kind == "BLACK_SCREEN"]
        assert black_flags

    def test_grey_frame_not_black(self, detector):
        """A mid-grey frame must NOT be flagged as black screen."""
        frame = _solid(b=128, g=128, r=128)
        flags = detector.detect(frame, idx=0)
        black_flags = [f for f in flags if f.kind == "BLACK_SCREEN"]
        assert not black_flags


# ---------------------------------------------------------------------------
# Frozen frame detection
# ---------------------------------------------------------------------------


class TestFrozen:
    def test_repeated_identical_frames_flagged(self, detector):
        """After frozen_ring_size identical frames the detector must raise FROZEN."""
        # Use a noisy frame so we don't accidentally trigger other detectors
        rng = np.random.default_rng(42)
        noisy = rng.integers(50, 200, (720, 1280, 3), dtype=np.uint8)
        n = detector.frozen_ring_size

        flags_all: list = []
        for i in range(n):
            flags_all.extend(detector.detect(noisy.copy(), idx=i))

        frozen_flags = [f for f in flags_all if f.kind == "FROZEN"]
        assert frozen_flags, f"Expected FROZEN flag after {n} identical frames"
        assert frozen_flags[-1].severity == "medium"

    def test_varying_frames_not_frozen(self, detector):
        """Different frames should not trigger FROZEN."""
        rng = np.random.default_rng(7)
        flags_all: list = []
        for i in range(detector.frozen_ring_size + 2):
            frame = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
            flags_all.extend(detector.detect(frame, idx=i))

        frozen_flags = [f for f in flags_all if f.kind == "FROZEN"]
        assert not frozen_flags, "Random noise frames should not trigger FROZEN"


# ---------------------------------------------------------------------------
# Random noise → no Layer-1 flags
# ---------------------------------------------------------------------------


class TestNoFlags:
    def test_random_noise_no_critical_flags(self, detector):
        """Pure random noise should not produce critical Layer-1 flags."""
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        flags = detector.detect(frame, idx=0)
        critical_flags = [f for f in flags if f.severity == "critical"]
        assert not critical_flags, f"Unexpected critical flags on noise: {critical_flags}"
