"""Layer-1 cheap heuristic detectors (target <2 ms/frame at 1280×720).

Detects: BSOD, black screen, frozen frames, tearing, NaN/Inf pixel blobs,
and missing-texture colors (magenta / black checker).
All checks use NumPy and OpenCV only — no deep learning.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import numpy as np

from graphics_validator.detectors.base import Detector, Flag, Severity


class Layer1Detector:
    """Cheap per-frame heuristic checks.

    Parameters
    ----------
    bsod_blue_dominance:
        Blue channel fraction threshold for BSOD detection.
    bsod_max_std:
        Maximum pixel std-dev for a BSOD (screens are mostly uniform).
    black_mean_threshold:
        Maximum mean pixel value for a black-screen flag.
    black_max_std:
        Maximum std-dev for a black screen.
    frozen_ring_size:
        Number of consecutive identical frame hashes that triggers a frozen flag.
    tearing_sobel_z_thresh:
        Z-score threshold for horizontal tear-line detection via Sobel-Y.
    missing_texture_threshold:
        Fraction of pixels that must be magenta/checker color to flag.
    """

    def __init__(
        self,
        bsod_blue_dominance: float = 0.55,
        bsod_max_std: float = 45.0,
        black_mean_threshold: float = 20.0,
        black_max_std: float = 30.0,
        frozen_ring_size: int = 8,
        tearing_sobel_z_thresh: float = 4.0,
        missing_texture_threshold: float = 0.05,
    ) -> None:
        self.bsod_blue_dominance = bsod_blue_dominance
        self.bsod_max_std = bsod_max_std
        self.black_mean_threshold = black_mean_threshold
        self.black_max_std = black_max_std
        self.frozen_ring_size = frozen_ring_size
        self.tearing_sobel_z_thresh = tearing_sobel_z_thresh
        self.missing_texture_threshold = missing_texture_threshold

        # Ring buffer of recent frame hashes for frozen detection
        self._hash_ring: deque[int] = deque(maxlen=frozen_ring_size)

    # ------------------------------------------------------------------
    # Detector protocol
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray, idx: int, **kw: Any) -> list[Flag]:
        """Run all Layer-1 checks on *frame* and return any flags raised."""
        flags: list[Flag] = []

        flags.extend(self._check_bsod(frame))
        flags.extend(self._check_black(frame))
        flags.extend(self._check_frozen(frame))
        flags.extend(self._check_tearing(frame))
        flags.extend(self._check_nan_blobs(frame))
        flags.extend(self._check_missing_texture(frame))

        return flags

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_bsod(self, frame: np.ndarray) -> list[Flag]:
        """Detect Windows BSOD: dominant blue channel + low spatial variance within blue."""
        mean_bgr = frame.mean(axis=(0, 1))  # shape (3,)
        total = mean_bgr.sum() + 1e-6
        blue_frac = float(mean_bgr[0]) / total  # OpenCV: BGR

        # Use per-channel spatial std (not across channels) to check uniformity
        b_std = float(frame[:, :, 0].std())

        if blue_frac >= self.bsod_blue_dominance and b_std < self.bsod_max_std:
            score = min(1.0, blue_frac + (self.bsod_max_std - b_std) / self.bsod_max_std)
            return [
                Flag(
                    layer="layer1",
                    kind="BSOD",
                    severity="critical",
                    score=round(score, 4),
                    detail={"blue_dominance": round(blue_frac, 4), "std": round(b_std, 2)},
                )
            ]
        return []

    def _check_black(self, frame: np.ndarray) -> list[Flag]:
        """Detect black/blank screen: very low mean + low spatial variance."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_val = float(gray.mean())
        std = float(gray.std())

        if mean_val < self.black_mean_threshold and std < self.black_max_std:
            score = 1.0 - mean_val / (self.black_mean_threshold + 1e-6)
            return [
                Flag(
                    layer="layer1",
                    kind="BLACK_SCREEN",
                    severity="high",
                    score=round(min(1.0, score), 4),
                    detail={"mean": round(mean_val, 2), "std": round(std, 2)},
                )
            ]
        return []

    def _check_frozen(self, frame: np.ndarray) -> list[Flag]:
        """Detect frozen frame via perceptual hash ring buffer."""
        # Downsample to 16×16 grayscale and compute a simple hash
        small = cv2.resize(frame, (16, 16), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        h = int(np.packbits(gray > gray.mean()).tobytes().hex(), 16)

        self._hash_ring.append(h)

        if (
            len(self._hash_ring) == self._hash_ring.maxlen
            and len(set(self._hash_ring)) == 1
        ):
            return [
                Flag(
                    layer="layer1",
                    kind="FROZEN",
                    severity="medium",
                    score=1.0,
                    detail={"consecutive_identical": self.frozen_ring_size},
                )
            ]
        return []

    def _check_tearing(self, frame: np.ndarray) -> list[Flag]:
        """Detect horizontal screen-tear via Sobel-Y energy spike across rows."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        row_energy = np.abs(sobel_y).mean(axis=1)  # mean energy per row

        mean_e = float(row_energy.mean())
        std_e = float(row_energy.std()) + 1e-6
        z_scores = (row_energy - mean_e) / std_e
        max_z = float(z_scores.max())

        if max_z >= self.tearing_sobel_z_thresh:
            tear_row = int(np.argmax(z_scores))
            score = min(1.0, (max_z - self.tearing_sobel_z_thresh) / self.tearing_sobel_z_thresh + 0.5)
            return [
                Flag(
                    layer="layer1",
                    kind="TEARING",
                    severity="medium",
                    score=round(score, 4),
                    detail={"tear_row": tear_row, "z_score": round(max_z, 2)},
                )
            ]
        return []

    def _check_nan_blobs(self, frame: np.ndarray) -> list[Flag]:
        """Detect pure-white or pure-black contiguous blobs in complex scenes.

        In rendered content NaN/Inf GPU values typically saturate to pure white
        (0xFFFFFF) or pure black (0x000000). We look for large connected regions
        of those extreme colors in scenes that are otherwise complex.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        global_std = float(gray.std())

        # Only flag in complex scenes (std > 20) to avoid false positives on
        # legitimate all-black intros or cut scenes.
        if global_std < 20.0:
            return []

        flags: list[Flag] = []
        h, w = gray.shape
        frame_area = h * w

        for val, kind in [(255, "NAN_WHITE_BLOB"), (0, "NAN_BLACK_BLOB")]:
            mask = np.uint8(gray == val) * 255
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            for i in range(1, num_labels):
                blob_area = int(stats[i, cv2.CC_STAT_AREA])
                frac = blob_area / frame_area
                # Flag blobs that cover >1% of the frame in an otherwise complex scene
                if frac > 0.01:
                    flags.append(
                        Flag(
                            layer="layer1",
                            kind=kind,
                            severity="high",
                            score=round(min(1.0, frac * 10), 4),
                            detail={"blob_frac": round(frac, 4), "blob_area_px": blob_area},
                        )
                    )
        return flags

    def _check_missing_texture(self, frame: np.ndarray) -> list[Flag]:
        """Detect missing-texture sentinel colors: magenta and black/purple checker.

        Unreal Engine uses magenta (255, 0, 255) and a black/purple checker for
        missing textures. Unity uses a similar magenta or cyan checker.
        """
        # Convert to float for channel arithmetic
        b = frame[:, :, 0].astype(np.float32)
        g = frame[:, :, 1].astype(np.float32)
        r = frame[:, :, 2].astype(np.float32)

        # Magenta: high R, low G, high B
        magenta_mask = (r > 180) & (g < 80) & (b > 180)
        magenta_frac = float(magenta_mask.mean())

        flags: list[Flag] = []
        if magenta_frac >= self.missing_texture_threshold:
            flags.append(
                Flag(
                    layer="layer1",
                    kind="MISSING_TEXTURE",
                    severity="high",
                    score=round(min(1.0, magenta_frac * 5), 4),
                    detail={"magenta_frac": round(magenta_frac, 4)},
                )
            )

        # Purple/black checker: alternating dark purple pixels
        purple_mask = (r > 80) & (r < 180) & (g < 50) & (b > 80) & (b < 180)
        purple_frac = float(purple_mask.mean())
        if purple_frac >= self.missing_texture_threshold:
            flags.append(
                Flag(
                    layer="layer1",
                    kind="MISSING_TEXTURE",
                    severity="high",
                    score=round(min(1.0, purple_frac * 5), 4),
                    detail={"purple_checker_frac": round(purple_frac, 4)},
                )
            )

        return flags


# Convenience: make the class a singleton-friendly instance factory
def make_layer1(cfg: dict | None = None) -> Layer1Detector:
    """Create a :class:`Layer1Detector` from a config dict (``layer1`` key)."""
    c = (cfg or {}).get("layer1", {})
    return Layer1Detector(
        bsod_blue_dominance=c.get("bsod_blue_dominance", 0.55),
        bsod_max_std=c.get("bsod_max_std", 45.0),
        black_mean_threshold=c.get("black_mean_threshold", 20.0),
        black_max_std=c.get("black_max_std", 30.0),
        frozen_ring_size=c.get("frozen_ring_size", 8),
        tearing_sobel_z_thresh=c.get("tearing_sobel_z_thresh", 4.0),
        missing_texture_threshold=c.get("missing_texture_threshold", 0.05),
    )
