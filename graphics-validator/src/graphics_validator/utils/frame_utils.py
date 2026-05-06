"""Frame utility functions: colour stats, dominant colour, edge/tearing helpers,
NaN-blob detector."""
from __future__ import annotations

import cv2
import numpy as np


def mean_bgr(frame: np.ndarray) -> tuple[float, float, float]:
    """Return per-channel mean (B, G, R) for *frame*."""
    m = frame.mean(axis=(0, 1))
    return float(m[0]), float(m[1]), float(m[2])


def frame_std(frame: np.ndarray) -> float:
    """Return the global pixel standard deviation of *frame*."""
    return float(frame.std())


def luminance_mean(frame: np.ndarray) -> float:
    """Return mean luminance (Y channel of YCrCb) for *frame*."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    return float(ycrcb[:, :, 0].mean())


def dominant_color(frame: np.ndarray, k: int = 3) -> np.ndarray:
    """Return the dominant BGR colour of *frame* via k-means clustering.

    Parameters
    ----------
    frame:
        BGR uint8 image.
    k:
        Number of colour clusters.

    Returns
    -------
    np.ndarray
        Shape ``(3,)`` BGR uint8 array of the most common cluster centre.
    """
    data = frame.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten())
    return centers[np.argmax(counts)].astype(np.uint8)


def row_edge_energy(frame: np.ndarray) -> np.ndarray:
    """Compute per-row Sobel-Y energy (mean absolute gradient) for *frame*.

    Returns
    -------
    np.ndarray
        1-D float32 array of shape ``(height,)``.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.abs(sobel).mean(axis=1)


def find_nan_blobs(
    frame: np.ndarray,
    min_frac: float = 0.01,
) -> list[dict]:
    """Find pure-white (255) or pure-black (0) connected blobs in *frame*.

    Parameters
    ----------
    frame:
        BGR uint8 image.
    min_frac:
        Minimum fraction of total pixels for a blob to be reported.

    Returns
    -------
    list[dict]
        Each entry has keys ``kind`` (``"white"`` or ``"black"``),
        ``area_px``, ``frac``, ``bbox`` (x, y, w, h).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    total = h * w
    results: list[dict] = []

    for val, kind in [(255, "white"), (0, "black")]:
        mask = np.uint8(gray == val) * 255
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            frac = area / total
            if frac >= min_frac:
                results.append(
                    {
                        "kind": kind,
                        "area_px": area,
                        "frac": round(frac, 4),
                        "bbox": (
                            int(stats[i, cv2.CC_STAT_LEFT]),
                            int(stats[i, cv2.CC_STAT_TOP]),
                            int(stats[i, cv2.CC_STAT_WIDTH]),
                            int(stats[i, cv2.CC_STAT_HEIGHT]),
                        ),
                    }
                )
    return results


def magenta_fraction(frame: np.ndarray) -> float:
    """Return the fraction of pixels that are magenta (R>180, G<80, B>180)."""
    b = frame[:, :, 0].astype(np.int32)
    g = frame[:, :, 1].astype(np.int32)
    r = frame[:, :, 2].astype(np.int32)
    mask = (r > 180) & (g < 80) & (b > 180)
    return float(mask.mean())
