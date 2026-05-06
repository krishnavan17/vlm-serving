"""Layer-3 reference comparison detector.

Compares a test frame against the corresponding golden (reference) frame using:
- SSIM (always available via scikit-image)
- LPIPS (optional, try/except import)
- NVIDIA FLIP (optional, try/except import, falls back to SSIM)
- PatchCore on DINOv2 features (stub, TODO)

Only active when a golden video source is provided to the pipeline.
"""
from __future__ import annotations

import warnings
from typing import Any

import cv2
import numpy as np

from graphics_validator.detectors.base import Flag
from graphics_validator.utils.logging import get_logger

log = get_logger(__name__)

# Optional imports
try:
    from skimage.metrics import structural_similarity as _ssim

    _SKIMAGE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKIMAGE_AVAILABLE = False
    log.warning("scikit-image not available — SSIM disabled.")

try:
    import lpips as _lpips_module  # noqa: PLC0415

    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False
    log.warning("lpips not installed — falling back to SSIM for perceptual similarity.")

try:
    import flip  # type: ignore  # noqa: PLC0415

    _FLIP_AVAILABLE = True
except ImportError:
    _FLIP_AVAILABLE = False
    # nvidia-flip is optional — no warning needed (it's commented out in requirements.txt)


class Layer3Detector:
    """Reference-based perceptual difference detector.

    Parameters
    ----------
    ssim_min:
        SSIM below this value triggers a flag.
    lpips_max:
        LPIPS above this value triggers a flag (when library available).
    """

    def __init__(self, ssim_min: float = 0.80, lpips_max: float = 0.30) -> None:
        self.ssim_min = ssim_min
        self.lpips_max = lpips_max
        self._lpips_fn: Any = None

        if _LPIPS_AVAILABLE:
            try:
                self._lpips_fn = _lpips_module.LPIPS(net="alex", verbose=False)
            except Exception as exc:
                log.warning(f"Could not initialise LPIPS model: {exc}")

    # ------------------------------------------------------------------
    # Detector protocol
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        idx: int,
        golden_frame: np.ndarray | None = None,
        **kw: Any,
    ) -> list[Flag]:
        """Compare *frame* against *golden_frame* and return perceptual-diff flags."""
        if golden_frame is None:
            return []

        # Resize golden to match test frame if needed
        if frame.shape != golden_frame.shape:
            golden_frame = cv2.resize(golden_frame, (frame.shape[1], frame.shape[0]))

        flags: list[Flag] = []
        flags.extend(self._check_ssim(frame, golden_frame))
        flags.extend(self._check_lpips(frame, golden_frame))
        return flags

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_ssim(self, frame: np.ndarray, golden: np.ndarray) -> list[Flag]:
        if not _SKIMAGE_AVAILABLE:
            return []

        gray_test = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_golden = cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY)
        score = float(_ssim(gray_test, gray_golden, data_range=255))

        if score < self.ssim_min:
            severity = "high" if score < 0.6 else "medium"
            return [
                Flag(
                    layer="layer3",
                    kind="PERCEPTUAL_DIFF",
                    severity=severity,  # type: ignore[arg-type]
                    score=round(1.0 - score, 4),
                    detail={"ssim": round(score, 4), "metric": "ssim"},
                )
            ]
        return []

    def _check_lpips(self, frame: np.ndarray, golden: np.ndarray) -> list[Flag]:
        if self._lpips_fn is None:
            return []

        try:
            import torch  # noqa: PLC0415

            def _to_tensor(img: np.ndarray) -> Any:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
                return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)

            with torch.no_grad():
                dist = float(self._lpips_fn(_to_tensor(frame), _to_tensor(golden)).item())

            if dist > self.lpips_max:
                severity = "high" if dist > 0.5 else "medium"
                return [
                    Flag(
                        layer="layer3",
                        kind="PERCEPTUAL_DIFF",
                        severity=severity,  # type: ignore[arg-type]
                        score=round(min(1.0, dist), 4),
                        detail={"lpips": round(dist, 4), "metric": "lpips"},
                    )
                ]
        except Exception as exc:
            log.warning(f"LPIPS check failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# PatchCore stub
# ---------------------------------------------------------------------------


class PatchCoreDetector:
    """PatchCore anomaly detector trained on DINOv2 features extracted from the golden video.

    TODO: Implement full PatchCore pipeline:
        1. ``fit(golden_frames)``:
           - Extract DINOv2 patch features for every frame in the golden video.
           - Build a coreset (random subsampling or greedy k-center) of the patch
             feature memory bank.
        2. ``score(frame)``:
           - Extract DINOv2 patch features for the test frame.
           - For each patch, compute the nearest-neighbour distance to the memory bank.
           - Return the maximum patch distance as the anomaly score.
        3. Set a threshold on validation data to separate normal / anomalous frames.

    Reference:
        Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022.
        https://arxiv.org/abs/2106.08265
    """

    def fit(self, golden_frames: list[np.ndarray]) -> None:
        """Fit the memory bank on golden frames.

        TODO: implement with DINOv2 feature extractor and k-center coreset.
        """
        raise NotImplementedError("PatchCore.fit is not yet implemented.")

    def detect(self, frame: np.ndarray, idx: int, **kw: Any) -> list[Flag]:
        """Score *frame* against the fitted memory bank.

        TODO: implement with DINOv2 feature extractor and NN distance.
        """
        raise NotImplementedError("PatchCore.detect is not yet implemented.")


def make_layer3(cfg: dict | None = None) -> Layer3Detector:
    """Create a :class:`Layer3Detector` from a config dict (``layer3`` key)."""
    c = (cfg or {}).get("layer3", {})
    return Layer3Detector(
        ssim_min=c.get("ssim_min", 0.80),
        lpips_max=c.get("lpips_max", 0.30),
    )
