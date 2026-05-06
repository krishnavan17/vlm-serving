"""Layer-2 temporal anomaly detectors.

Operates over a sliding window of sampled frames to detect:
- Flicker (FFT on per-frame luminance)
- Optical-flow anomalies (cv2.calcOpticalFlowFarneback)
- Embedding jumps (DINOv2, optional — lazy-imported)
"""
from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import numpy as np

from graphics_validator.detectors.base import Flag


class Layer2Detector:
    """Temporal anomaly detector using a sliding frame window.

    Parameters
    ----------
    window_seconds:
        Length of the analysis window in seconds.
    fps_sample:
        Frames per second at which the pipeline samples the video.
    flicker_min_hz / flicker_max_hz:
        Frequency band for flicker detection (Hz).
    flicker_z_thresh:
        Z-score threshold for spectral energy to trigger a flicker flag.
    flow_z_thresh:
        Z-score for optical-flow magnitude anomaly.
    embedding_cos_thresh:
        Cosine distance threshold for DINOv2 embedding jump.
    """

    def __init__(
        self,
        window_seconds: float = 2.0,
        fps_sample: float = 4.0,
        flicker_min_hz: float = 3.0,
        flicker_max_hz: float = 50.0,
        flicker_z_thresh: float = 3.5,
        flow_z_thresh: float = 3.0,
        embedding_cos_thresh: float = 0.25,
    ) -> None:
        self.fps_sample = fps_sample
        self.flicker_min_hz = flicker_min_hz
        self.flicker_max_hz = flicker_max_hz
        self.flicker_z_thresh = flicker_z_thresh
        self.flow_z_thresh = flow_z_thresh
        self.embedding_cos_thresh = embedding_cos_thresh

        win_len = max(4, int(round(window_seconds * fps_sample)))
        self._lum_window: deque[float] = deque(maxlen=win_len)
        self._prev_gray: np.ndarray | None = None
        self._flow_history: deque[float] = deque(maxlen=win_len * 4)

    # ------------------------------------------------------------------
    # Detector protocol
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray, idx: int, **kw: Any) -> list[Flag]:
        """Run all Layer-2 checks and return any flags raised."""
        flags: list[Flag] = []

        # Accumulate luminance for flicker FFT
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lum = float(gray.mean())
        self._lum_window.append(lum)

        flags.extend(self._check_flicker())
        flags.extend(self._check_optical_flow(gray))

        # Optional DINOv2 embedding jump — only if torch is available
        embedder = kw.get("embedder")
        if embedder is not None:
            prev_emb = kw.get("prev_embedding")
            flags.extend(self._check_embedding_jump(frame, embedder, prev_emb))

        self._prev_gray = gray
        return flags

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_flicker(self) -> list[Flag]:
        """FFT-based flicker detection over the luminance window."""
        if len(self._lum_window) < self._lum_window.maxlen:
            return []

        signal = np.array(self._lum_window, dtype=np.float32)
        signal -= signal.mean()

        freqs = np.fft.rfftfreq(len(signal), d=1.0 / self.fps_sample)
        spectrum = np.abs(np.fft.rfft(signal))

        band_mask = (freqs >= self.flicker_min_hz) & (freqs <= self.flicker_max_hz)
        if not band_mask.any():
            return []

        band_energy = float(spectrum[band_mask].max())
        all_energy = float(spectrum.mean()) + 1e-6
        z = band_energy / all_energy

        if z >= self.flicker_z_thresh:
            peak_hz = float(freqs[band_mask][np.argmax(spectrum[band_mask])])
            score = min(1.0, (z - self.flicker_z_thresh) / self.flicker_z_thresh + 0.5)
            return [
                Flag(
                    layer="layer2",
                    kind="FLICKER",
                    severity="medium",
                    score=round(score, 4),
                    detail={"peak_hz": round(peak_hz, 2), "band_z": round(z, 2)},
                )
            ]
        return []

    def _check_optical_flow(self, gray: np.ndarray) -> list[Flag]:
        """Optical-flow anomaly detection using Farneback dense flow."""
        if self._prev_gray is None:
            return []

        prev = self._prev_gray
        if prev.shape != gray.shape:
            return []

        flow = cv2.calcOpticalFlowFarneback(
            prev, gray,
            None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0,
        )
        magnitude = float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())
        self._flow_history.append(magnitude)

        if len(self._flow_history) < 4:
            return []

        hist = np.array(self._flow_history)
        mean_flow = float(hist.mean())
        std_flow = float(hist.std()) + 1e-6
        z = (magnitude - mean_flow) / std_flow

        if z >= self.flow_z_thresh:
            score = min(1.0, (z - self.flow_z_thresh) / self.flow_z_thresh + 0.5)
            return [
                Flag(
                    layer="layer2",
                    kind="FLOW_ANOMALY",
                    severity="medium",
                    score=round(score, 4),
                    detail={"flow_mag": round(magnitude, 3), "z": round(z, 2)},
                )
            ]
        return []

    def _check_embedding_jump(
        self,
        frame: np.ndarray,
        embedder: Any,
        prev_embedding: Any | None,
    ) -> list[Flag]:
        """Detect sudden jump in DINOv2 feature space (lazy torch import)."""
        try:
            import torch  # noqa: PLC0415

            curr_emb = embedder.embed(frame)
            if prev_embedding is None:
                return []

            cos_sim = float(
                torch.nn.functional.cosine_similarity(
                    curr_emb.unsqueeze(0), prev_embedding.unsqueeze(0)
                ).item()
            )
            cos_dist = 1.0 - cos_sim
            if cos_dist >= self.embedding_cos_thresh:
                score = min(1.0, cos_dist / self.embedding_cos_thresh)
                return [
                    Flag(
                        layer="layer2",
                        kind="EMBEDDING_JUMP",
                        severity="low",
                        score=round(score, 4),
                        detail={"cosine_distance": round(cos_dist, 4)},
                    )
                ]
        except Exception:
            pass
        return []


def make_layer2(cfg: dict | None = None, fps_sample: float = 4.0) -> Layer2Detector:
    """Create a :class:`Layer2Detector` from a config dict (``layer2`` key)."""
    c = (cfg or {}).get("layer2", {})
    return Layer2Detector(
        window_seconds=c.get("window_seconds", 2.0),
        fps_sample=fps_sample,
        flicker_min_hz=c.get("flicker_min_hz", 3.0),
        flicker_max_hz=c.get("flicker_max_hz", 50.0),
        flicker_z_thresh=c.get("flicker_z_thresh", 3.5),
        flow_z_thresh=c.get("flow_z_thresh", 3.0),
        embedding_cos_thresh=c.get("embedding_cos_thresh", 0.25),
    )
