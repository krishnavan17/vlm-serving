"""VideoSource: cv2.VideoCapture iterator with FPS subsampling."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoSource:
    """Wraps cv2.VideoCapture and subsamples to *fps_sample* frames per second.

    Parameters
    ----------
    path:
        Path to a video file (or integer camera index).
    fps_sample:
        Target sampling rate. Frames whose timestamps don't fall on a
        multiple of ``1/fps_sample`` seconds are skipped.
    """

    def __init__(self, path: Path | str | int, fps_sample: float = 4.0) -> None:
        self._path = path
        self.fps_sample = fps_sample
        self._cap = cv2.VideoCapture(str(path) if not isinstance(path, int) else path)
        if not self._cap.isOpened():
            raise OSError(f"Cannot open video source: {path!r}")
        self._native_fps: float = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_frames: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._step: int = max(1, round(self._native_fps / self.fps_sample))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def native_fps(self) -> float:
        return self._native_fps

    @property
    def frame_count(self) -> int:
        """Approximate number of *sampled* frames."""
        return max(1, self._total_frames // self._step)

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def timestamp(self, idx: int) -> float:
        """Return wall-clock timestamp (seconds) for sampled frame *idx*."""
        return (idx * self._step) / self._native_fps

    def frame_at(self, idx: int) -> np.ndarray:
        """Seek and return the *idx*-th sampled frame (BGR uint8)."""
        native_idx = idx * self._step
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, native_idx)
        ok, frame = self._cap.read()
        if not ok:
            raise IndexError(f"Cannot read frame at sampled index {idx}")
        return frame

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(sampled_idx, frame_bgr)`` tuples."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        native_idx = 0
        sampled_idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            if native_idx % self._step == 0:
                yield sampled_idx, frame
                sampled_idx += 1
            native_idx += 1

    def __len__(self) -> int:
        return self.frame_count

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
